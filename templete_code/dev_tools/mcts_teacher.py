#!/usr/bin/env python3
"""Full MCTS-style offline teacher for policy data generation.

Dev-only. This module must not be imported by submission runtime code.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithm import BoxInput, Candidate, PalletConfig, Palletizer
from evaluate import buffer_bonus_from_capacity


ActionKey = Tuple[int, int, int, int]


@dataclass
class MCTSConfig:
    num_simulations: int = 128
    max_depth: int = 30
    c_puct: float = 1.5
    prior: str = "uniform"
    rollout_policy: str = "heuristic"
    rollout_max_steps: int = 100
    use_baseline_delta: bool = True
    baseline_policy: str = "heuristic"
    max_actions_per_node: int = 64
    time_budget_sec_per_decision: float = 2.0
    fallback_on_timeout: str = "heuristic"
    random_seed: int = 0


@dataclass
class MCTSAction:
    key: ActionKey
    buffer_index: int
    box_id: int
    rotation: int
    x_index: int
    y_index: int
    x: float
    y: float
    z: float
    candidate: Candidate
    prior: float = 0.0
    heuristic_score: float = 0.0


@dataclass
class MCTSState:
    palletizer: Palletizer
    buffer: List[BoxInput]
    source: List[BoxInput]
    next_index: int
    done: bool = False

    def clone(self) -> "MCTSState":
        return MCTSState(
            palletizer=copy.deepcopy(self.palletizer),
            buffer=[dict(box) for box in self.buffer],
            source=self.source,
            next_index=int(self.next_index),
            done=bool(self.done),
        )

    def indexed_buffer(self) -> List[Tuple[int, BoxInput]]:
        return list(enumerate(self.buffer))

    def consume(self, buffer_index: int) -> None:
        if not self.buffer:
            self.done = True
            return
        self.buffer.pop(int(buffer_index))
        if self.next_index < len(self.source):
            self.buffer.append(dict(self.source[self.next_index]))
            self.next_index += 1
        if not self.buffer and self.next_index >= len(self.source):
            self.done = True

    def apply(self, action: MCTSAction) -> None:
        self.palletizer._place_candidate(action.candidate)
        self.consume(action.buffer_index)


@dataclass
class MCTSNode:
    state: MCTSState
    parent: Optional["MCTSNode"] = None
    action_from_parent: Optional[MCTSAction] = None
    prior: float = 1.0
    n: int = 0
    w: float = 0.0
    q: float = 0.0
    children: Dict[ActionKey, "MCTSNode"] = field(default_factory=dict)
    untried_actions: Optional[List[MCTSAction]] = None

    def is_expanded(self) -> bool:
        return self.untried_actions is not None

    def backup(self, value: float) -> None:
        self.n += 1
        self.w += float(value)
        self.q = self.w / max(self.n, 1)


@dataclass
class MCTSDecision:
    best_action: Optional[MCTSAction]
    mcts_policy: np.ndarray
    visit_counts: np.ndarray
    q_values: np.ndarray
    rollout_return: float
    baseline_return: float
    delta_return: float
    root_actions: List[MCTSAction]
    timed_out: bool


def config_from_dict(root: Dict[str, Any]) -> MCTSConfig:
    cfg = MCTSConfig()
    data = root.get("mcts", {})
    return MCTSConfig(
        num_simulations=int(data.get("num_simulations", cfg.num_simulations)),
        max_depth=int(data.get("max_depth", cfg.max_depth)),
        c_puct=float(data.get("c_puct", cfg.c_puct)),
        prior=str(data.get("prior", cfg.prior)),
        rollout_policy=str(data.get("rollout_policy", cfg.rollout_policy)),
        rollout_max_steps=int(data.get("rollout_max_steps", cfg.rollout_max_steps)),
        use_baseline_delta=bool(data.get("use_baseline_delta", cfg.use_baseline_delta)),
        baseline_policy=str(data.get("baseline_policy", cfg.baseline_policy)),
        max_actions_per_node=int(data.get("max_actions_per_node", cfg.max_actions_per_node)),
        time_budget_sec_per_decision=float(
            data.get("time_budget_sec_per_decision", cfg.time_budget_sec_per_decision)
        ),
        fallback_on_timeout=str(data.get("fallback_on_timeout", cfg.fallback_on_timeout)),
        random_seed=int(data.get("random_seed", cfg.random_seed)),
    )


def initial_state(palletizer: Palletizer, boxes: Sequence[BoxInput]) -> MCTSState:
    capacity = max(int(palletizer.algo.buffer_size), 0)
    source = [dict(box) for box in boxes]
    if capacity == 0:
        buffer = [dict(source[0])] if source else []
        next_index = 1 if source else 0
    else:
        buffer = [dict(box) for box in source[:capacity]]
        next_index = min(capacity, len(source))
    return MCTSState(
        palletizer=copy.deepcopy(palletizer),
        buffer=buffer,
        source=source,
        next_index=next_index,
        done=not bool(buffer),
    )


def utilization_score(state: MCTSState) -> float:
    pallet = state.palletizer.pallet
    volume = sum(float(np.prod(item["size"])) for item in state.palletizer.sequence)
    pallet_volume = pallet.length * pallet.width * pallet.height
    return (volume / pallet_volume * 100.0) if pallet_volume > 0.0 else 0.0


def final_score(state: MCTSState) -> float:
    return utilization_score(state) + buffer_bonus_from_capacity(state.palletizer.algo.buffer_size)


def grid_aligned(value: float, resolution: float) -> Optional[int]:
    idx = int(round(float(value) / float(resolution)))
    if abs(idx * resolution - value) <= 1e-6:
        return idx
    return None


def generate_actions(state: MCTSState, cfg: MCTSConfig) -> List[MCTSAction]:
    if state.done or not state.buffer:
        return []

    palletizer = state.palletizer
    actions: List[MCTSAction] = []
    seen: set[ActionKey] = set()

    for buffer_index, box in state.indexed_buffer():
        for dims, rotation in palletizer._candidate_orientations(box["size"]):
            xs = palletizer._axis_positions(palletizer.pallet.length, dims[0])
            ys = palletizer._axis_positions(palletizer.pallet.width, dims[1])
            rotation_index = palletizer._rotation_index_for_degrees(rotation)
            for y in ys:
                y_index = grid_aligned(y, palletizer.grid_resolution)
                if y_index is None or y_index < 0 or y_index >= palletizer.n_rows:
                    continue
                for x in xs:
                    x_index = grid_aligned(x, palletizer.grid_resolution)
                    if x_index is None or x_index < 0 or x_index >= palletizer.n_cols:
                        continue
                    candidate = palletizer._evaluate_candidate(
                        box=box,
                        buffer_index=buffer_index,
                        dims=dims,
                        rotation=rotation,
                        x=x,
                        y=y,
                    )
                    if candidate is None:
                        continue
                    key = (buffer_index, rotation_index, x_index, y_index)
                    if key in seen:
                        continue
                    seen.add(key)
                    actions.append(MCTSAction(
                        key=key,
                        buffer_index=buffer_index,
                        box_id=int(box["id"]),
                        rotation=int(rotation),
                        x_index=x_index,
                        y_index=y_index,
                        x=float(x),
                        y=float(y),
                        z=float(candidate.z),
                        candidate=candidate,
                        heuristic_score=float(candidate.score),
                    ))

    actions.sort(key=lambda action: action.heuristic_score, reverse=True)
    if cfg.max_actions_per_node > 0:
        actions = actions[: cfg.max_actions_per_node]
    prior = 1.0 / len(actions) if actions else 0.0
    for action in actions:
        action.prior = prior
    return actions


def heuristic_action(state: MCTSState) -> Optional[MCTSAction]:
    actions = generate_actions(state, MCTSConfig(max_actions_per_node=1))
    return actions[0] if actions else None


def rollout(state: MCTSState, cfg: MCTSConfig, rng: random.Random) -> float:
    sim = state.clone()
    steps = 0
    while not sim.done and steps < cfg.rollout_max_steps:
        actions = generate_actions(sim, cfg)
        if not actions:
            sim.done = True
            break
        if cfg.rollout_policy == "random-valid":
            action = rng.choice(actions)
        elif cfg.rollout_policy == "mixed" and rng.random() < 0.2:
            action = rng.choice(actions)
        else:
            action = actions[0]
        sim.apply(action)
        steps += 1
    return utilization_score(sim)


def select_child(node: MCTSNode, cfg: MCTSConfig) -> MCTSNode:
    parent_n = max(node.n, 1)
    best_score = -float("inf")
    best_child: Optional[MCTSNode] = None
    for child in node.children.values():
        u = cfg.c_puct * child.prior * math.sqrt(parent_n) / (1 + child.n)
        score = child.q + u
        if score > best_score:
            best_score = score
            best_child = child
    if best_child is None:
        raise RuntimeError("select_child called with no children")
    return best_child


def expand(node: MCTSNode, cfg: MCTSConfig) -> Optional[MCTSNode]:
    if node.untried_actions is None:
        node.untried_actions = generate_actions(node.state, cfg)
        if not node.untried_actions:
            node.state.done = True
            return None
    if not node.untried_actions:
        return None

    action = node.untried_actions.pop(0)
    child_state = node.state.clone()
    child_state.apply(action)
    child = MCTSNode(
        state=child_state,
        parent=node,
        action_from_parent=action,
        prior=action.prior,
    )
    node.children[action.key] = child
    return child


def backup(path: Sequence[MCTSNode], value: float) -> None:
    for node in path:
        node.backup(value)


def dense_stats(root: MCTSNode, state: MCTSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (
        max(int(state.palletizer.algo.buffer_size), len(state.buffer), 1),
        2,
        state.palletizer.n_rows,
        state.palletizer.n_cols,
    )
    visits = np.zeros(shape, dtype=np.float32)
    q_values = np.zeros(shape, dtype=np.float32)
    policy = np.zeros(shape, dtype=np.float32)
    total_visits = 0.0
    for key, child in root.children.items():
        b, r, x, y = key
        visits[b, r, y, x] = float(child.n)
        q_values[b, r, y, x] = float(child.q)
        total_visits += float(child.n)
    if total_visits > 0.0:
        policy = visits / total_visits
    return policy, visits, q_values


def decide(root_state: MCTSState, cfg: MCTSConfig) -> MCTSDecision:
    rng = random.Random(cfg.random_seed)
    root = MCTSNode(state=root_state.clone(), prior=1.0)
    start = time.perf_counter()
    timed_out = False
    last_value = utilization_score(root_state)
    last_baseline = last_value

    for _ in range(max(1, cfg.num_simulations)):
        if time.perf_counter() - start > cfg.time_budget_sec_per_decision:
            timed_out = True
            break

        node = root
        path = [node]
        depth = 0

        while (
            node.is_expanded()
            and not node.untried_actions
            and node.children
            and not node.state.done
            and depth < cfg.max_depth
        ):
            node = select_child(node, cfg)
            path.append(node)
            depth += 1

        if not node.state.done and depth < cfg.max_depth:
            child = expand(node, cfg)
            if child is not None:
                node = child
                path.append(node)

        rollout_return = rollout(node.state, cfg, rng)
        baseline_return = 0.0
        if cfg.use_baseline_delta:
            baseline_return = rollout(node.state, MCTSConfig(
                rollout_policy="heuristic",
                rollout_max_steps=cfg.rollout_max_steps,
                max_actions_per_node=1,
            ), rng)
            value = rollout_return - baseline_return
        else:
            value = rollout_return
        last_value = rollout_return
        last_baseline = baseline_return
        backup(path, value)

    policy, visits, q_values = dense_stats(root, root_state)
    best_child: Optional[MCTSNode] = None
    if root.children:
        best_child = max(root.children.values(), key=lambda child: (child.n, child.q))
    elif cfg.fallback_on_timeout == "heuristic":
        fallback = heuristic_action(root_state)
        if fallback is not None:
            # Build a one-hot root policy for fallback-only decisions.
            b, r, x, y = fallback.key
            visits[b, r, y, x] = 1.0
            policy[b, r, y, x] = 1.0
            q_values[b, r, y, x] = utilization_score(root_state)
            return MCTSDecision(
                best_action=fallback,
                mcts_policy=policy,
                visit_counts=visits,
                q_values=q_values,
                rollout_return=last_value,
                baseline_return=last_baseline,
                delta_return=last_value - last_baseline,
                root_actions=[fallback],
                timed_out=timed_out,
            )

    root_actions = [
        child.action_from_parent
        for child in root.children.values()
        if child.action_from_parent is not None
    ]
    return MCTSDecision(
        best_action=None if best_child is None else best_child.action_from_parent,
        mcts_policy=policy,
        visit_counts=visits,
        q_values=q_values,
        rollout_return=last_value,
        baseline_return=last_baseline,
        delta_return=last_value - last_baseline,
        root_actions=root_actions,
        timed_out=timed_out,
    )


def state_from_existing(
    palletizer: Palletizer,
    boxes: Sequence[BoxInput],
    buffer: Sequence[BoxInput],
    next_index: int,
) -> MCTSState:
    return MCTSState(
        palletizer=copy.deepcopy(palletizer),
        buffer=[dict(box) for box in buffer],
        source=[dict(box) for box in boxes],
        next_index=int(next_index),
        done=not bool(buffer),
    )
