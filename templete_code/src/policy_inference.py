from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnxruntime as ort


PolicyAction = Tuple[int, int, int, int, float]


class PolicyModel:
    """Small ONNX policy wrapper.

    Inputs:
      - height_map_input: [N, 1, H, W]
      - buffer_input: [N, B, F]
      - action_mask_input: [N, B, 2, H, W]

    Output:
      - logits: [N, B, 2, H, W]
    """

    def __init__(self, model_path: str | Path, top_k: int = 32) -> None:
        self.model_path = str(model_path)
        self.top_k = max(1, int(top_k))
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]

    def _input_feed(
        self,
        height_map: np.ndarray,
        buffer_features: np.ndarray,
        action_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        height = np.asarray(height_map, dtype=np.float32)
        buffer_arr = np.asarray(buffer_features, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=np.float32)

        if height.ndim == 3:
            height = height[None, ...]
        if buffer_arr.ndim == 2:
            buffer_arr = buffer_arr[None, ...]
        if mask.ndim == 4:
            mask = mask[None, ...]

        by_known_name = {
            "height_map_input": height,
            "buffer_input": buffer_arr,
            "action_mask_input": mask,
        }
        if all(name in by_known_name for name in self.input_names):
            return {name: by_known_name[name] for name in self.input_names}

        values = [height, buffer_arr, mask]
        return {name: values[idx] for idx, name in enumerate(self.input_names[:3])}

    def recommend(
        self,
        *,
        height_map: np.ndarray,
        buffer_features: np.ndarray,
        action_mask: np.ndarray,
        top_k: int | None = None,
    ) -> List[PolicyAction]:
        mask = np.asarray(action_mask, dtype=np.float32)
        if mask.ndim != 4:
            raise ValueError(f"action_mask must be [B,2,H,W], got shape={mask.shape}")
        if float(np.sum(mask)) <= 0.0:
            return []

        outputs = self.session.run(
            None,
            self._input_feed(height_map, buffer_features, mask),
        )
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim == 5:
            logits = logits[0]
        if logits.shape != mask.shape:
            raise ValueError(f"logits shape {logits.shape} does not match mask {mask.shape}")

        masked_logits = np.where(mask > 0.0, logits, -1.0e9)
        flat = masked_logits.reshape(-1)
        k = min(max(1, int(top_k or self.top_k)), flat.size)
        top_indices = np.argpartition(flat, -k)[-k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

        actions: List[PolicyAction] = []
        for flat_index in top_indices:
            if flat[flat_index] <= -1.0e8:
                continue
            buffer_index, rotation_index, y_index, x_index = np.unravel_index(
                int(flat_index),
                masked_logits.shape,
            )
            actions.append((
                int(buffer_index),
                int(rotation_index),
                int(y_index),
                int(x_index),
                float(flat[flat_index]),
            ))
        return actions


def top_k_from_logits(
    logits: np.ndarray,
    action_mask: np.ndarray,
    top_k: int,
) -> List[PolicyAction]:
    """Utility used by dev tools/tests without loading ONNX."""

    mask = np.asarray(action_mask, dtype=np.float32)
    masked = np.where(mask > 0.0, np.asarray(logits, dtype=np.float32), -1.0e9)
    flat = masked.reshape(-1)
    k = min(max(1, int(top_k)), flat.size)
    top_indices = np.argpartition(flat, -k)[-k:]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

    actions: List[PolicyAction] = []
    for flat_index in top_indices:
        if flat[flat_index] <= -1.0e8:
            continue
        b, r, y, x = np.unravel_index(int(flat_index), masked.shape)
        actions.append((int(b), int(r), int(y), int(x), float(flat[flat_index])))
    return actions
