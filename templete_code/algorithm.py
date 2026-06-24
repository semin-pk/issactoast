from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import yaml

from buffer_manager import BufferManager


# ---------------------------------------------------------------------------
# 입출력 스키마  (수정 금지)
# ---------------------------------------------------------------------------

class BoxInput(TypedDict):
    step: int
    id: int
    size: List[float]   # [length, width, height]
    mass: float


class PlacedBox(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float
    position: List[float]
    rotation: int       # 0 또는 90


class RunResult(TypedDict):
    buffer_size: int
    sequence: List[PlacedBox]

    # 더 이상 적재 가능한 박스를 찾지 못해 자동 종료된 경우 True
    terminated: bool
    terminated_step: Optional[int]

    # 참가자 알고리즘이 명시적으로 적재 종료를 선언한 경우 True
    finished_by_user: bool


# ---------------------------------------------------------------------------
# 설정 dataclass  (수정 금지)
# ---------------------------------------------------------------------------

@dataclass
class PalletConfig:
    length: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# 참가자 개발 영역
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    allow_rotation: bool
    buffer_size: int


@dataclass
class HeuristicWeights:
    """후보 배치 점수 계산에 사용하는 가중치 묶음."""

    w_height: float = 1.0
    w_support: float = 2.0
    w_contact: float = 0.7
    w_flat: float = 0.4
    w_mass: float = 0.15


@dataclass
class HeuristicConfig:
    """YAML의 heuristic 섹션을 코드에서 쓰기 쉬운 형태로 보관한다."""

    grid_resolution_m: float = 0.02
    candidate_step_m: float = 0.02
    support_threshold: float = 0.8
    support_z_tol_m: float = 0.003
    max_consecutive_failures: int = 40
    min_remaining_height_m: float = 0.03
    weights: HeuristicWeights = field(default_factory=HeuristicWeights)


@dataclass
class Candidate:
    """평가를 통과한 배치 후보 하나.

    x, y, z는 출력용 중심 좌표가 아니라 팔레트 좌하단 기준의
    bottom-left-bottom 좌표이다. 출력 JSON으로 저장할 때만 중심 좌표로 바꾼다.
    """

    score: float
    buffer_index: int
    box: BoxInput
    dims: Tuple[float, float, float]
    rotation: int
    x: float
    y: float
    z: float
    support_ratio: float
    cell_slice: Tuple[slice, slice]


@dataclass
class PlacedAABB:
    """이미 배치한 박스의 실제 3D 경계 상자.

    heightmap은 빠른 후보 평가용 격자 근사이고, 이 AABB 목록은 충돌 검사와
    실제 지지 면적 검증에 사용한다.
    """

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float


class Palletizer:
    """
    Heightmap + 점수 기반 Best-Fit 팔레타이저.

    팔레트 바닥을 2D 격자로 나누고 각 셀의 현재 윗면 높이를 유지한다.
    매 스텝마다 버퍼 안의 박스와 허용 회전, 후보 좌표를 모두 점수화한 뒤
    안정성 하드 제약을 통과한 최고 후보 하나를 배치한다.
    """

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg
        self.heuristic = self._load_heuristic_config()
        self._reset_state()

    def _reset_state(self) -> None:
        """한 번의 run() 실행에서 사용하는 모든 내부 상태를 초기화한다."""

        self.grid_resolution = max(float(self.heuristic.grid_resolution_m), 0.005)
        self.n_cols = int(np.ceil(self.pallet.length / self.grid_resolution))
        self.n_rows = int(np.ceil(self.pallet.width / self.grid_resolution))

        # heightmap[row, col]은 해당 X/Y 셀에서 현재 가장 높은 윗면 z 값이다.
        # row는 Y축, col은 X축에 대응한다. 초기에는 팔레트 바닥 높이 0이다.
        self.heightmap = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)

        self.sequence: List[PlacedBox] = []
        self._placed_aabbs: List[PlacedAABB] = []
        self._consecutive_failures = 0

        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False

    # -----------------------------------------------------------------------
    # 설정 로드
    # -----------------------------------------------------------------------

    def _load_heuristic_config(self) -> HeuristicConfig:
        """config/algorithm_config.yaml에서 heuristic 값을 읽는다.

        main.py의 AlgorithmConfig 생성 코드는 기존 구조를 유지해야 하므로,
        추가 튜닝 파라미터는 Palletizer가 직접 YAML을 읽어 로드한다.
        YAML에 특정 키가 없으면 dataclass 기본값을 사용한다.
        """

        cfg = HeuristicConfig()
        config_path = Path(__file__).resolve().parent / "config" / "algorithm_config.yaml"

        if not config_path.exists():
            return cfg

        with config_path.open("r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}

        heuristic = root.get("heuristic", {})
        weights = heuristic.get("weights", {})

        cfg.grid_resolution_m = float(
            heuristic.get("grid_resolution_m", cfg.grid_resolution_m)
        )
        cfg.candidate_step_m = float(
            heuristic.get("candidate_step_m", cfg.candidate_step_m)
        )
        cfg.support_threshold = float(
            heuristic.get("support_threshold", cfg.support_threshold)
        )
        cfg.support_z_tol_m = float(
            heuristic.get("support_z_tol_m", cfg.support_z_tol_m)
        )
        cfg.max_consecutive_failures = int(
            heuristic.get("max_consecutive_failures", cfg.max_consecutive_failures)
        )
        cfg.min_remaining_height_m = float(
            heuristic.get("min_remaining_height_m", cfg.min_remaining_height_m)
        )
        cfg.weights = HeuristicWeights(
            w_height=float(weights.get("w_height", cfg.weights.w_height)),
            w_support=float(weights.get("w_support", cfg.weights.w_support)),
            w_contact=float(weights.get("w_contact", cfg.weights.w_contact)),
            w_flat=float(weights.get("w_flat", cfg.weights.w_flat)),
            w_mass=float(weights.get("w_mass", cfg.weights.w_mass)),
        )
        return cfg

    # -----------------------------------------------------------------------
    # 참가자 수정 가능 함수
    # -----------------------------------------------------------------------

    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        """안정성 관점에서 더 진행하지 않을지 판단한다.

        이 함수가 True를 반환하면 run()은 finished_by_user=True로 끝난다.
        현재 구현은 다음 두 경우에 종료한다.

        1. heightmap의 최고점 기준으로 남은 높이가 현재 버퍼의 가장 낮은 박스보다 작다.
        2. 유효 배치를 찾지 못한 실패가 설정값만큼 연속 발생했다.
        """

        if not current_buffer:
            return False

        remaining_height = self.pallet.height - float(np.max(self.heightmap))
        smallest_box_height = min(float(box["size"][2]) for box in current_buffer)

        if remaining_height < min(smallest_box_height, self.heuristic.min_remaining_height_m):
            return True

        return self._consecutive_failures >= self.heuristic.max_consecutive_failures

    # -----------------------------------------------------------------------
    # 후보 생성 및 점수화
    # -----------------------------------------------------------------------

    def _candidate_orientations(
        self,
        size: List[float],
    ) -> List[Tuple[Tuple[float, float, float], int]]:
        """박스 하나에 대해 가능한 바닥면 회전 후보를 만든다.

        문제 조건상 Z축 기준 0도/90도 회전만 허용된다. 90도 회전은
        length와 width를 바꾸고 height는 그대로 둔다.
        """

        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

        if not self.algo.allow_rotation or abs(sx - sy) < 1e-9:
            return [((sx, sy, sz), 0)]

        return [
            ((sx, sy, sz), 0),
            ((sy, sx, sz), 90),
        ]

    def _axis_positions(self, limit: float, span: float) -> List[float]:
        """한 축에서 후보 시작 좌표 목록을 생성한다.

        모든 실수 좌표를 탐색할 수는 없으므로 candidate_step_m 간격으로
        시작 좌표를 만든다. 마지막 위치는 팔레트 끝에 딱 붙일 수 있도록
        max_start를 별도로 추가한다.
        """

        max_start = limit - span
        if max_start < -1e-9:
            return []

        step = max(float(self.heuristic.candidate_step_m), self.grid_resolution)
        count = int(np.floor((max_start + 1e-9) / step))
        values = [round(i * step, 6) for i in range(count + 1)]

        if not values or abs(values[-1] - max_start) > 1e-6:
            values.append(round(max_start, 6))

        return values

    def _cell_slice(self, x: float, y: float, dx: float, dy: float) -> Tuple[slice, slice]:
        """실제 meter 좌표의 footprint를 heightmap slice로 변환한다.

        heightmap은 격자 근사이므로 footprint에 조금이라도 걸치는 셀을 포함한다.
        이 보수적인 변환 덕분에 높은 셀을 놓치지 않고 z_place를 계산할 수 있다.
        """

        col0 = max(0, int(np.floor((x + 1e-9) / self.grid_resolution)))
        row0 = max(0, int(np.floor((y + 1e-9) / self.grid_resolution)))
        col1 = min(self.n_cols, int(np.ceil((x + dx - 1e-9) / self.grid_resolution)))
        row1 = min(self.n_rows, int(np.ceil((y + dy - 1e-9) / self.grid_resolution)))
        return slice(row0, row1), slice(col0, col1)

    def _aabb_intersects_existing(
        self,
        x: float,
        y: float,
        z: float,
        dims: Tuple[float, float, float],
    ) -> bool:
        """후보 박스가 기존 박스와 3D 공간에서 겹치는지 검사한다.

        heightmap의 z_place 계산으로 대부분의 수직 침투는 방지되지만,
        격자 근사나 반올림 오차를 대비해 실제 AABB 기준으로 한 번 더 확인한다.
        면이 맞닿는 것은 정상 접촉이므로 겹침으로 보지 않는다.
        """

        dx, dy, dz = dims
        x1, y1, z1 = x + dx, y + dy, z + dz
        eps = 1e-6

        for placed in self._placed_aabbs:
            overlaps = (
                x < placed.x1 - eps and x1 > placed.x0 + eps
                and y < placed.y1 - eps and y1 > placed.y0 + eps
                and z < placed.z1 - eps and z1 > placed.z0 + eps
            )
            if overlaps:
                return True

        return False

    def _contact_ratio(
        self,
        x: float,
        y: float,
        z: float,
        dims: Tuple[float, float, float],
    ) -> float:
        """팔레트 벽 또는 이웃 박스 측면과 맞닿는 정도를 계산한다.

        값은 0~1 범위의 둘레 비율이다. 벽에 붙거나 같은 높이대의 박스와
        측면 접촉이 많은 배치를 선호하면 빈 공간이 줄고 안정성이 좋아진다.
        """

        dx, dy, dz = dims
        x1, y1, z1 = x + dx, y + dy, z + dz
        perimeter = 2.0 * (dx + dy)
        if perimeter <= 0.0:
            return 0.0

        contact = 0.0
        tol = max(self.heuristic.support_z_tol_m, 1e-4)

        if x <= tol:
            contact += dy
        if y <= tol:
            contact += dx
        if abs(x1 - self.pallet.length) <= tol:
            contact += dy
        if abs(y1 - self.pallet.width) <= tol:
            contact += dx

        for placed in self._placed_aabbs:
            z_overlap = min(z1, placed.z1) - max(z, placed.z0)
            if z_overlap <= tol:
                continue

            y_overlap = max(0.0, min(y1, placed.y1) - max(y, placed.y0))
            x_overlap = max(0.0, min(x1, placed.x1) - max(x, placed.x0))

            if y_overlap > tol and (
                abs(x1 - placed.x0) <= tol or abs(x - placed.x1) <= tol
            ):
                contact += y_overlap

            if x_overlap > tol and (
                abs(y1 - placed.y0) <= tol or abs(y - placed.y1) <= tol
            ):
                contact += x_overlap

        return min(contact / perimeter, 1.0)

    def _exact_support_ratio(
        self,
        x: float,
        y: float,
        z: float,
        dims: Tuple[float, float, float],
    ) -> float:
        """실제 AABB overlap 면적 기준 지지율을 계산한다.

        heightmap 셀 비율은 빠르지만 격자 근사라서 실제 접촉 면적보다
        약간 낙관적일 수 있다. 최종 후보 판단과 결과 검증에서는 이 값을
        함께 사용해 공중부양이나 모서리 걸침을 줄인다.
        """

        dx, dy, _ = dims
        footprint_area = dx * dy
        if footprint_area <= 0.0:
            return 0.0

        if z <= self.heuristic.support_z_tol_m:
            return 1.0

        support_area = 0.0
        x1 = x + dx
        y1 = y + dy

        for placed in self._placed_aabbs:
            if abs(placed.z1 - z) > self.heuristic.support_z_tol_m:
                continue

            overlap_x = max(0.0, min(x1, placed.x1) - max(x, placed.x0))
            overlap_y = max(0.0, min(y1, placed.y1) - max(y, placed.y0))
            support_area += overlap_x * overlap_y

        return min(support_area / footprint_area, 1.0)

    def _score_candidate(
        self,
        box: BoxInput,
        dims: Tuple[float, float, float],
        x: float,
        y: float,
        z: float,
        support_ratio: float,
        region: np.ndarray,
    ) -> float:
        """하드 제약을 통과한 후보의 선호도를 점수화한다.

        점수가 높을수록 좋은 후보이다. 안정성을 위해 지지율과 낮은 높이를
        강하게 반영하고, 같은 수준의 후보 사이에서는 벽/이웃 접촉과 평탄한
        영역을 선호한다. mass 항은 무거운 박스를 낮게 배치하도록 돕는다.
        """

        dx, dy, dz = dims
        weights = self.heuristic.weights
        top_z = z + dz
        contact = self._contact_ratio(x, y, z, dims)

        if region.size <= 1:
            flatness = 1.0
        else:
            flatness = 1.0 - min(float(np.std(region)) / self.pallet.height, 1.0)

        mass = float(box.get("mass", 0.0))
        mass_term = mass * (1.0 - min(z / self.pallet.height, 1.0))

        return (
            -weights.w_height * top_z
            + weights.w_support * support_ratio
            + weights.w_contact * contact
            + weights.w_flat * flatness
            + weights.w_mass * mass_term
        )

    def _evaluate_candidate(
        self,
        box: BoxInput,
        buffer_index: int,
        dims: Tuple[float, float, float],
        rotation: int,
        x: float,
        y: float,
    ) -> Optional[Candidate]:
        """박스 하나의 특정 회전/좌표 후보가 배치 가능한지 평가한다.

        순서:
        1. 팔레트 X/Y 경계 검사
        2. footprint heightmap 영역에서 z_place 계산
        3. 높이 제한 검사
        4. 셀 기반 지지율과 실제 면적 기반 지지율 검사
        5. 기존 박스와 AABB 충돌 검사
        6. 점수 계산 후 Candidate 반환
        """

        dx, dy, dz = dims

        if (
            x < -1e-9
            or y < -1e-9
            or x + dx > self.pallet.length + 1e-9
            or y + dy > self.pallet.width + 1e-9
        ):
            return None

        row_slice, col_slice = self._cell_slice(x, y, dx, dy)
        region = self.heightmap[row_slice, col_slice]

        if region.size == 0:
            return None

        z = float(np.max(region))
        if z + dz > self.pallet.height + 1e-9:
            return None

        # footprint 안에서 가장 높은 높이 z와 같은 높이인 셀이 실제로 박스를
        # 받쳐주는 셀이다. 이 비율이 낮으면 박스가 한쪽 모서리에 걸친 상태가 된다.
        support_mask = np.abs(region - z) <= self.heuristic.support_z_tol_m
        cell_support_ratio = float(np.count_nonzero(support_mask) / region.size)
        exact_support_ratio = self._exact_support_ratio(x, y, z, dims)
        support_ratio = min(cell_support_ratio, exact_support_ratio)

        if support_ratio + 1e-9 < self.heuristic.support_threshold:
            return None

        if self._aabb_intersects_existing(x, y, z, dims):
            return None

        score = self._score_candidate(box, dims, x, y, z, support_ratio, region)

        return Candidate(
            score=score,
            buffer_index=buffer_index,
            box=box,
            dims=dims,
            rotation=rotation,
            x=x,
            y=y,
            z=z,
            support_ratio=support_ratio,
            cell_slice=(row_slice, col_slice),
        )

    def _find_position(
        self,
        box: BoxInput,
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
        """단일 박스에 대한 최적 위치를 찾는 호환용 helper.

        기존 코드가 가지고 있던 함수 이름을 유지한다. 실제 run()에서는
        버퍼 전체를 한 번에 평가하는 _best_candidate()를 직접 사용한다.
        """

        best = self._best_candidate([(0, box)])
        if best is None:
            return None
        return best.x, best.y, best.z, best.dims, best.rotation

    def _best_candidate(self, indexed_boxes: List[Tuple[int, BoxInput]]) -> Optional[Candidate]:
        """현재 버퍼 전체에서 가장 좋은 후보 하나를 찾는다."""

        best: Optional[Candidate] = None

        for buffer_index, box in indexed_boxes:
            for dims, rotation in self._candidate_orientations(box["size"]):
                dx, dy, _ = dims
                xs = self._axis_positions(self.pallet.length, dx)
                ys = self._axis_positions(self.pallet.width, dy)

                for y in ys:
                    for x in xs:
                        candidate = self._evaluate_candidate(
                            box=box,
                            buffer_index=buffer_index,
                            dims=dims,
                            rotation=rotation,
                            x=x,
                            y=y,
                        )

                        if candidate is None:
                            continue

                        if best is None or self._is_better_candidate(candidate, best):
                            best = candidate

        return best

    def _is_better_candidate(self, candidate: Candidate, best: Candidate) -> bool:
        """두 후보를 비교한다.

        1차 기준은 score이다. score가 사실상 같으면 bottom-left 원칙을 적용해
        작은 y, 작은 x, 작은 z, 더 앞쪽 버퍼 인덱스 순으로 선택한다.
        """

        eps = 1e-9
        if candidate.score > best.score + eps:
            return True
        if abs(candidate.score - best.score) > eps:
            return False

        # 동점은 bottom-left, 더 낮은 위치, 먼저 온 버퍼 순으로 정한다.
        cand_key = (
            round(candidate.y, 6),
            round(candidate.x, 6),
            round(candidate.z, 6),
            candidate.buffer_index,
        )
        best_key = (
            round(best.y, 6),
            round(best.x, 6),
            round(best.z, 6),
            best.buffer_index,
        )
        return cand_key < best_key

    # -----------------------------------------------------------------------
    # 배치/검증
    # -----------------------------------------------------------------------

    def _append_placed(
        self,
        box: BoxInput,
        dims: Tuple[float, float, float],
        rotation: int,
        x: float,
        y: float,
        z: float,
    ) -> None:
        """선택된 후보를 결과 sequence와 내부 상태에 반영한다.

        입력 x/y/z는 bottom-left-bottom 좌표지만, 출력 JSON의 position은
        기존 규격대로 박스 중심 좌표이다. size와 position은 main.py 출력
        규칙에 맞춰 소수점 3자리로 반올림한다.
        """

        dx, dy, dz = dims

        self.sequence.append({
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": [
                round(dx, 3),
                round(dy, 3),
                round(dz, 3),
            ],
            "mass": float(box["mass"]),
            "position": [
                round(x + dx / 2.0, 3),
                round(y + dy / 2.0, 3),
                round(z + dz / 2.0, 3),
            ],
            "rotation": int(rotation),
        })

        self._placed_aabbs.append(
            PlacedAABB(
                x0=x,
                y0=y,
                z0=z,
                x1=x + dx,
                y1=y + dy,
                z1=z + dz,
            )
        )

        row_slice, col_slice = self._cell_slice(x, y, dx, dy)
        # 후보 footprint 영역의 새 윗면 높이는 박스 top z이다.
        # 이후 박스는 이 갱신된 heightmap을 기준으로 다시 z_place를 계산한다.
        self.heightmap[row_slice, col_slice] = z + dz

    def _place_candidate(self, candidate: Candidate) -> None:
        """최고 후보를 실제 배치하고 연속 실패 카운터를 초기화한다."""

        self._append_placed(
            box=candidate.box,
            dims=candidate.dims,
            rotation=candidate.rotation,
            x=candidate.x,
            y=candidate.y,
            z=candidate.z,
        )
        self._consecutive_failures = 0

    def _assert_valid_result(self) -> None:
        """run() 종료 후 결과의 기본 물리/기하 제약을 자체 검증한다.

        이 검증은 제출 포맷을 바꾸지 않고 콘솔에만 OK를 출력한다.
        assert가 실패하면 알고리즘 내부의 배치 판단과 실제 결과 사이에
        불일치가 있다는 뜻이므로 즉시 수정해야 한다.
        """

        eps = 1e-6
        support_threshold = self.heuristic.support_threshold

        for i, placed in enumerate(self._placed_aabbs):
            assert placed.x0 >= -eps
            assert placed.y0 >= -eps
            assert placed.z0 >= -eps
            assert placed.x1 <= self.pallet.length + eps
            assert placed.y1 <= self.pallet.width + eps
            assert placed.z1 <= self.pallet.height + eps

            support_area = 0.0
            footprint_area = (placed.x1 - placed.x0) * (placed.y1 - placed.y0)

            if placed.z0 <= self.heuristic.support_z_tol_m:
                support_area = footprint_area
            else:
                for below in self._placed_aabbs[:i]:
                    if abs(below.z1 - placed.z0) > self.heuristic.support_z_tol_m:
                        continue
                    overlap_x = max(0.0, min(placed.x1, below.x1) - max(placed.x0, below.x0))
                    overlap_y = max(0.0, min(placed.y1, below.y1) - max(placed.y0, below.y0))
                    support_area += overlap_x * overlap_y

            support_ratio = support_area / footprint_area if footprint_area > 0.0 else 0.0
            assert support_ratio + 1e-9 >= support_threshold

            for other in self._placed_aabbs[i + 1:]:
                overlaps = (
                    placed.x0 < other.x1 - eps and placed.x1 > other.x0 + eps
                    and placed.y0 < other.y1 - eps and placed.y1 > other.y0 + eps
                    and placed.z0 < other.z1 - eps and placed.z1 > other.z0 + eps
                )
                assert not overlaps

        max_top = max((box.z1 for box in self._placed_aabbs), default=0.0)
        print(
            "[CHECK] OK "
            f"placed_count={len(self.sequence)} "
            f"max_top_height={max_top:.3f} "
            f"support_threshold={support_threshold:.2f}"
        )

    def run(self, boxes: List[BoxInput]) -> RunResult:
        """전체 박스 시퀀스에 대해 팔레타이징을 수행한다.

        run() 시그니처와 반환 구조는 평가 시스템과 연결되므로 유지한다.
        내부적으로는 BufferManager에서 현재 후보 창을 가져오고, 매 스텝마다
        버퍼 전체의 최고 점수 후보를 하나 배치한다.
        """

        self._reset_state()

        buf = BufferManager(self.algo.buffer_size)
        buf.reset(boxes)

        while buf.has_pending():
            if self.algo.buffer_size == 0:
                current = [buf.peek_next()]
            else:
                current = buf.get_buffer()

            # 높이 부족 또는 연속 실패 누적처럼 알고리즘이 직접 종료를 선언하는
            # 경우이다. 이때는 자동 실패 종료가 아니므로 terminated는 False로 둔다.
            if self.should_finish(current):
                self.finished_by_user = True
                self.terminated_step = None
                break

            indexed = list(enumerate(current))
            best = self._best_candidate(indexed)

            if best is not None:
                self._place_candidate(best)

                # buffer_size=0이면 peek_next()로 본 박스를 pop_next()로 소비한다.
                # buffer를 쓰는 경우에는 최고 후보의 buffer_index를 선택 소비한다.
                if self.algo.buffer_size == 0:
                    buf.pop_next()
                else:
                    buf.pop_selected(best.buffer_index)

                continue

            self._consecutive_failures += 1

            # 버퍼가 있으면 현재 창을 전부 검사했으므로 가장 앞 박스 하나를
            # 소비해 다음 입력을 노출한다. buffer_size=0도 같은 방식으로 skip한다.
            if self.algo.buffer_size == 0:
                buf.pop_next()
            else:
                buf.pop_selected(0)

            if self.should_finish(current):
                self.finished = True
                if current:
                    self.terminated_step = int(current[0]["step"])
                break

        self._assert_valid_result()

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }
