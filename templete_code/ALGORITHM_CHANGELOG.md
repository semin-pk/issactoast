# Palletizing Algorithm 변경 기록

이 문서는 `algorithm.py`의 알고리즘 변경 사항과 현재 구현을 이해하기 위한 설명을 기록한다.
알고리즘을 수정할 때마다 아래의 "변경 이력" 섹션에 날짜, 목적, 핵심 변경 사항, 검증 결과를 계속 추가한다.

## 변경 이력

### 2026-07-04 - Stage 8 ONNX 정책 채택 검증

#### 변경 목적

Stage 5의 `final_heuristic_v2` baseline과 비교해 새 ONNX 정책망을 채택할지 결정했다.

#### 변경 내용

- `policy_inference.enabled: true`로 전환했다.
- `models/policy_net.onnx`를 사용해 `onnx_policy_holdout_v2` holdout 평가를 실행했다.
- 제출 경로의 금지 import(`torch`, `pybullet`, `optuna`, `cma`)를 확인했다.

#### 검증 결과

| label | physics_mode | mean_score | worst_score | fail_rate | mean_runtime_sec | mean_utilization_pct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `final_heuristic_v2` | pybullet | 55.497069 | 0.000000 | 0.050000 | 17.443906 | 42.693819 |
| `onnx_policy_holdout_v2` | geom | 59.154477 | 45.838596 | 0.000000 | 19.283605 | 43.154477 |

ONNX 평가에서 max runtime은 24.994초로 90초 미만이었다. 정책 inference는 총 1144회 시도 중 1131회 성공, fallback 13회로 fallback rate는 약 1.1%였다. 금지 import 스캔 결과는 비어 있었다. 채택 기준을 만족하므로 ONNX 정책을 채택한다.

### 2026-07-04 - Stage 7 정책망 재학습 및 ONNX export

#### 변경 목적

`buffer.size: 4` shape와 호환되는 정책망을 재학습하고 ONNX 모델을 갱신했다.

#### 변경 내용

- `data/mcts_dataset/train_sku_mcts_aug.npz`와 `valid_sku_mcts_aug.npz`로 `dev_tools/train_policy.py`를 실행했다.
- 첫 export는 Windows CP949 콘솔 인코딩 문제로 실패했으나, `PYTHONIOENCODING=utf-8`로 재실행해 ONNX export를 완료했다.
- 회귀 체크로 주요 Python 파일을 `py_compile`했다.

#### 검증 결과

UTF-8 재실행 기준 주요 validation 로그:

| epoch | valid_ce | top1 | top5 |
| ---: | ---: | ---: | ---: |
| 34 | 4.2036 | 0.179 | 0.436 |
| 35 | 4.2233 | 0.171 | 0.424 |
| 36 | 4.1948 | 0.180 | 0.431 |
| 37 | 4.2379 | 0.176 | 0.432 |
| 38 | 4.2038 | 0.179 | 0.435 |
| 39 | 4.2037 | 0.180 | 0.433 |

Best checkpoint는 `valid_ce=4.1948`인 epoch 36이며, `models/policy_net.onnx`와 `models/policy_net.onnx.data`가 생성되었다. `py_compile`은 오류 없이 통과했다.

### 2026-07-04 - Stage 6 MCTS 데이터셋 증강 갱신

#### 변경 목적

Stage 1~5에서 확정된 `buffer.size: 4` 기준으로 정책망 학습 데이터의 shape 호환성을 확인하고, 학습/검증 증강 데이터셋을 다시 생성했다.

#### 변경 내용

- `data/mcts_dataset/train_sku_mcts.npz`, `valid_sku_mcts.npz`의 주요 tensor shape를 확인했다.
  - `buffer_features`: `(N, 4, 6)`
  - `action_mask`: `(N, 4, 2, 50, 60)`
  - `mcts_policy`: `(N, 4, 2, 50, 60)`
- 원 지시의 full MCTS 재수집(`--num-simulations 128 --max-sequences 50`)은 현재 환경에서 6시간 이상 완료되지 않아 중단했다.
- 기존 raw MCTS 데이터가 확정 buffer size와 shape가 맞아 이를 기준으로 증강을 다시 수행했다.

#### 검증 결과

| artifact | samples | 결과 |
| --- | ---: | --- |
| `data/mcts_dataset/train_sku_mcts_aug.npz` | 2972 | `AssertionError` 없이 생성 |
| `data/mcts_dataset/valid_sku_mcts_aug.npz` | 1096 | `AssertionError` 없이 생성 |

### 2026-07-04 - Stage 5 최종 휴리스틱 baseline 수립

#### 변경 목적

Stage 1~4에서 확정한 휴리스틱 config를 기준으로 PyBullet strict 최종 baseline을 재수립하고, 이후 ONNX 정책 채택 여부 판단 기준으로 고정했다.

#### 변경 내용

- 확정 config:
  - `buffer.size: 4`
  - `policy_inference.enabled: false`
  - `heuristic.min_remaining_height_m: 0.015`
  - `heuristic.max_consecutive_failures: 40`
  - `physics_mask.load_safety_margin: 0.90`
  - `heuristic.candidate_step_m: 0.02`
- `evaluate.py --refresh-results --physics --physics-strict --label final_heuristic_v2`로 holdout_sku 20개를 재평가했다.

#### 검증 결과

| label | physics_mode | mean_score | worst_score | fail_rate | mean_runtime_sec | mean_utilization_pct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `final_heuristic_v2` | pybullet | 55.497069 | 0.000000 | 0.050000 | 17.443906 | 42.693819 |

### 2026-07-04 - Stage 4 candidate_step_m 해상도 스윕

#### 변경 목적

후보 좌표 스캔 해상도를 0.02m보다 촘촘하게 만들 때 점수 개선이 있는지 확인하고, 점수와 런타임의 균형을 비교했다.

#### 변경 내용

- Stage 1~3 확정 config를 기준으로 `candidate_step_m` 후보 0.02, 0.015, 0.01을 각각 평가했다.
- 세 후보의 `mean_score`, `worst_score`, `fail_rate`가 모두 동일했다.
- 더 촘촘한 해상도는 런타임만 소폭 증가했으므로 `candidate_step_m: 0.02`를 유지했다.

#### 검증 결과

| label | mean_score | worst_score | fail_rate | mean_runtime_sec |
| --- | ---: | ---: | ---: | ---: |
| `step_tune_0_02` | 58.693819 | 45.348719 | 0.000000 | 16.571230 |
| `step_tune_0_015` | 58.693819 | 45.348719 | 0.000000 | 16.742166 |
| `step_tune_0_01` | 58.693819 | 45.348719 | 0.000000 | 16.812937 |

### 2026-07-03 - Stage 3 physics_mask 안전마진 검증

#### 변경 목적

휴리스틱의 `physics_mask` 안전마진이 과도하게 보수적인지 확인하되, geom 점수만이 아니라 PyBullet strict 검증을 함께 적용해 물리 실패 위험을 통제했다.

#### 변경 내용

- Stage 2 확정 config를 baseline으로 사용했다.
- `com_margin_m`, `corner_tolerance_m`, `min_supported_corners`, `min_supported_edges`, `load_safety_margin` 후보를 하나씩만 변경했다.
- 각 후보마다 geom 평가는 `--refresh-results`, strict 평가는 같은 결과에 `--physics --physics-strict`를 적용했다.
- `load_safety_margin: 0.90`만 geom `mean_score`가 상승했고 strict `fail_rate`, `worst_score`가 baseline보다 나빠지지 않아 채택했다.
- `com_margin_m`, `corner_tolerance_m`, `min_supported_corners`, `min_supported_edges`, `load_safety_margin: 0.95`는 채택 기준 미달로 원복했다.

#### 검증 결과

| label | geom_mean_score | strict_worst_score | strict_fail_rate | 판정 |
| --- | ---: | ---: | ---: | --- |
| `mask_tune_com_margin_0_003` | 58.298415 | 0.000000 | 0.050000 | 기각: geom 개선 없음 |
| `mask_tune_com_margin_0_008` | 58.298415 | 0.000000 | 0.050000 | 기각: geom 개선 없음 |
| `mask_tune_corner_tol_0_02` | 58.298415 | 45.348719 | 0.000000 | 기각: geom 개선 없음 |
| `mask_tune_corner_tol_0_04` | 58.298415 | 0.000000 | 0.050000 | 기각: geom 개선 없음 |
| `mask_tune_min_corners_1` | 58.298415 | 0.000000 | 0.050000 | 기각: geom 개선 없음 |
| `mask_tune_min_edges_0` | 58.298415 | 0.000000 | 0.050000 | 기각: geom 개선 없음 |
| `mask_tune_load_margin_0_90` | 58.693819 | 0.000000 | 0.050000 | 채택 |
| `mask_tune_load_margin_0_95` | 58.187873 | 0.000000 | 0.050000 | 기각: geom 점수 하락 |

### 2026-07-03 - Stage 2 종료조건 튜닝

#### 변경 목적

`should_finish()`가 남은 높이 또는 연속 실패 조건 때문에 조기에 종료되는지 확인하고, holdout_sku 기준 점수를 해치지 않는 범위에서 종료조건을 완화했다.

#### 변경 내용

- Stage 1 확정값인 `buffer.size: 4`와 `policy_inference.enabled: false`를 유지했다.
- `min_remaining_height_m` 후보 0.02, 0.015와 `max_consecutive_failures` 후보 60, 80을 각각 하나씩 변경해 평가했다.
- `min_remaining_height_m: 0.015`만 `mean_score`를 개선하고 `worst_score`, `fail_rate`를 유지했으므로 채택했다.
- `max_consecutive_failures`는 40을 유지했다.

#### 검증 결과

| label | mean_score | worst_score | fail_rate | mean_runtime_sec |
| --- | ---: | ---: | ---: | ---: |
| `stop_tune_baseline_b4` | 57.799909 | 45.348719 | 0.000000 | 16.990853 |
| `stop_tune_min_remaining_0_02` | 57.799909 | 45.348719 | 0.000000 | 16.856657 |
| `stop_tune_min_remaining_0_015` | 58.298415 | 45.348719 | 0.000000 | 17.240209 |
| `stop_tune_max_failures_60` | 57.799909 | 45.348719 | 0.000000 | 16.665060 |
| `stop_tune_max_failures_80` | 57.799909 | 45.348719 | 0.000000 | 16.547663 |
| `stop_tune_final_min_remaining_0_015` | 58.298415 | 45.348719 | 0.000000 | 16.655644 |

Outlier seed 9002/9007/9011의 utilization은 각각 46.95%, 37.10%, 29.35%로 유지되었다. 개선은 주로 seed 9009, 9016, 9017에서 추가 배치가 가능해진 데서 발생했다.

### 2026-07-03 - Stage 1 버퍼 크기 스윕

#### 변경 목적

holdout_sku 20개(seed 9000~9019) 기준으로 `buffer.size` 후보 4, 6, 8, 10, 12, 14, 16, 18, 20을 비교해, utilization 증가와 buffer bonus 감소를 함께 반영한 `mean_score` 최적점을 확정했다.

#### 변경 내용

- Stage 0 지시에 따라 `policy_inference.enabled`를 `false`로 변경해 휴리스틱만 평가했다.
- 각 후보마다 `evaluate.py --refresh-results --results algorithm_results --label buffer_sweep_<N> --seed-set holdout_sku --bounds-tol 0.001 --epsilon 0.0011`로 결과 JSON을 갱신한 뒤 평가했다.
- 최신 실행 기준 최고 `mean_score`는 `buffer_sweep_4`였으므로 `config/algorithm_config.yaml`의 `buffer.size`를 4로 확정했다.

#### 검증 결과

| label | mean_score | worst_score | fail_rate | mean_runtime_sec |
| --- | ---: | ---: | ---: | ---: |
| `buffer_sweep_4` | 57.799909 | 45.348719 | 0.000000 | 16.826443 |
| `buffer_sweep_6` | 56.167238 | 40.505036 | 0.000000 | 26.555574 |
| `buffer_sweep_8` | 54.942920 | 42.234553 | 0.000000 | 35.203147 |
| `buffer_sweep_10` | 52.965917 | 31.736029 | 0.000000 | 43.255980 |
| `buffer_sweep_12` | 53.829194 | 31.941854 | 0.000000 | 54.328468 |
| `buffer_sweep_14` | 52.375253 | 37.377151 | 0.000000 | 66.091435 |
| `buffer_sweep_16` | 50.913412 | 35.637011 | 0.000000 | 76.665843 |
| `buffer_sweep_18` | 48.170656 | 33.637011 | 0.000000 | 86.572603 |
| `buffer_sweep_20` | 46.844879 | 27.958597 | 0.000000 | 97.727608 |

### 2026-06-24 - 평가 도구 실행 가이드 문서 추가

#### 변경 목적

`evaluate.py`와 `physics_check.py`의 실행 방법, 옵션 의미, 결과 확인 방법을 한 곳에서 볼 수 있도록 별도 가이드 문서를 추가했다.

#### 변경 내용

- `EVALUATION_GUIDE.md`를 새로 추가했다.
- 다음 내용을 정리했다.
  - 가장 많이 쓰는 `evaluate.py --refresh-results --physics` 실행 명령
  - `evaluate.py` 단독 평가, refresh 평가, PyBullet 평가, 버퍼 시각화 평가 방법
  - `physics_check.py` 단일 파일/디렉토리 검증 방법
  - GUI에서 버퍼를 보는 방법
  - 주요 옵션 설명
  - 콘솔 표, `benchmark_log.csv`, `eval_report.json` 확인 방법
  - `buffer.size` 변경 시 stale result를 피하는 방법
  - 추천 워크플로우와 한계
- `README.md`의 프로젝트 구조 표에 `EVALUATION_GUIDE.md` 항목을 추가했다.

#### 검증 결과

문서 추가 작업이라 코드 실행 결과는 변하지 않는다. 문서 위치:

```text
EVALUATION_GUIDE.md
```

### 2026-06-24 - evaluate.py에 --refresh-results 추가

#### 변경 목적

`algorithm_config.yaml`에서 `buffer.size`를 변경해도 `algorithm_results/*.json`은 자동으로 바뀌지 않는다. 기존 `evaluate.py`는 이미 생성된 결과 JSON만 읽었기 때문에, config 변경 후 `main.py`를 먼저 실행하지 않으면 예전 `buffer_size` 기준 결과를 평가했다.

사용자가 `main.py`와 `evaluate.py`를 따로 실행하지 않고, 하나의 명령으로 "현재 config 기준 결과 생성 + 평가"를 수행할 수 있도록 `evaluate.py`에 refresh 옵션을 추가했다.

#### 변경 내용

- `evaluate.py`에 `--refresh-results` 옵션을 추가했다.
- 이 옵션을 사용하면 `evaluate.py`가 먼저 같은 디렉토리의 `main.py`를 현재 Python 인터프리터로 실행한다.
- `main.py` 실행이 끝난 뒤 갱신된 `algorithm_results/*.json`을 평가한다.
- 옵션 없이 실행할 때는 기존처럼 결과 JSON만 읽는다.
- 옵션 없이 실행했는데 config의 `buffer.size`와 결과 JSON의 `buffer_size`가 다르면 stale result 경고를 출력한다.
- `evaluate.py` docstring에 기본 동작과 `--refresh-results` 동작 차이를 명시했다.

#### 검증 결과

현재 config:

```yaml
buffer:
  size: 4
```

검증 명령:

```bash
.venv/bin/python evaluate.py --refresh-results --results algorithm_results --label "refresh_results_test" --bounds-tol 0.001 --epsilon 0.0011 --no-report-json
```

결과:

- `evaluate.py`가 먼저 `main.py`를 실행함.
- `main.py` 출력에서 `buffer_size: 4` 확인.
- 결과 JSON 평가도 `buffer=4`로 수행됨.

| 입력 파일 | placed_count | utilization_percent | buffer_bonus | final_score | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| `box_sequence_0.json` | 60 | 43.11% | 16.00 | 59.11 | PASS |
| `box_sequence_1.json` | 62 | 47.87% | 16.00 | 63.87 | PASS |

앞으로 config 변경 후 한 번에 실행하려면 다음 명령을 사용한다.

```bash
.venv/bin/python evaluate.py --refresh-results --results algorithm_results --label "my_run" --physics --physics-show-buffer --bounds-tol 0.001 --epsilon 0.0011
```

### 2026-06-24 - PyBullet 버퍼 시각화 추가 및 buffer.size=16 결과 갱신

#### 변경 목적

PyBullet GUI에서 팔레트뿐 아니라 버퍼 영역에 대기 중인 박스들도 보고 싶다는 요청을 반영했다. 또한 `config/algorithm_config.yaml`의 `buffer.size`를 변경했는데 결과 JSON에는 이전 값이 남아 있던 문제를 확인했다.

#### 변경 내용

- `physics_check.py`에 `--show-buffer` 옵션을 추가했다.
- `evaluate.py`에 `--physics-show-buffer` 옵션을 추가했다.
- 버퍼 시각화는 결과 JSON의 `sequence`를 기준으로 개발용 sliding-window staging buffer를 구성한다.
  - 초기 `buffer_size`개 박스를 팔레트 오른쪽 버퍼 플랫폼 슬롯에 생성한다.
  - 배치 순서에 따라 버퍼 슬롯의 박스를 팔레트 목표 위치로 옮긴다.
  - 비워진 슬롯에는 다음 예정 박스를 refill한다.
- 결과 JSON에는 원본 입력에서 스킵된 박스 정보가 없으므로, 이 버퍼 시각화는 공식 버퍼 전체를 완벽히 재현하는 것이 아니라 "출력 sequence 기준의 예정 배치 버퍼"를 보여주는 개발용 뷰이다.
- `--gui-step-delay` 옵션을 추가해 GUI에서 각 physics step 사이에 지연을 줄 수 있게 했다.
- `buffer.size` 변경 후 결과 JSON이 그대로였던 이유를 확인했다.
  - `algorithm_results/*.json`은 `main.py`를 다시 실행해야 갱신된다.
  - 기존 결과는 `buffer_size=8`로 생성된 stale output이었다.
  - 현재 active config는 `templete_code/config/algorithm_config.yaml`이며, 여기에 `buffer.size: 16`이 설정되어 있다.

#### 검증 결과

현재 config 확인:

```yaml
buffer:
  size: 16
```

`main.py` 재실행:

```bash
.venv/bin/python main.py
```

결과 JSON 확인:

| 입력 파일 | result buffer_size | placed_count | utilization_percent | max_top_height | 자체 검증 |
| --- | ---: | ---: | ---: | ---: | --- |
| `box_sequence_0.json` | 16 | 88 | 60.44% | 1.2380 | OK |
| `box_sequence_1.json` | 16 | 71 | 53.67% | 1.2380 | OK |

PyBullet 버퍼 시각화 경로 검증:

```bash
.venv/bin/python physics_check.py --result algorithm_results/box_sequence_0.json --show-buffer
```

결과:

- PASS
- `success_rate=100.00%`
- `max_drift=0.0077`
- `max_top=1.2380`

`evaluate.py` 연동 검증:

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "v3_buffer16_pybullet_buffer_view" --physics --physics-show-buffer --bounds-tol 0.001 --epsilon 0.0011
```

결과:

| 입력 파일 | placed_count | utilization_percent | buffer_bonus | final_score | physics_mode | 판정 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `box_sequence_0.json` | 88 | 60.44% | 4.00 | 64.44 | pybullet | PASS |
| `box_sequence_1.json` | 71 | 53.67% | 4.00 | 57.67 | pybullet | PASS |

평균:

- `mean_final_score=61.06`
- `pass_rate=100.00%`
- `mean_utilization=57.06%`
- `mean_buffer_bonus=4.00`

GUI로 버퍼를 보려면 다음처럼 실행한다.

```bash
.venv/bin/python physics_check.py --result algorithm_results/box_sequence_0.json --gui --show-buffer --gui-step-delay 0.02
```

### 2026-06-24 - .venv Python 3.12 전환 및 PyBullet 물리 검증 실행

#### 변경 목적

기존 `.venv`가 Python 3.14라 PyBullet wheel이 없고 소스 빌드도 실패했다. 사용자의 요청에 따라 `.venv`를 Python 3.12 기반으로 다시 만들고, PyBullet 기반 물리 검증을 실제 실행했다.

#### 변경 내용

- 기존 Python 3.14 `.venv`는 `.venv_py314_backup`으로 백업했다.
- `/Users/parksemin/.pyenv/versions/3.12.0/bin/python`으로 새 `.venv`를 생성했다.
- `requirements.txt`의 기존 의존성을 새 `.venv`에 설치했다.
- `requirements.txt`는 수정하지 않았다.
- `pybullet`은 로컬 검증용으로만 `.venv`에 설치했다.
- macOS arm64 환경에서 PyPI binary wheel이 없어 소스 빌드가 필요했다.
- PyBullet 3.2.7 소스의 bundled zlib이 macOS SDK의 `fdopen` 선언과 충돌해 빌드가 실패했다.
- `/tmp`에 받은 PyBullet 소스에서 `examples/ThirdPartyLibs/zlib/zutil.h`의 macOS `fdopen` 매크로 정의를 비활성화하고, 로컬 wheel을 빌드해 설치했다.
- 이 패치는 `/tmp`의 설치용 소스에만 적용했으며, 프로젝트 제출 코드나 requirements에는 반영하지 않았다.

#### 검증 결과

환경 확인:

```bash
.venv/bin/python --version
```

결과: `Python 3.12.0`

의존성 확인:

```bash
.venv/bin/python - <<'PY'
import numpy, yaml, pybullet
print(numpy.__version__)
PY
```

결과:

- `numpy 2.4.3`
- `PyYAML` import OK
- `pybullet` import OK

문법 검사:

```bash
.venv/bin/python -m py_compile algorithm.py evaluate.py physics_check.py main.py
```

결과: 통과.

PyBullet 단독 검증:

```bash
.venv/bin/python physics_check.py --result algorithm_results/box_sequence_0.json
```

결과:

- `pallet_pass=True`
- `success_rate=100.00%`
- `max_drift=0.0067`
- `max_top=1.2380`

`evaluate.py --physics` 연동 검증:

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "v2_pybullet" --physics --bounds-tol 0.001 --epsilon 0.0011
```

결과:

| 입력 파일 | placed_count | utilization_percent | buffer_bonus | final_score | physics_mode | 판정 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `box_sequence_0.json` | 69 | 48.50% | 12.00 | 60.50 | pybullet | PASS |
| `box_sequence_1.json` | 67 | 51.08% | 12.00 | 63.08 | pybullet | PASS |

평균:

- `mean_final_score=61.79`
- `pass_rate=100.00%`
- `mean_utilization=49.79%`

새 Python 3.12 `.venv`에서 `main.py`도 다시 실행했다.

```bash
.venv/bin/python main.py
```

결과:

| 입력 파일 | placed_count | utilization_percent | max_top_height | 자체 검증 |
| --- | ---: | ---: | ---: | --- |
| `box_sequence_0.json` | 69 | 48.50% | 1.2380 | OK |
| `box_sequence_1.json` | 67 | 51.08% | 1.2380 | OK |

이후 재생성된 결과에 대해 PyBullet 평가를 다시 실행했다.

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "v2_pybullet_after_main_312" --physics --bounds-tol 0.001 --epsilon 0.0011
```

결과는 동일하게 두 파일 모두 PASS, `mean_final_score=61.79`이다.

더미 실패 검증:

- 공중에 둔 박스와 팔레트 밖 박스를 포함한 임시 JSON을 생성해 검사했다.
- `DROP/COLLAPSE drift=0.7000 >= 0.4000`
- `OUT_OF_BOUNDS`
- 최종 점수 0점 처리 확인.
- 임시 테스트 폴더는 검증 후 삭제했다.

### 2026-06-24 - PyBullet 물리 검증기 physics_check.py 추가 및 evaluate.py 연동

#### 변경 목적

기하학적 지지율 검사는 빠르지만 실제 물리 붕괴나 드랍을 완전히 대체하지 못한다. 공식 Isaac Sim 평가의 로컬 근사 프록시로 PyBullet 기반 물리 검증기를 추가하고, `evaluate.py --physics` 옵션으로 최종 점수 계산에 결합할 수 있게 했다.

#### 변경 내용

- `physics_check.py`를 새로 추가했다.
- `algorithm.py`와 `main.py`는 수정하지 않았고, 제출 런타임에서 `physics_check.py`를 import하지 않는다.
- `pybullet`은 로컬 개발용 optional dependency로만 취급한다. `requirements.txt`에는 추가하지 않았다.
- `palletizing_simulator/config/sim_config.yaml`에서 다음 값을 읽어 사용한다.
  - `pallet.size`
  - `settling.max_steps`, `min_frames`, `velocity_threshold`, `final_steps`, `drop_offset`
  - `physics.box`, `physics.ground`, `physics.pallet`
  - `evaluation.drift_threshold_m`, `bounds_tolerance_m`, `episode_success_min_rate`
- PyBullet 환경은 `DIRECT` 모드 기본이며, `--gui` 옵션으로 GUI 실행을 요청할 수 있게 했다.
- 팔레트 상면이 z=0이 되도록 팔레트 고정 박스를 `z=-thickness/2`에 생성한다.
- 결과 JSON의 `position`은 centroid로 사용하고, spawn 시 z에 `drop_offset`을 더한다.
- 결과 JSON의 `size`는 이미 회전 반영된 world 크기이므로, `rotation=90`일 때 PyBullet local shape의 X/Y를 되돌린 뒤 Z축 quaternion을 적용한다.
- 박스별 settle 후 전체 `final_steps`만큼 추가 안정화한다.
- 최종 판정은 다음 기준을 사용한다.
  - `DROP/COLLAPSE`: 의도 위치 대비 최종 위치 drift가 `drift_threshold_m` 이상
  - `OUT_OF_BOUNDS`: 최종 AABB가 팔레트 XY 밖
  - `HEIGHT_OVERFLOW`: 최종 top z가 팔레트 높이 초과
  - 성공률이 `episode_success_min_rate` 미만이면 pallet fail
- `evaluate.py`에 `--physics`, `--gui`, `--time-step`, `--solver-iterations` 옵션을 추가했다.
- `evaluate.py --physics` 사용 시 기존 geometric stability warning 대신 PyBullet 결과를 사용한다.
- 기하 HARD FAIL은 그대로 유지하고, PyBullet fail도 HARD FAIL로 결합해 최종 점수를 0점 처리한다.
- `benchmark_log.csv`에 `physics_mode` 컬럼을 추가하고, 기존 로그는 `geom` 값으로 자동 마이그레이션한다.

#### 검증 결과

문법 검사:

```bash
.venv/bin/python -m py_compile evaluate.py physics_check.py
```

결과: 통과.

PyBullet 미설치 처리:

```bash
.venv/bin/python evaluate.py --results algorithm_results/box_sequence_0.json --label physics_missing_test --physics --no-report-json
```

결과: traceback 없이 `[ERROR] PyBullet is not installed...` 설치 안내를 출력한다.

기존 기하 평가 경로 회귀 검증:

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "geom_after_physics_integration" --bounds-tol 0.001 --epsilon 0.0011 --no-report-json
```

결과:

| 입력 파일 | placed_count | utilization_percent | buffer_bonus | final_score | physics_mode | 판정 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `box_sequence_0.json` | 60 | 43.11% | 16.00 | 59.11 | geom | PASS |
| `box_sequence_1.json` | 62 | 47.87% | 16.00 | 63.87 | geom | PASS |

PyBullet 실제 실행 검증은 현재 macOS arm64 + Python 3.14 `.venv`에서 `pybullet` wheel이 없어 소스 빌드로 진행되었고, clang 빌드 실패로 설치가 완료되지 않았다. Python 3.12 임시 venv(`/tmp/issac_pybullet312`)에서도 동일하게 소스 빌드 실패가 발생했다. 따라서 이번 턴에서는 실제 PyBullet 시뮬레이션 PASS/FAIL까지는 실행하지 못했다.

실제 실행은 PyBullet wheel이 설치 가능한 환경에서 다음 명령으로 검증한다.

```bash
python evaluate.py --results algorithm_results --label "v2_pybullet" --physics --bounds-tol 0.001 --epsilon 0.0011
python physics_check.py --result algorithm_results/box_sequence_0.json
python physics_check.py --result algorithm_results/box_sequence_0.json --gui
```

### 2026-06-24 - 공식 평가 근사 채점기 evaluate.py 추가

#### 변경 목적

알고리즘 버전과 설정을 바꿀 때마다 같은 기준으로 결과 JSON을 비교할 수 있도록 독립 실행형 평가 도구를 추가했다. 제출 알고리즘과 실행 파이프라인은 수정하지 않고, `algorithm_results/*.json`을 읽어 공식 점수 구조를 기하학적으로 근사한다.

#### 변경 내용

- `evaluate.py`를 새로 추가했다.
- 결과 JSON의 `buffer_size`, `sequence`, `size`, `position`, `rotation`, `mass` 키를 검증한다.
- `position`은 centroid, `size`는 회전이 이미 반영된 값으로 보고 AABB를 복원한다.
- 점수는 `util_points + buffer_bonus`로 계산한다.
- HARD FAIL 조건을 검사한다.
  - 팔레트 경계 이탈
  - 높이 초과
  - 3D AABB 침투
- STABILITY RISK 조건을 별도 경고로 보고한다.
  - 지지율 부족
  - 무게중심이 지지 영역 범위 밖
- `--strict-stability` 옵션을 주면 stability risk를 HARD FAIL로 승격한다.
- `benchmark_log.csv`에 label별 집계 결과를 append한다.
- `eval_report.json`에 파일별 상세 결과를 저장한다.
- 새 의존성은 추가하지 않았다. YAML은 표준 라이브러리 기반의 단순 parser로 필요한 키만 읽는다.

#### 검증 결과

문법 검사:

```bash
.venv/bin/python -m py_compile evaluate.py
```

결과: 통과.

현재 결과 JSON을 기본 엄격 설정으로 평가:

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "v2_bestfit"
```

결과: `bounds_tol=0.0`, `epsilon=1e-6` 기준에서는 출력 좌표 3자리 반올림으로 생긴 `-0.0005m` 경계 오차와 `0.0005~0.001m` 얇은 AABB 겹침까지 HARD FAIL로 잡혀 평균 점수는 0점이다.

반올림 오차를 허용한 비교 실행:

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "v2_bestfit_roundtol_0011" --bounds-tol 0.001 --epsilon 0.0011 --no-report-json
```

결과:

| 입력 파일 | placed_count | utilization_percent | buffer_bonus | final_score | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| `box_sequence_0.json` | 60 | 43.11% | 16.00 | 59.11 | PASS |
| `box_sequence_1.json` | 62 | 47.87% | 16.00 | 63.87 | PASS |

더미 HARD FAIL 검증:

- 박스 하나가 팔레트 X 경계를 넘는 임시 JSON을 만들어 실행했다.
- `box[0] outside XY bounds` 사유로 `final_score=0.00` 처리됨을 확인했다.
- 임시 테스트 폴더는 검증 후 삭제했다.

### 2026-06-24 - 현재 설정(buffer.size=4) 기준 재검증

#### 변경 목적

현재 `config/algorithm_config.yaml`의 `buffer.size`가 4로 설정되어 있어, 이 설정 기준으로 `main.py` 전체 실행 결과를 다시 확인했다.

#### 변경 내용

알고리즘 코드는 변경하지 않았다. 현재 설정값 기준 결과를 기록한다.

#### 검증 결과

검증 명령:

```bash
.venv/bin/python main.py
```

결과:

| 입력 파일 | placed_count | utilization_percent | max_top_height | 자체 검증 |
| --- | ---: | ---: | ---: | --- |
| `box_sequence_0.json` | 60 | 43.11% | 1.2385 | OK |
| `box_sequence_1.json` | 62 | 47.87% | 1.2380 | OK |

전체 처리 시간은 약 13.94초였다. 두 입력 모두 `placed_count`가 20보다 크고, `max_top_height`는 팔레트 제한 높이 1.25m 이하이며, 자체 검증 `[CHECK] OK`를 통과했다.

### 2026-06-24 - algorithm.py 설명 주석 보강

#### 변경 목적

Heightmap + Best-Fit 알고리즘은 기존 커서 방식보다 상태와 판단 기준이 많다. 이후 튜닝하거나 디버깅하는 사람이 코드를 빠르게 이해할 수 있도록 주요 dataclass, helper 함수, 후보 평가 과정, 지지율 계산, 점수 계산, 종료 처리에 설명 주석과 docstring을 추가했다.

#### 변경 내용

- `HeuristicWeights`, `HeuristicConfig`, `Candidate`, `PlacedAABB`에 역할 설명을 추가했다.
- `_reset_state()`에 heightmap의 row/col 의미와 셀 값의 의미를 설명했다.
- `_load_heuristic_config()`에 main.py를 수정하지 않고 heuristic YAML을 직접 읽는 이유를 설명했다.
- `should_finish()`에 종료 조건을 명시했다.
- `_candidate_orientations()`, `_axis_positions()`, `_cell_slice()`에 후보 생성과 좌표 변환 방식을 설명했다.
- `_evaluate_candidate()`에 하드 제약 검사 순서를 단계별로 적었다.
- `_exact_support_ratio()`와 `_aabb_intersects_existing()`에 heightmap 근사와 실제 AABB 검사의 차이를 설명했다.
- `_score_candidate()`에 각 점수 항목의 의도를 설명했다.
- `_append_placed()`에 bottom-left 좌표를 출력용 centroid 좌표로 바꾸는 규칙을 설명했다.
- `run()`에 버퍼 조회, 선택 소비, 스킵, 종료 플래그 의미를 설명했다.

#### 검증 결과

문서와 주석만 변경했기 때문에 알고리즘 동작은 바뀌지 않는다.

검증 명령:

```bash
.venv/bin/python -m py_compile algorithm.py
```

결과: 통과.

### 2026-06-24 - Heightmap + 점수 기반 Best-Fit 알고리즘 도입

#### 변경 목적

기존 베이스라인은 단순 커서 기반 선반 채우기 방식이었다. 박스를 입력 순서대로 확인하면서 현재 `cursor_x`, `cursor_y`, `layer_z` 위치에 놓을 수 있는지만 검사하고, 공간이 부족하면 다음 row 또는 다음 layer로 이동했다.

이 방식은 구현은 단순하지만 다음 한계가 있었다.

- 팔레트 위의 실제 표면 높이 분포를 알지 못한다.
- 이미 놓인 박스의 위쪽 빈 공간을 적극적으로 활용하지 못한다.
- 버퍼 안의 여러 박스 중 어떤 박스를 먼저 놓는 것이 좋은지 점수화하지 않는다.
- 지지율, 접촉, 평탄도 같은 안정성 요소를 배치 판단에 반영하지 않는다.
- `len(sequence) >= 20` 조건으로 강제 종료되어 충분히 더 놓을 수 있어도 20개에서 멈춘다.

이번 변경의 목표는 안정성을 우선으로 유지하면서 더 많은 박스를 적재하는 것이다. 이를 위해 팔레트 바닥면을 격자로 나눈 heightmap을 만들고, 가능한 후보 배치를 점수화해서 가장 좋은 후보를 선택하는 Best-Fit 방식으로 교체했다.

#### 이전 알고리즘에서 변경된 부분

1. 커서 기반 상태 제거

   기존에는 다음 상태를 사용했다.

   ```python
   cursor_x
   cursor_y
   layer_z
   row_depth
   layer_height
   ```

   새 알고리즘에서는 이 값을 사용하지 않는다. 대신 팔레트 전체의 현재 표면 높이를 표현하는 2D numpy 배열 `heightmap`을 사용한다.

2. Heightmap 상태 추가

   `heightmap`은 팔레트 바닥면을 작은 셀로 나눈 2차원 배열이다.

   - 배열 shape: `(n_rows, n_cols)`
   - row 방향: 팔레트 Y축
   - col 방향: 팔레트 X축
   - 각 셀 값: 해당 위치에서 현재 쌓여 있는 박스의 윗면 높이 `z`
   - 초기값: 모든 셀 `0.0`

   예를 들어 `grid_resolution_m=0.02`이면 기본 팔레트 `1.2m x 1.0m`는 대략 `60 x 50` 셀로 표현된다.

3. 후보 위치 탐색 방식 변경

   기존 알고리즘은 현재 커서 위치, 다음 row, 다음 layer 정도만 확인했다.

   새 알고리즘은 각 박스와 회전 방향에 대해 팔레트 위 여러 `(x, y)` 후보 위치를 스캔한다.

   후보 위치 간격은 `heuristic.candidate_step_m` 설정값을 사용한다. 기본값은 `0.02m`이다.

4. 버퍼 전체 Best-Fit 선택

   기존 알고리즘은 버퍼 안의 박스를 앞에서부터 순서대로 보다가 첫 번째로 들어맞는 박스를 배치했다.

   새 알고리즘은 버퍼에 있는 모든 박스에 대해 다음 조합을 평가한다.

   ```text
   박스 x 회전 방향 x 후보 (x, y) 위치
   ```

   그중 하드 제약을 통과하고 점수가 가장 높은 후보 1개를 선택한다. 선택된 박스는 `BufferManager.pop_selected(index)`로 소비한다.

5. 지지율 기반 안정성 검사 추가

   새 알고리즘은 박스가 공중에 뜨거나 모서리에만 걸치는 상황을 막기 위해 지지율을 검사한다.

   후보 박스의 footprint 영역에서 `z_place`와 거의 같은 높이를 가진 셀 비율을 계산하고, 추가로 실제 배치된 박스 AABB 면적 기준의 지지율도 계산한다. 두 값 중 더 작은 값을 최종 지지율로 사용한다.

   최종 지지율이 `support_threshold`보다 낮으면 후보에서 제외된다.

6. 점수 기반 후보 선택 추가

   하드 제약을 통과한 후보는 다음 요소를 더해 점수화한다.

   ```text
   score =
     -w_height  * top_z
     +w_support * support_ratio
     +w_contact * contact_ratio
     +w_flat    * flatness
     +w_mass    * mass_term
   ```

   점수가 높을수록 좋은 후보로 본다.

7. 20개 강제 종료 제거

   기존 코드에는 다음 조건이 있었다.

   ```python
   if len(self.sequence) >= 20:
       self.finished_by_user = True
       break
   ```

   이 조건을 제거했다. 이제 20개 이후에도 유효한 배치가 있으면 계속 진행한다.

8. 자체 검증 추가

   `run()` 종료 직전에 `_assert_valid_result()`를 호출해 결과를 검증한다.

   검증 항목은 다음과 같다.

   - 모든 박스가 팔레트 경계 안에 있는지
   - 모든 박스의 top height가 팔레트 최대 높이 이하인지
   - 박스 간 3D AABB 충돌이 없는지
   - 각 박스의 실제 지지율이 `support_threshold` 이상인지

   통과하면 콘솔에 `[CHECK] OK ...`를 출력한다.

## 현재 알고리즘 구조

### 주요 데이터 구조

#### `HeuristicConfig`

`config/algorithm_config.yaml`의 `heuristic` 섹션을 읽어 구성된다.

```yaml
heuristic:
  grid_resolution_m: 0.02
  candidate_step_m: 0.02
  support_threshold: 0.8
  support_z_tol_m: 0.003
  max_consecutive_failures: 40
  min_remaining_height_m: 0.03
  weights:
    w_height: 1.0
    w_support: 2.0
    w_contact: 0.7
    w_flat: 0.4
    w_mass: 0.15
```

각 항목의 의미는 다음과 같다.

| 설정 | 의미 |
| --- | --- |
| `grid_resolution_m` | heightmap 셀 하나의 실제 크기 |
| `candidate_step_m` | 후보 `(x, y)` 위치를 생성할 때 사용하는 간격 |
| `support_threshold` | 후보 배치가 통과해야 하는 최소 지지율 |
| `support_z_tol_m` | 같은 높이로 인정할 z 오차 허용값 |
| `max_consecutive_failures` | 연속 배치 실패가 이 값을 넘으면 자동 종료 |
| `min_remaining_height_m` | 남은 높이가 너무 작을 때 종료 판단에 쓰는 최소 높이 |
| `weights.w_height` | 낮게 놓는 후보를 선호하는 가중치 |
| `weights.w_support` | 지지율이 높은 후보를 선호하는 가중치 |
| `weights.w_contact` | 벽 또는 이웃 박스와 접촉하는 후보를 선호하는 가중치 |
| `weights.w_flat` | footprint 영역이 평탄한 후보를 선호하는 가중치 |
| `weights.w_mass` | 무거운 박스를 낮게 놓는 후보를 선호하는 가중치 |

#### `Candidate`

후보 배치 하나를 표현한다.

주요 필드는 다음과 같다.

| 필드 | 의미 |
| --- | --- |
| `score` | 후보의 최종 점수 |
| `buffer_index` | 버퍼 안에서 이 박스가 몇 번째인지 |
| `box` | 원본 입력 박스 |
| `dims` | 회전이 반영된 박스 크기 `(dx, dy, dz)` |
| `rotation` | `0` 또는 `90` |
| `x`, `y`, `z` | 박스 bottom-left-bottom 좌표 |
| `support_ratio` | 최종 지지율 |
| `cell_slice` | heightmap에서 이 박스 footprint가 차지하는 영역 |

#### `PlacedAABB`

이미 배치된 박스의 실제 AABB 정보를 저장한다. 출력 JSON은 `sequence`에 저장하지만, 충돌 검사와 실제 지지율 검사는 `PlacedAABB` 목록을 사용한다.

## 실행 흐름

### 1. 초기화

`Palletizer.__init__()`에서 heuristic 설정을 로드하고 `_reset_state()`를 호출한다.

`_reset_state()`는 다음 상태를 초기화한다.

- `heightmap`
- 출력용 `sequence`
- 검사용 `_placed_aabbs`
- 종료 플래그
- 연속 실패 횟수

### 2. 현재 후보 박스 가져오기

`run()` 루프 안에서 `BufferManager`를 통해 현재 후보 박스를 가져온다.

버퍼 크기가 0이면 다음 박스 1개만 본다.

```python
current = [buf.peek_next()]
```

버퍼 크기가 0보다 크면 현재 버퍼 전체를 본다.

```python
current = buf.get_buffer()
```

### 3. 종료 여부 확인

`should_finish(current)`는 다음 조건을 본다.

- 현재 버퍼가 비어 있으면 종료하지 않는다.
- 현재 heightmap의 최대 높이를 기준으로 남은 높이가 너무 작으면 종료한다.
- 연속 배치 실패 횟수가 `max_consecutive_failures` 이상이면 종료한다.

이 함수가 `True`를 반환하면 `finished_by_user=True`로 종료한다.

### 4. 후보 평가

`_best_candidate(indexed_boxes)`가 현재 버퍼의 모든 박스를 평가한다.

평가 순서는 다음과 같다.

1. 박스 선택
2. 가능한 회전 방향 생성
3. 회전된 크기에 대해 후보 x 좌표 목록 생성
4. 후보 y 좌표 목록 생성
5. 각 `(x, y)`에 대해 `_evaluate_candidate()` 호출
6. 통과 후보 중 최고 점수 후보 선택

동점일 때는 다음 우선순위를 사용한다.

1. 작은 `y`
2. 작은 `x`
3. 작은 `z`
4. 버퍼에서 더 앞에 있는 박스

즉, 점수가 같으면 bottom-left에 가까운 배치를 선호한다.

### 5. 후보의 z 위치 계산

후보 박스의 footprint가 차지하는 heightmap 영역을 가져온다.

```python
region = self.heightmap[row_slice, col_slice]
z = float(np.max(region))
```

`z_place`는 footprint 영역에서 가장 높은 셀 값이다. 박스는 이 높이 위에 놓인다.

이 방식은 박스가 기존 물체를 뚫고 들어가는 것을 막아준다. footprint 안에 더 높은 부분이 하나라도 있으면 그 높이를 기준으로 박스가 올라가기 때문이다.

### 6. 하드 제약 검사

`_evaluate_candidate()`는 다음 조건을 모두 검사한다.

#### 경계 검사

```text
x >= 0
y >= 0
x + dx <= pallet.length
y + dy <= pallet.width
z + dz <= pallet.height
```

하나라도 위반하면 후보에서 제외한다.

#### 셀 기반 지지율 검사

footprint 영역에서 `z_place`와 거의 같은 높이에 있는 셀 비율을 계산한다.

```python
support_mask = np.abs(region - z) <= support_z_tol_m
cell_support_ratio = count_true / region.size
```

#### 실제 AABB 면적 기반 지지율 검사

셀 기반 지지율은 격자 근사이기 때문에 실제 박스 면적 기준보다 조금 낙관적일 수 있다. 그래서 배치된 박스 AABB 목록을 사용해 실제 지지 면적도 계산한다.

```text
exact_support_ratio =
  아래 박스들과 맞닿은 실제 overlap 면적 합
  /
  후보 박스 footprint 면적
```

바닥에 놓이는 박스는 `support_ratio=1.0`으로 본다.

최종 지지율은 다음처럼 보수적으로 정한다.

```python
support_ratio = min(cell_support_ratio, exact_support_ratio)
```

이 값이 `support_threshold`보다 낮으면 후보에서 제외한다.

#### 충돌 검사

`_aabb_intersects_existing()`에서 이미 배치된 박스들과 3D AABB 교차를 검사한다. 교차하면 후보에서 제외한다.

### 7. 후보 점수 계산

하드 제약을 통과한 후보만 `_score_candidate()`에서 점수화한다.

#### 낮게 놓기

```python
-w_height * top_z
```

`top_z = z + dz`가 낮을수록 점수가 높다. 높은 곳에 불필요하게 쌓는 배치를 줄인다.

#### 지지율

```python
+w_support * support_ratio
```

지지율이 높을수록 안정적이라고 보고 가산점을 준다.

#### 접촉 둘레 비율

```python
+w_contact * contact_ratio
```

팔레트 벽면 또는 같은 높이대의 이웃 박스 측면과 맞닿는 길이를 footprint 둘레로 나눈 값이다. 벽이나 이웃 박스에 붙는 배치를 선호해서 빈틈을 줄이고 안정성을 높인다.

#### 평탄도

```python
+w_flat * flatness
```

footprint 영역의 heightmap 표준편차가 작을수록 평탄하다고 본다. 평탄한 영역에 놓는 배치를 선호해 모서리 걸침을 줄인다.

#### 무게 항

```python
+w_mass * mass_term
```

무거운 박스를 낮은 위치에 놓는 후보를 선호한다. `mass_term`은 박스 질량과 현재 `z` 위치를 함께 고려한다.

### 8. 배치 적용

최고 후보가 있으면 `_place_candidate()`가 호출되고, 내부에서 `_append_placed()`가 실행된다.

`_append_placed()`는 다음 작업을 수행한다.

1. 출력 JSON용 `sequence`에 `PlacedBox` 추가
2. `position`을 bottom-left 좌표에서 centroid 좌표로 변환
3. `size`와 `position`을 기존 규칙대로 소수점 3자리로 반올림
4. `PlacedAABB` 목록에 실제 AABB 추가
5. 후보 footprint에 해당하는 heightmap 셀 값을 `z + dz`로 갱신

출력의 `position`은 항상 중심 좌표이다.

### 9. 배치 실패 처리

현재 버퍼의 어떤 박스도 유효한 후보를 만들지 못하면 연속 실패 횟수를 증가시킨다.

그 다음 가장 앞 박스 하나를 소비해 다음 입력이 보이게 한다.

- `buffer_size == 0`: `buf.pop_next()`
- `buffer_size > 0`: `buf.pop_selected(0)`

`BufferManager`에는 별도의 skip 메서드가 없기 때문에, 소비 메서드를 사용해 스킵을 구현한다.

연속 실패가 `max_consecutive_failures`에 도달하면 `terminated=True`로 종료한다.

## 주요 함수 요약

| 함수 | 역할 |
| --- | --- |
| `_load_heuristic_config()` | YAML에서 heuristic 설정 로드 |
| `_reset_state()` | heightmap, 결과 목록, 종료 상태 초기화 |
| `should_finish()` | 높이 부족 또는 연속 실패 기반 종료 판단 |
| `_candidate_orientations()` | 0도/90도 회전 후보 생성 |
| `_axis_positions()` | 후보 x 또는 y 좌표 목록 생성 |
| `_cell_slice()` | 실제 좌표를 heightmap slice로 변환 |
| `_evaluate_candidate()` | 후보 하나에 대해 하드 제약과 점수 계산 |
| `_best_candidate()` | 버퍼 전체에서 최고 후보 선택 |
| `_score_candidate()` | 높이, 지지율, 접촉, 평탄도, 무게 점수 계산 |
| `_exact_support_ratio()` | 실제 AABB 면적 기준 지지율 계산 |
| `_aabb_intersects_existing()` | 3D 충돌 검사 |
| `_append_placed()` | 출력 sequence, AABB, heightmap 갱신 |
| `_assert_valid_result()` | 최종 결과 자체 검증 |

## 현재 검증 결과

검증 명령:

```bash
.venv/bin/python main.py
```

일반 `python3 main.py`는 현재 시스템 Python 환경에 `PyYAML`이 없어 실패했다. 프로젝트 `.venv`에는 필요한 의존성이 설치되어 있어 `.venv/bin/python`으로 검증했다.

검증 결과:

| 입력 파일 | placed_count | utilization_percent | max_top_height | 자체 검증 |
| --- | ---: | ---: | ---: | --- |
| `box_sequence_0.json` | 69 | 48.50% | 1.238 | OK |
| `box_sequence_1.json` | 67 | 51.08% | 1.238 | OK |

생성된 시각화 파일:

- `algorithm_results/vis/box_sequence_0.png`
- `algorithm_results/vis/box_sequence_1.png`

## 향후 변경 기록 작성 규칙

알고리즘을 수정할 때마다 이 문서의 "변경 이력" 최상단에 새 항목을 추가한다.

권장 형식:

```markdown
### YYYY-MM-DD - 변경 제목

#### 변경 목적

왜 바꾸었는지 작성한다.

#### 변경 내용

- 어떤 파일을 바꾸었는지
- 어떤 함수나 설정을 바꾸었는지
- 기존 동작과 새 동작이 어떻게 다른지

#### 검증 결과

- 실행한 명령
- placed_count / utilization_percent / max_top_height
- 자체 검증 통과 여부
- 실패나 주의할 점
```

이 문서는 알고리즘 설명서이자 변경 이력이다. 단순히 결과 숫자만 남기지 말고, 왜 그런 선택을 했는지와 다음 사람이 튜닝할 때 봐야 할 포인트를 함께 적는다.
