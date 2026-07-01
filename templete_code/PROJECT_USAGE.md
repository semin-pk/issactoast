# Palletizing Project Usage

이 문서는 현재 프로젝트를 데이터 생성, 휴리스틱 평가, MCTS teacher 데이터 수집,
정책망 학습, ONNX 추론 채택 순서로 사용하는 방법을 정리합니다.

## 1. 기본 실행

```bash
cd /Users/parksemin/Documents/Cursor/issactoast/templete_code
.venv/bin/python main.py
```

`config/algorithm_config.yaml`의 `input_path`에 지정된 입력을 읽고,
`algorithm_results/`에 결과 JSON과 시각화 이미지를 저장합니다.

현재 제출 runtime의 기본 의사결정은 다음 순서입니다.

```text
policy_inference.enabled == true 이고 models/policy_net.onnx 존재
→ ONNX policy top-K 추천
→ 기존 action mask / feasibility 재검증
→ 실패 시 휴리스틱 fallback
→ 후보가 없으면 stop
```

기본값은 `policy_inference.enabled: false`이므로 휴리스틱만 사용합니다.

## 2. 자가 데이터 생성

단일 입력 파일:

```bash
.venv/bin/python src/box_generator.py \
  --seed 1000 \
  --count 120 \
  --mode uniform \
  --output box_sequence/generated.json
```

tuning / holdout seed set:

```bash
.venv/bin/python src/box_generator.py \
  --count 120 \
  --mode sku \
  --seed-set both \
  --output generated_sequences
```

`uniform`과 `sku`는 숨은 평가 분포의 정답 모델이 아닙니다. 두 분포 모두에서
교차 검증해 한쪽 분포에만 맞는 값을 피합니다.

## 3. 휴리스틱 기준 평가

`config/algorithm_config.yaml`의 `input_path`를 평가할 폴더로 맞춥니다.

```yaml
input_path: generated_sequences/holdout
```

그 다음:

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label heuristic_holdout \
  --seed-set holdout_sku \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

PyBullet strict 검증:

```bash
.venv/bin/python evaluate.py \
  --results algorithm_results \
  --label heuristic_holdout_strict \
  --seed-set holdout_sku \
  --physics \
  --physics-strict \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

주요 지표는 `benchmark_log.csv`의 `mean_score`, `worst_score`,
`fail_rate`, `mean_runtime_sec`입니다.

## 4. MCTS Teacher 데이터 수집

Full MCTS는 제출 중 실행하지 않습니다. `dev_tools/` 아래의 오프라인
teacher로만 사용합니다.

작은 smoke test:

```bash
.venv/bin/python dev_tools/collect_mcts_policy_data.py \
  --config config/algorithm_config.yaml \
  --input box_sequence/generated.json \
  --output data/mcts_dataset/generated_mcts.npz \
  --num-simulations 8 \
  --max-depth 6 \
  --max-sequences 1 \
  --label mcts_smoke
```

본격 수집:

```bash
.venv/bin/python dev_tools/collect_mcts_policy_data.py \
  --config config/algorithm_config.yaml \
  --input generated_sequences/tuning \
  --output data/mcts_dataset/train_sku_mcts.npz \
  --num-simulations 128 \
  --max-depth 30 \
  --max-sequences 50 \
  --label mcts_teacher_v1
```

저장되는 주요 필드:

```text
height_map
buffer_features
action_mask
action
mcts_policy
visit_counts
q_values
baseline_score
mcts_score
delta_score
```

`action`은 hard-label 학습용이고, `mcts_policy`는 root visit count 기반
soft-label 학습 확장용입니다.

## 5. 데이터 증강

지원 증강은 직사각형 팔레트에 안전한 세 가지입니다.

```text
flip_x
flip_y
rot180
```

실행:

```bash
.venv/bin/python dev_tools/augment_policy_data.py \
  --input data/mcts_dataset/train_sku_mcts.npz \
  --output data/mcts_dataset/train_sku_mcts_aug.npz
```

증강 시 `action`, `action_mask`, `mcts_policy`, `visit_counts`, `q_values`를
같이 좌표 보정합니다. 보정 후 `action_mask[action] == 1`을 assert합니다.

## 6. 정책망 학습 및 ONNX Export

학습은 dev-only이며 PyTorch가 필요합니다. 제출 코드에는 PyTorch를 넣지 않습니다.

```bash
.venv/bin/python dev_tools/train_policy.py \
  --config config/policy_train_config.yaml \
  --train data/mcts_dataset/train_sku_mcts_aug.npz \
  --valid data/mcts_dataset/valid_sku_mcts_aug.npz \
  --output models/policy_net.onnx
```

현재 학습 코드는 hard label cross entropy를 사용합니다.

```text
target = argmax/root best action
input = height_map + buffer_features + action_mask
output = [B, 2, H, W] logits
```

## 7. ONNX Policy 채택

`models/policy_net.onnx`가 준비된 뒤에만 설정을 켭니다.

```yaml
policy_inference:
  enabled: true
  model_path: models/policy_net.onnx
  top_k: 32
  fallback_to_heuristic: true
  stop_if_no_safe_action: true
```

평가:

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label onnx_policy_holdout \
  --seed-set holdout_sku \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

채택 기준:

```text
fail_rate가 휴리스틱보다 높지 않음
mean_score가 휴리스틱보다 높거나 runtime 이점이 명확함
worst_score가 낮아지지 않음
max runtime < 90 sec
fallback이 과도하지 않음
제출 경로 금지 import 없음
```

기준을 만족하지 못하면 `policy_inference.enabled: false`로 두고 휴리스틱을
사용합니다.

## 8. 제출 경로 주의사항

제출 ZIP에는 다음을 포함하지 않습니다.

```text
dev_tools/
physics_check.py
PyBullet 관련 파일
학습 스크립트
torch / optuna / cma import
```

제출 runtime에 허용되는 정책 관련 코드는 다음뿐입니다.

```text
src/policy_inference.py
models/policy_net.onnx
onnxruntime
numpy
```

금지 import 스캔:

```bash
rg -n "^(import|from) (torch|pybullet|optuna|cma)|^\\s*(import|from) (torch|pybullet|optuna|cma)" \
  algorithm.py main.py buffer_manager.py visualize.py src config
```

## 9. 회귀 체크

```bash
.venv/bin/python -m py_compile \
  algorithm.py evaluate.py main.py \
  src/policy_inference.py \
  dev_tools/mcts_teacher.py \
  dev_tools/collect_mcts_policy_data.py

.venv/bin/python src/regression_checks.py
```

MCTS smoke dataset:

```bash
.venv/bin/python src/box_generator.py \
  --seed 7 --count 12 --mode uniform \
  --output /tmp/mcts_tiny_sequence.json

.venv/bin/python dev_tools/collect_mcts_policy_data.py \
  --config config/algorithm_config.yaml \
  --input /tmp/mcts_tiny_sequence.json \
  --output /tmp/mcts_dataset_tiny.npz \
  --num-simulations 4 \
  --max-depth 4 \
  --max-sequences 1 \
  --min-utilization-threshold 0.0 \
  --label smoke_mcts
```
