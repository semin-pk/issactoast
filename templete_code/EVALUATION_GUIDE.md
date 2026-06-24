# Evaluation Guide

이 문서는 `evaluate.py`와 `physics_check.py`를 실행하고 결과를 확인하는 방법을 정리한다.

두 파일은 제출용 알고리즘이 아니라 로컬 개발/검증 도구이다.

- `evaluate.py`: 결과 JSON을 채점하고 버전별 점수를 `benchmark_log.csv`에 기록한다.
- `physics_check.py`: PyBullet으로 결과 JSON을 물리 시뮬레이션해서 붕괴/드랍/이탈 위험을 확인한다.

## 준비

프로젝트 디렉토리로 이동한다.

```bash
cd /Users/parksemin/Documents/Cursor/issactoast/templete_code
```

가상환경 Python을 사용한다.

```bash
.venv/bin/python --version
```

현재 권장 환경:

```text
Python 3.12.0
```

PyBullet import 확인:

```bash
.venv/bin/python - <<'PY'
import pybullet
print("pybullet ok")
PY
```

## 가장 많이 쓰는 실행 명령

config를 바꾼 뒤에는 아래 명령 하나만 실행하면 된다.

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label "my_run" \
  --physics \
  --physics-show-buffer \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

이 명령은 다음을 한 번에 수행한다.

1. `main.py` 실행
2. 현재 `config/algorithm_config.yaml` 기준으로 `algorithm_results/*.json` 재생성
3. 기하 HARD FAIL 검사
4. PyBullet 물리 검사
5. 점수 출력
6. `benchmark_log.csv`에 결과 append
7. `eval_report.json`에 상세 결과 저장

## evaluate.py 사용법

### 1. 이미 생성된 결과만 평가

```bash
.venv/bin/python evaluate.py \
  --results algorithm_results \
  --label "geom_check" \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

주의: 이 명령은 `main.py`를 실행하지 않는다. 이미 존재하는 `algorithm_results/*.json`만 읽는다.

`config/algorithm_config.yaml`의 `buffer.size`를 바꿨는데 결과 JSON을 새로 만들지 않았다면 이전 `buffer_size`가 그대로 평가된다.

### 2. config 변경까지 반영해서 평가

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label "buffer_test" \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

`--refresh-results`를 붙이면 `evaluate.py`가 먼저 `main.py`를 실행한다.

config 변경 후에는 이 방식을 권장한다.

### 3. PyBullet 물리 검증까지 포함해서 평가

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label "pybullet_test" \
  --physics \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

`--physics`를 붙이면 geometric stability warning 대신 PyBullet 결과를 사용한다.

기하 HARD FAIL과 PyBullet FAIL 중 하나라도 발생하면 해당 팔레트 점수는 0점이다.

### 4. PyBullet 버퍼 시각화까지 포함

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label "pybullet_buffer_view" \
  --physics \
  --physics-show-buffer \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

`--physics-show-buffer`는 PyBullet 시뮬레이션 내부에서 버퍼 플랫폼과 대기 박스를 함께 만든다.

GUI가 없으면 화면에는 보이지 않지만, 같은 물리 경로로 검증된다.

### 5. GUI로 직접 보기

`evaluate.py`에서도 GUI를 켤 수 있다.

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label "gui_check" \
  --physics \
  --physics-show-buffer \
  --gui \
  --gui-step-delay 0.02 \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

GUI로 볼 때는 `physics_check.py`를 단일 파일 대상으로 실행하는 편이 더 가볍다.

## physics_check.py 사용법

### 1. 단일 결과 파일 물리 검증

```bash
.venv/bin/python physics_check.py \
  --result algorithm_results/box_sequence_0.json
```

예상 출력 형태:

```text
box_sequence_0.json: PASS success_rate=100.00% max_drift=0.0067 max_top=1.2380 reasons=-
```

### 2. 결과 디렉토리 전체 물리 검증

```bash
.venv/bin/python physics_check.py \
  --results algorithm_results
```

`algorithm_results` 아래의 `.json` 결과 파일을 모두 검사한다.

### 3. 버퍼까지 시뮬레이션

```bash
.venv/bin/python physics_check.py \
  --result algorithm_results/box_sequence_0.json \
  --show-buffer
```

`--show-buffer`는 팔레트 오른쪽 버퍼 플랫폼에 대기 박스를 만들고, 배치 순서대로 팔레트로 이동시킨다.

결과 JSON에는 스킵된 원본 박스 정보가 없기 때문에, 이 버퍼 시각화는 출력 `sequence` 기준의 개발용 버퍼 뷰이다.

### 4. GUI로 버퍼 보기

```bash
.venv/bin/python physics_check.py \
  --result algorithm_results/box_sequence_0.json \
  --gui \
  --show-buffer \
  --gui-step-delay 0.02
```

`--gui-step-delay` 값이 클수록 움직임이 천천히 보인다.

빠르게 보고 싶으면 값을 줄인다.

```bash
--gui-step-delay 0.005
```

멈춤 없이 최대 속도로 돌리고 싶으면 생략하거나 0으로 둔다.

```bash
--gui-step-delay 0
```

## 주요 옵션 설명

### evaluate.py 옵션

| 옵션 | 의미 |
| --- | --- |
| `--refresh-results` | 평가 전에 `main.py`를 먼저 실행해 결과 JSON을 최신 config 기준으로 갱신 |
| `--results` | 평가할 결과 JSON 파일 또는 디렉토리 |
| `--label` | `benchmark_log.csv`에 기록할 실험 이름 |
| `--physics` | PyBullet 물리 검증 결과를 점수에 반영 |
| `--physics-show-buffer` | PyBullet 검증 시 버퍼 플랫폼과 대기 박스도 생성 |
| `--gui` | PyBullet GUI 사용 |
| `--gui-step-delay` | GUI에서 각 simulation step 후 대기 시간 |
| `--bounds-tol` | 경계 검사 허용오차 |
| `--epsilon` | AABB 겹침 검사 허용오차 |
| `--strict-stability` | geometric stability risk를 HARD FAIL로 승격 |
| `--benchmark-log` | 비교 로그 CSV 경로 |
| `--report-json` | 상세 평가 리포트 JSON 경로 |
| `--no-report-json` | `eval_report.json` 저장 생략 |

### physics_check.py 옵션

| 옵션 | 의미 |
| --- | --- |
| `--result` | 단일 결과 JSON 파일 |
| `--results` | 결과 JSON 디렉토리 |
| `--sim-config` | PyBullet에 매핑할 simulator config |
| `--gui` | PyBullet GUI 사용 |
| `--show-buffer` | 버퍼 플랫폼과 대기 박스 표시 |
| `--gui-step-delay` | GUI에서 각 simulation step 후 대기 시간 |
| `--time-step` | PyBullet simulation timestep |
| `--solver-iterations` | PyBullet solver iteration 수 |

## 출력 확인 방법

### 콘솔 표

`evaluate.py` 실행 후 파일별 표가 출력된다.

중요하게 볼 컬럼:

| 컬럼 | 의미 |
| --- | --- |
| `placed` | 적재된 박스 수 |
| `util%` | 팔레트 부피 대비 적재율 |
| `buffer` | 결과 JSON에 기록된 buffer_size |
| `bonus` | `20 - buffer_size` 버퍼 가산점 |
| `pass/fail` | HARD FAIL 또는 PyBullet FAIL 여부 |
| `score` | 최종 점수 |
| `phys` | `geom` 또는 `pybullet` |
| `reasons` | 실패 또는 risk 사유 |

### 집계 행

마지막 SUMMARY를 본다.

```text
SUMMARY n_files=2.0 mean_final_score=61.06 pass_rate=100.00% mean_utilization=57.06% mean_buffer_bonus=4.00 physics_mode=pybullet
```

대표 지표는 `mean_final_score`이다.

### benchmark_log.csv

실행할 때마다 한 줄씩 추가된다.

```bash
tail -n 10 benchmark_log.csv
```

컬럼:

| 컬럼 | 의미 |
| --- | --- |
| `timestamp` | 실행 시각 |
| `label` | 실행 label |
| `physics_mode` | `geom` 또는 `pybullet` |
| `n_files` | 평가 파일 수 |
| `mean_final_score` | 평균 최종 점수 |
| `pass_rate` | PASS 비율 |
| `mean_utilization_pct` | 평균 적재율 |
| `mean_buffer_bonus` | 평균 버퍼 보너스 |

### eval_report.json

파일별 상세 결과가 저장된다.

```bash
sed -n '1,120p' eval_report.json
```

확인할 항목:

- `summary.mean_final_score`
- `summary.pass_rate`
- `files[].final_score`
- `files[].reasons`
- `files[].stability_reasons`
- `files[].physics_mode`
- `files[].max_final_drift`
- `files[].max_final_top_z`

## buffer.size 변경 시 주의

`config/algorithm_config.yaml`에서 buffer size를 바꿨다면 반드시 결과 JSON을 다시 만들어야 한다.

좋은 방법:

```bash
.venv/bin/python evaluate.py --refresh-results --results algorithm_results --label "buffer16" --physics --bounds-tol 0.001 --epsilon 0.0011
```

나쁜 방법:

```bash
.venv/bin/python evaluate.py --results algorithm_results --label "buffer16"
```

위 명령은 기존 `algorithm_results`만 읽기 때문에, 예전 buffer size 결과를 평가할 수 있다.

`evaluate.py`는 config의 `buffer.size`와 결과 JSON의 `buffer_size`가 다르면 경고를 출력한다.

```text
[WARN] result JSON buffer_size does not match config buffer.size=...
[WARN] Run evaluate.py with --refresh-results to execute main.py first.
```

## 추천 워크플로우

1. `config/algorithm_config.yaml`에서 튜닝값 변경
2. 아래 명령 한 번 실행

```bash
.venv/bin/python evaluate.py \
  --refresh-results \
  --results algorithm_results \
  --label "experiment_name" \
  --physics \
  --physics-show-buffer \
  --bounds-tol 0.001 \
  --epsilon 0.0011
```

3. 콘솔의 `mean_final_score` 확인
4. `benchmark_log.csv`에서 이전 실험과 비교
5. 필요하면 GUI로 한 파일 확인

```bash
.venv/bin/python physics_check.py \
  --result algorithm_results/box_sequence_0.json \
  --gui \
  --show-buffer \
  --gui-step-delay 0.02
```

## 한계

- PyBullet은 Isaac Sim PhysX의 근사이다.
- 마찰, 접촉, solver 차이 때문에 경계선 케이스는 공식 평가와 다를 수 있다.
- `--show-buffer`는 결과 JSON의 `sequence`만으로 구성한 개발용 버퍼 뷰이다.
- 공식 최종 판단은 운영 서버 기준이다.
