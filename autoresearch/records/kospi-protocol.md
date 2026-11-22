# Autoresearch

> **English note.** Original protocol, kept in Korean as written. These are the rules the coding agent followed for the KOSPI research: reference documents, session-start checklist, the experiment loop, and the rule that every planned experiment is run to a conclusion and recorded. File paths refer to the original `BRAIN` workspace, most of which is not included in this repository. English summary: [AutoResearch README](../README.md).

## 참조 문서

| 문서 | 내용 |
|------|------|
| `reference/data.md` | 원천 데이터 (raw candle) |
| `reference/problem.md` | 문제 정의, 평가 방식 |
| `reference/features.md` | Feature/Label 정의, 코드 구조 |
| `reference/npy_management.md` | NPY 생성 규칙 |
| `reference/backtest.md` | Rust 백테스트 엔진 가이드 |
| `dont_repeat.md` | 반복 금지 (실패한 실험 + 실수) |
| `request.md` | 수정/인프라 요청 |
| `autoresearch/YYYY-MM-DD.md` | 일일 실험 계획 + 결과 |
| `BRAIN/codebase_architecture.md` | 코드 구조 (프로젝트 루트 기준) |

---

## 세션 시작 준비

1. 오늘 날짜 확인
2. **`dont_repeat.md` 읽기** — 이미 실패한 접근/실수 확인
3. `autoresearch/YYYY-MM-DD.md` 읽고 실험 방향 파악
4. **이전 베스트 재현 확인** — 파이프라인 문제인지 모델 문제인지 구분
5. **리팩토링 체크** — `BRAIN/refactor_principle.md` 기준으로 현재 코드 상태 간단 점검 후, 정리가 필요하면 "리팩토링 먼저 할까요?"라고 한 번 물어본다. 실험 루프 시작 후에는 다시 묻지 않는다.

---

## 실험 루프

```
1. autoresearch/YYYY-MM-DD.md에서 다음 실험 확인
2. 목록에 있는 실험만 수행. 목록에 없는 실험은 금지.
3. dont_repeat.md mistake 항목 확인
4. **Plan vs Actual 대조** — 실험 스펙과 실제 config 대조, 불일치 시 실행 금지
5. 구현
6. 학습 실행
7. 평가: trading sim + 백테스트
8. 결과 기록
9. 판정: keep / discard / crash
10. 다음 실험으로 (목록 끝나면 자유 탐색)
```

### 실험 원칙: 절대 드랍하지 않는다

- 각 실험은 반드시 학습 → 백테스트까지 완료하고 결과를 기록한다.
- "안 될 것 같아서" 중간에 건너뛰거나 다른 실험으로 바꾸지 않는다.
- 확실히 **된다/안 된다** 결론을 내고 다음으로 넘어간다.
- 한 실험이 실패해도 계획된 다음 실험을 그대로 진행한다.

### 기록 규칙

실험 1개 eval 완료 즉시, 아래 **2곳**에 기록 후 다음 실험:

1. **`experiments/results_all.csv`** — 한 줄 요약 (숫자)
2. **`autoresearch/YYYY-MM-DD.md`** — 상세 분석, 판정, 다음 방향

여러 실험 모아서 한꺼번에 기록하지 마. eval 1개 → 기록 → 다음.

### results_all.csv 컬럼

```
dataset,date,model,direction,L_CumAlpha,LS_CumAlpha,L_WR,L_Trades,status,description,inference_path
```
- status: `keep`, `discard`, `crash`

### backtest 결과

백테스트 결과는 `experiments/backtest.csv`에 기록. 방법: `backtest.md` 참조.

---

## 인퍼런스 parquet 고정 규칙

인퍼런스 parquet 생성 시 항상 적용:

- **KST 타임존 필수**: `start_timestamp`, `end_timestamp` 모두 `Asia/Seoul` tz-aware
- **end_timestamp 필수**: `start_timestamp + bar duration` (1분봉: +1분, 1초봉: +1초)
- **15:20 제외**: 인퍼런스 대상은 15:19까지. 15:20봉 포함하지 않음
- **저장 경로**: `/shared/inference_kospi/<모델명>/`

---

## 제약 조건

- **primary target**: c2c_120 고정. 다른 horizon은 auxiliary loss로만
- **평가 기준**: CumAlpha (beta 제거 후 누적 alpha)
  - 우선순위: L CumAlpha > L+S CumAlpha > L AvgAlpha > L+S AvgAlpha
- **sample_ratio**:
  - 탐색 단계: `sample_ratio=0.05`. 앙상블 안 함. 다양한 모델/feature 개별 실험
  - 성능 최적화 단계: sample_ratio 높이고, 앙상블 구성. **지시가 있을 때만 진행**
  - 기본값: 0.05
- **Feature에서 미래 데이터 금지** (label은 미래 참조 정상)
- Feature/NPY 상세 규칙: `features.md`, `npy_management.md` 참조

---

## 할 수 있는 것

- 새 모델 아키텍처 추가 (`src/models/architectures/`)
- 새 trainer 추가 (`src/training/trainers/`)
- 새 config 추가 (`src/config/models/`, `configs/`)
- 새 feature 추가 (`src/features/` + `features.md` 업데이트)
- 새 NPY config 추가 (`configs/npy/`)
- 하이퍼파라미터 변경
- loss 함수 변경/추가

---

## 할 수 없는 것

아래 항목은 임의 수정 금지. 실험 목적으로 필요하면 **`request.md`에 요청 기록**.

- 공유 코드 수정 (`registry.py`, `split.py` 등) — 기존 모델이 깨질 수 있음
- 평가 코드 수정 (`src/evaluation/`) — 메트릭 비교가 무의미해짐
- 패키지 추가/삭제
- 데이터 원본 수정 (`/shared/vendor_kospi/`)
- `src/features/` 기존 함수 원본 수정 — 새 함수/파일로 추가해서 실험
- `src/features/rolling.py` 수정 — 모든 feature 값이 달라짐
- `reference/` 직접 수정 — `src/features/`가 canonical, reference에 배포 시 복사
- `class_bins` 원본 수정 — 다른 bins 시도 시 별도 config로
- **백테스트 전략 파라미터 변경** — `reference/260330_vendor_ultimate_v3/config.py`의 `STRATEGY_PARAMS` 고정. 지시가 있을 때만 변경. 그 외 절대 건드리지 말 것.

---

## GPU 제약

- GPU 1장 (RTX 5090 32GB)
- 학습은 반드시 하나씩 — 동시에 돌리면 OOM
- batch_size 조절로 메모리 관리

---

## 실험 우선순위

1. **지시된 실험 우선**: `autoresearch/`에서 가장 최신 날짜 md를 읽고 명시된 실험을 먼저 수행
2. **자유 탐색**: 지시된 실험이 끝나면 백테스트 성능을 올리기 위해 자유롭게 탐색

---

## 논문 구현 주의

AI에게 "A 논문을 구현해줘"라고 하면 핵심 contribution을 빠뜨리거나 아키텍처를 엉터리로 구현하는 경우가 많다.

### 필수 프로세스

1. **논문 이해**: 구현 전에 핵심 contribution과 아키텍처 특징 정리
2. **체크리스트 작성**: 정리한 내용 기반으로 구현 체크리스트
3. **구현 후 검증**: 체크리스트 항목별로 논문과 일치하는지 확인

체크리스트 예시:
- [ ] 핵심 contribution이 반영되었는가?
- [ ] 아키텍처 주요 구성요소가 논문과 일치하는가?
- [ ] 특수 기법 (초기화, dropout, loss 등)이 포함되었는가?
- [ ] HP 기본값이 논문 권장값과 일치하는가?

---

## 실행 방법

### 학습

```bash
PYTHONPATH="." uv run python scripts/train.py -c configs/<model>.yaml data.data_path=/shared/vendor_kospi/npy/<version>/ data.sample_ratio=0.05
```

- 인퍼런스(test)는 항상 전체 데이터. sample_ratio는 학습/검증에만 적용

### 평가 (trading sim)

```bash
PYTHONPATH="." uv run python scripts/trading_sim_eval_vendor.py \
    --probs outputs/inference/<model>/test_probs.npy \
    --npy_dir /shared/vendor_kospi/npy/<version>/
```

### 백테스트 parquet 변환

```bash
PYTHONPATH="." uv run python scripts/probs_to_parquet.py \
    --probs outputs/inference/<model>/test_probs.npy \
    --npy_dir /shared/vendor_kospi/npy/<version>/ \
    --output_dir /shared/inference_kospi/<model>_parquet
```

백테스트 실행: `backtest.md` 참조.

---

## 멈추지 마

실험 루프가 시작되면 사람이 중단할 때까지 계속 돌린다.
"계속할까요?" 같은 질문 하지 말 것.
아이디어가 떨어지면 참조 문서를 다시 읽고, 이전 실험 결과를 분석하고, 더 과감한 시도를 해볼 것.
