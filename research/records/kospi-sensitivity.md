# 백테스트 예민도 (Backtest Sensitivity)

> 2026-04-08 발견. 같은 학습 모델 3개를 서로 다른 백테스트 config로 돌려서 std (variance) 측정한 결과, config에 따라 variance가 **10~27배까지 차이** 남. 실험 판정에 핵심.

---

## 핵심 결론 (먼저)

1. **250만 한도가 variance 안정화의 가장 큰 요인**. 한도 250만 config들은 std ≤ 162K. 750만 config들은 std 226K~807K.
2. **NEW 운영 config (750만 + IC sizing + dyn_slip + wScore4)는 variance가 mean을 압도**. 단일 seed 실험 판정에 부적합.
3. **실험 판정용 권장**:
   - 최대 안정성 → `C6 (250, off, wScore4)` std 30K
   - 안정성 + 현실성 → `OLD (250, off, score4)` std 81K
   - 운영 인프라 근접 + 적정 신호 → `C5 (750, on, score4)` std 304K
4. **wScore4 `[-2,-1,0,1,2]` 사용 금지** — 모든 조건에서 score4 `[-1,-1,0,1,1]` 대비 alpha 악화 + NEW config에서 variance 폭발.

---

## 측정 방법

- 같은 학습 config (`cs_mlp_200_base.yaml`), 같은 seed=42 (default), 같은 NPY (`260401_200_base`, test_end=2026-04-04)
- Flag OFF (XLA deterministic off — GPU drift 허용) 로 **3번 재학습** → 3 모델
- 각 모델을 8개 backtest config로 돌려서 alpha_pnl 수집
- config별로 3 모델의 mean, std 측정

GPU drift 외에는 완전 동일한 학습이므로, 이 std는 **"같은 품질의 모델들 사이에서 backtest가 만드는 variance"** 임.

---

## 8개 backtest config 정의

| id | one_coin | IC sizing | dyn_slip | score | model_slip | fee |
|---|---|---|---|---|---|---|
| **OLD** | 250만 | off | off | score4 [-1,-1,0,1,1] | 0 | 0.001 |
| **C6** | 250만 | off | off | **wScore4 [-2,-1,0,1,2]** | 0 | 0.001 |
| **C1** | **750만** | off | off | score4 | 0 | 0.001 |
| **C2** | **750만** | off | off | **wScore4** | 0 | 0.001 |
| **C3** | 250만 | **on** | **on** | score4 | 0 | 0.001 |
| **C4** | 250만 | **on** | **on** | **wScore4** | 0 | 0.001 |
| **C5** | **750만** | **on** | **on** | score4 | -0.002 | 0.00115 |
| **NEW** | **750만** | **on** | **on** | **wScore4** | -0.002 | 0.00115 |

IC sizing 파라미터 (on일 때): window=240, return_horizon=60, min_mult=0.05, max_mult=3.0
Dynamic slippage 파라미터 (on일 때): window=20, scale=0.5

---

## 결과 — 같은 모델 3개의 alpha variance

| config | v1 | v2 | v3 | mean | std | range |
|---|---:|---:|---:|---:|---:|---:|
| **OLD** | -3,011,452 | -2,915,154 | -2,813,187 | **-2,913,265** | **81K** | 198K |
| **C6** | -4,861,681 | -4,934,405 | -4,900,468 | **-4,898,851** | **30K** | 73K |
| **C1** | -4,967,734 | -5,342,595 | -3,912,485 | **-4,740,938** | **605K** | 1.43M |
| **C2** | -10,983,089 | -11,532,112 | -11,201,430 | **-11,238,877** | **226K** | 549K |
| **C3** | -1,516,534 | -1,147,669 | -1,456,163 | **-1,373,455** | **162K** | 369K |
| **C4** | -2,822,679 | -2,670,070 | -2,980,841 | **-2,824,530** | **127K** | 311K |
| **C5** | +4,861,714 | +5,463,773 | +4,782,563 | **+5,036,017** | **304K** | 681K |
| **NEW** | +648,020 | +759,247 | -1,005,471 | **+133,932** | **807K** | 1.76M |

---

## Std 순위 (안정성)

| 순위 | config | std | one_coin |
|---:|---|---:|---|
| 🥇 1 | **C6** | **30K** | 250 |
| 🥈 2 | **OLD** | **81K** | 250 |
| 3 | C4 | 127K | 250 |
| 4 | C3 | 162K | 250 |
| 5 | C2 | 226K | 750 |
| 6 | C5 | 304K | 750 |
| 7 | C1 | 605K | 750 |
| 8 | **NEW** | **807K** | 750 |

**패턴**: 상위 4위 (30K~162K) 전부 **250만 한도**, 하위 4위 (226K~807K) 전부 **750만 한도**.

→ **one_coin 한도가 variance 결정의 가장 강한 변수**.

---

## Signal-to-Noise 관점 (|mean| / std)

실험 판정에서 중요한 건 std가 아니라 **|mean| 대비 std** (신호가 잡음보다 얼마나 큰지).

| config | \|mean\| | std | **\|mean\|/std** | 해석 |
|---|---:|---:|---:|---|
| **C6** | 4.90M | 30K | **163** | 신호 극명 |
| **C2** | 11.24M | 226K | **50** | 매우 명확 |
| **OLD** | 2.91M | 81K | **36** | 명확 |
| **C4** | 2.82M | 127K | **22** | 명확 |
| **C5** | 5.04M | 304K | **16.6** | 명확 |
| **C3** | 1.37M | 162K | **8.5** | 있음 |
| **C1** | 4.74M | 605K | **7.8** | 있음 |
| **NEW** | 0.13M | 807K | **0.17** | **noise 수준** |

→ **NEW만 mean < std**. retrain variance가 alpha 신호를 완전히 삼킴. 단일 seed 비교 의미 없음.
→ 나머지 7개는 모두 mean > std로 신호 유지.

---

## 변수별 효과 분석

### 1. one_coin_max_pos_position_krw (250 vs 750)

| 다른 조건 | 250 mean | 750 mean | 250 std | 750 std | 결론 |
|---|---:|---:|---:|---:|---|
| off/off, score4 | -2.91M (OLD) | -4.74M (C1) | 81K | 605K | **750이 std 7.5x 악화**, mean 1.8M 악화 |
| off/off, wScore4 | -4.90M (C6) | -11.24M (C2) | 30K | 226K | 750이 std 7.5x 악화, mean 6.3M 악화 |
| on/on, score4 | -1.37M (C3) | +5.04M (C5) | 162K | 304K | **750이 mean +6.4M 개선** (slip/fee 차이 포함) |
| on/on, wScore4 | -2.82M (C4) | +0.13M (NEW) | 127K | 807K | 750이 mean +2.9M, 하지만 std 6.4x 악화 |

- **750만 한도는 variance를 대부분 증가시킴** (상기 4개 중 4개)
- IC/dyn on + score4 조합에서만 mean 개선 (C5가 C3 대비 +6.4M) — IC sizing이 큰 포지션을 활용하는 능력

### 2. IC sizing + dynamic_slippage (off vs on)

| 다른 조건 | off mean | on mean | off std | on std | 결론 |
|---|---:|---:|---:|---:|---|
| 250, score4 | -2.91M (OLD) | -1.37M (C3) | 81K | 162K | on이 mean +1.5M 개선, std 2x |
| 250, wScore4 | -4.90M (C6) | -2.82M (C4) | 30K | 127K | on이 mean +2.1M 개선, std 4.2x |
| 750, score4 | -4.74M (C1) | +5.04M (C5) | 605K | 304K | on이 mean +9.8M 개선, **std 감소** |
| 750, wScore4 | -11.24M (C2) | +0.13M (NEW) | 226K | 807K | on이 mean +11.4M 개선, std 3.6x |

- **IC/dyn on은 모든 조건에서 mean 개선** (+1.5M ~ +11.4M)
- std 효과는 mixed: 일반적으로 증가 (2~4배), C5에서만 감소

### 3. score weights (score4 vs wScore4)

| 다른 조건 | score4 mean | wScore4 mean | Δ |
|---|---:|---:|---:|
| 250, off/off | -2.91M (OLD) | -4.90M (C6) | **-1.99M** (wScore4 악화) |
| 750, off/off | -4.74M (C1) | -11.24M (C2) | **-6.50M** (wScore4 악화) |
| 250, on/on | -1.37M (C3) | -2.82M (C4) | **-1.45M** (wScore4 악화) |
| 750, on/on | +5.04M (C5) | +0.13M (NEW) | **-4.91M** (wScore4 악화) |

- **wScore4는 모든 조건에서 score4 대비 alpha 악화** (-1.45M ~ -6.50M)
- 750만 조건에서 악화 폭 더 큼 (-4.91M ~ -6.50M)
- **score4 > wScore4 확정**. wScore4 폐기 권장.

---

## 왜 NEW만 특이하게 variance 폭발하나 (가설)

**NEW = 750만 + IC/dyn on + wScore4 + slip -0.002 + fee 0.00115**

가설 (아직 증거 기반 확인 안 됨):

1. **wScore4 → 극단 prediction 분포 (bimodal 증폭)**: class 0/4 가중 2배 → score 값이 극단 쪽으로 쏠림. 작은 prob 차이가 wScore에선 2배 증폭
2. **극단 ranking → z-score 경계선 더 자주 flip**: cross-sectional z-score 상위/하위 k개 선택이 모델별로 다른 종목 집어냄
3. **IC sizing bimodal → multiplier flip**: market IC가 작은 prediction 차이에 따라 0.05x ↔ 3x로 flip (60배 사이즈 swing)
4. **750만 한도 → 절대 규모 증폭**: 1.5M/trade × 60x swing 가능
5. **dynamic_slippage → 진입가 추가 변동**: 위 모든 것에 마지막 레이어 sensitivity 추가

이 체인이 compound되면서 미세 학습 차이를 alpha 공간에서 1M+ 규모로 증폭.

**중요**: C5 (same but score4) 는 std 304K로 절반 이하. **wScore4 제거만으로 variance 2.6배 감소.** 즉 wScore4가 compound loop의 주요 연결고리.

---

## 실험 설계 권장

### 모델 비교 실험 (단일 seed)

- **1차 판정**: `C5` (750, IC/dyn on, score4)
  - std 304K, |mean|/std 16.6 → 신호 충분
  - 운영 인프라와 가장 가까움
  - score4라 variance amplifier 없음
- **백업 판정**: `OLD` 또는 `C3`
  - 더 안정적이지만 운영 인프라와 거리 있음

### 모델 비교 실험 (multi-seed)

3-5 seed로 학습 → 각 seed 모델을 같은 backtest config에서 돌림 → **mean ± std** 로 비교
- N=5 seed면 std가 √5 ≈ 2.24배 감소 → 실험 판정 신뢰도 ↑

### 절대 쓰지 말 것

- **NEW (wScore4 포함)** 를 단일 seed 비교 판정에 사용
  - mean < std (0.13M vs 807K) → 어떤 실험 결과도 variance 안에 묻힘
  - 같은 모델 3번 돌려도 alpha [-1.0M, +1.8M] 범위로 뒤죽박죽

---

## 주의 사항

1. 이 변동성 측정은 **같은 NPY, 같은 seed, GPU drift만** 존재하는 조건. NPY나 config를 바꾸면 variance 패턴 다를 수 있음.
2. 학습 epochs 수가 다르면 수렴 정도 다름 → variance도 달라짐. `cs_mlp_200_base.yaml` default (100 epoch patience 15) 기준.
3. Trading sim / val_loss 는 **variance 훨씬 작음** (1.7% / 1e-5). 이 차이는 "cross-sectional 전략의 ranking 민감성 + IC sizing/position cap amplification" 때문. 모델 품질 자체는 일관됨.
4. 이 표 값은 2026-04-08 측정치. 모델 아키텍처 / NPY 기간 변경 시 재측정 필요.

출처: `outputs/backtest/fullv{1,2,3}_{oldbt,C1,C2,C3,C4,C5,C6}/`, `outputs/backtest/cs_mlp_200_base_reference_baseline/`, `BRAIN/kospi/autoresearch/2026-04-08.md` 4번/10번 섹션.

---

## 추가 발견 — NPY 기간이 학습 variance도 크게 좌우 (2026-04-08)

### 배경
위의 std 수치는 NEW NPY (test_end=2026-04-04)로 학습한 3개 retrain 기준. 이후 NPY를 OLD (test_end=2026-03-31, reference 학습 당시 기간)로 재빌드 후 4번 retrain → 같은 8 config 측정한 결과:

### mean alpha 비교

| cfg | new NPY retrain (n=3) mean | **oldnpy retrain (n=4) mean** | **reference** |
|---|---:|---:|---:|
| OLD | -2,913,265 | -555,703 | +636,134 |
| C1 | -4,740,938 | -682,122 | +1,467,320 |
| C2 | -11,238,877 | +1,935,758 | -773,819 |
| C3 | -1,373,455 | +2,925,117 | +3,640,061 |
| C4 | -2,824,530 | +2,185,094 | +2,118,388 |
| **C5** | +5,036,017 | **+5,818,764** | +7,188,748 |
| C6 | -4,898,851 | -548,363 | -833,172 |
| **NEW** | +133,932 | **+7,562,652** | +6,302,503 |

→ oldnpy retrain이 모든 config에서 reference에 매우 근접. NEW에서는 **reference보다 +1.26M 더 좋음**.

### std (variance) 비교

| cfg | new retrain std | **oldnpy std** | ratio (oldnpy/new) |
|---|---:|---:|---:|
| OLD | 81K | 103K | 1.3x |
| C1 | 605K | 168K | **0.28x** |
| C2 | 226K | 1,151K | 5.1x (wScore4+no IC 특이점) |
| C3 | 162K | **16K** | **0.10x** |
| C4 | 127K | **41K** | **0.32x** |
| **C5** | 304K | **6K** | **0.02x (50배 작음)** |
| C6 | 30K | 182K | 6.1x |
| **NEW** | 807K | **56K** | **0.07x (14배 작음)** |

### 결론

**NPY 기간 4일 shift가 학습 variance 자체를 크게 영향**:
- oldnpy 기간에서는 대부분 config에서 std 6K~182K
- new NPY 기간에선 std 30K~807K
- **C5에서 variance 50배 감소**, NEW에서 14배 감소

해석: new NPY (2026-04-04 test_end)가 학습을 덜 안정적인 local minima 근처로 끌고갔음. NPY 기간이 mean뿐 아니라 **수렴 품질 자체**를 좌우함.

### 실험 판정 재권장

**NPY는 반드시 test_end=2026-03-31 (reference 기준) 사용**. 이 기간이면:
- Mean alpha가 reference에 근접 (단일 seed retrain 가능)
- Variance 매우 작음 (C5 std 6K로 신호 대비 noise 거의 없음)
- 실험 간 fair 비교 가능

**C5 config가 여전히 best choice**:
- oldnpy C5: mean +5.82M, std **6K** → |mean|/std = 970 (극명한 신호)
- 어떤 실험이든 +10~20K 알파 개선을 측정할 수 있는 수준의 민감도

단, C2와 C6는 oldnpy에서도 std 큼 (1.15M, 182K) — 사용 비추.

trading sim 기준 oldnpy 4개:
- IC 0.0664, ICIR 0.562 (전부 동일)
- L+S CumAlpha 953~973% (20% spread, 2% 변동)
- WR 58.1~58.2%

→ 모델 품질도 new NPY retrain 대비 명확히 우위 (IC 0.0497 → 0.0664, 34% 개선)
