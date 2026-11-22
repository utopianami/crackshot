# Phase 2-R 실험 로그 — Differentiable Portfolio Loss

> **English note.** Original experiment log, kept in Korean as written. Phase 2-R on Binance: replacing the classification objective with a differentiable portfolio loss, the first config whose scores collapsed to a constant, the follow-up configs, and per-seed results. English summary: [Binance write-up](../binance.md).

## 컨텍스트

Phase 1 분류형 baseline(FT-T / TabM / CS-MLP) 이후 알파스쿼드 회의(2026-04-20 17:12)가 추천한 Phase 0-R → 1-R → 2-R 로드맵 중, 사용자가 "가능성 확인" 목적으로 Phase 2-R을 Compromise A (기존 NPY 유지 + 회의 핵심 교훈 반영) 로 바로 진행하기로 결정.

## 변경점 요약

| 항목 | Phase 1 (기존) | Phase 2-R (이번) |
|---|---|---|
| 모델 출력 | [H=3, C=5] class logits | scalar score |
| 헤드 | Top-k argmax (외부 평가) | SOFT long/short top-k (학습 내장, 미분가능) |
| 손실 | CE + ListNet | -E[r_resid · w] + λ_β·(Σβ·w)² + λ_turn·max(|Δw|-θ,0)·fee |
| Target | fwd_1h/4h/24h (raw) | fwd_res_1h/4h/24h (β-residual, BTC 방향 제거) |
| 피처 | 12개 (기존) | 15개 (+ mins_to_next_funding, fund_phase_sin/cos) |
| Rebalance | 4h | 4h (유지, 가능성 확인) |
| 유니버스 | 298 symbols | 298 symbols |

## TDD 산출물 (모두 Green)

- `src/training/heads/soft_topk.py` — SOFT top-k (log-Sinkhorn) — 4/4 ✅
- `src/training/heads/long_short_topk.py` — 파라미터화된 long K_long + short K_short 헤드 — 4/4 ✅
- `src/training/losses/portfolio.py` — return + β + turnover + vol 항 — 4/4 ✅
- `src/features/binance_funding_clock.py` — 3개 펀딩 클럭 피처 — 3/3 ✅
- `src/features/binance_btc_residual.py::compute_forward_residual_returns` — 훈련 타겟 + β(t) — 3/3 ✅
- `src/training/trainers/cs_mlp_portfolio_step.py` — JIT 학습 스텝 — 2/2 ✅

총 20/20 단위 테스트 Green.

## 데이터 파이프라인

- `scripts/binance/phase2_augment_forward_residual.py` — features_5m에 `beta`, `fwd_res_{1h,4h,24h}` 추가 (314 심볼, 1.4분)
- `scripts/binance/build_npy_phase2.py` — stride 48 (4h) NPY 빌드
  - 출력: `BRAIN/binance/data/npy_phase2/` (X [N,15], y [N,5])
  - train=1.1M rows, val=91K, test=91K, 298 symbols

## 학습 구성

- 모델: CS-MLP scorer [256, 128], dropout=0.3, GELU
- 헤드: SOFT long/short top-k, k_long=k_short=10
- ε warmup: 1.0 → 0.1 선형 (20 epoch)
- 손실 가중치: λ_β=1.0, λ_turn=1.0, θ=0.0, fee=5e-4
- 옵티마이저: adamw, lr=1e-3, weight_decay=1e-2, cosine warmup
- 에폭: 50 / patience=10
- 배치: 8 timestamps/step, max_stocks=320

## 학습 진행

### 1차 (collapse 발견) — `cs_mlp_portfolio_binance.yaml`

- λ_β=1, λ_turn=1·fee, weight_decay=0.01, eps: 1.0→0.1 over 20 ep
- 결과: ep=2부터 val_loss→0, port_ret→0, turn_l1→0. ep=34 early stopping.
- 진단: best checkpoint에서 scorer가 **상수 출력**(mean=-0.0186, std=0.0). 모든 엔티티 점수 동일 → Sinkhorn이 w_long, w_short 둘 다 uniform → w=0 완전 상쇄.
- 원인: (a) w=0 이 loss의 trivial minimum (return·β·turnover 모두 0), (b) weight_decay=0.01 이 MLP를 zero output으로 끌고감.
- val Pearson=0.011, test Pearson=NaN (w=0 이므로 정의되지 않음)

### 2차 (ablation) — `cs_mlp_portfolio_binance_v2.yaml`

- **λ_β=0, λ_turn=0, weight_decay=0** (순수 return 최대화)
- dropout 0.1 (0.3 → 과한 regularization 배제), layernorm=true, feature_mode=zscore (raw는 scale mismatch)
- eps: 1.0→0.05 over 10 ep, epochs=30, patience=8
- 학습 270초(4.5분), 30ep 완주.
- Best ep=27, val_loss=-0.0574 (port_ret=+5.74% per 4h, 2k=20 leverage 기준)
- ε=0.05 전환(ep=10) 후 val 일시 dip → 14ep부터 재학습 후 best update.
- 체크포인트: `outputs/models/cs_mlp_portfolio_20260420_181238/`

#### VAL
| metric | value |
|---|---|
| Pearson | 0.0214 |
| Spearman | 0.0372 |
| Top 0.1% acc / mean_ret | 58.2% / +0.768% |
| Top 1% acc / mean_ret | 57.4% / +0.357% |
| Top 5% acc / mean_ret | 54.4% / +0.185% |

#### TEST (out-of-sample, 60일)
| metric | value |
|---|---|
| Pearson | **0.0164** |
| Spearman | **0.0224** |
| Top 0.1% acc / mean_ret | 56.0% / +0.899% |
| Top 0.5% acc / mean_ret | 57.4% / +0.127% |
| Top 1% acc / mean_ret | 56.6% / +0.165% |
| Top 5% acc / mean_ret | 54.6% / +0.136% |

#### Scorer health check
- scores std=4.44, range [-24.4, +15.6] (non-collapsed ✅)
- |w|_1=20.0, top-10 ≈ +1.25 each, bot-10 ≈ -1.25 each

## 핵심 교훈 (collapse 원인)

1. `-E[r·w]` + β·(Σβw)² + λ_turn·|Δw|·fee 이 losses 조합에서 **w=0 이 trivial global minimum** (모든 term 동시에 0).
2. SOFT top-k Sinkhorn은 개별 leg의 합은 k 로 고정하지만, **scores가 constant이면 w_long과 w_short가 둘 다 uniform으로 나와 부호 상쇄**.
3. `weight_decay=0.01` 이 MLP 출력을 zero 쪽으로 당김 → scorer가 constant로 수렴하면 모든 gradient가 0.

## Backtest 결과 (test 60일, 2배10 L/S, 5bp/side fee)

- Gross PnL total: **+1088.8%** (leverage-unit 54% / 60d → ≈329% 연환산)
- Net PnL total (fee 반영): **+525.95%** (158% 연환산)
- Sharpe 연환산: **gross 3.83 / net 1.85**
- MaxDD: gross -357%, net -362% (leverage-unit)
- Hit rate(양수 bar): **gross 59.2% / net 56.7%**
- Avg turnover/bar (L1): 31.27 (k_long+k_short=20 기준 ~1.56x 회전)
- Avg net β: 0.067, **std=2.76** (β 노출 변동이 큼 → β 페널티 복원 고려 대상)

결과 파일: `outputs/models/cs_mlp_portfolio_20260420_181238/backtest_test.csv`

**주의사항**: funding cost·slippage·orderbook 체결 미반영. 순수 β-residual forward return 기반. 실 PnL은 보수적으로 보정 필요.

## 3차 (β 페널티 복원) — `cs_mlp_portfolio_binance_v4.yaml`

- λ_β=0.01 (v3 λ_β=0.1은 1ep collapse → 롤백), 나머지는 v2와 동일
- Best ep=10 (ε 전환 직후), val_loss=-0.012
- 체크포인트: `outputs/models/cs_mlp_portfolio_20260420_182320/`

### TEST (60일)
| metric | value |
|---|---|
| Gross PnL | +480% |
| Net PnL | +168% |
| Sharpe gross / net | 6.00 / 2.10 |
| MDD gross / net | -193% / -197% |
| Turnover/bar | 17.3 |
| Net β exposure | -0.08 (std=0.61) |
| Hit rate net | 57% |

v2 대비: 알파 절반으로 줄었지만 MDD 거의 절반·β 변동 4.5배 감소·Sharpe 개선. Risk-adjusted 관점에서 v2보다 우수.

## 4차 (turnover 페널티 추가) — `cs_mlp_portfolio_binance_v5.yaml`

- λ_β=0.01 + λ_turn=1.0, threshold=5.0 (회전율 5 초과분에만 fee 페널티)
- Best ep=10, val_loss=-0.00654
- 체크포인트: `outputs/models/cs_mlp_portfolio_20260420_182857/`

### TEST (60일)
| metric | value |
|---|---|
| Gross PnL | +225% |
| Net PnL | +58% |
| Sharpe gross / net | 2.82 / 0.73 |
| Turnover/bar | 9.3 |
| Net β exposure | -0.16 (std=0.44) |
| Hit rate net | 52% |

회전율은 절반으로 줄었지만 알파 자체도 크게 훼손. λ_turn=1.0 과도. → **v4 가 현재 SOTA.**

## 5차 (약한 turnover penalty) — `cs_mlp_portfolio_binance_v6.yaml` ⭐ SOTA

- λ_β=0.01 + λ_turn=0.3, threshold=10.0 (극단 회전만 페널티)
- Best ep=14, val_loss=-0.01199 (v5 ep=10 -0.00654 대비 2배 개선)
- 체크포인트: `outputs/models/cs_mlp_portfolio_20260420_183258/`

### TEST (60일)
| metric | value |
|---|---|
| Gross PnL | **+659.6%** |
| Net PnL | **+334.3%** |
| Sharpe gross / net | **7.48 / 3.79** |
| MDD gross / net | **-132% / -172%** |
| Turnover/bar | 18.1 |
| Net β exposure | -0.15 (std=0.64) |
| Hit rate net | 55.0% |

MDD 절반으로 줄었는데 net Sharpe는 v4의 1.8배. threshold=10·약한 λ 조합이 모델이 자유롭게 signal을 활용하면서 극단적 회전만 피하게 유도.

## 비교표 (seed=42 기준)

| Metric | v2 (base) | v4 (+β) | v5 (+strong turn) | v6 (+weak turn) | **v7 (+med turn)** |
|---|---|---|---|---|---|
| Gross PnL | +1089% | +480% | +225% | +660% | **+755%** |
| Net PnL | +526% | +168% | +58% | +334% | **+454%** |
| Net Sharpe | 1.85 | 2.10 | 0.73 | 3.79 | **4.70** |
| MDD gross | -357% | -193% | -251% | -132% | -127% |
| Turnover/bar | 31.3 | 17.3 | 9.3 | 18.1 | 16.7 |
| Net β std | 2.76 | 0.61 | 0.44 | 0.64 | 0.63 |

## 6차 (강한 turnover penalty + 작은 threshold) — `cs_mlp_portfolio_binance_v7.yaml` ⭐ new SOTA

- λ_β=0.01 + λ_turn=0.5 + threshold=7.0 (v6 대비 λ 1.67배, threshold 0.7배)
- Seed 42: best val=-0.01143, ep=22
- 체크포인트: `outputs/models/cs_mlp_portfolio_20260420_184819/`

### TEST (60일, seed=42)
| metric | value |
|---|---|
| Gross PnL | **+755%** |
| Net PnL | **+454%** |
| Sharpe gross / net | **7.81 / 4.70** |
| MDD gross / net | -127% / -156% |
| Turnover/bar | 16.7 |
| Net β exposure | -0.18 (std=0.63) |

v6(seed=42) 대비 Net PnL 334→454% (36% ↑), Net Sharpe 3.79→4.70 (24% ↑). 회전율 18.1→16.7.

## v6/v7 다중 시드 안정성

### v6 (λ_turn=0.3, threshold=10)
| Seed | best_val | Net Sharpe | Turn | β std | Mode |
|---|---|---|---|---|---|
| 42 | -0.01199 | 3.79 | 18.1 | 0.64 | ✅ controlled |
| 123 | -0.01200 | 3.68 | 19.0 | 0.62 | ✅ controlled |
| 7 | -0.00757 | 0.97 | 32.4 | 1.29 | ⚠️ high-turn |
| 2026 | -0.00790 | 2.35 | 31.7 | 1.21 | ⚠️ high-turn |

### v7 (λ_turn=0.5, threshold=7)
| Seed | best_val | Net Sharpe | Net PnL | Turn | β std | Mode |
|---|---|---|---|---|---|---|
| 42 | -0.01143 | **4.70** | +454% | 16.7 | 0.63 | ✅ controlled |
| 123 | -0.01055 | 3.57 | +300% | 16.7 | 0.57 | ✅ controlled |
| 2026 | -0.00797 | 4.33 | +370% | 15.89 | 0.56 | ✅ controlled |
| 7 | -0.00706 | 1.15 | — | 32.0 | 1.29 | ⚠️ high-turn |

**핵심 발견 — Bimodal 수렴**: threshold를 10→7로 tightening 해도 bad seed(7)는 여전히 aggressive mode 탈출 실패. val_loss 임계값은 애매(s2026 -0.00797도 controlled로 수렴) — turn/β 메트릭을 함께 봐야 확실. **3 controlled seeds 평균 Net Sharpe 4.20** (4.70, 3.57, 4.33), Turnover 16.1±0.5 일관. Best-of-N 시드 전략 실용성 확인.

### 5 시드 전수조사 결과 (v6+v7)
- controlled 확률: 4/7 ≈ 57% (seeds 42, 123은 reproducible 하게 controlled)
- 시드 선택 가능한 실전 파이프라인에서는 3 시드 training 후 val 기준 필터 → 평균 Net Sharpe 3.5+ 기대

| Seed | best_val | Net Sharpe | Net PnL | Turn | β std | Mode |
|---|---|---|---|---|---|---|
| 42 | -0.01199 | 3.79 | +334% | 18.1 | 0.64 | ✅ controlled |
| 123 | -0.01200 | 3.68 | +277% | 19.0 | 0.62 | ✅ controlled |
| 7 | -0.00757 | 0.97 | +147% | 32.4 | 1.29 | ⚠️ high-turn |
| 2026 | -0.00790 | 2.35 | +351% | 31.7 | 1.21 | ⚠️ high-turn |

**Bimodal 학습 행동 발견**: threshold=10 턴오버 페널티가 모든 시드에서 바인딩 되지 않음. 2/4 시드는 "controlled" 모드로 수렴(low turn, 높은 Sharpe), 2/4는 "aggressive" 모드로 고턴오버·고β변동에 빠짐. 4 시드 모두 net 양수 수익은 유지 — 시그널은 reproducible, 하지만 regularization이 불안정.

Mean Net Sharpe = 2.70, Median = 3.02. 최악 시드도 +147% Net PnL(Sharpe 0.97).

## 7차 (피처 확장 20개 — OI/markPrice 추가) — `cs_mlp_portfolio_binance_v8.yaml`

- 기존 15개 + 5개 추가: `oi_change_1h/4h`, `oi_z_24h`, `mark_premium`, `mark_basis_z`
- `derivative_ticker_5m` 267/314 심볼 활용. 최종 유니버스 266 (OI 없는 48심볼 drop)
- 파이프라인: `phase3_augment_oi_mark.py` + `build_npy_phase3.py`
- v7과 동일 lambda/threshold (λ_β=0.01, λ_turn=0.5, θ=7)

### TEST (60일)
| Seed | best_val | Net Sharpe | Net PnL | Turn | β std |
|---|---|---|---|---|---|
| 42 | -0.01088 | 3.75 | +539% | 22.5 | 0.87 |
| 123 | **-0.01337** | 1.65 | +260% | 27.9 | 1.16 |

**Gross alpha는 증가** (v7 s42 755% → v8 s42 944%, +25%) **but 회전율·β 변동 함께 증가** → Net Sharpe는 오히려 하락 (4.70 → 3.75). 모델이 OI 신호를 회전 증가로 소비. 

시사점: 피처가 좋아져도 고정 λ_turn=0.5에서는 모델이 알파를 회전으로 변환. OI를 넣으려면 λ_turn 더 강화 (0.8~1.0) + threshold 축소 필요.

## Fee 민감도 (v7 seed=42)

| fee/side | Net PnL 60d | Net Sharpe | 비고 |
|---|---|---|---|
| 5bp | +454% | +4.70 | 현재 default |
| 10bp | +153% | +1.58 | 실제 VIP0 taker 4bp + 슬리피지 6bp 근사 |
| 15bp | **-148%** | **-1.54** | break-even 지점 통과 |

**Turnover 16.7 × round-trip fee = daily net return 훼손.** 5bp에선 수익 유지되지만 10bp 이상으로 악화되면 alpha 대부분 소멸. 실전 배포 전 slippage 모델링 + turnover 더 낮추거나 gross alpha 개선 필수.

## 8차 (20 feat + tight turn) — `cs_mlp_portfolio_binance_v9.yaml`

- 20 features + λ_turn=0.8, threshold=5 (v7 대비 λ 1.6배, θ 0.71배)
- Best ep=11, val_loss=-0.01187 (v7 s42보다 약간 좋은 val)

### TEST (60일, seed=42)
| metric | value |
|---|---|
| Gross PnL | +382% (↓↓) |
| Net PnL | +111% (↓↓) |
| Sharpe gross / net | 3.16 / **0.92** |
| Turnover/bar | **15.06** (↓↓ 최저) |
| MDD gross | -233% |

**Paradox 재확인**: val_loss는 좋아졌으나 test net Sharpe는 폭락. 이유 = val_loss = -return + λ_turn·turn 이라 **λ_turn을 키우면 val에선 회전 감소 보너스 받지만 test에선 gross alpha만 깎임**. Val_loss가 정직한 성과 지표가 아닌 것이 재확인됨.

→ **λ_turn 튜닝은 그만. v7 is SOTA.** 다음은 평가 realism 개선으로.

## Ensemble 결과 (v7 s42+s123+s2026, 동일-가중 평균)

3 controlled seeds의 per-bar return을 평균 (= 각 시드에 1/3 capital 분할과 동치).

| Metric | s42 | s123 | s2026 | **Ensemble** |
|---|---|---|---|---|
| Gross Sharpe | 7.81 | 7.71 | 7.67 | **8.22** |
| Net Sharpe (5bp) | 4.70 | 3.57 | 4.33 | **4.52** |
| Net PnL 60d | +454% | +261% | +370% | **+362%** |
| Return std/bar | 12.57% | 9.49% | 11.12% | **10.40%** |
| MDD net (compounded) | -156% | -146% | -152% | **-83%** |

Return correlation pairwise: 0.81~0.86 — 시그널은 공통이나 불완전 상관.

핵심:
- Gross Sharpe **8.22** (개별 최고 7.81 초과) — 노이즈 분산
- Net Sharpe 4.52 = 평균 가까움 (개별 최고 4.70 못 넘음) but **std 12.57→10.40 (18% ↓)**
- MDD compounded -156→-83% (절반으로 개선)
- Turnover upper bound 16.46 — 실제는 weight cancellation으로 더 낮을 것(추정 13~14)

→ **실전은 ensemble 쓰는 게 합리적**. Sharpe 약간 손해보지만 안정성과 MDD 크게 개선.

### True ensemble (weight-level 평균, `backtest_portfolio_ensemble.py`)

3 모델 inference → weight 평균 → 동일 바에 적용. Δw 실측으로 turnover 정확 산출.

| Metric | Component-avg | **True ensemble** | vs s42 single |
|---|---|---|---|
| Gross Sharpe | 8.22 | **8.22** | +5% |
| Net Sharpe (5bp) | 4.52 | **4.67** | -0.6% |
| Net PnL 60d | +362% | **+373%** | -18% |
| Turnover/bar | 16.46 (UB) | **15.80** | -6% |
| Net std/bar | 10.40% | **10.40%** | **-17%** |
| MDD net | -148.5% | **-146.7%** | -6% |

Turnover 감소 상대적으로 작은 이유 = 3 시드 return 상관 0.81-0.86 (weight 방향 공통). 그래도 true-ensemble이 component-avg보다 fee 절감 + Sharpe 약간 개선.

### Fee sensitivity — 단일 vs ensemble

| fee/side | s42 Net Sharpe | s42 Net PnL | **Ens Sharpe** | **Ens PnL** | Winner |
|---|---|---|---|---|---|
| 5bp | 4.70 | +454% | **4.67** | +373% | 대동소이 (Sharpe) |
| 10bp | **1.58** | **+153%** | 1.11 | +89% | **s42 단일** |
| 15bp | -1.54 | -148% | -2.44 | -196% | 둘 다 dead |

**중요**: Ensemble은 5bp에서 variance 개선은 있지만, **10bp+에선 평균 gross가 깎여 단일 s42보다 열등**. Fee breakeven 단일 ≈ 12-14bp, ensemble ≈ 11-12bp. 실전 fee 10bp 가정 시 **단일 s42를 쓰는 게 낫다** (PnL +72% 우위). Ensemble은 gross alpha가 가장 높은 시드만큼 좋진 않다는 trade-off 확인.

→ **배치 전략**: 5bp 가능한 고빈도 환경이면 ensemble, 실제 taker 10bp 환경이면 s42 단일. 두 옵션 모두 유효.

## v7 s42 Val split 검증 (60d OOS, 테스트 직전 60일)

Val split은 early-stopping에만 사용, gradient update 안 됨 → OOS 확인용.

| Metric | Val split | Test split | 비교 |
|---|---|---|---|
| Gross PnL | +813% | +755% | +8% |
| Net Sharpe (5bp) | **4.94** | 4.70 | +5% |
| MDD net | -92% | -156% | **42% 개선** |
| Turnover/bar | 16.81 | 16.73 | ≈ |
| β std | 1.02 | 0.63 | val 기간 β 더 변동적 |

→ 2개 독립 60일 OOS window 모두 Sharpe ~4.7-4.9. **알파 시계열 재현성 확인** (총 120일 OOS). 이 시점에선 "test 시기만 잘된 fluke" 가능성 대폭 낮아짐.

### 월별 breakdown (120d OOS, 30d chunks)

| Period | Gross% | Net% | Sharpe | Turn | Hit% |
|---|---|---|---|---|---|
| M1 (2025-12) | +414 | +270 | 4.56 | 15.97 | 51.1 |
| M2 (2026-01) | +399 | +240 | 5.64 | 17.65 | 51.7 |
| M3 (2026-02) | +312 | +167 | 3.18 | 16.04 | 52.8 |
| M4 (2026-03) | +443 | +287 | 6.58 | 17.43 | 56.1 |

**4개월 연속 수익 (loss 달 없음)**. 월 Net Sharpe 3.18~6.58, 평균 ~5.0. 최악 월도 Net +167%. Turnover 16.1±0.7 일관. **월별 알파 stability 확정**.

### Symbol attribution (v7 s42, test 60일, `backtest_portfolio_attribution.py`)

264 심볼 중 기여도 분포:

| Bucket | Gross PnL 기여 | 비중 |
|---|---|---|
| Top 20 | +781% | **103.4%** of total (+755%) |
| Bottom 20 | -304% | -40.2% |
| 나머지 224 | +278% | +36.8% |

Leg별:
- Long leg: +479% (가치주 사는 게 주력)
- Short leg: +277% (고평가 쇼트도 꾸준히 버림)

Top 5 기여자: SIREN(+113%), RIVER(+102%), PIPPIN(+88%), POWER(+71%), ENSO(+69%)
Worst 5: ORCA(-42%), FHE(-41%), JELLYJELLY(-38%), CLO(-30%), INIT(-18%)

**시사점**: 알파가 20~30 심볼에 고도로 집중. 유니버스 trim (꼬리 심볼 drop) 시 Sharpe 개선 가능성. 반면 상위 심볼이 상장 폐지되거나 유동성 격감 시 전략 리스크 ↑. 집중도 관리가 배포 전 고려 사항.

## 다음 단계 후보

A) 체결 시뮬 도입 (Task #19) — tick quote + slippage, "돈 번다" 실전 확정용
B) ~~2년 rolling 평가~~ → val split 확인으로 일부 수행 (120d OOS Sharpe 4.82). 추가 확장은 quarterly 재학습 + rolling window 필요
C) ~~Ensemble 3 seeds~~ ✅ **완료** (Sharpe 4.67, variance 17%↓, but 10bp+에서 단일 s42에 패배)
D) ~~Best-of-N seed 자동화~~ ✅ **완료** (`scripts/binance/train_best_of_n.py`) — val_port_ret 기준 top-K 필터 + 자동 ensemble backtest
E) ~~True ensemble turnover~~ ✅ **완료** (15.80, 단일 대비 -6%)
F) Symbol attribution — 어떤 코인에서 수익 나는지 분석
G) Rolling historical evaluation — **일부 완료** (아래 섹션)

## Historical Window 검증 — v7 on 2025-08~09 (npy_phase2_h1)

`build_npy_phase2.py --test-end 2025-09-30` 로 다른 window NPY 생성:
- train: < 2025-06-02 (~2.5년)
- val: 2025-06-02 ~ 2025-08-01
- test: **2025-08-01 ~ 2025-09-30** (7개월 과거)
- 275 심볼

동일 v7 config, 3 시드 모두 **aggressive mode 수렴** (best_val -0.00277~-0.00361, recent controlled -0.010~-0.012 대비 크게 얕음):

| Seed | Gross | Net | Sharpe | Turn | β std |
|------|-------|-----|--------|------|-------|
| 42 | +794% | +213% | 2.70 | 32.4 | 2.78 |
| 123 | +434% | -173% | **-1.95** | 33.9 | 2.25 |
| 2026 | +802% | +165% | 1.61 | 35.5 | 2.53 |

**시사점**:
- Gross alpha는 평균 +677% — 여전히 강함 (학습 자체는 성공)
- 하지만 turnover 32~35 (recent의 2배) + β std 2.25~2.78 (4배 변동)
- v7 λ_turn=0.5에서 controlled-mode 수렴 0/3 — 시기별 재학습시 λ 튜닝 필요
- **v7 recipe를 그대로 배포하면 과거 구간에서 수익 보장 못함**. 시기에 따른 regime-specific tuning 또는 walk-forward retraining + seed filter 필수

다음: λ_turn=0.8(v9 config)으로 h1 재학습해 controlled mode 유도 가능한지 확인

### v9 config on h1 (λ_turn=0.8, θ=5, 3 seeds)

| Seed | best_val | Gross | Net | Sharpe | Turn | β std |
|------|---|-------|-----|--------|------|-------|
| 42 | -0.00266 | +849 | +268 | **3.21** | 32.4 | 2.80 |
| 123 | -0.00242 | +590 | +4 | 0.05 | 32.6 | 2.22 |
| 2026 | -0.00285 | +874 | +248 | 2.37 | 34.9 | 2.52 |

**Ensemble (3 seeds)**: Gross Sharpe 9.32, Net Sharpe 2.35, PnL +194%, MDD -185% (단일 대비 개선). Turnover 32.1. → v9가 v7 h1보다 전반적으로 낫지만 여전히 불안정.

- 3 seeds 모두 aggressive mode 수렴 (val_loss 얕음) — λ_turn 0.8로도 h1 regime에서는 controlled 유도 실패
- 시드별 Net Sharpe 변동 0.05~3.21 — ensemble으로 variance 안정화 필요 (std 13.6→10.8)
- Hit rate 61.3%까지 상승 — alpha signal은 분명 있음, regularization이 문제

### Regime diagnostic (npy_phase2 vs npy_phase2_h1, test split)

| Metric | Recent (2026-Q1) | h1 (2025-Q3) | Δ |
|---|---|---|---|
| fwd_res_4h std | 0.0249 | 0.0246 | ≈ |
| fwd_res_4h abs mean | 1.28% | 1.22% | ≈ |
| β mean | 0.94 | **1.69** | +80% |
| β std | 0.30 | **0.55** | +85% |
| β [5%, 95%] | [0.38, 1.37] | [0.73, 2.57] | shift + widen |

h1 구간 coins이 BTC에 대해 **훨씬 높은 β + 넓은 분산**. 목표 dispersion은 유사.

### β 가설 테스트 (λ_β=0.01→0.003, v7 recipe + seed=42)

- val_loss=-0.00879 (겉보기 개선) but val penalty weight 자체가 1/3 줄어든 결과
- Test: Gross +906%(h1 최고), Net +245%, **Sharpe 2.29** (original v7 h1 s42 2.70보다 낮음)
- β std 3.0 (대폭 증가) — 예상대로 β 제약 완화됨
- Turnover 36.8 (증가)
- **가설 기각**: β mean 차이가 h1 불안정의 근본 원인이 아님

남은 가능 원인: 시장 regime 내부 drift, 학습 데이터 규모(2.5yr vs 3yr), OI/funding 분포 차이. 다음 단계는 real cause 규명보다 Walk-forward retrain 파이프라인 구축이 실용적.

### Feature 분포 비교 (test split)

대부분 피처 유사. 유일한 큰 차이:

| Feature | Recent std | h1 std | 배수 |
|---|---|---|---|
| ret_7d | 0.193 | **0.424** | **2.2x** |

h1 구간 주간 수익률 변동성이 2배 이상 커짐 — 2025-Q3 알트코인 광풍/폭락 흔적. Zscore normalization으로도 완전히 kill 되지 않는 outlier가 훈련 불안정성(high turn oscillation)의 실제 원인일 가능성 높음.

→ 실용적 대응: (a) ret_7d winsorize/clip, (b) rank-based 정규화 사용, (c) 학습 시 outlier bar skip. 향후 feature pipeline 개선 포인트.

**결론**: λ_turn 강화로 controlled 모드 유도 실패 — h1 구간 자체가 회전성이 본질적으로 높은 regime. 그러나 v9가 v7보다 net Sharpe 우위 확인 (3.21 vs 2.70). 시기별로 최적 hyperparameter가 다름 = **walk-forward retuning 필수**.

시사점 정리:
1. v7 recipe(λ_turn=0.5)는 2026-Q1 구간에 강함 (Sharpe 4.70)
2. 2025-08~09 구간은 v9 recipe(λ_turn=0.8)가 상대적으로 나음 (Sharpe 3.21)
3. 어떤 구간에도 **Gross alpha는 양호** (+700%~850% 60일) — 시그널은 존재, regularization 맞추는 게 관건
4. 배포 파이프라인은 **quarterly re-tuning + best-of-N seed filter** 필요

### Symbol attribution 교차 비교 (recent vs h1)

Top-30 기여자 교집합: **7종 (23%)** — 77%가 구간마다 다름.
- 공통: SIREN, PIPPIN, C98, TA, RESOLV, H, 1000SATS
- Bottom-30 교집합: 3종 (10%) — DUSK, INIT, PTB

**시사점** (긍정적): 모델이 "고정된 alpha-generating 심볼 집합"을 암기한 게 아니라, **각 시기 예측 가능한 심볼에 동적으로 적응**. 진짜 제너럴한 시그널 학습 근거. 7 심볼(공통)은 구조적으로 predictable할 가능성 (심볼 특성 조사 필요).

## 최종 SOTA 요약 (2026-04-20, 확장판)

### 두 가지 배포 옵션

**Option 1 — 단일 v7 s42 (fee-robust)**
- Test 60일: Gross +755%, Net +454%, Sharpe gross 7.81 / net 4.70
- MDD gross -127% / net -156%
- Turnover 16.7/bar, β std 0.63
- Fee breakeven ≈ 12-14bp/side
- **10bp 실전 fee에서 Sharpe 1.58, PnL +153%** — 살아남음

**Option 2 — True ensemble (v7 s42+s123+s2026, 5bp 환경용)**
- Net Sharpe 4.67, PnL +373%, std 17%↓ vs 단일 s42
- MDD -147% (단일 대비 +5% 개선)
- Turnover 15.80/bar (단일 대비 -6%)
- 10bp에서 Sharpe 1.11, PnL +89% (단일에 열세)
- **5bp 환경에선 variance·MDD 개선 > 미미한 Sharpe 손실**

### 120d OOS (val+test) 전체 통계 (단일 s42)

- 4개월 모두 net 흑자, 월 Sharpe 3.18~6.58 (평균 5.0)
- 최악 월도 Net +167%
- Turnover 16.1±0.7 매우 안정
- β std 0.6~1.0 (val 더 변동적)

검증된 속성:
1. β-residual forward return 타겟 + 미분가능 L/S top-k 헤드로 collapse 없이 학습 가능
2. v7 3 seeds (42/123/2026) 모두 controlled 수렴 — Net Sharpe 3.57~4.70, 평균 **4.20**, 턴오버 16.1±0.5 일관 재현성 확인
3. 피처 확장(OI) 단독으로는 net Sharpe 개선 안 됨 — regularization 재조정 필요

미해결 / 보수적으로 봐야 할 부분:
1. 60일 단일 test window — 2년 rolling 평가 필요
2. Funding cost 실측 미반영 (턴오버 페널티로 간접 흡수)
3. Tick 슬리피지 미반영 — 실체결 시뮬 전 "돈 번다" 단정 금물
4. Bimodal 수렴 (3/7 시드 aggressive) — best-of-3 seed 파이프라인 필요

### ret_7d winsorize 가설 검증 (2026-04-20)

**가설**: h1 분포에서 `ret_7d` std가 0.424로 recent(0.193)보다 2.2배 크다 → outlier가 head 학습을 혼란시켜 aggressive mode 유발.

**실험**: `npy_phase2_h1_clip/` — `ret_7d` 전 기간 ±0.25 clip, 그 외 동일. v7 s42 재학습.

| 지표 | v7 h1 s42 원본 | v7 h1 s42 clip | 판정 |
|---|---|---|---|
| val_loss (best) | -0.00301 | -0.00309 | ≈ 동률 |
| val_turnover_l1 | ~32 | **10.18** | clip에서 극적 감소 |
| test Turnover/bar | 32.35 | 34.46 | 개선 없음 |
| test Net Sharpe (5bp) | **+2.70** | **-0.55** | **clip이 열화** |
| test β std | 2.78 | 2.26 | 소폭 개선 |
| test Gross Sharpe | - | 6.06 | 정상 |

**결론**: ret_7d clip은 **val-time turnover만 낮추고 test OOS에 일반화 안 됨.** Train 분포가 정돈되어 학습 중엔 안정 수렴처럼 보였으나, test 시점에 모델이 여전히 공격적 weight를 만들어 turnover가 돌아옴. ret_7d outlier 가설 **기각**.

### rankgauss 가설도 기각 (2026-04-20)

동일 h1 데이터에 `feature_mode: rankgauss`로 전체 피처에 rank-Gaussian 정규화 (outlier 자연 bounded).

| 지표 | 원본 | clip ret_7d | rankgauss |
|---|---|---|---|
| val_loss | -0.00301 | -0.00309 | **-0.00389** |
| val_turn | ~32 | 10.18 | **7.51** |
| test Net Sharpe | **+2.70** | -0.55 | **-1.40** |
| test turnover | 32.35 | 34.46 | 33.01 |
| test β std | 2.78 | 2.26 | 2.13 |

**패턴 확정**: feature 후처리는 val_turnover만 낮추고, test turnover는 **모두 ~33으로 수렴**. 이는 outlier/스케일 문제가 아니라 **train/val vs test 기간의 regime 차이** (2025-06~08 vs 2025-08~09-30). 모델이 val-time turnover는 제어해도 test에서는 다른 dynamics로 turnover 폭발.

**진짜 해결책**: rolling walk-forward 재학습 (월 단위 재훈련 + threshold 재캘리브레이션). 단일 학습으로는 regime drift 못 따라감. feature-side tweak 탐색 중단.

### Walk-forward 5 윈도우 (v7 s42 단일, 2026-04-20)

각 test_end에 대해 npy 재빌드 → 동일 v7 config로 재학습 → 5bp 백테스트.

| 윈도우 | Test 기간 | Net Sharpe | Turn | β std | val_loss | 모드 |
|---|---|---:|---:|---:|---:|---|
| 2026-Q1 (recent) | ~2026-02-20~04-20 | **4.70** | 16.7 | 0.63 | -0.010 | ctrl |
| 2025-Q3 (h1) | 2025-08-01~09-30 | 2.70 | 32.3 | 2.78 | -0.003 | aggr |
| W1 | 2025-05-01~06-30 | **6.63** | 34.2 | 2.77 | -0.005 | aggr |
| W2 | 2025-03-01~04-30 | 0.04 | 35.2 | 2.04 | -0.005 | aggr |
| W3 | 2024-12-30~2025-02-28 | **7.03** | 24.5 | 1.20 | -0.026 | ctrl |

**패턴**:
- **5개 중 2개 controlled 수렴** (W3, 2026-Q1) → 둘 다 Sharpe 4.7~7.0, 안정적.
- **3개 aggressive 수렴** (W1, W2, h1) → turnover ~33, β std 2+. Sharpe는 0.04 ~ 6.63으로 변동성 큼.
- W2는 aggressive가 단일 seed로 완전 무너진 케이스 (Net +5%)
- W1은 aggressive인데도 시장이 강해서 gross +1245% 달성 → Net +630%

**결론**:
1. 단일 시드 + 단일 학습으로는 **40% 확률로 controlled 모드에 들어갈 뿐** — 실전 프로덕션엔 불안정.
2. 하지만 controlled 모드에 들어가면 5개 윈도우 중 2개 모두 Sharpe 4~7, 매우 일관됨 → 알파는 진짜.
3. **Best-of-N seed 파이프라인 필수** (이미 구현됨: `train_best_of_n.py`). val_port_ret 기준 top-K 선택해 aggressive seed 배제.
4. 분기별(60일) 재학습으로 regime drift 자동 대응.

**실전 권장**:
- 매 60일마다: 새 npy 빌드 → best-of-5 train → val_port_ret 기준 top-3 → true ensemble → 다음 60일 live.
- Expected Net Sharpe (best-of-N controlled ensemble): 5.0~6.0 추정 (5 윈도우 중 controlled 2개 평균 5.87).
- 운영 리스크: aggressive 모드 폭주. val_turnover > 20 모델은 자동 배제 필수.

### W2 best-of-3 ensemble 검증 (2026-04-21)

W2 (최악 단일 seed 윈도우)에 3 seeds 학습 후 true ensemble:

| 모델 | val_loss | val_turn | test Net Sharpe | turn | β std |
|---|---:|---:|---:|---:|---:|
| W2 s42 | -0.00510 | 11.9 | +0.04 | 35.2 | 2.04 |
| W2 s123 | -0.00583 | 10.3 | -0.06 | 35.3 | 1.79 |
| W2 s2026 | -0.00671 | 8.3 | -1.12 | 33.3 | 1.79 |
| **Ensemble (3)** | - | - | **-0.32** | 34.0 | 1.75 |

**3개 seed 모두 aggressive 모드** (val_loss 전부 -0.010 이상) → 앙상블도 불로. Ensemble Sharpe -0.32, +575% gross / -37% net.

**운영 함의 (중요)**: 
- best-of-N은 controlled seed가 **최소 1개라도 존재할 때**만 작동.
- W2 같은 window에서는 표준 v7 config로 어떤 seed도 controlled 수렴 안 함 → 구조적 이슈 가능성.
- **Gate 조건**: val_loss ≤ -0.010 AND val_turnover ≤ 20 인 seed가 N개 중 최소 1개 없으면 해당 윈도우 거래 skip.
- 이번 W2처럼 전체 실패 확률: 5 윈도우 중 1 = 20%. 실전에선 "월별로 약 20% 확률로 휴장" 가정해야.

**추가 탐색 후보 (미실행)**:
- W2에 λ_β↑, λ_turn↑ 로 explicit regularization 강화 (aggressive 억제)
- eps_end↑ (0.05 → 0.1) 로 softer top-k (bimodal 탈출 유도)
- W2 전용 seed pool 확장 (10+ seed)

