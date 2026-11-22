# v7 → v8 리서치 로그 (stride anchor 수정 이후)

> **English note.** Original record, kept in Korean as written. An English index of all records is in the [README](../../README.md).

작성: 2026-04-22
트리거: Opus/Sonnet 감사로 stride 4h UTC 앵커 버그 발견. 기존 λ_fund 실험 결과 신뢰성 재검증 필요.

---

## 평가 기준 (확정 2026-04-22)

각 실험은 3윈도우(insample / h1 / w1)에서 평가하고, 한 윈도우당 PnL 그래프에 **두 개의 선**을 그린다:

- **net_wo_fund** = 수수료 + bid/ask 테이커 슬리피지 (실측 bookticker)
- **net_w_fund** = net_wo_fund + funding 현금흐름 (받거나 주거나)

**산출물 구조**: λ/피쳐/모델 버전 1개당 **1장의 사진 × 3 subplot** (insample/h1/w1 시간 순서).
한 Stage에 후보 4개면 **4장 × 3 subplot = 12 그래프**, 각 그래프에 net_wo_fund + net_w_fund 두 선.

**스크립트**: `scripts/binance/analyze_bidask_pnl.py` (수수료+실측 스프레드+펀딩 모두 반영)

---

## 확정 계획 (6단계)

각 단계 끝날 때 3윈도우 결과 표 + 그래프 공유.

### 1단계 — λ_fund sweep (진행 중)
- NPY: phase2 (15 features: 12 base + 3 clock)
- λ ∈ {0.0, 0.5, 1.0, 2.0} × 3 윈도우 (insample/h1/w1) = 12 학습
- 산출: 4장 그래프 → **best λ 확정**
- 상태: insample 4개 학습 완료, h1/w1 학습 필요

### 2단계 — Tier-A 13 피쳐 추가
- NPY: phase3_tierA (33 features = 12 base + 3 clock + 13 Tier-A + 5 CS)
- 모델/λ: 1단계 best λ 고정
- 윈도우: insample/h1/w1
- 산출: 1장 × 3 subplot (phase2 대비 비교 — 1단계 best와 overlay)

### 3단계 — Auto-feature gate (VIF + IC)
- 33 피쳐에 통계 필터 (VIF 다중공선성 + block-bootstrap IC 유의성)
- 생존 subset 으로 재학습 × 3윈도우
- 산출: 1장 × 3 subplot (2단계 대비 비교)

### 4단계 — 모델 아키텍처 4개 비교
- 모델: TabM / FT-Transformer / Tabsyn-latent / cs-attn
- 피쳐: 3단계 gate-survivor 고정
- λ: 1단계 best λ 고정
- 윈도우: insample/h1/w1
- 산출: 4장 × 3 subplot

### 5단계 — AlphaForge 피쳐 투입
- 4단계 best 모델 + AlphaForge 생성 피쳐 추가
- 윈도우: insample/h1/w1
- 산출: 1장 × 3 subplot

### 6단계 — 최종 v8
- best λ × gate-survivor 피쳐 × best 모델 (+ AlphaForge 피쳐)
- 3윈도우 전부 안정적 net_w_fund 양수 확인
- 산출: 1장 × 3 subplot + 최종 리포트

---

## 원래 계획 (참조)

### Stage 1 — 피처 개선
- **1-a** ✅ 펀딩피 피쳐 추가 (ret/residual/clock 8개)
- **1-b** 🔄 재평가 중 (1단계) — 원래 "λ=2.0 best" 결론은 stride 버그 데이터 위 결과
- **1-c** ✅ heimerdinger 논문 피쳐 카탈로그 → Tier-A 13 (phase3_tierA)
- **1-d** ⏳ Auto-feature gate (3단계에 해당)

### Stage 2 — 모델 비교 → 4단계에 해당
### Stage 3 — AlphaForge → 5단계에 해당

---

## 버그 발견 후 점검 계획

### Phase A — 감사 (완료)
- Opus + Sonnet 병렬 리키지 감사 → critical 0, minor 14
- Opus 코드 품질 감사 → critical 0, minor 5

### Phase B — 수정 및 재검증
- Phase 1 ✅ leakage invariant 테스트 9+3=12개 추가
- Phase 2 ✅ 코드 수정 (stride anchor, Corwin NaN 등 마이너 5건)
- Phase 2.5 ✅ phase2/phase3 NPY 리빌드 (3윈도우 × 2 = 6 NPY)
- **Phase 2.6** ⏳ v7 λ_fund 재평가 (현재, = 1단계)
- Phase 3 ⏳ 테스트 품질 개선 (ddof 고정, 참조값 비교)
- Phase 4 ⏳ 문서/docstring (bar-close 관례, warmup, beta 이중 역할)

---

## 1단계 중간 결과 — 간이 백테스트 (참고용, 공식 아님)

⚠️ 아래는 `backtest_portfolio.py` 기준(fwd_res 알파 + 고정 5bp 수수료 모델)으로 **펀딩 현금흐름 반영 안 됨**. 공식 평가 기준은 `analyze_bidask_pnl.py` 의 net_wo_fund / net_w_fund.

insample (2026-01-30 ~ 2026-03-31, 361 × 4h bars)

| λ_fund | 데이터 | Gross Sharpe | Net Sharpe (5bp) | Net PnL | β | Hit(net) |
|---|---|---|---|---|---|---|
| 2.0 | preanchor | +2.40 | -0.31 | -32% | -0.01 | 48.8% |
| 0.0 | anchor | +6.50 | +4.20 | +757% | -0.03 | 52.6% |
| 0.5 | anchor | +0.81 | -0.29 | -40% | +0.00 | 52.4% |
| 1.0 | anchor | (학습 완료, 백테스트 대기) | | | | |
| 2.0 | anchor | -3.16 | -4.21 | -616% | -0.32 | 44.0% |

공식 비교는 analyze_bidask_pnl.py 결과 나오면 별도 표로.

---

## 현재 상태 & 다음 액션

### 완료
- [x] 평가 기준 확정 (net_wo_fund / net_w_fund)
- [x] 6단계 계획 확정
- [x] 1단계 insample 학습 4개 완료 + weights 덤프

### 다음 (순서)
1. 1단계 h1/w1 학습 (λ × 2 window × 4 = 8 학습)
2. 각 모델 → analyze_bidask_pnl.py (net_wo_fund + net_w_fund)
3. 4장 PnL 그래프 생성 + 표 공유 → best λ 확정
4. 2단계(Tier-A 33피쳐) 진입

---

## 1단계 확정 결과 (2026-04-22)

실측 백테스트 (bid/ask spread + funding, leverage-normalized 1x, cumulative %):

| λ | insample (wo/w) | h1 (wo/w) | w1 (wo/w) |
|---|---|---|---|
| 0.0 | +33.4 / +13.1 | −5.5 / −7.8 | +1.2 / −8.5 |
| 0.5 | −17.0 / −17.2 | +24.7 / +43.3 | −2.9 / −3.2 |
| **1.0** ★ | **+12.5 / +42.6** | **+3.4 / +32.0** | **+2.9 / +9.5** |
| 2.0 | −90.9 / −30.2 | +12.7 / +46.5 | −13.9 / +1.1 |

**Best: λ=1.0** — 3윈도우 전부 net_w_fund > 0 (유일함).

원래 Stage 1-b (stride 버그) 결론은 λ=2.0 이었는데, 앵커 수정 후 λ=1.0 으로 변경.

PNG: `BRAIN/binance/experiments/fund_sweep_anchor/plots/stage1_<lam>_<window>.png`

---

## 2단계 확정 결과 (2026-04-22)

Tier-A 33 피쳐 × λ=1.0, 3윈도우 (net_wo_fund / net_w_fund 1x lev-norm):

| window | 1단계 (15피) | 2단계 (33피) |
|---|---|---|
| recent | +12.5 / +42.6 | +2.7 / +22.7 |
| h1 | +3.4 / +32.0 | +3.5 / +22.5 |
| w1 | +2.9 / +9.5 | **+24.7 / +32.6** |

Top 수익은 살짝 감소했지만 최저 윈도우가 +9.5 → +22.5 로 크게 개선. 3윈도우 net_w_fund 전부 +22% 이상으로 안정화.

PNG: `BRAIN/binance/experiments/stage2_tierA/plots/stage2_tierA_combined.png`

---

## 3단계 확정 결과 (2026-04-22)

VIF + IC gate: 33 피쳐 → 26 IC-pass → 23 survivors (VIF drop 3: ret_7d, fund_phase_cos, fund_phase_sin).

23 피쳐 × λ=1.0:

| window | 2단계 (33피) | 3단계 (23피) |
|---|---|---|
| recent | +2.7 / +22.7 | +1.2 / **+24.8** |
| h1 | +3.5 / +22.5 | −0.7 / +18.0 ⬇️ |
| w1 | +24.7 / +32.6 | +17.0 / +29.9 |

**판정**: gate 적용 시 recent는 +2.1pp 개선, 그러나 h1 −4.5pp / w1 −2.7pp 악화. 3윈도우 net_w_fund 평균 Stage 2: 25.9% → Stage 3: 24.2%.

최종 v8 피쳐셋: **Stage 2 (33피) 유지가 유리**. gate 미적용.

PNG: `BRAIN/binance/experiments/stage3_gated/plots/stage3_gated_combined.png`

Gate result JSON: `BRAIN/binance/experiments/stage3_gate/gate_result.json`

---

## 변경 이력

- 2026-04-22 16:40 초안 (stride anchor 수정 후 v7 λ=0 insample Sharpe 4.20 확인)
- 2026-04-22 17:05 평가 기준 확정 + 6단계 계획 확정 (net_wo_fund / net_w_fund, 3윈도우 × 2선)
- 2026-04-22 18:10 1단계 확정 — λ=1.0 best (3윈도우 전부 net_w_fund 양수)
- 2026-04-22 18:25 2단계 완료 — 33 피쳐 × λ=1.0, 최저 윈도우 +9.5→+22.5
- 2026-04-22 18:36 3단계 완료 — gate(23피) h1 악화로 기각. 최종 v8 = 33피 유지
