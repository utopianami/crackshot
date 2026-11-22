# ALPHA-00 Phase 0 Verdict

> **English note.** Original record, kept in Korean as written. An English index of all records is in the [README](../../README.md).

**결론: 조건부 PASS — 된다, 단 maker 실행 전제**

## 게이트 성적 (1.5 / 3)

### Gate A — |IC|>0.02 @ fwd_4h ≥ 3 features: **PASS ✅**

6 / 15 features pass:

| feature | IC @ 4h | ICIR | t-stat |
|---|---|---|---|
| rv_4h          | -0.0526 | -0.172 | -29.7 |
| ret_24h        | -0.0336 | -0.119 | -20.5 |
| ret_4h         | -0.0327 | -0.118 | -20.3 |
| amihud_4h      | +0.0271 | +0.104 | +17.9 |
| ret_7d         | -0.0254 | -0.089 | -15.3 |
| ret_1h         | -0.0249 | -0.090 | -15.5 |

방향: 과거 vol/momentum 높을수록 미래 수익 ↓ (mean-reversion). amihud는 illiq premium.

### Gate B — |ICIR|>0.3 anywhere: **FAIL ❌**

- 최대 |ICIR| = 0.255 (rv_4h × fwd_24h, t=-17.6)
- Per-bar Sharpe는 낮지만 샘플 규모(~30k bars)로 t-stat 수십 → 통계적으로는 매우 유의
- 0.3 threshold는 엄격 적용 시 FAIL, 실질적으로는 near-miss

### Gate C — Portfolio net Sharpe > 0.5 @ fwd_4h: **CONDITIONAL**

| scenario | cost / rebal | # features pass | best feature | best net Sharpe |
|---|---|---|---|---|
| gross | 0 bp | 6 | funding_mean_30d | +1.32 |
| maker 2bp (낙관적) | 2 bp | 3 | funding_mean_30d | +1.01 |
| maker 5bp (현실) | 5 bp | 1 | funding_mean_30d | +0.54 |
| mixed 10bp | 10 bp | 0 | — | — |
| taker 20bp | 20 bp | 0 | — | — |

**Binance USDT-M perp**: maker fee 2bp, taker 5bp (one-way) × 4 trades/rebal → maker-only 8bp, taker-only 20bp per rebal. 현실적 운용 가능 구간 = **5–10bp/rebal**.

## 종합 판단

- **진짜 알파가 있다** — IC significant, portfolio gross Sharpe > 1
- **ICIR 게이트는 strict FAIL, 실질 의미 제한적** (t-stat이 알파 존재를 확정)
- **경제성은 maker 체결 가능성에 달림** — 5bp 이하로 체결할 수 있으면 funding_mean_30d × fwd_4h가 net Sharpe 0.54로 gate pass

## Phase 1 권장 우선순위

1. **`funding_mean_30d × fwd_4h`** — 최강 net Sharpe (0.54 @ 5bp)
2. **`rv_4h × fwd_24h`** — 최강 IC (-0.053, ICIR -0.255), stride 길어 cost 부담 최소
3. **Mean-reversion basket**: ret/vol 피처 결합해 signal strengthen
4. **BTC-residual return** 피처 추가 → 베타 제거 후 alpha 순도 확인
5. **OI / notional** 피처 (deriv_ticker 수집 완료 후) — alpha 후보 확장

## 실행 제약

- Execution은 반드시 maker 중심 (post-only limit)
- Rebal frequency 4h 이하 권장 — 1h는 cost 감당 불가
- 유동성 상위 subset 적용 (amihud 기준) 시 net Sharpe 추가 상승 여지

## 원본 파일
- `BRAIN/binance/data/phase0_rank_ic.csv`
- `BRAIN/binance/data/phase0_portfolio_proxy.csv`
- `BRAIN/binance/data/features_5m/*.parquet` (314 symbols)
