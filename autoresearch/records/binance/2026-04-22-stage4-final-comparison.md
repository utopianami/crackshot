# Stage 4 — 6 scorer 아키텍처 × 3 윈도우 최종 비교 (2026-04-22)

> **English note.** Original record, kept in Korean as written. An English index of all records is in the [README](../../README.md).

**평가 기준**: bid/ask + funding 포함 1x leverage NAV (실제 투입 가능 PnL)

| Model        | recent (2026-01~03) | h1 (2025-08~09) | w1 (2025-05~06) | 평균    | Sharpe 평균 |
|--------------|---------------------|------------------|------------------|---------|-------------|
| cs_mlp (baseline) | +23.4%         | +23.5%           | +37.3%           | +28.1%  | 4.11        |
| cs_attn      | **-24.8%**          | +4.0%            | +9.8%            | -3.7%   | 0.05        |
| ft_t         | +36.1%              | +11.8%           | +4.6%            | +17.5%  | 1.66        |
| resmlp       | -6.6%               | +25.0%           | +7.7%            | +8.7%   | 1.20        |
| tabm (k=4)   | +11.6%              | +19.2%           | +24.7%           | +18.5%  | 2.72        |
| tabmk8       | +25.7%              | +17.0%           | +28.6%           | +23.8%  | 3.56        |
| **tabsyn**   | **+19.5%**          | **+35.9%**       | **+39.1%**       | **+31.5%** | **4.52** ⭐ |

## 핵심 관찰

1. **tabsyn-latent (VAE encoder) 전체 1위**
   - 3윈도우 모두 양수, Sharpe 평균 4.52 — 가장 안정적
   - 특히 w1(+39.1%)·h1(+35.9%) 고성과 + recent도 +19.5% 유지
   - VAE 잠재공간 regularization이 피처 간 구조를 압축해서 오버피팅 억제

2. **tabmk8 (BatchEnsemble k=8) 2위**
   - 평균 +23.8%, Sharpe 3.56
   - k=4(tabm)에 비해 recent(+25.7% vs +11.6%)에서 큰 개선, ensemble 효과 뚜렷

3. **cs_attn 최악**
   - recent -24.8%로 큰 손실. Cross-entity self-attention이 recent 체제에 과적합
   - 같은 VAE(tabsyn)처럼 Transformer 쓰지만 regularization 없어서 무너짐

## 패러다임 별 교훈

| 패러다임 | 모델 | 평균 PnL | 해석 |
|---|---|---|---|
| Plain MLP | (baseline) | - | 기준선 |
| Residual MLP | resmlp | +8.7% | Vision ResMLP 스킵 연결은 별 도움 안 됨 |
| BatchEnsemble | tabmk8 | +23.8% | k 증가로 안정성 개선 |
| Feature Tokenizer + Transformer | ft_t | +17.5% | CLS 기반 스코어링, w1에서 약함 |
| Cross-entity Attention | cs_attn | -3.7% | MASTER 스타일 과적합 위험 큼 |
| **VAE encoder + MLP head** | **tabsyn** | **+31.5%** | **잠재공간 regularization 성공** |

## 다음 단계 제안

1. **tabsyn을 기본 스코어로 승격** — v8 sweep은 tabsyn 위에서 λ/피처/seed 스위프
2. tabsyn + tabmk8 앙상블 실험 — top-2 모델 상관도 체크 후 blending
3. cs_attn은 기각 (recent window에서 구조적으로 불안정)
4. resmlp는 보류 (효과 제한적, 변형 필요)

## 산출물

각 모델별 combined PNG:
- `BRAIN/binance/experiments/stage4_{cs_attn,ft_t,resmlp,tabm,tabmk8,tabsyn}/plots/*_combined.png`

## 재현

```bash
# 단일 모델: 3윈도우 학습 + 백테스트 + PNG
bash scripts/binance/run_stage4_model.sh <tag>
bash scripts/binance/stage4_backtest.sh <tag>
PYTHONPATH=. uv run python scripts/binance/plot_stage4_model.py <tag>

# 오버나이트 체인 (cs_attn 완료 대기 후 나머지 순차)
bash scripts/binance/stage4_overnight_chain.sh
```
