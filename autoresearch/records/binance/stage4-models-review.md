# Stage 4 — 추가 모델 논문 복기 + 구현 체크리스트

> **English note.** Original record, kept in Korean as written. An English index of all records is in the [README](../../README.md).

작성: 2026-04-22 (jayce autonomous night)

Stage 4 1차 3모델 (TabM/FT-T/cs_attn) 결과가 전반적으로 MLP baseline을 일관되게 넘지 못함.
사용자 요청: 추가 모델 구현 + 논문 컨트리뷰션 체크 + 구현 체크리스트.

---

## 1. Tabsyn-latent (arxiv 2310.09656, ICLR 2024)

### 컨트리뷰션 (논문 원본)
1. **혼합 자료형 통합 잠재 공간**: 수치/범주 특성을 단일 latent z로 매핑 → 기존 CTGAN/TabDDPM은 자료형별 별도 처리, Tabsyn은 Transformer-based VAE로 통합.
2. **Latent-space 기반 Score diffusion**: raw feature가 아닌 VAE latent 위에서 score SDE를 풀어서 diffusion step 수를 크게 줄임 (수천 → 수십 step).
3. **Column-wise distribution 오차 86% 감소**: prior arts 대비.
4. **Conditioning 단순화**: latent space 동질성 덕분에 row/col conditioning을 한 틀로 처리.

### 전체 아키텍처 복기
```
Raw (x_num, x_cat)
  → Tokenizer: 각 feature에 d_token 짜리 개별 임베딩 (x_num * W_i, x_cat_i → Emb_i(x_cat_i)), [CLS] 프리펜드
  → Encoder (2개의 Transformer, d_token=192, n_heads=8, layers=2~4)
      - encoder_mu  : 각 feature position마다 μ
      - encoder_logvar : 각 feature position마다 log σ²
  → Reparameterize: z = μ + σ * ε
  → Decoder (Transformer 1개)
  → Reconstructor: 수치는 linear projection, 범주는 per-column linear
VAE Loss = MSE(x_num, x̂_num) + Σ CE(x_cat_i, x̂_cat_i) + β · KL(q(z|x) || N(0,I))
  β 어닐링: 0.01 시작, val loss 정체 시 ×0.7
Stage 2 (synthesis only): latent z에 score SDE → 신규 샘플 디코딩
```

### 우리 목적(scoring)에 맞춘 적응
| 논문 요소 | 우리 버전 | 이유 |
|---|---|---|
| 범주형 임베딩 | 제거 | 우리 33 feat 전부 수치 |
| Reconstructor per-column CE | 제거 | 범주 없음 |
| Score diffusion stage | 제거 | 생성 아니라 예측만 |
| VAE encoder | 유지 (핵심 컨트리뷰션) | 비선형 잠재 표현 |
| β 어닐링 | 상수 β (0.001~0.01) | 간단화 |
| 2-stage pretrain | **유지** | encoder 사전학습 → scorer head 학습. 논문 정신 유지 |
| [CLS] 토큰 | 유지 | scoring용 pooled 표현 |

### 구현 체크리스트
- [ ] `src/models/architectures/portfolio/tabsyn_vae.py`
  - [ ] `Tokenizer(n_features, d_token)`: per-feature weight [n+1, d_token], CLS 프리펜드
  - [ ] `TransformerBlock(d_token, n_heads, ffn_mult, dropout)` — MHA + FFN + LN+residual (기존 FT-T 블록 재사용 가능)
  - [ ] `TabsynEncoder(d_num, d_token, n_layers, n_heads)` — encoder_mu, encoder_logvar 두 트랜스포머
  - [ ] `TabsynDecoder(d_num, d_token, n_layers, n_heads)` — 역방향 + Reconstructor linear
  - [ ] `TabsynVAE`: Tokenizer + Encoder + Decoder + `reparameterize()` + `vae_loss(x)` (MSE + β·KL)
- [ ] `scripts/binance/pretrain_tabsyn_vae.py`
  - [ ] Train split 불러서 stock-stacked 배치로 VAE만 학습 (약 10 epochs, lr 1e-3)
  - [ ] Best val loss 체크포인트 저장 → `outputs/tabsyn_vae/<window>/encoder.eqx`
- [ ] `src/models/architectures/portfolio/tabsyn_latent_scorer.py`
  - [ ] `TabsynLatentScorer(in_features, encoder_path, hidden_units, dropout)`
  - [ ] 시작 시 pretrained encoder 로드, 학습 중 freeze (또는 fine-tune 옵션)
  - [ ] forward: x → Tokenizer → Encoder → z (CLS pooled) → MLP head → score
- [ ] factory 등록 (`scorer_name: tabsyn_latent`)
- [ ] 테스트
  - [ ] `test_tabsyn_vae_reconstruction`: pretrain 후 recon MSE가 raw < std보다 작아야 함 (실제 학습된 VAE 맞는지)
  - [ ] `test_tabsyn_latent_scorer_scalar_output`: 입력 [D] → 스칼라
  - [ ] `test_tabsyn_encoder_frozen_when_freeze_true`: grad가 head로만 흘러야 함
  - [ ] factory test 파라미터에 tabsyn_latent 추가
- [ ] configs: `cs_mlp_portfolio_binance_v7_tabsyn.yaml` (+ _h1, _w1)
- [ ] 학습 & 백테스트 (3 윈도우, 다른 모델과 동일 스크립트)

### 기각 가능성
- VAE recon loss가 충분히 내려가지 않으면 latent이 의미 없음 → fallback으로 non-pretrained joint training 옵션 남기기
- latent dim 너무 크면 overfitting, 너무 작으면 정보 손실 → d_token=32, n_layers=2 로 시작

---

## 2. ResMLP (간이 변형)

### 컨트리뷰션 (ResMLP 원논문, 2021)
- MLP에 residual connection + LayerNorm 정규화 → 깊게 쌓아도 학습 안정
- 이미 우리 `CrossSectionalScorerMLP`는 LayerNorm 있으므로 핵심 추가는 **residual skip**

### 체크리스트
- [ ] 기존 MLP 블록 수정 최소화: scorer 옵션 `residual: true` 추가 시 각 hidden 블록 출력 = `block(x) + x` (차원 동일할 때만)
- [ ] 서로 다른 hidden dim이면 linear projection으로 맞춤
- [ ] 테스트: residual=True면 x를 zero block로 전해도 x가 거의 유지
- [ ] config: `cs_mlp_portfolio_binance_v7_resmlp.yaml` (scorer_name=mlp, residual=true, hidden=[256,256,256])

### 예상 효과
MLP baseline을 깊게 쌓아 capacity 확대. 학습 안정성↑.

---

## 3. TabM k=8 variant (config only)

### 컨트리뷰션 재확인
TabM = BatchEnsemble 기반. k=4는 Stage 4 1차 TabM 실험. 논문에서는 보통 k=8~32가 SOTA. 

### 체크리스트
- [ ] configs 복사 (tabm → tabm_k8), tab_m_k: 8로 변경
- [ ] 재학습 3 윈도우, backtest

코드 변경 불필요. 실험만.

---

## 우선순위 & 일정
1. cs_attn 학습 완료 → backtest + PNG (30분)
2. ResMLP 구현 + 학습 + backtest (1시간)
3. TabM k=8 config + 학습 + backtest (30분)
4. Tabsyn 논문 구현 (2시간) + pretrain + 학습 + backtest (1시간)
5. 종합 비교 테이블 + PNG 푸시

모든 결과는 net_w_fund 1x 누적% + Sharpe (recent/h1/w1) 로 요약.
