# JAX 실험 구조

같은 데이터에서 모델만 바꾸려면 모델 구조, 학습 방법, 데이터셋에 관한 선택을 분리해야 합니다. 이 저장소에서는 Equinox가 순전파를, JAX와 Optax를 쓰는 트레이너가 학습을, OmegaConf 설정이 실험별 선택을 담당합니다.

| 변경할 것 | 수정할 곳 | 예시 |
| --- | --- | --- |
| 피처를 확률로 바꾸는 방법 | 모델별 Equinox 모듈 | [TabM](../src/models/architectures/tab_m.py) · [FT-Transformer](../src/models/architectures/ft_transformer.py) |
| 손실, 최적화, 추론 방법 | `BaseTrainer`를 구현하는 모델별 트레이너 | [TabM 학습](../src/training/trainers/tab_m.py) · [FT-Transformer 학습](../src/training/trainers/ft_transformer.py) |
| 모델 기본값과 실험 조건 | 기본값은 dataclass, 실험별 변경은 YAML과 CLI | [TabM 기본값](../src/config/models/tab_m.py) · [실험 설정](../configs/tab_m_binance_phase1.yaml) |

## 새 모델을 연결하는 순서

1. `src/models/architectures/`에 순전파 모듈을 만듭니다. 학습·추론 과정에서 필요한 난수 키와 출력 형태를 정합니다.
2. `src/config/models/`에 모델 설정을 정의합니다. `src/training/trainers/`에 `fit_from_npy`, `predict_npy`, `save`, `load`를 구현하고, `@register_trainer`에 모델 이름과 기본 설정을 전달합니다.
3. YAML의 `model.name`에 등록 이름을 넣습니다. 실행 시 `discover_trainers()`가 모듈을 불러오고 `load_config()`가 모델의 기본값, YAML, CLI 변경값을 합칩니다. [공통 인터페이스와 자동 등록](../src/training/registry.py) · [설정 병합](../src/config/__init__.py)

## 같은 조건에서 교체하기

아래 두 설정은 같은 분할, 수익률 구간, 평가 horizon을 사용합니다. 준비한 Phase 1 NPY 디렉터리를 지정하면 같은 학습 진입점으로 실행할 수 있습니다.

```bash
uv run --locked python -m scripts.train -c configs/tab_m_binance_phase1.yaml data.data_path=/path/to/npy
uv run --locked python -m scripts.train -c configs/ft_transformer_binance_phase1.yaml data.data_path=/path/to/npy
```

시세 데이터 없이 교체 과정을 확인하려면 `uv run --locked python -m examples.compare_trainers`를 실행합니다. 이 예제는 두 모델에 같은 합성 표본을 제공하고 학습 후 예측 형태와 확률 합계를 검사합니다. [실행 예제](../examples/compare_trainers.py) · [자동 등록과 설정 우선순위 테스트](../tests/strict/test_trainer_discovery.py)
