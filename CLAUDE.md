# 연구 세션 시작

새 연구 회차를 요청받으면 [AutoResearch 절차](autoresearch/README.md)와 직전 회차의 일지를 읽는다. 이전 결정의 근거는 `autoresearch/records/`와 `research/records/`에서 찾을 수 있다.

- **회의 주제**에는 관찰한 결과와 이번에 결정할 질문을 함께 쓴다. Alpha Squad를 사용할 때는 이전 결과를 `--context-file`로 전달한다. [안건 예시](alpha_squad/README.md)
- **실험 전**에 기준선과 변경 조건, 기간, 시드, 비용 가정, 판정 지표를 회차 일지에 적는다. 모델을 바꿀 때는 같은 데이터·평가 설정을 사용하고 실행한 YAML과 CLI 변경값을 남긴다. [JAX 설정 방식](research/jax.md)
- **실험 후**에는 측정값과 `keep`, `discard`, `crash` 중 한 가지 판정을 결과표에 적는다. 일지에는 이유와 다음에 바꿀 조건을 기록한다. 새 회차의 기록은 `autoresearch/runs/`에 작성한다.
- 합성 데이터 데모는 학습·예측 경로를 검증한다. 시장 성과를 기록할 때는 해당 시세 구간에서 수행한 실험 결과를 사용한다.
