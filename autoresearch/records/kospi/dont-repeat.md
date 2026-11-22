# Don't Repeat

> **English note.** Original record, kept in Korean as written. An English index of all records is in the [README](../../README.md).

실험 시작 전 반드시 읽는다. 이미 시도해서 안 되는 걸 확인한 것들.

| category | what | why | evidence | date |
|----------|------|-----|----------|------|
| mistake | vwap_dev cumsum 구현 | rolling 360-window여야 하는데 cumsum으로 구현. NPY + reference 둘 다 | 2026-03-30 수정 완료 | 2026-03-30 |
| mistake | NPY enhanced feature 날짜별 리셋 | 종목별 전체 히스토리로 한번에 계산해야 하는데 날짜마다 초기화됨 | scripts/create_enhanced_npy.py | 2026-03-30 |
| mistake | create_enhanced_npy.py _rolling_sum 미정의 | 함수 참조만 있고 구현 없었음 | 2026-03-30 추가 | 2026-03-30 |
| mistake | feature에 cumsum/expanding 사용 | reference는 state 없이 유한 윈도우만 받음. 누적 연산 사용 시 학습/추론 불일치 | vwap_dev 버그 | 2026-03-30 |
| mistake | NPY extend/append | split 경계 rolling 오염 위험. 기간 바뀌면 무조건 새로 생성 | npy_management 원칙 | 2026-03-31 |
| mistake | 백테스트 initial_balances에 KRW 누락 | `initial_balances`에 `'KRW': 1e9`를 안 넣으면 현금 잔고 0 → 매수 불가 → 0 executions. 에러 없이 조용히 0이 나와서 디버깅에 수시간 소모 | 2026-04-03 0 executions 사태 | 2026-04-03 |
| mistake | 백테스트 관련 코드/데이터 수정 | `backtest_candle_1520` 파일 덮어쓰기, `probs_to_parquet.py`가 벤더 캔들 폴더에 자동 생성 등으로 기존 데이터 오염. 백테스트 캔들/인퍼런스/Rust 코드는 절대 건드리지 말 것. 읽기 전용으로만 사용 | 2026-04-03 캔들 오염 사태 | 2026-04-03 |
| mistake | 서빙 캔들 수를 600개로 설정 | 중첩 rolling window(`rolling_mean(10)` → `rolling_std(600)`)는 최소 609개 캔들이 있어야 마지막 row의 feature가 캔들 수와 무관하게 동일. 600개면 expanding window 구간이 포함되어 캔들 수에 따라 feature가 달라짐. **해결: 캔들 입력 스펙을 610개로 변경**. 609개 이상이면 610이든 700이든 1000이든 결과 동일 | 260406 cs_mlp_200_base 배포 시 발견. 610으로 수정 완료 | 2026-04-06 |
| mistake | `predict_npy`에서 vol==0/target==0 필터링 후 uniform [0.2,...] 채움 | inference 시 **모든 row를 예측해야 함**. 학습 loss용 필터를 inference에 그대로 적용하면 안 됨. `_build_timestamp_groups`의 필터가 `predict_npy`까지 전파되어 필터된 종목+시점에 uniform prediction이 들어감. reference pipeline과 불일치 원인. 학습 필터링은 학습 시에만 적용할 것 | cs_mlp_200_base reference 배포 시 발견: 14060 timestamps 중 2568개에 uniform 삽입 | 2026-04-06 |
| mistake | reference feature_extractor에 filter_after_time(15:20) 누락 | NPY builder는 closing auction(15:20~15:30) 11 bars/day 자동 필터하는데 reference loader는 미적용. 어제 closing auction bars가 오늘 inference 610-bar window에 섞여서 tv_normalized/amihud 등 volume feature 오염. cs_mlp diff 0.70, tabm 0.22 발생. fix: reference candle loader에서 동일 필터 적용 | 2026-05-13 verify_e2e.py 진단 | 2026-05-13 |
| data | candle_vendor 006360/2021-03-10, 035420/2023-04-06, 035420/2023-04-07 corrupt | thrift/snappy decompression 오류. 이 파일들이 있으면 _load_all_dates_for_stock이 전체 종목을 skip → verify_e2e에서 198/200 종목만 로드 → cs_mlp cross-sectional mean/std 오차로 diff 0.027/0.020. 2026-05-13 synthetic 파일로 복구 (원본 .corrupt.bak으로 보존). 복구 source: pykrx daily OHLCV 기반 합성, 전일 bars에 timestamp만 변경 | 2026-05-13 candle repair | 2026-05-13 |
| data | 047040 amihud 계산 오류 (cumsum catastrophic cancellation) | 047040 test period에 zero-volume bars 13개 존재. NPY builder의 rolling_mean은 cumsum 기반이라 zero-vol 시 amihud_raw=|ret|/EPS=1e10 수준의 spike → cumsum이 8.6e7 수준으로 누적 → 이후 rolling window에서 (cum[i]-cum[i-30])/30 에서 catastrophic cancellation. verify_e2e의 610-bar slice는 해당 spike bar를 포함하지 않아 오차 없음. 두 경로 간 amihud_30 diff ~0.17로 tabm diff 9.2e-5 발생. 해결하려면 NPY 재빌드 + rolling_mean을 직접 창 평균으로 변경 필요 | 2026-05-13 verify_e2e Tier2B WARN 진단 | 2026-05-13 |
