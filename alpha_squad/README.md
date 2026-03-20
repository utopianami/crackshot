# AlphaSquad: Multi-Agent Research Meeting System

KOSPI 트레이딩 모델 개선을 위한 멀티에이전트 연구 회의 시스템.
AutoGen GroupChat + Claude CLI 기반으로, 5명의 전문 에이전트가 체계적으로 실험 방향을 도출한다.

## 구조

```
alpha_squad/
├── run.py                    ← CLI 진입점 (topic, --rounds, --context-file)
├── config.py                 ← AlphaSquadConfig (모델, 경로, context files)
│
├── agents/                   ← 5명의 에이전트 (AutoGen ConversableAgent), 발언 순서:
│   ├── researcher2.py        ← ① 최신 딥러닝 아키텍처 리서처 (아이디어 제안)
│   ├── expert1.py            ← ② 피처 엔지니어링 전문가 (leakage·redundancy 비판)
│   ├── expert2.py            ← ③ 실험 설계 전문가 (loss·label·regime 비판)
│   ├── researcher1.py        ← ④ 금융 시계열 ML 엔지니어 (실현 가능성 검증)
│   └── researcher3.py        ← ⑤ PM (종합 판단 + 실험 스펙 작성) [Opus]
│
├── research/                 ← 사전 리서치 페이즈
│   └── phase.py              ← 5 에이전트 병렬 리서치 (코드분석, 데이터분석, 논문검색)
│
├── meeting/                  ← 회의 오케스트레이터
│   └── orchestrator.py       ← AutoGen GroupChat + 고정 발언 순서
│
├── llm/                      ← LLM 백엔드
│   └── claude_client.py      ← Claude CLI 어댑터 (Max 구독, API키 불필요)
│
├── memory/                   ← 영속 메모리
│   └── store.py              ← Mem0 + ChromaDB (에이전트별 namespace)
│
└── meetings/                 ← 회의록 저장 디렉토리
    └── meeting_YYYYMMDD_HHMMSS.md
```

## 회의 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRE-MEETING: Research Phase                   │
│  5 에이전트가 병렬로 사전 조사 (ThreadPoolExecutor)               │
│                                                                 │
│  Researcher2 → 최신 논문/아키텍처 검색 (claude CLI)               │
│  Expert1     → 피처 데이터 통계 분석 (numpy 코드 실행)            │
│  Expert2     → label 분포 + loss 코드 분석 (코드 실행)            │
│  Researcher1 → 실험 결과/모델 config 분석 (파일 읽기)             │
│  Researcher3 → 코드베이스 구조 + 데이터 디렉토리 파악              │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEETING: GroupChat (고정 순서)                 │
│                                                                 │
│  ① Researcher2 (AI아키텍처)  ──→  아이디어 제안                   │
│         │                                                       │
│  ② Expert1 (피처전문가)      ──→  피처 관점 비판                   │
│         │                                                       │
│  ③ Expert2 (문제정의전문가)   ──→  문제 정의 관점 비판              │
│         │                                                       │
│  ④ Researcher1 (금융모델링)  ──→  실현 가능성 검토                 │
│         │                                                       │
│  ⑤ Researcher3 (PM) [Opus]  ──→  수용/기각 + 실험 스펙 작성       │
│                                                                 │
│  ※ --rounds N 으로 이 사이클을 N번 반복                           │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-MEETING                                  │
│                                                                 │
│  1. PM 발언에서 실험 스펙 추출 ("실험:" 키워드)                    │
│  2. PM 결정 + Expert 비판 → Mem0 영속 저장                       │
│  3. 전체 회의록 → meetings/meeting_YYYYMMDD.md 저장              │
└─────────────────────────────────────────────────────────────────┘
```

## 설계 포인트

| 항목 | 설계 |
|------|------|
| **LLM 호출** | `claude` CLI subprocess → Max 구독 그대로 사용, API키 불필요 |
| **에이전트 역할** | 제안(R2) → 비판(E1,E2) → 검증(R1) → 결정(R3) — 체계적 검증 루프 |
| **PM 모델** | Opus (나머지는 Sonnet) — 최종 결정권자에게 강한 모델 |
| **메모리** | Mem0 + ChromaDB, 에이전트별 namespace — 회의간 맥락 유지 |
| **리서치** | 회의 전 실제 코드 실행/파일 분석 — system prompt 지식이 아닌 실데이터 기반 |
| **출력** | PM이 `실험:` 형식의 구체적 실험 스펙 생성 → autoresearch 루프에 바로 투입 가능 |

## 사용법

```bash
# 기본 회의
PYTHONPATH="." uv run python alpha_squad/run.py "다음 실험 방향은?"

# 추가 컨텍스트 파일 포함
PYTHONPATH="." uv run python alpha_squad/run.py "TabM 개선 방안" \
  --context-file BRAIN/kospi/experiments/results_all.tsv

# 3라운드 토론
PYTHONPATH="." uv run python alpha_squad/run.py "새로운 아키텍처 탐색" --rounds 3

# 메모리 없이 실행
PYTHONPATH="." uv run python alpha_squad/run.py "피처 엔지니어링" --no-memory
```
