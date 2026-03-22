# AlphaSquad: Multi-Agent Research Meeting System

KOSPI 트레이딩 모델 개선을 위한 멀티에이전트 연구 회의 시스템.
AutoGen GroupChat + Claude CLI 기반으로, 5명의 전문 에이전트가 체계적으로 실험 방향을 도출한다.

## 구조

```
alpha_squad/
├── run.py                    ← CLI 진입점 (study, meet, refresh)
├── config.py                 ← AlphaSquadConfig (모델, 경로, context files)
│
├── agents/                   ← 5명의 에이전트 (AutoGen ConversableAgent), 발언 순서:
│   ├── researcher2.py        ← ① 최신 딥러닝 아키텍처 리서처 (아이디어 제안)
│   ├── expert1.py            ← ② 피처 엔지니어링 전문가 (leakage·redundancy 비판)
│   ├── expert2.py            ← ③ 실험 설계 전문가 (loss·label·regime 비판)
│   ├── researcher1.py        ← ④ 금융 시계열 ML 엔지니어 (실현 가능성 검증)
│   └── researcher3.py        ← ⑤ PM (종합 판단 + 실험 스펙 작성) [Opus]
│
├── knowledge/                ← 지식 기반 (LightRAG)
│   └── lightrag_backend.py   ← LightRAG + Claude CLI + sentence-transformers
│
├── research/                 ← 사전 리서치 페이즈
│   ├── phase.py              ← 5 에이전트 병렬 리서치 (코드분석, 데이터분석, 논문검색)
│   └── knowledge.py          ← 지식 축적/회상 (study, recall, post_meeting_learn)
│
├── meeting/                  ← 회의 오케스트레이터
│   └── orchestrator.py       ← AutoGen GroupChat + 고정 발언 순서
│
├── llm/                      ← LLM 백엔드
│   └── claude_client.py      ← Claude CLI 어댑터 (Max 구독, API키 불필요)
│
├── memory/                   ← 회의 메모리 (ChromaDB)
│   └── store.py              ← ChromaDB 벡터 DB (에이전트별 collection)
│
├── knowledge_store/          ← LightRAG 에이전트별 지식 저장소
│   ├── researcher1/          ← 금융 모델링 지식 (논문, 벤치마크)
│   ├── researcher2/          ← AI 아키텍처 지식 (논문, 최신 기법)
│   ├── expert1/              ← 피처 엔지니어링 지식
│   ├── expert2/              ← 문제 정의/loss 지식
│   └── researcher3/          ← 실험 관리 지식
│
├── memory_store/             ← ChromaDB 영속 저장소
│   └── chroma/
│
├── meetings/                 ← 회의록 저장
│   └── meeting_YYYYMMDD_HHMMSS.md
│
├── research_cache/           ← 일일 static 리서치 캐시
│   └── static_YYYYMMDD.json
│
└── research_results/         ← 사전 리서치 결과 저장
    └── YYYYMMDD_HHMMSS_*.md
```

## 지식 시스템 (2계층)

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Knowledge Base (LightRAG)                             │
│  ─ 에이전트별 독립된 지식 그래프                                   │
│  ─ 논문/문서 → 엔티티+관계 자동 추출 → 그래프 저장                  │
│  ─ 쿼리 시 그래프 탐색 + 벡터 검색 조합 (hybrid retrieval)          │
│  ─ 저장소: knowledge_store/{agent}/                              │
│                                                                 │
│  LLM: Claude CLI (Max)  |  임베딩: all-MiniLM-L6-v2 (로컬)       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Meeting Memory (ChromaDB)                             │
│  ─ 회의 발언/결정/비판을 벡터로 저장                                │
│  ─ 다음 회의에서 시맨틱 검색으로 과거 맥락 회상                      │
│  ─ 저장소: memory_store/chroma/                                  │
└─────────────────────────────────────────────────────────────────┘
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
│                                                                 │
│  + Knowledge Base에서 주제 관련 지식 recall (LightRAG hybrid)     │
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
│  2. PM 결정 + Expert 비판 → ChromaDB 영속 저장                   │
│  3. 핵심 인사이트 → Knowledge Base에 축적 (LightRAG)             │
│  4. 전체 회의록 → meetings/meeting_YYYYMMDD.md 저장              │
└─────────────────────────────────────────────────────────────────┘
```

## 설계 포인트

| 항목 | 설계 |
|------|------|
| **LLM 호출** | `claude` CLI subprocess → Max 구독 그대로 사용, API키 불필요 |
| **에이전트 역할** | 제안(R2) → 비판(E1,E2) → 검증(R1) → 결정(R3) — 체계적 검증 루프 |
| **PM 모델** | Opus (나머지는 Sonnet) — 최종 결정권자에게 강한 모델 |
| **지식 기반** | LightRAG (에이전트별 지식 그래프) — 논문/문서에서 엔티티·관계 자동 추출 |
| **회의 메모리** | ChromaDB (에이전트별 collection) — 회의간 맥락 유지 |
| **임베딩** | sentence-transformers `all-MiniLM-L6-v2` (로컬, API키 불필요) |
| **리서치** | 회의 전 실제 코드 실행/파일 분석 — system prompt 지식이 아닌 실데이터 기반 |
| **출력** | PM이 `실험:` 형식의 구체적 실험 스펙 생성 → autoresearch 루프에 바로 투입 가능 |

## 사용법

### 지식 수집 (study-agent)

에이전트별 지식 그래프를 웹 검색 기반으로 구축한다. Snowball 방식으로 키워드를 확장하며 수집.

```bash
# 특정 에이전트에 특정 주제 수집 (depth=2: 시드 + 1단계 확장)
PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 "alpha factor design" --depth 2

# 에이전트의 사전 정의된 시드 키워드 전체 수집
PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 --depth 2

# 4개 에이전트 전체 일괄 수집 (밤새 돌리기용)
nohup bash alpha_squad/knowledge/run_all_study.sh > alpha_squad/study_all.log 2>&1 &
```

사전 정의된 시드 키워드 (knowledge/study_agent.py `SEED_TOPICS`):
- **expert1**: alpha factor, microstructure, IC/feature selection, leakage, normalization
- **expert2**: ranking loss, focal/label noise, backtest overfitting, validation, multi-task
- **researcher1**: tree vs DL benchmark, CS vs per-stock, regularization, ensemble, non-stationarity
- **researcher3**: HPO, reproducibility, model decay, experiment management

### 지식 업데이트

LightRAG는 incremental — 새 문서를 넣으면 기존 그래프에 엔티티/관계가 추가되고, 중복 엔티티는 자동 merge.

```bash
# 새 주제 추가 (기존 지식 그래프에 merge)
PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 "order flow imbalance" --depth 2

# 시드 키워드 추가: knowledge/study_agent.py의 SEED_TOPICS dict에 추가 후 재실행
PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 --depth 2
```

- 주기적으로 돌릴 필요 없음 — 새 주제가 생기거나 지식이 부족할 때만
- 삭제 없이 쌓이기만 하므로 부담 없이 실행 가능
- 지식 그래프 초기화가 필요하면: `rm -rf alpha_squad/knowledge_store/{agent}/`

### 회의

```bash
# 기본 회의 (LightRAG knowledge + ChromaDB memory + research phase 자동 실행)
PYTHONPATH="." uv run python alpha_squad/run.py meet "다음 실험 방향은?"

# 추가 컨텍스트 파일 포함
PYTHONPATH="." uv run python alpha_squad/run.py meet "TabM 개선 방안" \
  --context-file BRAIN/kospi/experiments/results_all.tsv

# 3라운드 토론
PYTHONPATH="." uv run python alpha_squad/run.py meet "새로운 아키텍처 탐색" --rounds 3

# 메모리 없이 실행
PYTHONPATH="." uv run python alpha_squad/run.py meet "피처 엔지니어링" --no-memory
```

### 기타

```bash
# Static 리서치 캐시 갱신 (일 1회)
PYTHONPATH="." uv run python alpha_squad/run.py refresh

# 구 study 명령 (ChromaDB용, deprecated)
PYTHONPATH="." uv run python alpha_squad/run.py study "tabular deep learning for finance 2025"
```

### 지식 그래프 상태 확인

```bash
# 에이전트별 노드/엣지 수 확인
for agent in expert1 expert2 researcher1 researcher3; do
    nodes=$(grep -c '<node' alpha_squad/knowledge_store/$agent/graph_chunk_entity_relation.graphml)
    edges=$(grep -c '<edge' alpha_squad/knowledge_store/$agent/graph_chunk_entity_relation.graphml)
    echo "$agent: ${nodes} nodes, ${edges} edges"
done
```
