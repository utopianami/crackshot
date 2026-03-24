# AlphaSquad: Multi-Agent Research Meeting System

KOSPI 트레이딩 모델 개선을 위한 멀티에이전트 연구 회의 시스템.
AutoGen GroupChat + Claude CLI 기반으로, 5명의 전문 에이전트가 체계적으로 실험 방향을 도출한다.

## 에이전트 구성

| 에이전트 | 역할 | 모델 |
|---------|------|------|
| **Researcher2** (AI아키텍처 전문가) | 아이디어 제안 + 반박/보강. 3소스 리서치 파이프라인 보유. **연구의 정점** | Opus |
| **Expert1** (금융 데이터 피처 전문가) | 피처 관점 비판 (leakage, redundancy, normalization) | Sonnet |
| **Expert2** (금융 ML 전략 PM) | 문제 정의·loss·평가 관점 비판 | Sonnet |
| **Researcher1** (금융 데이터 ML 엔지니어) | 실전 경험 기반 실현 가능성 검토 | Sonnet |
| **Researcher3** (실험 PM) | 종합 판단 + 구체적 실험 스펙 작성 | Opus |

## 핵심: Researcher2 리서치 파이프라인

Researcher2는 "이 문제에 어떤 아키텍처를 써야 하는가?"에 대해 **논문 기반으로 근거 있는 제안**을 한다.
3소스 병렬 검색 + Opus 순위 매기기 + 논문 요약으로 구성된 리서치 파이프라인을 사용.

```
문제 정의 (쿼리)
    │
    ├─ 이전 리서치 참조 (중복 검색 방지)
    │
    ├─── ① HyDE + 로컬 임베딩 검색 (42K+ 학회 논문)
    │    가상 abstract 생성 → 임베딩 유사도로 관련 논문 탐색
    │
    ├─── ② Elicit Search API (학술 논문 전문 검색)
    │    Semantic Scholar 넘어서는 학술 검색
    │
    ├─── ③ Opus 직접 추천 (LLM 지식 기반)
    │    검색 API가 못 찾는 논문을 Opus가 추천
    │
    │    ※ 이전에 찾은 논문은 제외하고 새 논문만 검색
    │
    └─── 3소스 병렬 실행 → 결과 합치기 + 중복 제거
                    │
                    ▼
         Opus가 문제 적합도 순위 매기기
                    │
                    ▼
         상위 N편 논문 요약 (이미 요약 있으면 재사용)
                    │
                    ▼
         최종 컨텍스트 생성 → 저장 + 에이전트 주입
```

### 리서치 결과 누적 관리

리서치 결과는 1회성이 아니라 **누적 자산**으로 관리된다.

```
research_store/
├── papers/          ← 논문 단위 저장 (paperId 기준, 중복 없음)
│   └── {paperId}.json  (title, abstract, summary, sources, topics)
│
├── queries/         ← 쿼리별 검색 결과
│   └── {id}.json       (query, hyde_abstract, 3소스 결과, ranked)
│
├── contexts/        ← 회의용 컨텍스트 캐시
│   ├── index.json      (주제별 인덱스 — topic, keywords, date)
│   └── {id}.md         (최종 컨텍스트 텍스트)
│
└── embeddings/      ← 학회 논문 임베딩 인덱스
    ├── abstract_embeddings.npz  (42K+ 논문, all-MiniLM-L6-v2)
    └── abstract_meta.json
```

- **논문 요약은 한 번만**: 같은 논문이 여러 회의에서 나와도 기존 요약 재사용
- **이전 결과 제외 검색**: 3소스에 "이전에 찾은 논문 외에 추천해줘"로 요청 → 매번 새 논문 발견
- **컨텍스트 캐시**: 유사 주제로 회의하면 이전 리서치 재활용

### 학회 논문 인덱스

Semantic Scholar API로 주요 학회 논문을 수집하고 임베딩하여 로컬 검색에 사용.

**수집 대상** (2022~2025):
ICML, NeurIPS, ICLR, KDD, AAAI, IJCAI, WSDM, WWW, SIGIR, CIKM, ECML, AISTATS, ICAIF

**현재**: 42,531편 수집 + 임베딩 완료

## 구조

```
alpha_squad/
├── run.py                        ← CLI 진입점 (study, meet, refresh)
├── config.py                     ← AlphaSquadConfig (모델, 경로, context files)
│
├── agents/                       ← 5명의 에이전트 (AutoGen ConversableAgent)
│   ├── researcher2.py            ← Researcher2: AI아키텍처 전문가 [Opus] ★
│   ├── expert1.py                ← Expert1: 금융 데이터 피처 전문가
│   ├── expert2.py                ← Expert2: 금융 ML 전략 PM
│   ├── researcher1.py            ← Researcher1: 금융 데이터 ML 엔지니어
│   └── researcher3.py            ← Researcher3: 실험 PM [Opus]
│
├── knowledge/                    ← 지식 시스템
│   ├── researcher2_pipeline.py   ← ★ 3소스 리서치 파이프라인 (핵심)
│   ├── lightrag_backend.py       ← LightRAG + Claude CLI + sentence-transformers
│   ├── study_agent.py            ← snowball 웹 검색 지식 수집
│   ├── collect_papers.py         ← 학회 논문 수집 + 임베딩
│   └── run_all_study.sh          ← 전체 에이전트 일괄 수집 스크립트
│
├── research/                     ← 사전 리서치 페이즈
│   ├── phase.py                  ← 5 에이전트 병렬 리서치
│   └── knowledge.py              ← ChromaDB 지식 축적/회상
│
├── meeting/                      ← 회의 오케스트레이터
│   └── orchestrator.py           ← AutoGen GroupChat + Researcher2 강화
│
├── llm/                          ← LLM 백엔드
│   └── claude_client.py          ← Claude CLI 어댑터 (Max 구독)
│
├── memory/                       ← 회의 메모리
│   └── store.py                  ← ChromaDB (에이전트별 collection)
│
├── research_store/               ← ★ Researcher2 리서치 결과 누적 저장
│   ├── papers/                   ← 논문 단위 (paperId 기준)
│   ├── queries/                  ← 쿼리별 검색 결과
│   ├── contexts/                 ← 회의용 컨텍스트 캐시 + index.json
│   └── embeddings/
│
├── papers/                       ← 학회 논문 인덱스 (42K+ abstract + 임베딩)
│
├── knowledge_store/              ← LightRAG 에이전트별 지식 그래프
│   ├── expert1/                  ← 1,545 nodes (금융 데이터 피처)
│   ├── expert2/                  ← 1,300 nodes (금융 ML 전략)
│   ├── researcher1/              ← 1,261 nodes (금융 ML 실전)
│   ├── researcher2/              ← 10,122 nodes (AI아키텍처, 부분 구축)
│   └── researcher3/              ← 1,263 nodes (실험 관리)
│
└── memory_store/                 ← ChromaDB 영속 저장소
    └── chroma/
```

## 회의 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRE-MEETING                                   │
│                                                                 │
│  1. Research Phase                                               │
│     5 에이전트 병렬 사전 조사 (코드분석, 데이터분석 등)              │
│                                                                 │
│  2. Knowledge Recall                                            │
│     ChromaDB: 과거 회의 기억 회상                                 │
│     LightRAG: 에이전트별 지식 그래프 검색                          │
│                                                                 │
│  3. ★ Researcher2 리서치 파이프라인                               │
│     HyDE + 로컬 임베딩 / Elicit / Opus 추천 → 3소스 병렬          │
│     Opus 순위 → 논문 요약 → 컨텍스트 생성 (10,000+ chars)         │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEETING: GroupChat                             │
│                                                                 │
│  ① Researcher2 (AI아키텍처 전문가) ──→  아이디어 제안 (논문 근거)  │
│         │                                                       │
│  ② Expert1 (금융 피처 전문가)      ──→  피처 관점 비판             │
│         │                                                       │
│  ③ Expert2 (금융 ML 전략 PM)     ──→  문제정의·loss·평가 비판      │
│         │                                                       │
│  ④ Researcher1 (금융 ML 엔지니어) ──→  실현 가능성 검토            │
│         │                                                       │
│  ⑤ Researcher2 (AI아키텍처 전문가) ──→  반박/보강 (논문 근거)      │
│         │                                                       │
│  ⑥ Researcher3 (실험 PM)         ──→  수용/기각 + 실험 스펙 작성  │
│                                                                 │
│  ※ --rounds N 으로 이 사이클을 N번 반복                            │
│  ※ Researcher2는 토론 중 추가 리서치 가능 (3소스 재실행)            │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-MEETING                                  │
│                                                                 │
│  1. Researcher3 발언에서 실험 스펙 추출                            │
│  2. 결정 + 비판 → ChromaDB 영속 저장                              │
│  3. 핵심 인사이트 → Knowledge Base에 축적                         │
│  4. 전체 회의록 → meetings/meeting_YYYYMMDD.md 저장              │
└─────────────────────────────────────────────────────────────────┘
```

## 설계 포인트

| 항목 | 설계 |
|------|------|
| **Researcher2** | Opus + 3소스 리서치 + 2회 발언/라운드 + 10,000+ chars 컨텍스트 |
| **Researcher3** | Opus — 최종 결정권자, 구체적 실험 스펙 작성 |
| **Expert1, Expert2, Researcher1** | Sonnet — LightRAG 지식 그래프 기반 |
| **LLM 호출** | `claude` CLI subprocess → Max 구독 그대로 사용 |
| **Researcher2 지식** | 3소스 파이프라인 (HyDE+임베딩 / Elicit / Opus 추천) |
| **나머지 에이전트 지식** | LightRAG 지식 그래프 (hybrid: 벡터+그래프 탐색) |
| **회의 메모리** | ChromaDB — 회의간 맥락 유지 |
| **리서치 누적** | research_store/ — 논문 요약/컨텍스트 재활용 |

## 사용법

### Researcher2 리서치 (핵심)

```bash
# 주제에 대해 3소스 리서치 실행
PYTHONPATH="." uv run python alpha_squad/knowledge/researcher2_pipeline.py research "non-stationary tabular deep learning"

# 이전 리서치 목록
PYTHONPATH="." uv run python alpha_squad/knowledge/researcher2_pipeline.py list

# 이전 컨텍스트 보기
PYTHONPATH="." uv run python alpha_squad/knowledge/researcher2_pipeline.py show {id}
```

### 학회 논문 인덱스

```bash
# 학회 논문 수집 (Semantic Scholar API)
PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py collect

# 임베딩 생성
PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py embed

# 수집 + 임베딩 한번에
PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py all

# 논문 검색 테스트
PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py search "tabular deep learning" --top-k 10
```

### 에이전트 지식 수집 (LightRAG)

Expert1, Expert2, Researcher1, Researcher3용 지식 그래프 구축.

```bash
# 특정 에이전트에 특정 주제 수집
PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 "alpha factor design" --depth 2

# 에이전트의 시드 키워드 전체 수집
PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 --depth 2

# 4개 에이전트 전체 일괄 수집
nohup bash alpha_squad/knowledge/run_all_study.sh > alpha_squad/study_all.log 2>&1 &
```

### 회의

```bash
# 기본 회의 (Researcher2 리서치 + LightRAG + ChromaDB + research phase 자동)
PYTHONPATH="." uv run python alpha_squad/run.py meet "다음 실험 방향은?"

# 추가 컨텍스트 파일 포함
PYTHONPATH="." uv run python alpha_squad/run.py meet "TabM 개선 방안" \
  --context-file BRAIN/kospi/experiments/results_all.tsv

# 3라운드 토론
PYTHONPATH="." uv run python alpha_squad/run.py meet "새로운 아키텍처 탐색" --rounds 3
```

### 지식 그래프 상태 확인

```bash
for agent in expert1 expert2 researcher1 researcher3; do
    nodes=$(grep -c '<node' alpha_squad/knowledge_store/$agent/graph_chunk_entity_relation.graphml)
    edges=$(grep -c '<edge' alpha_squad/knowledge_store/$agent/graph_chunk_entity_relation.graphml)
    echo "$agent: ${nodes} nodes, ${edges} edges"
done
```
