"""Knowledge building system: agents accumulate expertise via Mem0.

Modes:
1. build_knowledge(topic) — deep research on a topic, store in Mem0
2. recall_knowledge(agent, topic) — retrieve relevant past research
3. post_meeting_learn(messages) — extract insights from meeting transcript
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from alpha_squad.memory.store import AlphaSquadMemory

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _claude_query(prompt: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text", "--model", "sonnet", "--tools", ""],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() or "(응답 없음)"
    except subprocess.TimeoutExpired:
        return "(타임아웃)"
    except Exception as e:
        return f"(에러: {e})"


# ─── Deep research prompts per agent ─────────────────────────────

RESEARCH_PROMPTS = {
    "Researcher2_AI아키텍처": """주제: {topic}

금융 tabular/시계열 예측에 적용 가능한 최신 아키텍처 5개를 조사하세요.
각각 논문명(영어), 핵심 아이디어 2줄, 금융 적용 장단점을 포함.
SSM(Mamba), MoE, Retrieval-augmented, Foundation model for tabular 등 포함.
금융에서 실패한 사례도 1-2개 포함. 한국어, 논문명은 영어.""",

    "Expert1_피처전문가": """주제: {topic}

금융 1분봉 데이터의 피처 엔지니어링을 조사하세요:
1. 가격/거래량/변동성/마이크로구조 피처 최신 기법 (수식 포함)
2. IC 분석, feature decay half-life 등 품질 평가법
3. Look-ahead bias 방지 패턴
4. 피처 선택/압축 기법 (LASSO, PCA, tree importance)
한국어로 간결하게.""",

    "Expert2_문제정의전문가": """주제: {topic}

금융 ML 문제 정의를 조사하세요:
1. 타겟 설계: regression vs classification vs ranking, multi-horizon
2. Loss: CE, Focal, ListNet, LambdaRank, IC-based loss, Sharpe loss
3. SNR 문제, non-stationarity, label noise 대응법
4. Evaluation: walk-forward, purged k-fold, backtest overfitting 방지
한국어로 간결하게, 논문 reference 포함.""",

    "Researcher1_금융모델링": """주제: {topic}

금융 tabular 모델링 실전을 조사하세요:
1. TabM, FT-Transformer, GBM 비교 (2024-2025 벤치마크)
2. Cross-sectional vs per-stock learning 트레이드오프
3. Regularization: dropout, weight decay, label smoothing 금융 최적 범위
4. 앙상블: feature diversity vs model diversity, averaging vs stacking
한국어로 간결하게, 실전 판단 포함.""",

    "Researcher3_PM": """주제: {topic}

ML 실험 관리를 조사하세요:
1. 효율적 HPO: Bayesian opt, Hyperband, multi-fidelity
2. 금융 ML 프로젝트: 재현성, 통계적 유의성 검증, backtest overfitting 방지
3. 리소스 최적화: GPU 시간 대비 정보량, 빠른 실패 전략
4. 위험 관리: test set overfitting 탐지, 모델 decay 모니터링
한국어로 간결하게, 도구/방법론 이름 포함.""",
}


def _load_project_context() -> str:
    """Load actual project code structure + autoresearch documents."""
    parts = []

    # 1. Autoresearch documents (all of them)
    ar_dir = PROJECT_ROOT / "BRAIN" / "kospi" / "autoresearch"
    if ar_dir.exists():
        for f in sorted(ar_dir.glob("*.md")):
            content = f.read_text(encoding="utf-8", errors="replace")
            parts.append(f"--- {f.name} ---\n{content[:2000]}")

    # 2. Key BRAIN docs
    for doc in ["autoresearch_common.md", "data.md", "program.md",
                "data_feature_extraction.md", "feature_analysis.md", "literature_survey.md"]:
        path = PROJECT_ROOT / "BRAIN" / "kospi" / doc
        if path.exists():
            parts.append(f"--- {doc} ---\n{path.read_text(encoding='utf-8', errors='replace')[:2000]}")

    # 3. Experiment results
    results_path = PROJECT_ROOT / "BRAIN" / "kospi" / "experiments" / "results_all.tsv"
    if results_path.exists():
        parts.append(f"--- results_all.tsv ---\n{results_path.read_text()[:3000]}")

    # 4. Model architectures (list + key files)
    arch_dir = PROJECT_ROOT / "src" / "models" / "architectures"
    if arch_dir.exists():
        models = [f.stem for f in arch_dir.glob("*.py") if f.name != "__init__.py"]
        parts.append(f"--- 구현된 모델 ---\n{', '.join(sorted(models))}")

    # 5. Configs
    cfg_dir = PROJECT_ROOT / "configs"
    if cfg_dir.exists():
        configs = [f.stem for f in cfg_dir.glob("*.yaml")]
        parts.append(f"--- Config 파일 ---\n{', '.join(sorted(configs))}")

    # 6. Loss functions
    loss_dir = PROJECT_ROOT / "src" / "training" / "losses"
    if loss_dir.exists():
        for f in sorted(loss_dir.glob("*.py")):
            if f.name != "__init__.py":
                parts.append(f"--- loss: {f.name} ---\n{f.read_text()[:1000]}")

    return "\n\n".join(parts)


def build_knowledge(topic: str, memory: AlphaSquadMemory | None = None) -> dict[str, str]:
    """Deep research on a topic. Store findings in Mem0.

    Reads actual project code/documents and injects them into research prompts.

    Args:
        topic: Research topic
        memory: AlphaSquadMemory instance (creates one if None)

    Returns:
        Dict mapping agent name to research findings
    """
    if memory is None:
        memory = AlphaSquadMemory()

    print(f"[Knowledge] Loading project context...")
    project_context = _load_project_context()
    print(f"[Knowledge] Project context: {len(project_context)} chars from code + BRAIN docs")
    print(f"[Knowledge] Deep research: {topic}")
    results = {}

    # Run all agent research in parallel
    def _research(agent_name: str) -> tuple[str, str]:
        base_prompt = RESEARCH_PROMPTS[agent_name].format(topic=topic)
        # Inject actual project data so agents ground their research in reality
        prompt = f"""{base_prompt}

---
## 현재 프로젝트 실제 데이터 (이 정보를 기반으로 구체적인 조언을 해주세요)

{project_context[:2000]}
"""
        return agent_name, _claude_query(prompt, timeout=300)

    # Run 2-3 at a time to avoid rate limiting
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_research, name) for name in RESEARCH_PROMPTS]
        for future in as_completed(futures):
            agent_name, content = future.result()
            results[agent_name] = content
            print(f"  [{agent_name}] researched ({len(content)} chars)")

            # Store in Mem0
            memory.add(
                agent_name=agent_name,
                content=f"[{datetime.now().strftime('%Y-%m-%d')}] 주제: {topic}\n\n{content[:2000]}",
                metadata={
                    "type": "research",
                    "topic": topic,
                    "date": datetime.now().isoformat(),
                },
            )

    print(f"[Knowledge] Complete. {len(results)} agents' research stored in Mem0.")
    return results


def recall_knowledge(agent_name: str, topic: str, memory: AlphaSquadMemory | None = None, limit: int = 5) -> str:
    """Recall relevant past research for an agent.

    Returns formatted string of past knowledge.
    """
    if memory is None:
        memory = AlphaSquadMemory()

    memories = memory.search(agent_name, topic, limit=limit)
    if not memories:
        return ""

    lines = [f"[축적된 전문 지식 — {len(memories)}건]"]
    for m in memories:
        text = m.get("memory", m.get("text", ""))
        lines.append(f"\n---\n{text[:500]}")

    return "\n".join(lines)


def post_meeting_learn(
    messages: list[dict],
    topic: str,
    memory: AlphaSquadMemory | None = None,
) -> list[str]:
    """Extract key insights from meeting transcript and store in Mem0.

    Stores:
    - PM decisions (always)
    - Expert critiques that were accepted
    - New ideas that were validated
    """
    if memory is None:
        memory = AlphaSquadMemory()

    saved = []
    saved_agents: dict[str, str] = {}  # name → type (에이전트당 1건만 저장, 가장 긴 발언)
    date = datetime.now().strftime("%Y-%m-%d")

    # 에이전트별 가장 긴 발언을 선택
    best_per_agent: dict[str, tuple[str, str]] = {}  # name → (type, content)
    for msg in messages:
        name = msg.get("name", "")
        content = msg.get("content", "")
        if not content or len(content) < 100:
            continue

        if name == "Researcher3_PM":
            msg_type = "decision"
        elif "Expert" in name and len(content) > 300:
            msg_type = "critique"
        elif "Researcher" in name and len(content) > 300:
            msg_type = "insight"
        else:
            continue

        prev = best_per_agent.get(name)
        if prev is None or len(content) > len(prev[1]):
            best_per_agent[name] = (msg_type, content)

    # 에이전트당 최고 발언 1건만 저장
    for name, (msg_type, content) in best_per_agent.items():
        if msg_type == "decision":
            memory.add(
                agent_name=name,
                content=f"[{date}] 회의 결정 — {topic}\n\n{content[:2000]}",
                metadata={"type": "decision", "topic": topic, "date": date},
            )
        elif msg_type == "critique":
            memory.add(
                agent_name=name,
                content=f"[{date}] 비판/제안 — {topic}\n\n{content[:1500]}",
                metadata={"type": "critique", "topic": topic, "date": date},
            )
        else:
            memory.add(
                agent_name=name,
                content=f"[{date}] 분석 — {topic}\n\n{content[:1500]}",
                metadata={"type": "insight", "topic": topic, "date": date},
            )
        saved.append(f"[{name}] {msg_type}")

    print(f"[Knowledge] Post-meeting: stored {len(saved)} insights in Mem0")
    return saved
