"""AlphaSquad meeting orchestrator using AutoGen GroupChat + claude CLI."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from autogen import GroupChat, GroupChatManager

from alpha_squad.agents import (
    create_expert1,
    create_expert2,
    create_researcher1,
    create_researcher2,
    create_researcher3,
)
from alpha_squad.config import AlphaSquadConfig
from alpha_squad.llm import ClaudeCLIClient
from alpha_squad.memory.store import AlphaSquadMemory
from alpha_squad.knowledge.lightrag_backend import AlphaSquadKnowledge
from alpha_squad.knowledge.researcher2_pipeline import get_research_context
from alpha_squad.research.phase import run_research_phase
from alpha_squad.research.knowledge import recall_knowledge, post_meeting_learn


# Mapping: AutoGen agent name → LightRAG agent name
_AGENT_TO_KNOWLEDGE = {
    "Researcher1_금융모델링": "researcher1",
    "Researcher2_AI아키텍처": "researcher2",
    "Researcher3_PM": "researcher3",
    "Expert1_피처전문가": "expert1",
    "Expert2_문제정의전문가": "expert2",
}


# Meeting flow: R2 speaks twice (제안 + 비판 후 반박)
SPEAKER_ORDER = [
    "Researcher2_AI아키텍처",     # 1. 아이디어 제안
    "Expert1_피처전문가",          # 2. 피처 관점 비판
    "Expert2_문제정의전문가",      # 3. 문제 정의 관점 비판
    "Researcher1_금융모델링",      # 4. 실현 가능성 검토
    "Researcher2_AI아키텍처",     # 5. 반박/보강 (논문 근거)
    "Researcher3_PM",             # 6. 수용/기각 + 실험 설계
]


def _build_llm_config(model: str, max_tokens: int) -> dict:
    """Build AutoGen LLM config using custom Claude CLI client."""
    return {
        "config_list": [
            {
                "model": model,
                "model_client_cls": "ClaudeCLIClient",
                "max_tokens": max_tokens,
            }
        ],
        "cache_seed": None,
    }


def _make_speaker_selector():
    """Create a stateful speaker selector that advances through SPEAKER_ORDER.

    메시지 카운트 대신 호출 횟수 기반으로 다음 발언자를 결정.
    이전 방식(len(agent_messages) % N)은 AutoGen이 에러/시스템 메시지를
    추가할 때 카운트가 깨져서 같은 에이전트만 반복 선택되는 버그가 있었음.
    """
    state = {"call_count": 0}

    def selector(last_speaker, groupchat):
        agents_by_name = {a.name: a for a in groupchat.agents}
        pos = state["call_count"] % len(SPEAKER_ORDER)
        state["call_count"] += 1
        next_speaker = agents_by_name[SPEAKER_ORDER[pos]]
        print(f"  [Speaker] Turn {state['call_count']}: {SPEAKER_ORDER[pos]}")
        return next_speaker

    return selector


def _load_context(config: AlphaSquadConfig) -> str:
    """Load project context files for the meeting."""
    parts = []
    for rel_path in config.context_files:
        full_path = config.project_root / rel_path
        if full_path.exists():
            parts.append(f"--- {rel_path} ---\n{full_path.read_text()[:3000]}")
    return "\n\n".join(parts)


def _load_latest_research(config: AlphaSquadConfig) -> str:
    """Load the latest autoresearch file for context."""
    research_dir = config.brain_dir / "autoresearch"
    if not research_dir.exists():
        return ""
    files = sorted(research_dir.glob("*.md"), reverse=True)
    if not files:
        return ""
    latest = files[0]
    return f"--- 최신 실험 기록: {latest.name} ---\n{latest.read_text()[:4000]}"


def _save_meeting_log(
    config: AlphaSquadConfig,
    topic: str,
    messages: list[dict],
    memories_saved: list[str],
) -> Path:
    """Save meeting transcript and results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.output_dir / f"meeting_{timestamp}.md"

    lines = [
        f"# AlphaSquad Meeting — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\n## Topic\n{topic}\n",
        "## Transcript\n",
    ]

    for msg in messages:
        name = msg.get("name", msg.get("role", "unknown"))
        content = msg.get("content", "")
        lines.append(f"### {name}\n{content}\n")

    if memories_saved:
        lines.append("## Memories Saved\n")
        for m in memories_saved:
            lines.append(f"- {m}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _extract_and_save_memories(
    memory: AlphaSquadMemory,
    messages: list[dict],
    topic: str,
) -> list[str]:
    """Extract key decisions and save to Mem0."""
    saved = []

    for msg in messages:
        name = msg.get("name", "")
        content = msg.get("content", "")
        if not content:
            continue

        if name == "Researcher3_PM":
            memory.add(
                agent_name=name,
                content=content,
                metadata={"type": "decision", "topic": topic},
            )
            saved.append(f"[{name}] decision saved")

        elif name.startswith("Expert"):
            if len(content) > 200:
                memory.add(
                    agent_name=name,
                    content=content[:1000],
                    metadata={"type": "critique", "topic": topic},
                )
                saved.append(f"[{name}] critique saved")

    return saved


def run_meeting(
    topic: str,
    config: AlphaSquadConfig | None = None,
    extra_context: str = "",
    use_memory: bool = True,
    rounds: int = 2,
    skip_r2_research: bool = False,
) -> dict:
    """Run an AlphaSquad research meeting.

    Args:
        topic: Meeting agenda / research question
        config: AlphaSquadConfig (uses defaults if None)
        extra_context: Additional context to inject
        use_memory: Whether to use Mem0 for persistent memory
        rounds: Number of full discussion rounds (each round = 5 speakers)

    Returns:
        dict with keys: transcript, log_path, experiment_specs
    """
    config = config or AlphaSquadConfig()
    config.meeting_topic = topic

    # Build LLM configs — uses claude CLI (Max subscription)
    standard_llm = _build_llm_config(config.model, config.max_tokens)
    pm_llm = _build_llm_config(config.pm_model, config.max_tokens)

    # Load context
    project_context = _load_context(config)
    latest_research = _load_latest_research(config)

    # Research phase: each agent gathers data before the meeting
    print("[AlphaSquad] Running research phase...")
    research_results = run_research_phase(topic, parallel=True)

    # Memory context per agent
    memory = None
    memory_contexts = {name: "" for name in SPEAKER_ORDER}
    if use_memory:
        try:
            memory = AlphaSquadMemory()
            for name in SPEAKER_ORDER:
                memory_contexts[name] = memory.get_context_string(name, topic)
        except Exception as e:
            print(f"[AlphaSquad] Mem0 init failed ({e}), proceeding without memory")
            memory = None

    # Recall accumulated knowledge from Mem0 per agent
    knowledge_contexts = {name: "" for name in SPEAKER_ORDER}
    if memory is not None:
        print("[AlphaSquad] Recalling accumulated knowledge from ChromaDB...")
        for name in SPEAKER_ORDER:
            knowledge_contexts[name] = recall_knowledge(name, topic, memory)
            if knowledge_contexts[name]:
                print(f"  [{name}] ChromaDB: {len(knowledge_contexts[name])} chars")

    # Recall from LightRAG knowledge base per agent
    lightrag_contexts = {name: "" for name in SPEAKER_ORDER}
    try:
        print("[AlphaSquad] Querying LightRAG knowledge base...")
        kg = AlphaSquadKnowledge()

        async def _query_all():
            for name in SPEAKER_ORDER:
                kg_name = _AGENT_TO_KNOWLEDGE.get(name)
                if not kg_name:
                    continue
                try:
                    ctx = await kg.query_context(kg_name, topic, mode="hybrid", max_chars=3000)
                    if ctx:
                        lightrag_contexts[name] = ctx
                        print(f"  [{name}] LightRAG: {len(ctx)} chars")
                except Exception as e:
                    print(f"  [{name}] LightRAG query failed: {e}")
            await kg.close()

        asyncio.run(_query_all())
    except Exception as e:
        print(f"[AlphaSquad] LightRAG init failed ({e}), proceeding without knowledge base")

    # Researcher2 전용 리서치 파이프라인 (3소스 검색 + Opus 순위)
    r2_research_context = ""
    try:
        if skip_r2_research:
            # 최신 캐시된 컨텍스트 재사용
            ctx_dir = Path(__file__).parent.parent / "research_store" / "contexts"
            ctx_files = sorted(ctx_dir.glob("*.md"), reverse=True)
            md_files = [f for f in ctx_files if f.name != "index.json"]
            if md_files:
                r2_research_context = md_files[0].read_text(encoding="utf-8")
                print(f"  [Researcher2] 캐시된 리서치 재사용: {md_files[0].name} ({len(r2_research_context)} chars)")
            else:
                print("  [Researcher2] 캐시 없음 — 새 리서치 실행")
                r2_research_context = get_research_context(topic)
        else:
            print("[AlphaSquad] Researcher2 리서치 파이프라인 실행...")
            r2_research_context = get_research_context(topic)
            print(f"  [Researcher2] 리서치 컨텍스트: {len(r2_research_context)} chars")
    except Exception as e:
        print(f"  [Researcher2] 리서치 파이프라인 실패: {e}")

    # Combine memory + lightrag knowledge + research context per agent
    def _agent_context(name: str) -> str:
        parts = []
        if memory_contexts.get(name):
            parts.append(memory_contexts[name])
        if knowledge_contexts.get(name):
            parts.append(knowledge_contexts[name][:2000])
        if lightrag_contexts.get(name):
            parts.append(lightrag_contexts[name])
        # R2는 전용 리서치 파이프라인 결과를 받음 (다른 에이전트보다 훨씬 많은 컨텍스트)
        if name == "Researcher2_AI아키텍처" and r2_research_context:
            parts.append(f"\n{r2_research_context[:10000]}")
        if research_results.get(name):
            parts.append(f"\n[사전 리서치 결과]\n{research_results[name][:3000]}")
        return "\n".join(parts)

    # Build per-agent project context from static cache
    # (replaces hardcoded stats that were previously in agent system prompts)
    static = research_results  # static cache is already in research_results from run_research_phase
    def _project_context(name: str) -> str:
        parts = []
        # Common: experiment results + latest research
        if static.get("Researcher1_금융모델링"):
            # Extract experiment results and model config from R1's research
            r1_data = static["Researcher1_금융모델링"]
            if "[실험 결과]" in r1_data:
                results_section = r1_data.split("[실험 결과]")[1].split("[")[0]
                parts.append(f"[실험 결과]\n{results_section[:2000]}")
            if "[현재 SOTA Config]" in r1_data:
                config_section = r1_data.split("[현재 SOTA Config]")[1].split("[")[0]
                parts.append(f"[현재 SOTA Config]\n{config_section[:1500]}")

        if name == "Researcher1_금융모델링":
            if static.get("Researcher1_금융모델링"):
                if "[최신 실험 기록]" in static["Researcher1_금융모델링"]:
                    latest = static["Researcher1_금융모델링"].split("[최신 실험 기록]")[1].split("[")[0]
                    parts.append(f"[최신 실험 기록]\n{latest[:2000]}")

        elif name == "Researcher2_AI아키텍처":
            if static.get("Researcher3_PM"):
                pm_data = static["Researcher3_PM"]
                if "[코드베이스 구조]" in pm_data:
                    codebase = pm_data.split("[코드베이스 구조]")[1].split("[")[0]
                    parts.append(f"[코드베이스 구조]\n{codebase[:1500]}")

        elif name == "Researcher3_PM":
            if static.get("Researcher3_PM"):
                pm_data = static["Researcher3_PM"]
                if "[코드베이스 구조]" in pm_data:
                    codebase = pm_data.split("[코드베이스 구조]")[1].split("[")[0]
                    parts.append(f"[코드베이스 구조]\n{codebase[:1500]}")
                if "[프로젝트 제약]" in pm_data:
                    constraints = pm_data.split("[프로젝트 제약]")[1].split("[")[0]
                    parts.append(f"[프로젝트 제약]\n{constraints[:1500]}")

        elif name == "Expert1_피처전문가":
            if static.get("Expert1_피처전문가"):
                e1_data = static["Expert1_피처전문가"]
                if "[피처 데이터 분석]" in e1_data:
                    feat = e1_data.split("[피처 데이터 분석]")[1].split("[")[0]
                    parts.append(f"[피처 데이터 분석]\n{feat[:2000]}")
                if "[피처 문서]" in e1_data:
                    docs = e1_data.split("[피처 문서]")[1].split("[")[0]
                    parts.append(f"[피처 문서]\n{docs[:1500]}")

        elif name == "Expert2_문제정의전문가":
            if static.get("Expert2_문제정의전문가"):
                e2_data = static["Expert2_문제정의전문가"]
                if "[Loss 구현]" in e2_data:
                    loss = e2_data.split("[Loss 구현]")[1].split("[")[0]
                    parts.append(f"[Loss 구현]\n{loss[:1500]}")
                if "[Label 분포]" in e2_data:
                    label = e2_data.split("[Label 분포]")[1].split("[")[0]
                    parts.append(f"[Label 분포]\n{label[:1500]}")

        return "\n\n".join(parts) if parts else "(프로젝트 데이터 로딩 중)"

    # R2 = opus (연구의 정점), PM = opus, 나머지 = sonnet
    r2_llm = _build_llm_config(config.pm_model, config.max_tokens)  # opus

    # Create agents with dynamic project context + memory context
    agents = [
        create_researcher2(r2_llm, _agent_context("Researcher2_AI아키텍처"), _project_context("Researcher2_AI아키텍처")),
        create_expert1(standard_llm, _agent_context("Expert1_피처전문가"), _project_context("Expert1_피처전문가")),
        create_expert2(standard_llm, _agent_context("Expert2_문제정의전문가"), _project_context("Expert2_문제정의전문가")),
        create_researcher1(standard_llm, _agent_context("Researcher1_금융모델링"), _project_context("Researcher1_금융모델링")),
        create_researcher3(pm_llm, _agent_context("Researcher3_PM"), _project_context("Researcher3_PM")),
    ]

    # Register custom client on all agents
    for agent in agents:
        agent.register_model_client(model_client_cls=ClaudeCLIClient)

    # GroupChat with custom speaker selection
    max_rounds = len(SPEAKER_ORDER) * rounds + 1
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=_make_speaker_selector(),
        allow_repeat_speaker=True,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=standard_llm,
    )
    manager.register_model_client(model_client_cls=ClaudeCLIClient)

    # Compose the meeting kickoff message
    kickoff = f"""# AlphaSquad Research Meeting

## Agenda
{topic}

## Project Context
{project_context[:2000]}

## Latest Experiment Results
{latest_research[:2000]}

{f"## Additional Context{chr(10)}{extra_context}" if extra_context else ""}

---
회의를 시작합니다. Researcher2가 먼저 아이디어를 제안해 주세요.
"""

    # Run the meeting — Researcher2 starts first
    print(f"[AlphaSquad] Meeting started: {topic}")
    print(f"[AlphaSquad] {rounds} round(s), {len(SPEAKER_ORDER)} speakers per round")
    print()

    first_agent = agents[0]  # Researcher2
    result = first_agent.initiate_chat(
        manager,
        message=kickoff,
    )

    # Collect transcript
    all_messages = groupchat.messages

    # Post-meeting learning: store insights in Mem0
    memories_saved = []
    if memory is not None:
        memories_saved = post_meeting_learn(all_messages, topic, memory)

    # Save meeting log
    log_path = _save_meeting_log(config, topic, all_messages, memories_saved)

    # Extract experiment specs from PM's messages
    experiment_specs = []
    for msg in all_messages:
        if msg.get("name") == "Researcher3_PM" and "실험:" in msg.get("content", ""):
            experiment_specs.append(msg["content"])

    print(f"\n[AlphaSquad] Meeting complete. Log: {log_path}")
    print(f"[AlphaSquad] {len(experiment_specs)} experiment spec(s) produced")

    return {
        "transcript": all_messages,
        "log_path": str(log_path),
        "experiment_specs": experiment_specs,
        "memories_saved": memories_saved,
    }
