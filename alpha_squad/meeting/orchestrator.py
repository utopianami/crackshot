"""AlphaSquad meeting orchestrator using AutoGen GroupChat + claude CLI."""

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
from alpha_squad.research.phase import run_research_phase
from alpha_squad.research.knowledge import recall_knowledge, post_meeting_learn


# Meeting flow: who speaks after whom
SPEAKER_ORDER = [
    "Researcher2_AI아키텍처",     # 1. 아이디어 제안
    "Expert1_피처전문가",          # 2. 피처 관점 비판
    "Expert2_문제정의전문가",      # 3. 문제 정의 관점 비판
    "Researcher1_금융모델링",      # 4. 실현 가능성 검토
    "Researcher3_PM",             # 5. 수용/기각 + 실험 설계
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


def _custom_speaker_selection(last_speaker, groupchat):
    """Enforce the meeting flow order, allowing multiple rounds."""
    agents_by_name = {a.name: a for a in groupchat.agents}
    messages = groupchat.messages

    if not messages or last_speaker is None:
        return agents_by_name[SPEAKER_ORDER[0]]

    last_name = last_speaker.name
    try:
        idx = SPEAKER_ORDER.index(last_name)
        next_idx = (idx + 1) % len(SPEAKER_ORDER)
    except ValueError:
        next_idx = 0

    return agents_by_name[SPEAKER_ORDER[next_idx]]


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
        print("[AlphaSquad] Recalling accumulated knowledge...")
        for name in SPEAKER_ORDER:
            knowledge_contexts[name] = recall_knowledge(name, topic, memory)
            if knowledge_contexts[name]:
                print(f"  [{name}] recalled {len(knowledge_contexts[name])} chars")

    # Combine memory + knowledge + research context per agent
    def _agent_context(name: str) -> str:
        parts = []
        if memory_contexts.get(name):
            parts.append(memory_contexts[name])
        if knowledge_contexts.get(name):
            parts.append(knowledge_contexts[name][:2000])
        if research_results.get(name):
            parts.append(f"\n[사전 리서치 결과]\n{research_results[name][:3000]}")
        return "\n".join(parts)

    # Create agents with research context injected
    agents = [
        create_researcher2(standard_llm, _agent_context("Researcher2_AI아키텍처")),
        create_expert1(standard_llm, _agent_context("Expert1_피처전문가")),
        create_expert2(standard_llm, _agent_context("Expert2_문제정의전문가")),
        create_researcher1(standard_llm, _agent_context("Researcher1_금융모델링")),
        create_researcher3(pm_llm, _agent_context("Researcher3_PM")),
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
        speaker_selection_method=_custom_speaker_selection,
        allow_repeat_speaker=False,
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
