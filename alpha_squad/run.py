#!/usr/bin/env python3
"""AlphaSquad: multi-agent research meeting system.

Usage:
    # Build knowledge (deep research, stored in Mem0)
    PYTHONPATH="." uv run python alpha_squad/run.py study "tabular deep learning for finance 2025"
    PYTHONPATH="." uv run python alpha_squad/run.py study "ranking loss functions"

    # Run a meeting (auto-recalls past knowledge)
    PYTHONPATH="." uv run python alpha_squad/run.py meet "다음 실험 방향은?"
    PYTHONPATH="." uv run python alpha_squad/run.py meet "TabM 개선 방안" --rounds 3

    # Refresh static cache (data analysis, codebase structure)
    PYTHONPATH="." uv run python alpha_squad/run.py refresh
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_study(args):
    """Deep research on a topic — builds knowledge in Mem0."""
    from alpha_squad.memory.store import AlphaSquadMemory
    from alpha_squad.research.knowledge import build_knowledge

    print(f"=== AlphaSquad Study: {args.topic} ===\n")

    try:
        memory = AlphaSquadMemory()
    except Exception as e:
        print(f"[WARN] Mem0 init failed ({e}), research will not be persisted")
        memory = None

    results = build_knowledge(args.topic, memory=memory)

    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)
    for agent, content in results.items():
        preview = content[:200].replace("\n", " ")
        print(f"\n[{agent}] {preview}...")

    if memory:
        print("\nAll findings stored in Mem0. Will be recalled in future meetings.")


def cmd_meet(args):
    """Run a research meeting."""
    from alpha_squad.config import AlphaSquadConfig
    from alpha_squad.meeting.orchestrator import run_meeting

    config = AlphaSquadConfig()
    if args.model:
        config.model = args.model
    if args.pm_model:
        config.pm_model = args.pm_model

    extra_context = ""
    if args.context_file:
        ctx_path = Path(args.context_file)
        if ctx_path.exists():
            extra_context = ctx_path.read_text()[:5000]
        else:
            print(f"[WARN] Context file not found: {ctx_path}")

    result = run_meeting(
        topic=args.topic,
        config=config,
        extra_context=extra_context,
        use_memory=not args.no_memory,
        rounds=args.rounds,
    )

    print("\n" + "=" * 60)
    print("MEETING SUMMARY")
    print("=" * 60)

    if result["experiment_specs"]:
        print("\n## Experiment Specs from PM:\n")
        for i, spec in enumerate(result["experiment_specs"], 1):
            print(f"--- Spec {i} ---")
            print(spec)
            print()
    else:
        print("\nNo experiment specs produced in this meeting.")

    if result["memories_saved"]:
        print(f"\n## Knowledge stored: {len(result['memories_saved'])} items")

    print(f"\nFull log: {result['log_path']}")


def cmd_refresh(args):
    """Refresh static research cache."""
    from alpha_squad.research.phase import run_static_research

    print("=== Refreshing static research cache ===\n")
    results = run_static_research(force=True)
    print(f"\nCached {len(results)} data sources for today.")


def main():
    parser = argparse.ArgumentParser(description="AlphaSquad Research System")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # study
    p_study = subparsers.add_parser("study", help="Deep research on a topic (builds Mem0 knowledge)")
    p_study.add_argument("topic", help="Research topic")

    # meet
    p_meet = subparsers.add_parser("meet", help="Run a research meeting")
    p_meet.add_argument("topic", help="Meeting agenda")
    p_meet.add_argument("--rounds", type=int, default=2, help="Discussion rounds (default: 2)")
    p_meet.add_argument("--context-file", type=str, help="Extra context file")
    p_meet.add_argument("--no-memory", action="store_true", help="Disable Mem0")
    p_meet.add_argument("--model", type=str, default=None, help="Override LLM model")
    p_meet.add_argument("--pm-model", type=str, default=None, help="Override PM model")

    # refresh
    subparsers.add_parser("refresh", help="Refresh static research cache")

    args = parser.parse_args()

    if args.command == "study":
        cmd_study(args)
    elif args.command == "meet":
        cmd_meet(args)
    elif args.command == "refresh":
        cmd_refresh(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
