"""AlphaSquad configuration."""

from dataclasses import dataclass, field
from pathlib import Path


def latest_research_note(project_root: Path) -> Path | None:
    """Prefer new dated session notes, falling back to the archived research log."""
    research = project_root / "autoresearch"
    runs = sorted((research / "runs").glob("????-??-??*.md"))
    if runs:
        return runs[-1]
    archived = research / "records" / "binance-2026-04-28.md"
    return archived if archived.is_file() else None


@dataclass
class AlphaSquadConfig:
    # LLM
    model: str = "sonnet"
    pm_model: str = "opus"  # PM gets stronger model
    temperature: float = 0.7
    max_tokens: int = 4096

    # Meeting
    max_rounds: int = 15  # max conversation rounds per meeting
    meeting_topic: str = ""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "meetings")
    memory_dir: Path = field(default_factory=lambda: Path(__file__).parent / "memory_store")

    # Context files to load for meetings
    context_files: list[str] = field(default_factory=lambda: [
        "autoresearch/README.md",
        "research/binance.md",
        "research/kospi.md",
    ])

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
