"""AlphaSquad configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlphaSquadConfig:
    # LLM
    model: str = "claude-sonnet-4-20250514"
    pm_model: str = "claude-opus-4-20250514"  # PM gets stronger model
    temperature: float = 0.7
    max_tokens: int = 4096

    # Meeting
    max_rounds: int = 15  # max conversation rounds per meeting
    meeting_topic: str = ""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    brain_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "BRAIN" / "kospi")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "meetings")
    memory_dir: Path = field(default_factory=lambda: Path(__file__).parent / "memory_store")

    # Context files to load for meetings
    context_files: list[str] = field(default_factory=lambda: [
        "BRAIN/kospi/autoresearch_common.md",
        "BRAIN/kospi/data.md",
        "BRAIN/kospi/program.md",
    ])

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
