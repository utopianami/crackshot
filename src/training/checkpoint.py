"""Crash-safe best-model checkpointing.

Trainers call `save_best_checkpoint()` whenever val metric improves. The helper
temporarily swaps `trainer.model` to the best model so the trainer's existing
`save()` method serializes the right weights, then restores. A `best_meta.json`
is written alongside.

If training crashes mid-run, the last-best snapshot is already on disk at the
same path that would have been used at training's end.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def save_best_checkpoint(
    trainer: Any,
    save_path: Path | None,
    best_model: Any,
    epoch: int,
    val_loss: float,
    extra_meta: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Serialize best_model via trainer.save() to save_path, write meta.

    No-op if save_path is None (e.g. when running without an orchestrator).
    """
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    original = getattr(trainer, "model", None)
    trainer.model = best_model
    try:
        trainer.save(save_path)
        meta: dict[str, Any] = {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "trainer_class": type(trainer).__name__,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(save_path / "best_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    finally:
        trainer.model = original

    lg = logger or log
    lg.info(
        f"[checkpoint] best saved (epoch={epoch}, val_loss={val_loss:.6f}) → {save_path}"
    )
