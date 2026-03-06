"""Trainer registry and base class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl


class BaseTrainer(ABC):
    """모든 Trainer의 추상 기반 클래스."""

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
        ...

    def fit_from_disk(
        self,
        train_paths: list[tuple["Path", "Path"]],
        val_paths: list[tuple["Path", "Path"]],
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support disk-based training"
        )

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        ...


TRAINER_REGISTRY: dict[str, type] = {}


def register_trainer(name: str) -> Callable:
    """Trainer 클래스를 레지스트리에 등록하는 데코레이터."""
    def decorator(cls: type) -> type:
        if name in TRAINER_REGISTRY:
            raise ValueError(f"Trainer '{name}' already registered")
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


def create_trainer(name: str, **kwargs: Any) -> Any:
    """이름으로 Trainer 인스턴스 생성."""
    if name not in TRAINER_REGISTRY:
        raise KeyError(
            f"Trainer '{name}' not found. Available: {list(TRAINER_REGISTRY.keys())}"
        )
    return TRAINER_REGISTRY[name](**kwargs)


def list_trainers() -> list[str]:
    return list(TRAINER_REGISTRY.keys())
