"""Trainer registry and base class."""

import importlib
import json
import logging
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """모든 Trainer의 추상 기반 클래스 (npy 기반)."""

    @abstractmethod
    def fit_from_npy(self, npy_dir: str | Path) -> None:
        ...

    @abstractmethod
    def predict_npy(
        self, npy_dir: str | Path, split: str = "test"
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        ...

    def _load_npy_data(
        self, npy_dir: Path,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load memmap data, filter, sample.

        Requires self.cfg, self._vol_idx, self._h1, self.feature_columns to be set.

        Returns: (in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices)
        """
        from src.data.split import build_splits

        log.info(f"Class bins (interior): {list(self._class_interior_bins)} → {self._num_classes} classes")

        with open(npy_dir / "meta.json") as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]
        in_features = meta["n_features"]
        log.info(f"Features: {in_features}")

        use_offsets = (npy_dir / "X.npy").exists() and (npy_dir / "offsets.json").exists()

        if use_offsets:
            X = np.load(npy_dir / "X.npy", mmap_mode="r")
            y = np.load(npy_dir / "y.npy", mmap_mode="r")
            log.info(f"Memmap loaded (offsets mode): X={X.shape[0]:,}")
            tr_indices, va_indices = build_splits(
                json.load(open(npy_dir / "offsets.json")), y,
                vol_idx=self._vol_idx, target_idx=self._h1,
            )
            log.info(f"After filter: train={len(tr_indices):,}, val={len(va_indices):,}")
            X_train, y_train = X, y
            X_val, y_val = X, y
        else:
            X_train = np.load(npy_dir / "X_train.npy", mmap_mode="r")
            y_train = np.load(npy_dir / "y_train.npy", mmap_mode="r")
            X_val = np.load(npy_dir / "X_val.npy", mmap_mode="r")
            y_val = np.load(npy_dir / "y_val.npy", mmap_mode="r")
            log.info(f"Memmap loaded (split mode): train={X_train.shape[0]:,}, val={X_val.shape[0]:,}")

            def _valid_mask(y_arr: np.ndarray) -> np.ndarray:
                return (y_arr[:, self._vol_idx] != 0) & (y_arr[:, self._h1] != 0)

            tr_indices = np.where(_valid_mask(y_train))[0]
            va_indices = np.where(_valid_mask(y_val))[0]

        rng = np.random.default_rng(self.cfg.seed)
        sample_ratio = self.cfg.data.sample_ratio
        n_tr = int(len(tr_indices) * sample_ratio)
        n_va = int(len(va_indices) * sample_ratio)
        tr_indices = rng.choice(tr_indices, n_tr, replace=False)
        va_indices = rng.choice(va_indices, n_va, replace=False)
        log.info(f"Sample ratio={sample_ratio}: train={n_tr:,}, val={n_va:,}")

        return in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices


TRAINER_REGISTRY: dict[str, type] = {}
TRAINER_META: dict[str, dict] = {}


def register_trainer(
    name: str,
    *,
    config_key: str = "",
    model_type: str = "cls",
    default_config: dict | None = None,
) -> Callable:
    """Trainer 클래스를 레지스트리에 등록하는 데코레이터.

    Args:
        name: 모델 이름 (e.g. "tab_m")
        config_key: cfg에서 모델 설정을 가져올 키 (e.g. "tab_m" → cfg.tab_m)
                     빈 문자열이면 name과 동일
        model_type: "cls" (classification) 또는 "reg" (regression)
        default_config: 새 모델용 기본 설정 dict. Config 루트 dataclass에
                        정의되지 않은 모델은 이걸로 기본값 제공.
    """
    def decorator(cls: type) -> type:
        if name in TRAINER_REGISTRY:
            raise ValueError(f"Trainer '{name}' already registered")
        TRAINER_REGISTRY[name] = cls
        TRAINER_META[name] = {
            "config_key": config_key or name,
            "model_type": model_type,
            "default_config": default_config,
        }
        return cls
    return decorator


def create_trainer(name: str, **kwargs: Any) -> Any:
    """이름으로 Trainer 인스턴스 생성."""
    if name not in TRAINER_REGISTRY:
        raise KeyError(
            f"Trainer '{name}' not found. Available: {list(TRAINER_REGISTRY.keys())}"
        )
    return TRAINER_REGISTRY[name](**kwargs)


def get_model_config(cfg: Any, model_name: str) -> Any:
    """모델별 config 섹션을 가져온다. (e.g. cfg.tab_m, cfg.nn)"""
    if model_name not in TRAINER_META:
        raise KeyError(f"Model '{model_name}' not registered")
    key = TRAINER_META[model_name]["config_key"]
    return getattr(cfg, key)


def is_classification(model_name: str) -> bool:
    """모델이 classification인지 확인."""
    return TRAINER_META.get(model_name, {}).get("model_type") == "cls"


def list_trainers() -> list[str]:
    return list(TRAINER_REGISTRY.keys())


def discover_trainers() -> None:
    """src/training/trainers/ 아래의 모듈을 자동 임포트하여 등록."""
    import src.training.trainers as pkg
    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"src.training.trainers.{module_name}")
