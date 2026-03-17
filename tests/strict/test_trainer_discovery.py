"""trainer auto-discovery + registry 기능 검증."""

import pytest

from src.training.registry import (
    discover_trainers, create_trainer, TRAINER_REGISTRY, TRAINER_META,
)


class TestTrainerDiscovery:

    def test_discovers_tab_m(self):
        discover_trainers()
        assert "tab_m" in TRAINER_REGISTRY

    def test_discovers_ae_mlp(self):
        discover_trainers()
        assert "ae_mlp" in TRAINER_REGISTRY

    def test_import_from_trainers_package(self):
        from src.training.trainers.tab_m import TabMTrainer
        assert TabMTrainer is not None

    def test_import_from_trainers_package_ae_mlp(self):
        from src.training.trainers.ae_mlp import AEMLPTrainer
        assert AEMLPTrainer is not None


class TestTrainerMeta:

    def test_tab_m_has_default_config(self):
        discover_trainers()
        meta = TRAINER_META["tab_m"]
        assert meta["default_config"] is not None
        assert "d_block" in meta["default_config"]
        assert "batch_size" in meta["default_config"]

    def test_ae_mlp_has_default_config(self):
        discover_trainers()
        meta = TRAINER_META["ae_mlp"]
        assert meta["default_config"] is not None
        assert "bottleneck_dim" in meta["default_config"]

    def test_create_trainer_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            create_trainer("nonexistent_model", cfg=None)


class TestLoadConfigDefaultInjection:

    def test_model_config_injected_via_registry(self):
        """Config root에 없는 모델 config가 default_config으로 주입되는지 검증."""
        discover_trainers()
        from src.config import load_config
        cfg = load_config(overrides=["model.name=tab_m"])
        # tab_m config should be injected from default_config
        assert hasattr(cfg, "tab_m")
        assert cfg.tab_m.d_block == 512  # TabMConfig default

    def test_override_injected_config(self):
        """CLI override가 default_config보다 우선하는지 검증."""
        discover_trainers()
        from src.config import load_config
        cfg = load_config(overrides=["model.name=tab_m", "tab_m.d_block=128"])
        assert cfg.tab_m.d_block == 128

    def test_ae_mlp_config_injected(self):
        discover_trainers()
        from src.config import load_config
        cfg = load_config(overrides=["model.name=ae_mlp"])
        assert hasattr(cfg, "ae_mlp")
        assert cfg.ae_mlp.bottleneck_dim == 64
