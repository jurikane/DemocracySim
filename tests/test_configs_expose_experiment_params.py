from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_MODEL_KEYS = {
    # Adaptive participation learning
    "participation_alpha",
    "participation_beta",
    "participation_init_q",
    "participation_q_max",
    "bias_toward_participation",
    # Personal preference intensity distribution
    "personal_preference_peakedness",
    # Adaptive altruism learning
    "altruism_alpha",
    "altruism_init",
    "altruism_clip_min",
    "altruism_clip_max",
    "altruism_mode",
    "altruism_response_gamma",
}


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def test_main_configs_expose_experiment_params() -> None:
    """Guardrail: ensure key experiment knobs are present in the main configs.

    These parameters are meant to be tuned frequently during thesis experiments.
    Keeping them explicit in YAML avoids relying on code defaults.
    """

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"

    for name in ["default.yaml", "toy.yaml", "test.yaml", "config.yaml"]:
        cfg = _load_yaml(configs_dir / name)
        assert "model" in cfg, f"{name} missing 'model' section"
        model_cfg = cfg["model"]

        missing = sorted(k for k in REQUIRED_MODEL_KEYS if k not in model_cfg)
        assert not missing, f"{name} missing model keys: {missing}"
