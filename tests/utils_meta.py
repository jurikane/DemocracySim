from __future__ import annotations

from pathlib import Path

import yaml

from src.config.schema import AppConfig


def load_appcfg_from_meta(run_dir: Path) -> AppConfig:
    meta_path = Path(run_dir) / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    config_ref = meta.get("config_ref")
    if not isinstance(config_ref, str) or not config_ref:
        raise ValueError("meta.yaml missing config_ref")
    cfg_path = (Path(run_dir) / config_ref).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    return AppConfig.model_validate(cfg)
