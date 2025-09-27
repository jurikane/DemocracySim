from pathlib import Path
from src.config.schema import AppConfig
import yaml
import os


def check_schema(open_file):
    """
    Load configuration from an open YAML file
    and validate against AppConfig schema.
    """
    raw = yaml.safe_load(open_file)
    return AppConfig.model_validate(raw)


def load_config(config_file=None):
    """
    Load configuration from a YAML file.
    """
    if config_file is None:
        config_file = os.environ.get("CONFIG_FILE", "default.yaml")

    cfg = Path(config_file)

    # Use absolute or direct if exists
    if cfg.is_absolute() and cfg.exists():
        with cfg.open("r") as f:
            return check_schema(f)
    if cfg.exists():
        with cfg.open("r") as f:
            return check_schema(f)

    # Try CWD (when invoked from project root)
    cwd_path = Path.cwd() / cfg
    if cwd_path.exists():
        with cwd_path.open("r") as f:
            return check_schema(f)

    # Try project-root `configs/`
    project_root = Path(__file__).resolve().parents[2]
    root_cfg = project_root / "configs" / cfg.name
    if root_cfg.exists():
        with root_cfg.open("r") as f:
            return check_schema(f)

    # Legacy fallback: src/configs/
    legacy = project_root / "src" / "configs" / cfg.name
    if legacy.exists():
        with legacy.open("r") as f:
            return check_schema(f)

    tried = [str(p) for p in [cfg, cwd_path, root_cfg, legacy]]
    raise FileNotFoundError(f"Config not found. Tried: {', '.join(tried)}")
