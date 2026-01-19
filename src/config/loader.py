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


def get_project_root() -> Path:
    """
    Returns the root folder of the project (two levels up from this file).
    """
    return Path(__file__).resolve().parents[2]


def get_project_subfolder(*subfolders, create_if_missing=False) -> Path:
    """
    Get a Path object pointing to a subfolder inside the project root.

    Args:
        *subfolders: Subfolder names to append to the project root.
        create_if_missing: If True, automatically create the folder (and parents)
        if it doesn't exist.

    Returns:
        Path object to the folder.
    """
    path = get_project_root().joinpath(*subfolders)
    if create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(f"File or subfolder does not exist: {path}")
    return path


def load_config(config_file=None) -> AppConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the YAML config file.

    Returns:
        AppConfig: Validated configuration object.
    """
    if config_file is None:
        config_file = os.environ.get("CONFIG_FILE", "default.yaml")
    print(f"Loading config from: {config_file}")
    cfg_path = Path(config_file)

    # 1) Absolute path
    if cfg_path.is_absolute() and cfg_path.exists():
        with cfg_path.open("r") as f:
            return check_schema(f)

    # 2) Relative path (CWD)
    cwd_path = Path.cwd() / cfg_path
    if cwd_path.exists():
        with cwd_path.open("r") as f:
            return check_schema(f)

    # 3) Project-root configs/
    root_cfg = get_project_subfolder("configs") / cfg_path.name
    if root_cfg.exists():
        with root_cfg.open("r") as f:
            return check_schema(f)

    # 4) Legacy src/configs/
    legacy_cfg = get_project_subfolder("src", "configs") / cfg_path.name
    if legacy_cfg.exists():
        with legacy_cfg.open("r") as f:
            return check_schema(f)

    # If nothing found
    tried = [str(p) for p in [cfg_path, cwd_path, root_cfg, legacy_cfg]]
    raise FileNotFoundError(f"Config not found. Tried: {', '.join(tried)}")
