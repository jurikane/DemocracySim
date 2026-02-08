import pytest
import tempfile
import os
from pathlib import Path
from src.config import loader
from unittest.mock import patch

@pytest.fixture
def mock_model_validate():
    with patch("src.config.schema.AppConfig.model_validate",
               return_value="validated") as mock:
        yield mock

def test_check_schema_valid(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("key: value")
    with cfg.open("r") as f:
        with patch("src.config.schema.AppConfig.model_validate", return_value="ok") as mock:
            result = loader.check_schema(f)
    assert result == "ok"
    mock.assert_called_once()

def test_load_config_default_yaml(monkeypatch, tmp_path):
    default = tmp_path / "default.yaml"
    default.write_text("key: value")
    monkeypatch.chdir(tmp_path)
    with patch("src.config.schema.AppConfig.model_validate", return_value="validated"):
        result = loader.load_config()
    assert result == "validated"

def test_load_config_direct_path(mock_model_validate):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write("key: value")
        tmp_path = Path(tmp.name)
    result = loader.load_config(str(tmp_path))
    assert result == "validated"
    tmp_path.unlink()

def test_load_config_cwd(mock_model_validate):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=Path.cwd(), delete=False) as tmp:
        tmp.write("key: value")
        tmp_path = Path(tmp.name)
    result = loader.load_config(tmp_path.name)
    assert result == "validated"
    tmp_path.unlink()

def test_load_config_project_root(mock_model_validate, tmp_path, monkeypatch):
    # Simulate project root two levels up
    project_root = tmp_path / "project"
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True)
    cfg_file = configs_dir / "test.yaml"
    cfg_file.write_text("key: value")

    # Patch __file__ to trick loader into thinking it's inside project_root/src/config
    fake_loader_file = project_root / "src" / "config" / "loader.py"
    fake_loader_file.parent.mkdir(parents=True)
    fake_loader_file.write_text("# fake loader file")

    monkeypatch.setattr(loader, "__file__", str(fake_loader_file))

    result = loader.load_config("test.yaml")
    assert result == "validated"

def test_load_config_env_var(mock_model_validate):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write("key: value")
        tmp_path = Path(tmp.name)
    os.environ["CONFIG_FILE"] = str(tmp_path)
    result = loader.load_config()
    assert result == "validated"
    tmp_path.unlink()
    del os.environ["CONFIG_FILE"]

def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        loader.load_config("nonexistent.yaml")
