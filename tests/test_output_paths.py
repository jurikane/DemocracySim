from pathlib import Path
import os

from src.config.loader import resolve_output_dir
from src.config.loader import get_project_root


class _Output:
    def __init__(self, directory):
        self.directory = directory


class _Conf:
    def __init__(self, output=None):
        self.output = output


def test_default_output_dir_is_project_root_data():
    conf = _Conf(output=None)
    base = resolve_output_dir(conf)
    assert base == get_project_root() / "data" / "simulation_output"


def test_relative_configured_output_dir_is_relative_to_project_root():
    conf = _Conf(output=_Output(directory=Path("data") / "simulation_output"))
    base = resolve_output_dir(conf)
    assert base == get_project_root() / "data" / "simulation_output"


def test_absolute_configured_output_dir_is_used_as_is(tmp_path: Path):
    conf = _Conf(output=_Output(directory=tmp_path / "out"))
    base = resolve_output_dir(conf)
    assert base == tmp_path / "out"


def test_tilde_and_env_in_output_dir_are_expanded(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DEMO_OUT", str(tmp_path / "env_out"))
    conf = _Conf(output=_Output(directory="~/demo/${DEMO_OUT}"))
    base = resolve_output_dir(conf)
    expected = Path(os.path.expanduser(os.path.expandvars("~/demo/${DEMO_OUT}")))
    assert base == expected
