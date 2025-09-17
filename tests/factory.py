from src.participation_model import ParticipationModel
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")

def create_default_model(**overrides):
    with open(DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    params = config["model"]
    params.update(overrides)
    return ParticipationModel(**params)
