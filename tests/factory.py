from src.models.participation_model import ParticipationModel
from src.model_setup import build_model_kwargs
from src.config.loader import load_config


def create_test_model(**overrides) -> tuple[ParticipationModel, dict]:
    """
    Create a ParticipationModel instance using the default config
    (set by DEFAULT_CONFIG, fallback to 'configs/default.yaml'),
    returning both the model and the parameter dictionary used.
    This is useful for tests that need to inspect model parameters.

    Args:
        **overrides: Any model parameters to override from defaults.
    Returns:
        tuple: (ParticipationModel instance, config dict)
    """
    if "altruism_learning" in overrides and "altruism_mode" not in overrides:
        overrides["altruism_mode"] = "surprise_learning" if bool(overrides["altruism_learning"]) else "static"

    model_app_cfg = load_config().model
    params = build_model_kwargs(model_app_cfg)
    params.update(overrides)
    model = ParticipationModel(**params)
    return model, params
