from tests.factory import create_test_model

from src.viz.visualization_elements import (
    AreaPuzzleColorDistributionElement,
    AreaPersonalityGroupDists,
    MainMetricsElement,
    PersonalityGroupDistribution,
)


def test_global_personality_group_plot_can_render_collapsible_panel() -> None:
    model, _cfg = create_test_model()

    html = PersonalityGroupDistribution(collapsible=True, default_open=False).render(model)

    assert "<details" in html
    assert "Global preference-group distribution" in html
    assert "<img" in html


def test_area_personality_group_plot_can_render_collapsible_panel() -> None:
    model, _cfg = create_test_model()

    html = AreaPersonalityGroupDists(collapsible=True, default_open=False).render(model)

    assert "<details" in html
    assert "Per-area preference-group distributions" in html
    assert "<img" in html


def test_main_metrics_panel_renders_after_model_steps() -> None:
    model, _cfg = create_test_model()
    for _ in range(3):
        model.step()

    html = MainMetricsElement().render(model)

    assert "<img" in html


def test_puzzle_color_distribution_panel_is_collapsible() -> None:
    model, _cfg = create_test_model()
    for _ in range(3):
        model.step()

    html = AreaPuzzleColorDistributionElement(collapsible=True, default_open=True).render(model)

    assert "<details" in html
    assert "Puzzle color distribution by area" in html
    assert "<img" in html
