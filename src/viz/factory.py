from __future__ import annotations
from src.config.loader import AppConfig
from mesa.visualization.modules import CanvasGrid, ChartModule
from src.agents.color_cell import ColorCell
from src.viz.color_palette import COLORS

def make_canvas(cfg: AppConfig) -> CanvasGrid:
    """
    Build a CanvasGrid using the current config.
    Expects cfg like: { 'model': {...}, 'visualization': {...}}.
    """
    model_cfg = cfg.model
    vis_cfg = cfg.visualization

    width = int(model_cfg.width)
    height = int(model_cfg.height)
    cell_px = int(vis_cfg.cell_size)
    draw_borders = bool(vis_cfg.draw_borders)

    def portrayal(agent):
        # We only draw ColorCell objects (grid contains one per cell)
        if not isinstance(agent, ColorCell):
            return None

        color_name = COLORS[agent.color] \
            if 0 <= agent.color < len(COLORS) else "Black"
        p = {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Layer": 0,
            "Color": color_name,
            # Hover fields:
            "Position": f"{agent.pos}",
            "Color - text": color_name,
        }

        # Mark area borders (except global area) as circles
        if draw_borders and agent.is_border_cell:
            p["Shape"] = "circle"
            p["r"] = 1
            if color_name == "LightGray":
                p["Color"] = "Gainsboro"

        # Show agent count in the cell
        if agent.num_agents_in_cell > 0:
            p["text"] = str(agent.num_agents_in_cell)
            p["text_color"] = "Black"

        # Add area info (tooltips)
        for a in agent.areas:
            aid = a.unique_id if a.unique_id != -1 else "global"
            p[f"Area {aid}"] = \
                f"{a.num_agents} agents, color dist: {a.color_distribution}"

        # Add agent info (tooltips)
        for voter in agent.agents:
            p[f"Agent {voter.unique_id}"] = \
                (f"pref-group-idx: {voter.personality_group_idx}, "
                 f"personality: {voter.personality}, assets: {voter.assets}")

        return p

    return CanvasGrid(portrayal, width, height, width * cell_px, height * cell_px)


def make_charts(cfg: AppConfig) -> list:
    """
    Build the list of chart/extra visualization elements.
    """
    model_cfg = cfg.model.model_dump()
    num_colors = model_cfg["num_colors"]
    vis_cfg = cfg.visualization
    calibration_mode = vis_cfg.calibration_mode
    show_area_stats = vis_cfg.show_area_stats
    show_agent_debug_panel = vis_cfg.show_agent_debug_panel
    show_static_infos = vis_cfg.show_static_infos

    color_distribution_chart = ChartModule(
        [{"Label": f"color_{i}",
          "Color": ("LightGrey" if COLORS[i] == "LightGray" else COLORS[i])}
         for i in range(num_colors)],
        data_collector_name="datacollector",
    )

    wealth_chart = ChartModule(
        [{"Label": "collective_assets", "Color": "Black"}],
        data_collector_name="datacollector",
    )

    voter_turnout = ChartModule(
        [
            {"Label": "turnout", "Color": "Black"},
            {"Label": "gini_index", "Color": "Red"},
        ],
        data_collector_name="datacollector",
    )

    learning_means_chart = ChartModule(
        [
            {"Label": "mean_p_participation", "Color": "Black"},
            {"Label": "mean_altruism", "Color": "Blue"},
        ],
        data_collector_name="datacollector",
    )
    dissatisfaction_chart = ChartModule(
        [
            {"Label": "mean_dissatisfaction", "Color": "Green"},
        ],
        data_collector_name="datacollector",
    )

    # Advanced matplotlib-based elements
    from src.viz.visualization_elements import (
        ReplayGridStepStatusElement,
        PersonalityGroupDistribution,
        AreaDiagnosticsPanel,
        MainMetricsElement,
        AreaPersonalityGroupDists,
        AreaPuzzleColorDistributionElement,
        CohortElectionLearningDiagnostics,
    )
    from src.viz.debug_viz import AreaAgentDebugPanel
    extras = [ReplayGridStepStatusElement()]
    if show_area_stats:
        extras.append(AreaDiagnosticsPanel())
    if show_agent_debug_panel:
        extras.append(
            AreaAgentDebugPanel(
                max_steps=int(getattr(vis_cfg, "agent_debug_max_steps", 1)),
                area_id=getattr(vis_cfg, "agent_debug_area_id", None),
                max_agents=int(getattr(vis_cfg, "agent_debug_max_agents", 50)),
                max_field_len=int(
                    getattr(vis_cfg, "agent_debug_max_field_len", 180)),
            )
        )
    if show_static_infos:
        extras.append(MainMetricsElement())
        extras.append(
            AreaPuzzleColorDistributionElement(
                collapsible=True,
                default_open=True,
            )
        )
        extras.append(
            PersonalityGroupDistribution(
                collapsible=True,
                default_open=False,
            )
        )
        extras.append(
            AreaPersonalityGroupDists(
                collapsible=True,
                default_open=True,
            )
        )
    if calibration_mode:
        extras.append(CohortElectionLearningDiagnostics())

    return [*extras, color_distribution_chart]
    # return [*extras, color_distribution_chart, wealth_chart, voter_turnout,
    #        learning_means_chart, dissatisfaction_chart]
