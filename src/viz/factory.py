from __future__ import annotations
from src.config.schema import VisualizationConfig
from src.config.loader import AppConfig
from mesa.visualization.modules import CanvasGrid, ChartModule
from src.agents.color_cell import ColorCell

# Central color palette (index-aligned with color IDs)
COLORS = [
    "LightGray",   # 0
    "Red",         # 1
    "Green",       # 2
    "Blue",        # 3
    "Yellow",      # 4
    "Aqua",        # 5
    "Fuchsia",     # 6
    "Lime",        # 7
    "Maroon",      # 8
    "Orange",      # 9
]

# Module-level store for visualization config
_VIS_CFG: VisualizationConfig | None = None


def get_vis_cfg() -> VisualizationConfig | None:
    return _VIS_CFG


def make_canvas(cfg: AppConfig) -> CanvasGrid:
    """
    Build a CanvasGrid using the current config.
    Expects cfg like: { 'model': {...}, 'visualization': {...}}.
    """
    global _VIS_CFG
    model_cfg = cfg.model
    vis_cfg = cfg.visualization
    _VIS_CFG = vis_cfg  # expose to other visualization modules

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
                (f"pers_group_idx: {voter.personality_group_idx}, "
                 f"personality: {voter.personality}, assets: {voter.assets}")

        return p

    return CanvasGrid(portrayal, width, height, width * cell_px, height * cell_px)


def make_charts(cfg: AppConfig) -> list:
    """
    Build the list of chart/extra visualization elements.
    """
    model_cfg = cfg.model.model_dump()
    num_colors = int(model_cfg["num_colors"])
    vis_cfg = cfg.visualization
    calibration_mode = bool(getattr(vis_cfg, "calibration_mode", False))

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

    # Advanced matplotlib-based elements
    from src.viz.visualisation_elements import (
        PersonalityGroupDistribution,
        AreaDiagnosticsPanel,
        AreaStats,
        VoterTurnoutElement,
        AreaGiniElement,
        AreaPersonalityGroupDists,
        CohortElectionLearningDiagnostics,
    )
    if calibration_mode:
        # Calibration layout: area-focused first, then histograms, then global charts.
        extras = [
            AreaDiagnosticsPanel(),
            CohortElectionLearningDiagnostics(),
        ]
        return [*extras, color_distribution_chart, wealth_chart, voter_turnout, learning_means_chart]

    extras = [
        CohortElectionLearningDiagnostics(),
        PersonalityGroupDistribution(),
        AreaStats(),
        VoterTurnoutElement(),
        AreaGiniElement(),
        AreaPersonalityGroupDists(),
    ]

    return [color_distribution_chart, wealth_chart, voter_turnout, learning_means_chart, *extras]
