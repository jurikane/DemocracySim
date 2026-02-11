from pydantic import BaseModel
from typing import Optional
from pathlib import Path

class ModelConfig(BaseModel):
    """
    Configuration for the core simulation model.
    """
    # Cost/effort rate for participating in an election, as a fraction of current assets (0..1).
    election_cost_rate: float
    election_impact_on_mutation: float  # Impact of election on mutation rate
    mu: float              # Mutation rate
    rule_idx: int          # Index of the voting rule to use
    distance_idx: int      # Index of the distance function to use

    # --- Reward magnitude scaling (v2 economics) ---
    reward_rate_common: float = 0.0  # Common reward magnitude in % of agent assets
    reward_rate_personal: float = 0.0  # Personal reward magnitude in % of agent assets
    reward_threshold_common: float = 0.5  # Common coeff = threshold - dist_to_reality
    reward_threshold_personal: float = 0.5  # Personal coeff = threshold - dist(personality, elected)
    abstention_share: float = 1.0  # Share of common reward given to abstainers (0..1)

    # --- Adaptive participation learning (schema v2 thesis) ---
    participation_alpha: float = 0.05
    participation_beta: float = 1.0
    participation_init_q: float = 0.0
    participation_q_max: float = 50.0
    bias_toward_participation: float = 0.0
    # EMA alpha for participation baseline (1.0 => baseline becomes last step's value).
    participation_baseline_alpha: float = 0.1

    # --- Adaptive altruism learning (reality-weight) ---
    altruism_alpha: float = 0.05
    altruism_init: float = 0.5
    altruism_clip_min: float = 0.0
    altruism_clip_max: float = 1.0
    altruism_learning: bool = False
    altruism_static: float = 0.5

    # --- Satisfaction value (learning signal stub; used by altruism learning) ---
    # Modes (planned): "global", "area", "knowledge", "combination"
    satisfaction_mode: str = "area"
    # EMA alpha for satisfaction baseline (1.0 => baseline becomes last step's value).
    satisfaction_baseline_alpha: float = 0.1

    num_agents: int        # Number of agents in the simulation
    # common_assets should simply be 100 per agent for now (automatically set if None)
    common_assets: float = None   # Initial collective assets
    num_colors: int        # Number of color options
    color_patches_steps: int  # Steps for color patch adjustment
    patch_power: float     # Power/radius of color patching
    heterogeneity: float   # Heterogeneity factor for color distribution
    known_cells: int       # Number of cells each agent knows
    num_personality_groups: int # Number of unique agent personality_groups
    height: int            # Grid height
    width: int             # Grid width
    num_areas: int         # Number of areas (territories)
    av_area_height: int    # Average area height
    av_area_width: int     # Average area width
    area_size_variance: float  # Variance in area sizes
    seed: Optional[int] = None # Random seed for reproducibility
    # Per-agent personal_opt_dist (static preference distribution) ---
    personal_opt_dist_concentration: float = 1.0  # Controls the heterogeneity/intensity of the derived distribution

class VisualizationConfig(BaseModel):
    """
    Configuration for visualization settings.
    """
    cell_size: int = 10                # Size of each grid cell in pixels
    draw_borders: bool                 # Whether to draw area borders
    show_area_stats: Optional[bool] = True  # Show area statistics overlay
    calibration_mode: bool = False     # Reorder UI for calibration-focused layout
    show_static_infos: bool = True  # Show static info (e.g. personality group) in agent tooltips
    show_agent_debug_panel: bool = False  # Show per-agent debug panel
    agent_debug_area_id: Optional[int] = None  # Area ID to show (None = first area)
    agent_debug_max_steps: int = 1  # How many steps to retain/show in debug panel
    agent_debug_max_agents: int = 50  # Limit agents rendered in debug panel
    agent_debug_max_field_len: int = 180  # Max chars per field in debug panel

class SimulationConfig(BaseModel):
    """
    Configuration for simulation runs and storage.
    """
    runs: int             # Number of simulation runs
    num_steps: int        # Number of steps per run
    processes: int        # Number of parallel processes
    store_grid: bool      # Whether to store grid state
    grid_interval: int    # Interval for storing grid state
    base_seed: Optional[int] = None    # Simulations base random seed


class OutputConfig(BaseModel):
    """Configuration for where run artifacts are written."""
    # Absolute path is used as-is; relative paths are interpreted relative to project root.
    directory: Path = Path("data") / "simulation_output"


class AppConfig(BaseModel):
    """
    Top-level application configuration.
    """
    model: ModelConfig
    visualization: VisualizationConfig
    simulation: SimulationConfig
    output: Optional[OutputConfig] = None
