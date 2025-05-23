"""
Launch a Mesa web-server that *replays* a single batch run.

Usage:
    python replay_server.py [--run-dir <dir>] [--run-id <int>]

If no arguments are given, uses the most recent run in ./runs/
"""

import argparse, json, numpy as np, pyarrow.parquet as pq
from pathlib import Path
import mesa
from mesa.visualization.ModularVisualization import ModularServer
from participation_model import ColorCell, VoteAgent, Area, distance_functions
from model_setup import (
    canvas_element,
    voter_turnout,
    wealth_chart,
    color_distribution_chart,
)
from itertools import combinations, permutations, product


# ────────────────────  REPLAY MODEL  ──────────────────────────
class ParticipationReplay(mesa.Model):
    """A lightweight wrapper that replays pre-computed states."""
    def __init__(self, run_path: Path):
        self.meta       = json.loads((run_path / "meta.json").read_text())
        self.model_df   = pq.read_table(next(run_path.glob("model_*.parquet"))
                                        ).to_pandas().sort_values("step")
        static_file = next(run_path.glob("static_*.npz"))
        static = np.load(static_file)
        agents_data = static["agents"]
        areas_data = static["areas"]
        grid_file = next(run_path.glob("grid_*.npz"), None)
        if grid_file is None:
            raise FileNotFoundError("Grids were not stored for this run.")
        self.grids = np.load(grid_file)["grid"]  # (T, H, W)

        # restore params
        p = self.meta["model"]
        self.height, self.width = p["height"], p["width"]
        self.schedule = mesa.time.BaseScheduler(self)
        self.draw_borders = p.get("draw_borders", True)
        # Add attributes needed by VoteAgent
        self.known_cells = p.get("known_cells", 10)  # Default value from model_setup.py
        self.num_colors = p.get("num_colors", 3)     # Default value from model_setup.py

        # Add more attributes needed by VoteAgent
        distance_idx = p.get("distance_idx", 0)  # Default to first distance function
        self.distance_func = distance_functions[distance_idx]
        self.options = self.create_all_options(self.num_colors)
        self.color_search_pairs = list(combinations(range(0, self.num_colors), 2))

        # build grid + cells
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        cells = []
        uid = 0
        for y in range(self.height):
            for x in range(self.width):
                cell = ColorCell(uid, self, (x, y), 0)
                self.grid.place_agent(cell, (x, y))
                cells.append(cell)
                uid += 1
        self._cells = cells

        # recreate agents
        self.voting_agents = []
        for uid, x, y, pers_idx in agents_data:
            a = VoteAgent(int(uid), self, (int(x), int(y)),
                          personality_idx=int(pers_idx))
            self.voting_agents.append(a)
            self.grid.place_agent(a, (int(x), int(y)))

        # recreate areas
        self.areas = []
        for uid, x, y, h, w in areas_data:
            var = self.meta["model"]["area_size_variance"]
            ar = Area(int(uid), self, int(h), int(w), var)
            ar.idx_field = (int(x), int(y))
            self.areas.append(ar)

        # fake DataCollector
        first_row = self.model_df.iloc[0]
        reporters = {k: (lambda m, k=k: m._current_row[k])
                     for k in first_row.keys() if k not in ("run_id", "step")}
        self.datacollector = mesa.DataCollector(model_reporters=reporters)
        self._current_row = first_row.to_dict()
        self.datacollector.collect(self)

        self._t = 0
        self.running = True

    def step(self):
        if self._t + 1 >= len(self.grids):
            self.running = False
            return
        self._t += 1
        self._current_row = self.model_df.iloc[self._t].to_dict()
        colours_flat = self.grids[self._t].ravel()
        for i, cell in enumerate(self._cells):
            cell.color = int(colours_flat[i])
        self.datacollector.collect(self)

    @staticmethod
    def create_all_options(n: int, include_ties=False):
        """
        Creates a matrix (an array of all possible ranking vectors),
        if specified including ties.
        Rank values start from 0.

        Args:
            n (int): The number of items to rank (number of colors in our case)
            include_ties (bool): If True, rankings include ties.

        Returns:
            np.array: A matrix containing all possible ranking vectors.
        """
        if include_ties:
            # Create all possible combinations and sort out invalid rankings
            # i.e. [1, 1, 1] or [1, 2, 2] aren't valid as no option is ranked first.
            r = np.array([np.array(comb) for comb in product(range(n), repeat=n)
                          if set(range(max(comb))).issubset(comb)])
        else:
            r = np.array([np.array(p) for p in permutations(range(n))])
        return r


# ─────────────────────── server launcher ──────────────────────
def launch(run_dir: str, run_id: int):
    run_path = Path(run_dir)
    if run_id >= 0:
        run_path = run_path / f"model_{run_id}.parquet"
        if run_path.exists():
            run_path = run_path.parent
        else:
            raise FileNotFoundError("run_id not found in that directory.")

    model_params = {"run_path": run_path}

    server = ModularServer(
        ParticipationReplay,
        [canvas_element, wealth_chart,
         color_distribution_chart, voter_turnout],
        "Democracy-Sim replay",
        model_params,
    )
    server.port = 8585
    server.launch()


# ────────────────────────── CLI ───────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    help="Path to a run folder (e.g. simulation_output/20250425_101530)")
    ap.add_argument("--run-id", type=int, default=-1,
                    help="Run index to replay (model_#.parquet). Default = -1 = first one")
    args = ap.parse_args()

    if args.run_dir is None:
        runs_root = Path("simulation_output")
        run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()],
                          key=lambda p: p.name, reverse=True)
        if not run_dirs:
            raise FileNotFoundError(
                "No run directories found in ./simulation_output/")

        print("Available simulation runs:")
        for i, run in enumerate(run_dirs):
            print(f"  [{i}] {run.name}")

        selection = input("Select a run by number: ").strip()
        if not selection.isdigit() or not (0 <= int(selection) < len(run_dirs)):
            raise ValueError("Invalid selection.")

        args.run_dir = str(run_dirs[int(selection)])
        print(f"[INFO] Selected: {args.run_dir}")

    launch(**vars(args))
