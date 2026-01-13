import matplotlib.pyplot as plt
from mesa.visualization import TextElement
import matplotlib.patches as patches
from src.viz.factory import COLORS, get_vis_cfg
import base64
import math
import io

# Visualization config (is set by make_canvas before these are instantiated)
vis_cfg = get_vis_cfg()
show_area_stats = bool(getattr(vis_cfg, 'show_area_stats', True))


def save_plot_to_base64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}" />'


class AreaStats(TextElement):
    def render(self, model):
        step = getattr(model.scheduler, 'steps', 0)
        if not show_area_stats or step == 0:
            return ""
        data = model.datacollector.get_agent_vars_dataframe()
        if data is None or len(data) == 0:
            return ""
        if 'ColorDistribution' not in data.columns or 'DistToReality' not in data.columns or 'ElectionResults' not in data.columns:
            return ""
        color_distribution = data['ColorDistribution'].dropna()
        dist_to_reality = data['DistToReality'].dropna()
        election_results = data['ElectionResults'].dropna()

        # Do not drop area 0: only exclude the optional global area id (-1) if present.
        try:
            area_ids_all = list(color_distribution.index.get_level_values(1).unique())
            area_ids = [aid for aid in area_ids_all if int(aid) != -1]
        except Exception:
            area_ids = color_distribution.index.get_level_values(1).unique()

        if len(color_distribution) == 0 or len(area_ids) == 0:
            return ""

        num_colors = len(color_distribution.iloc[0])
        num_areas = len(area_ids)
        fig, axes = plt.subplots(nrows=num_areas, ncols=2,
                                 figsize=(8, 4 * num_areas), sharex=True)

        # Handle case of single area (axes shape)
        if num_areas == 1:
            axes = [axes]

        for i, area_id in enumerate(area_ids):
            row = i
            ax1 = axes[row][0]
            try:
                area_data = color_distribution.xs(area_id, level=1)
                a_data = dist_to_reality.xs(area_id, level=1)
            except Exception:
                continue
            ax1.plot(a_data.index, a_data.values, color='Black', linestyle='--')
            for color_idx in range(num_colors):
                try:
                    cdata = area_data.apply(lambda x: x[color_idx])
                    ax1.plot(cdata.index, cdata.values, color=COLORS[color_idx])
                except Exception:
                    continue
            ax1.set_title(f'Area {area_id} \n--- deviation from voted distribution')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Color Distribution')

            ax2 = axes[row][1]
            try:
                area_data = election_results.xs(area_id, level=1)
            except Exception:
                continue
            for color_id in range(num_colors):
                try:
                    cdata = area_data.apply(lambda x: list(x).index(color_id) if color_id in x else None)
                    ax2.plot(cdata.index, cdata.values, marker='o',
                             label=f'Color {color_id}', color=COLORS[color_id],
                             linewidth=0.2)
                except Exception:
                    continue
            ax2.set_title(f'Area {area_id} \n')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Elected ranking (rank values)')
            ax2.invert_yaxis()

        plt.tight_layout()
        return save_plot_to_base64(fig)


class PersonalityDistribution(TextElement):
    def __init__(self):
        super().__init__()
        self.pers_dist_plot = None

    def create_once(self, model):
        try:
            dists = getattr(model, 'personality_distribution', None)
            personalities = getattr(model, 'personalities', None)
            if dists is None or personalities is None:
                self.pers_dist_plot = ""
                return
            num_personalities = personalities.shape[0]
            num_agents = getattr(model, 'num_agents', 0)
            colors = COLORS[:getattr(model, 'num_colors', len(COLORS))]
            num_colors = len(personalities[0])
        except Exception:
            self.pers_dist_plot = ""
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        heights = dists
        bars = ax.bar(range(num_personalities), heights, width=0.6)

        for bar, personality in zip(bars, personalities):
            height = bar.get_height()
            width = bar.get_width()
            for i, color_idx in enumerate(personality):
                rect_width = width / num_colors
                coords = (bar.get_x() + i * rect_width, 0)
                rect = patches.Rectangle(coords, rect_width, height,
                                         color=colors[color_idx])
                ax.add_patch(rect)

        ax.set_xlabel('"Personality" ID')
        ax.set_ylabel(f'Percentage of the {num_agents} Agents')
        ax.set_title('Global distribution of personalities among agents')
        plt.tight_layout()
        self.pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model):
        if getattr(model.scheduler, 'steps', 0) == 0:
            self.create_once(model)
        return self.pers_dist_plot or ""


class VoterTurnoutElement(TextElement):
    def render(self, model):
        step = getattr(model.scheduler, 'steps', 0)
        if not show_area_stats or step == 0:
            return ""
        try:
            data = model.datacollector.get_agent_vars_dataframe()
            if data is None or len(data) == 0 or 'VoterTurnout' not in data.columns:
                return ""
            voter_turnout = data['VoterTurnout'].dropna()
        except Exception:
            return ""
        if len(voter_turnout) == 0:
            return ""

        area_ids = voter_turnout.index.get_level_values(1).unique()
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, area_id in enumerate(area_ids):
            try:
                area_data = voter_turnout.xs(area_id, level=1)
            except Exception:
                continue
            if i < 10:
                line_style = '-'
            elif i < 20:
                line_style = ':'
            else:
                line_style = '--'
            ax.plot(area_data.index, area_data.values, label=f'Area {area_id}',
                    linestyle=line_style)
        ax.set_title('Voter Turnout by Area Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Voter Turnout (%)')
        ax.legend()
        return save_plot_to_base64(fig)


class MatplotlibElement(TextElement):
    def render(self, model):
        step = getattr(model.scheduler, 'steps', 0)
        if not show_area_stats or step == 0:
            return ""
        try:
            data = model.datacollector.get_model_vars_dataframe()
            collective_assets = data.get("Collective assets")
        except Exception:
            collective_assets = None
        if collective_assets is None:
            return ""
        fig, ax = plt.subplots()
        ax.plot(collective_assets, label="Collective assets")
        ax.set_title("Collective Assets Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Collective Assets")
        ax.legend()
        return save_plot_to_base64(fig)


class StepsTextElement(TextElement):
    def render(self, model):
        step = getattr(model.scheduler, 'steps', 0)
        first_agents = [str(a) for a in getattr(model, 'voting_agents', [])[:5]]
        return (f"Step: {step} | cells: {len(getattr(model, 'color_cells', []))} | "
                f"areas: {len(getattr(model, 'areas', []))} | First 5 voters of "
                f"{len(getattr(model, 'voting_agents', []))}: {first_agents}")


class AreaPersonalityDists(TextElement):
    def __init__(self):
        super().__init__()
        self.areas_pers_dist_plot = None

    def create_once(self, model):
        try:
            colors = COLORS[:getattr(model, 'num_colors', len(COLORS))]
            personalities = getattr(model, 'personalities', None)
            if personalities is None:
                self.areas_pers_dist_plot = ""
                return
            num_colors = len(personalities[0])
            num_personalities = personalities.shape[0]
            num_areas = len(getattr(model, 'areas', []))
        except Exception:
            self.areas_pers_dist_plot = ""
            return

        if num_areas == 0:
            self.areas_pers_dist_plot = ""
            return

        num_cols = math.ceil(math.sqrt(num_areas))
        num_rows = math.ceil(num_areas / num_cols)
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                                 figsize=(8, num_areas), sharex=True)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, area in zip(axes_flat, getattr(model, 'areas', [])):
            p_dist = getattr(area, 'personality_distribution', [])
            num_agents = getattr(area, 'num_agents', 0)
            heights = [int(val * num_agents) for val in p_dist] if p_dist else []
            bars = ax.bar(range(num_personalities), heights, color='skyblue')
            max_height = max(heights) if heights else 1
            p_top_hight = max_height * 0.02

            for bar, personality in zip(bars, personalities):
                height = bar.get_height()
                width = bar.get_width()
                for i, color_idx in enumerate(personality):
                    rect_width = width / num_colors
                    coords = (bar.get_x() + i * rect_width, height)
                    rect = patches.Rectangle(coords, rect_width, p_top_hight,
                                             color=colors[color_idx])
                    ax.add_patch(rect)

            ax.set_xlabel('"Personality" ID')
            ax.set_ylabel('Number of Agents')
            ax.set_title(f'Area {getattr(area, "unique_id", "?")}')

        plt.tight_layout()
        self.areas_pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model):
        if getattr(model.scheduler, 'steps', 0) == 0:
            self.create_once(model)
        return self.areas_pers_dist_plot or ""
