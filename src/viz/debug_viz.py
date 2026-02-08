from __future__ import annotations

from mesa.visualization import TextElement
from src.viz.factory import get_vis_cfg
import numpy as np
import html

# Visualization config (is set by make_canvas before these are instantiated)
vis_cfg = get_vis_cfg()
show_agent_debug_panel = bool(getattr(vis_cfg, 'show_agent_debug_panel', False))
agent_debug_area_id = getattr(vis_cfg, "agent_debug_area_id", None)
agent_debug_max_steps = int(getattr(vis_cfg, "agent_debug_max_steps", 1))
agent_debug_max_agents = int(getattr(vis_cfg, "agent_debug_max_agents", 50))
agent_debug_max_field_len = int(getattr(vis_cfg, "agent_debug_max_field_len", 180))


class AreaAgentDebugPanel(TextElement):
    """Step-wise per-agent debug panel for a single area.

    Renders a compact text overview of:
      - per-agent full state (including learning variables)
      - cast votes per agent
      - preference profile and aggregated ordering
      - reward components and deltas
    Intended for tiny configurations.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        area_id: int | None = None,
        max_agents: int | None = None,
        max_field_len: int | None = None,
    ):
        super().__init__()
        self.max_steps = max(1, int(max_steps if max_steps is not None else agent_debug_max_steps))
        self.area_id = area_id if area_id is not None else agent_debug_area_id
        self.max_agents = int(max_agents if max_agents is not None else agent_debug_max_agents)
        self.max_field_len = int(max_field_len if max_field_len is not None else agent_debug_max_field_len)
        self.max_list_items = 8
        self.agent_key_order = [
            "id",
            "pos",
            "assets",
            "assets_pre_est",
            "personality_group_idx",
            "personality_group",
            "personality",
            "est_real_dist",
            "confidence",
            "altruism_factor",
            "eligible",
            "participating",
            "q_participation",
            "p_participation",
            "fee",
            "num_elections_participated",
            "reward_common",
            "reward_personal",
            "delta_abs",
            "delta_rel",
            "known_cells_count",
            "known_cells",
            "participation_strategy",
            "voting_strategy",
            "award_history_tail",
        ]

    def _truncate(self, s: str) -> str:
        if len(s) <= self.max_field_len:
            return s
        return s[: max(0, self.max_field_len - 3)] + "..."

    def _fmt_value(self, value) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.6g}"
        if isinstance(value, (int, bool)):
            return str(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            items = []
            for v in list(value)[: self.max_list_items]:
                items.append(self._fmt_value(v))
            s = "[" + ", ".join(items)
            if len(value) > self.max_list_items:
                s += ", ..."
            s += "]"
            return self._truncate(s)
        if isinstance(value, dict):
            items = []
            for k in list(value.keys())[: self.max_list_items]:
                items.append(f"{k}={self._fmt_value(value[k])}")
            s = "{" + ", ".join(items)
            if len(value) > self.max_list_items:
                s += ", ..."
            s += "}"
            return self._truncate(s)
        return self._truncate(str(value))

    def _format_record(self, rec: dict) -> str:
        lines: list[str] = []
        lines.append(
            f"Agent Debug | area {rec.get('area_id')} | step {rec.get('step')}"
        )
        lines.append(
            "Summary: "
            f"agents={rec.get('num_agents')} "
            f"eligible={rec.get('num_eligible')} "
            f"participants={rec.get('num_participants')} "
            f"abstainers={rec.get('num_abstainers')}"
        )
        lines.append(
            "Outcome: "
            f"dist_to_reality={self._fmt_value(rec.get('dist_to_reality'))} "
            f"winning_option={rec.get('winning_option')} "
            f"aggregated_ordering={self._fmt_value(rec.get('aggregated_ordering'))}"
        )
        if rec.get("voted_ordering") is not None:
            lines.append(f"Voted ordering: {self._fmt_value(rec.get('voted_ordering'))}")
        if rec.get("real_color_distribution") is not None:
            lines.append(f"Actual color distribution: {self._fmt_value(rec.get('real_color_distribution'))}")
        lines.append(
            "Rewards: "
            f"rate_common={self._fmt_value(rec.get('reward_rate_common'))} "
            f"rate_personal={self._fmt_value(rec.get('reward_rate_personal'))} "
            f"thr_common={self._fmt_value(rec.get('reward_threshold_common'))} "
            f"thr_personal={self._fmt_value(rec.get('reward_threshold_personal'))} "
            f"abstention_share={self._fmt_value(rec.get('abstention_share'))}"
        )

        #pref = rec.get("preference_profile") or []
        #rows = len(pref)
        #cols = len(pref[0]) if rows > 0 else 0
        #lines.append(f"Preference profile shape: {rows} x {cols}")
        lines.append("Votes:")
        votes = rec.get("votes") or []
        if not votes:
            lines.append("  (no votes recorded)")
        else:
            for vote in votes:
                lines.append(
                    f"  Agent {vote.get('agent_id')}: "
                    f"scores={self._fmt_value(vote.get('scores'))} "
                    f"ordering (debug-tie-break!): "
                    f"{self._fmt_value(vote.get('ordering'))}"
                )

        lines.append("Agents:")
        agents = rec.get("agents") or []
        shown_agents = agents[: self.max_agents]
        for agent in shown_agents:
            agent_id = agent.get("id")
            lines.append(f"  Agent {agent_id}:")
            pair_of_entries = ""  # Save some lines by combining two fields
            for cnt, key in enumerate(self.agent_key_order):
                if key in agent:
                    pair_of_entries += f"{key}: {self._fmt_value(agent.get(key))}  "
                    if (cnt + 1) % 2 == 0:
                        lines.append(f"    {pair_of_entries}")
                        pair_of_entries = ""
            if len(pair_of_entries) > 0:  # In case of odd number of fields
                lines.append(f"    {pair_of_entries}")
        if len(agents) > self.max_agents:
            lines.append(f"  ... ({len(agents) - self.max_agents} more agents omitted)")

        return "\n".join(lines)

    def render(self, model) -> str:
        if not show_agent_debug_panel:
            return ""

        # Enable capture for subsequent steps.
        setattr(model, "_debug_agent_panel_enabled", True)
        setattr(model, "_debug_agent_panel_max_steps", int(self.max_steps))

        step = int(getattr(getattr(model, "scheduler", None), "steps", 0) or 0)
        if step == 0:
            return ""

        areas = [a for a in getattr(model, "areas", []) if a is not None and a.unique_id != -1]
        if not areas:
            return ""
        areas = sorted(areas, key=lambda a: int(getattr(a, "unique_id", 0)))

        area = None
        if self.area_id is None:
            area = areas[0]
        else:
            for a in areas:
                if int(getattr(a, "unique_id", -1)) == int(self.area_id):
                    area = a
                    break
        if area is None:
            return ""

        history = getattr(area, "debug_history", []) or []
        if not history:
            return "<pre>Agent debug capture not yet available (advance one step).</pre>"

        records = history[-self.max_steps :]
        text_blocks = [self._format_record(rec) for rec in records]
        text = ("\n\n" + ("-" * 60) + "\n\n").join(text_blocks)
        return f"<pre>{html.escape(text)}</pre>"

