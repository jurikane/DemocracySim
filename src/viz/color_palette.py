from __future__ import annotations

# Central color palette (index-aligned with simulation color IDs)
COLORS = [
    "LightGray",   # 0
    "Red",         # 1
    "Green",       # 2
    "Blue",        # 3
    "Orange",      # 4
    "Maroon",      # 5
    "Lime",        # 6
    "Yellow",      # 7
]

# Group palette is intentionally distinct from simulation grid colors.
# It is reused across analysis/summary views to keep group encoding stable.
GROUP_COLORS = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#e7298a",  # magenta
    "#66a61e",  # green
    "#e6ab02",  # mustard
    "#a6761d",  # brown
    "#666666",  # dark gray
]


def get_group_color(group_idx: int) -> str:
    if group_idx < 0:
        return "#666666"
    return GROUP_COLORS[group_idx % len(GROUP_COLORS)]
