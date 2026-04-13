from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np


def series_at(idx: int, seqs: list) -> list[float]:
    return [s[idx] if s is not None and len(s) > idx else float("nan") for s in seqs]


def save_plot_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}" />'


def float_array(values) -> np.ndarray:
    arr = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        arr[i] = np.nan if value is None else value
    return arr


def vector_matrix(values, width: int) -> np.ndarray:
    matrix = np.full((len(values), width), np.nan, dtype=np.float64)
    for row_idx, value in enumerate(values):
        if value is None:
            continue
        row = np.asarray(value, dtype=np.float64)
        matrix[row_idx, : min(width, row.size)] = row[:width]
    return matrix


def ordering_rank_matrix(values, width: int) -> np.ndarray:
    ranks = np.full((len(values), width), -1, dtype=np.int16)
    for row_idx, value in enumerate(values):
        if value is None:
            continue
        ordering = np.asarray(value, dtype=np.int16)
        limit = min(width, ordering.size)
        ranks[row_idx, ordering[:limit]] = np.arange(limit, dtype=np.int16)
    return ranks


def wrap_html_panel(
    content: str,
    *,
    title: str,
    collapsible: bool = True,
    default_open: bool = False,
) -> str:
    if not content:
        return ""
    if not collapsible:
        return content
    open_attr = " open" if default_open else ""
    return (
        f"<details{open_attr} style='margin:8px 0;'>"
        f"<summary style='cursor:pointer; font-weight:600;'>{title}</summary>"
        f"<div style='padding-top:8px'>{content}</div>"
        f"</details>"
    )


def step_axis(model, model_vars: dict[str, list[object]], num_steps: int) -> np.ndarray:
    if "step" in model_vars:
        return float_array(model_vars["step"])

    starts_at_zero = num_steps == model.scheduler.steps + 1
    start = 0.0 if starts_at_zero else 1.0
    return np.arange(start, start + num_steps, dtype=float)


def auto_ylim(
    values: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    min_span: float = 8.0,
    pad_fraction: float = 0.1,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return lower_bound, upper_bound

    lower = np.min(finite)
    upper = np.max(finite)
    span = upper - lower
    pad = max(min_span * 0.25, span * pad_fraction)
    lower -= pad
    upper += pad

    if upper - lower < min_span:
        center = 0.5 * (lower + upper)
        half_span = 0.5 * min_span
        lower = center - half_span
        upper = center + half_span

    lower = max(lower_bound, lower)
    upper = min(upper_bound, upper)

    if upper - lower < min_span:
        if lower <= lower_bound:
            return lower_bound, min(upper_bound, lower_bound + min_span)
        if upper >= upper_bound:
            return max(lower_bound, upper_bound - min_span), upper_bound

    return lower, upper


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")
