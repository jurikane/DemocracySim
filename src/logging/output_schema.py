"""Locked output schema v2 (contract).

This module is the *single source of truth* for the on-disk output format
produced by headless batch runs.

Key rules:
- Step indexing semantics: recorded step t is post-election/reward for step t,
  and pre-mutation of step t (mutation is applied at the start of step t+1).
  Grid snapshots are the election-time state for step t.
- Step-based data is stored in Parquet tables (no per-step JSON files).
- Dense arrays (grids/overlays) remain in separate artifacts (e.g. in .npy).
- No `run_id`. Every table includes:
    - run_seed (int32): the concrete RNG seed used for this run
    - rule_idx (int16): index of the voting rule used for this run

Validators in this module are intentionally tolerant to safe upcasts
(e.g., int16 -> int32, float32 -> float64) because Pandas/Arrow may widen
integer/float dtypes during IO.

This file is documentation + validation only. It must not perform any logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Iterable

import numpy as np
import pandas as pd


SCHEMA_NAME: Final[str] = "output_schema_v2"
SCHEMA_VERSION: Final[int] = 2

# Indexing meaning for all step-based tables in this schema.
# (Election has run, rewards distributed; mutation of step t is applied at start of t+1.)
STEP_INDEXING: Final[str] = "post_election_pre_mutation"


@dataclass(frozen=True)
class TableSchema:
    """A Parquet table schema contract.

    `dtypes` should use pandas/numpy dtype strings (e.g. "int32", "float32",
    "boolean") and is used by validators.
    """

    name: str
    primary_key: tuple[str, ...]
    columns: tuple[str, ...]
    dtypes: Mapping[str, str]


def _expanded(prefix: str, n: int) -> tuple[str, ...]:
    """Return (prefix_0, ..., prefix_{n-1})."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return tuple(f"{prefix}_{i}" for i in range(n))


def _dtypes_for_cols(cols: Iterable[str], dtype: str) -> dict[str, str]:
    return {str(c): str(dtype) for c in cols}


# Note: num_colors is model-config dependent.
# We document and validate the *base columns* here. Expanded by writers/tests.
COLOR_EXPANSION_NOTE: Final[str] = (
    "Color vector columns are expanded in-file as *_0..*_{C-1} where C=num_colors"
)


# -----------------
# steps.parquet
# -----------------
STEPS_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "run_seed",
    "rule_idx",
    "step",
    "collective_assets",
    "gini_index",
    "turnout",
    "mean_altruism",
    "mean_dissatisfaction",
)

STEPS_BASE_DTYPES: Final[dict[str, str]] = {
    "run_seed": "int32",
    "rule_idx": "int16",
    "step": "int32",
    "collective_assets": "float32",
    "gini_index": "int16",
    "turnout": "float32",
    "mean_altruism": "float32",
    "mean_dissatisfaction": "float32",
    # Optional per-color model series: color_0...color_{C-1} float32
}

STEPS_TABLE: Final[TableSchema] = TableSchema(
    name="steps",
    primary_key=("run_seed", "rule_idx", "step"),
    columns=STEPS_BASE_COLUMNS,
    dtypes=STEPS_BASE_DTYPES,
)


# -----------------
# area_steps.parquet
# -----------------
AREA_STEPS_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "run_seed",
    "rule_idx",
    "step",
    "area_id",
    # Participation / costs
    "eligible_voters",
    "participants",
    "turnout",
    "fee_pool",
    # Outcome
    "winning_option_id",
    "grid_ordering_id",
    "puzzle_ordering_id",
    # Vectors (expanded):
    # - elected_color_0.. elected_color_{C-1} (int16)
    # - area_color_0.. area_color_{C-1} (float32)
    # - puzzle_color_0.. puzzle_color_{C-1} (float32, optional; present in puzzle-mode runs)
    # Metrics
    "dist_to_reality",
    "puzzle_distance",
    "gini_index",
)

AREA_STEPS_BASE_DTYPES: Final[dict[str, str]] = {
    "run_seed": "int32",
    "rule_idx": "int16",
    "step": "int32",
    "area_id": "int32",
    "eligible_voters": "int32",
    "participants": "int32",
    "turnout": "float32",
    # fee_pool must match simulation internal type; allow float.
    "fee_pool": "float32",
    "winning_option_id": "int32",
    "grid_ordering_id": "int32",
    "puzzle_ordering_id": "int32",
    "dist_to_reality": "float32",
    "puzzle_distance": "float32",
    "gini_index": "int16",
    # expanded vectors documented but validated dynamically
}

AREA_STEPS_TABLE: Final[TableSchema] = TableSchema(
    name="area_steps",
    primary_key=("run_seed", "rule_idx", "step", "area_id"),
    columns=AREA_STEPS_BASE_COLUMNS,
    dtypes=AREA_STEPS_BASE_DTYPES,
)


# -----------------
# agents.parquet
# -----------------
AGENTS_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "run_seed",
    "rule_idx",
    "step",
    "agent_id",
    "assets",
    "num_elections_participated",
    "personality_group_idx",
    "eligible_for_election",
    "participating",
    "election_fee",
    "reward_personal",
    "election_delta_abs",
    "election_delta_rel",
    "participation_baseline",
    "participation_signal",
    "participation_signal_group_component",
    "participation_signal_fee_component",
    "q_participation",
    "participation_probability",
    "altruism_factor",
    "dissatisfaction_value",
    "dissatisfaction_baseline",
    "dissatisfaction_signal",
)

AGENTS_BASE_DTYPES: Final[dict[str, str]] = {
    "run_seed": "int32",
    "rule_idx": "int16",
    "step": "int32",
    "agent_id": "int32",
    # assets must match simulation internal type; allow float.
    "assets": "float32",
    "num_elections_participated": "int32",
    "personality_group_idx": "int16",
    "eligible_for_election": "boolean",
    "participating": "boolean",
    "election_fee": "float32",
    "reward_personal": "float32",
    "election_delta_abs": "float32",
    "election_delta_rel": "float32",
    "participation_baseline": "float32",
    "participation_signal": "float32",
    "participation_signal_group_component": "float32",
    "participation_signal_fee_component": "float32",
    "q_participation": "float32",
    "participation_probability": "float32",
    "altruism_factor": "float32",
    "dissatisfaction_value": "float32",
    "dissatisfaction_baseline": "float32",
    "dissatisfaction_signal": "float32",
}

AGENTS_TABLE: Final[TableSchema] = TableSchema(
    name="agents",
    primary_key=("run_seed", "rule_idx", "step", "agent_id"),
    columns=AGENTS_BASE_COLUMNS,
    dtypes=AGENTS_BASE_DTYPES,
)


# -----------------
# votes.parquet
# -----------------
# One row per participating agent per area per step.
VOTES_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "run_seed",
    "rule_idx",
    "step",
    "area_id",
    "agent_id",
    "participating",
    "confidence",
    "voted_altruistically",
    # Vector (expanded): estim_dst_color_0..estim_dst_color_{C-1}
    "rank_1_option_id",
    "rank_1_oppose_score",
    "rank_2_option_id",
    "rank_2_oppose_score",
    "rank_3_option_id",
    "rank_3_oppose_score",
)

VOTES_BASE_DTYPES: Final[dict[str, str]] = {
    "run_seed": "int32",
    "rule_idx": "int16",
    "step": "int32",
    "area_id": "int32",
    "agent_id": "int32",
    "participating": "boolean",
    "confidence": "float32",
    "voted_altruistically": "boolean",
    # estim_dst_color_* float32 validated dynamically
    # Use nullable ints for option ids (so missing ranks can be NA).
    "rank_1_option_id": "Int32",
    "rank_1_oppose_score": "float32",
    "rank_2_option_id": "Int32",
    "rank_2_oppose_score": "float32",
    "rank_3_option_id": "Int32",
    "rank_3_oppose_score": "float32",
}

VOTES_TABLE: Final[TableSchema] = TableSchema(
    name="votes",
    primary_key=("run_seed", "rule_idx", "step", "area_id", "agent_id"),
    columns=VOTES_BASE_COLUMNS,
    dtypes=VOTES_BASE_DTYPES,
)


def all_tables() -> tuple[TableSchema, ...]:
    return STEPS_TABLE, AREA_STEPS_TABLE, AGENTS_TABLE, VOTES_TABLE


# -----------------
# Validation helpers
# -----------------

class SchemaValidationError(ValueError):
    pass


def _normalize_pd_dtype(dtype: str) -> str:
    """Map various pandas dtype strings to a canonical form."""
    d = str(dtype)
    # pandas may report 'Int64' for nullable integer; normalize to 'int64'
    # but keep a label so we can allow it under the hood.
    return d.lower()


def _is_integer_kind(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.integer)


def _is_float_kind(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def _is_bool_kind(dtype: np.dtype) -> bool:
    # pandas BooleanDtype has kind 'b'? For safety, check both bool and pandas boolean.
    return dtype == np.dtype(bool)


def _allow_safe_cast(actual: np.dtype, expected: np.dtype) -> bool:
    """Return True if `actual` is an acceptable dtype for `expected`.

    Policy:
    - exact match is ok
    - integer upcasts are ok (int16 -> int32 -> int64)
    - float upcasts are ok (float32 -> float64)
    - integer -> float is ok (writer may store ints but pandas reads float)
      NOTE: we allow this only if expected is float.
    - pandas nullable boolean is acceptable for expected boolean
    """
    if actual == expected:
        return True

    # expected integer family
    if _is_integer_kind(expected):
        if _is_integer_kind(actual):
            return np.can_cast(actual, expected, casting="safe") or np.can_cast(expected, actual, casting="safe")
        # do NOT allow float as replacement for expected int
        return False

    # expected float family
    if _is_float_kind(expected):
        if _is_float_kind(actual):
            return np.can_cast(actual, expected, casting="safe") or np.can_cast(expected, actual, casting="safe")
        if _is_integer_kind(actual):
            # int -> float is safe
            return True
        return False

    # expected boolean
    if expected == np.dtype(bool):
        # pandas BooleanDtype arrives as 'boolean' extension; treat as acceptable
        if actual == np.dtype(bool):
            return True
        return False

    # fallback strict
    return False


def _expected_np_dtype(dtype_str: str) -> np.dtype:
    d = _normalize_pd_dtype(dtype_str)
    if d in {"boolean", "bool"}:
        return np.dtype(bool)
    # 'int16', 'int32', 'int64', 'float32', 'float64'
    return np.dtype(d)


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{table_name}: missing required columns: {missing}"
        )


def _validate_dtypes(df: pd.DataFrame, expected_dtypes: Mapping[str, str], table_name: str) -> None:
    bad: dict[str, str] = {}
    for col, exp in expected_dtypes.items():
        if col not in df.columns:
            continue
        actual = df[col].dtype

        # Normalize pandas dtypes to numpy dtype where possible.
        # Pandas extension dtypes (Int32Dtype, BooleanDtype) don't always expose
        # a .numpy_dtype attribute across versions.
        actual_str = str(actual).lower()
        if actual_str in {"boolean", "bool"}:
            actual_np = np.dtype(bool)
        elif actual_str.startswith("int") or actual_str.startswith("uint") or actual_str.startswith("float"):
            # covers both numpy dtypes and pandas extension dtypes like "Int32"
            actual_np = np.dtype(actual_str)
        else:
            # object/category/string/etc.
            actual_np = np.dtype(actual)

        exp_np = _expected_np_dtype(exp)

        # Special-case boolean
        if _normalize_pd_dtype(exp) in {"boolean", "bool"}:
            if actual_str in {"boolean", "bool"} or actual_np == np.dtype(bool):
                continue
            bad[col] = f"expected {exp} got {actual}"
            continue

        if not _allow_safe_cast(actual_np, exp_np):
            bad[col] = f"expected {exp} got {actual}"

    if bad:
        msg = "; ".join(f"{k} ({v})" for k, v in bad.items())
        raise SchemaValidationError(f"{table_name}: dtype mismatches: {msg}")


def _validate_no_unknown_columns(
    df: pd.DataFrame,
    *,
    table_name: str,
    exact_allowed: set[str],
    allowed_prefixes: tuple[str, ...] = (),
) -> None:
    unknown: list[str] = []
    for c in df.columns:
        cs = str(c)
        if cs in exact_allowed:
            continue
        if any(cs.startswith(p) for p in allowed_prefixes):
            continue
        unknown.append(cs)
    if unknown:
        unknown = sorted(unknown)
        raise SchemaValidationError(f"{table_name}: unknown columns not allowed: {unknown}")


def _validate_expanded_prefix(
    df: pd.DataFrame,
    prefix: str,
    dtype: str,
    table_name: str,
) -> None:
    """Validate expanded vector columns like prefix_0...prefix_{C-1}.

    We require:
    - at least one column exists (prefix_0)
    - all columns with that prefix are consecutive indices starting at 0
    - all present are of an acceptable dtype
    """
    prefix_with_sep = prefix + "_"
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix_with_sep)]
    if not cols:
        raise SchemaValidationError(f"{table_name}: missing expanded vector columns for '{prefix}_0..' ")

    # parse suffixes
    idxs: list[int] = []
    for c in cols:
        suffix = c[len(prefix_with_sep):]
        try:
            idxs.append(int(suffix))
        except ValueError:
            raise SchemaValidationError(f"{table_name}: vector column has non-integer suffix: {c}")

    idxs_sorted = sorted(idxs)
    if idxs_sorted[0] != 0:
        raise SchemaValidationError(f"{table_name}: vector columns for {prefix} must start at 0")
    # must be contiguous
    for a, b in zip(idxs_sorted, idxs_sorted[1:]):
        if b != a + 1:
            raise SchemaValidationError(
                f"{table_name}: vector columns for {prefix} must be contiguous (found {idxs_sorted})"
            )

    expected_map = _dtypes_for_cols((f"{prefix}_{i}" for i in idxs_sorted), dtype)
    _validate_dtypes(df, expected_map, table_name)


# ---- Public validators ----

def validate_steps_df(df: pd.DataFrame) -> None:
    """Validate a DataFrame read from steps.parquet."""
    table = STEPS_TABLE
    _validate_required_columns(df, table.columns, table.name)
    _validate_no_unknown_columns(
        df,
        table_name=table.name,
        exact_allowed=set(table.columns),
        allowed_prefixes=("color_",),
    )
    _validate_dtypes(df, table.dtypes, table.name)
    # optional: allow color_0... columns if present; validate if they exist
    color_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("color_")]
    if color_cols:
        _validate_expanded_prefix(df, prefix="color", dtype="float32", table_name=table.name)


def validate_area_steps_df(df: pd.DataFrame) -> None:
    """Validate a DataFrame read from area_steps.parquet."""
    table = AREA_STEPS_TABLE
    _validate_required_columns(df, table.columns, table.name)
    _validate_no_unknown_columns(
        df,
        table_name=table.name,
        exact_allowed=set(table.columns),
        allowed_prefixes=("elected_color_", "area_color_", "puzzle_color_"),
    )
    _validate_dtypes(df, table.dtypes, table.name)
    _validate_expanded_prefix(df, prefix="elected_color", dtype="int16", table_name=table.name)
    _validate_expanded_prefix(df, prefix="area_color", dtype="float32", table_name=table.name)
    puzzle_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("puzzle_color_")]
    if puzzle_cols:
        _validate_expanded_prefix(df, prefix="puzzle_color", dtype="float32", table_name=table.name)


def validate_agents_df(df: pd.DataFrame) -> None:
    """Validate a DataFrame read from agents.parquet."""
    table = AGENTS_TABLE
    _validate_required_columns(df, table.columns, table.name)
    _validate_no_unknown_columns(
        df,
        table_name=table.name,
        exact_allowed=set(table.columns),
    )
    _validate_dtypes(df, table.dtypes, table.name)


def validate_votes_df(df: pd.DataFrame) -> None:
    """Validate a DataFrame read from votes.parquet."""
    table = VOTES_TABLE
    _validate_required_columns(df, table.columns, table.name)
    _validate_no_unknown_columns(
        df,
        table_name=table.name,
        exact_allowed=set(table.columns),
        allowed_prefixes=("estim_dst_color_",),
    )
    _validate_dtypes(df, table.dtypes, table.name)

    # Validate estim_dst_color_* expansion (must exist for vote context)
    _validate_expanded_prefix(df, prefix="estim_dst_color", dtype="float32", table_name=table.name)

    # Basic sanity: participating should be True for all rows (participants-only table).
    if "participating" in df.columns:
        try:
            if (~df["participating"].fillna(False)).any():
                raise SchemaValidationError(f"{table.name}: participating must be True for all rows")
        except (TypeError, ValueError):
            # dtype validator should catch wild types; keep runtime robust
            pass

    if "voted_altruistically" in df.columns and len(df) > 0:
        mode = df["voted_altruistically"]
        invalid = ~mode.isin([True, False]) & mode.notna()
        if bool(invalid.any()):
            bad = sorted(set(mode[invalid].tolist()))
            raise SchemaValidationError(
                f"{table.name}: voted_altruistically contains invalid values: {bad} (allowed: [True, False, <NA>])"
            )
