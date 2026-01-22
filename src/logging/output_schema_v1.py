"""Stable output schema v1 for run step logs.

This module defines the canonical *table schemas* we write per run.

Goals:
- Efficient analysis for a thesis (filter by area_id and step).
- Still replayable later (via loading these tables + optional grids).
- Avoid per-step JSON files.

Schema v1 is intentionally minimal but extensible. If you change column names
or semantics, bump the schema version.

Tables (per run directory):

1) steps.parquet
   One row per simulation step (global model-level scalars).

2) areas.parquet
   One row per (step, area_id).

3) votes.parquet
   One row per (step, area_id, agent_id, candidate_id) capturing an agent's
   ballot / preference signal for that election.

Optional (not implemented in this step):
- grids/ grid snapshots (can stay as .npy for now).

Notes on election logging:
- We assume each election produces a ranked ordering of candidates per agent.
  If the model uses a different representation (approval scores, etc.), the
  writer should map it into this long format with columns like `rank` and/or
  `score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable


SCHEMA_VERSION: Final[int] = 1
SCHEMA_NAME: Final[str] = "output_schema_v1"


@dataclass(frozen=True)
class TableSchema:
    name: str
    # Ordered column names; types are documented but enforced by writer/tests.
    columns: tuple[str, ...]


STEPS_TABLE: Final[TableSchema] = TableSchema(
    name="steps",
    columns=(
        "run_id",
        "step",
        # Common global metrics (nullable)
        "collective_assets",
        "gini_index",
        "turnout",
    ),
)

AREAS_TABLE: Final[TableSchema] = TableSchema(
    name="areas",
    columns=(
        "run_id",
        "step",
        "area_id",
        # Area metrics
        "voter_turnout",
        "dist_to_reality",
        "gini_index",
        # Election
        "election_winner",  # candidate_id or color index (nullable)
        "election_results_json",  # JSON string for full result object
        # Color distribution (often vector-ish)
        "color_distribution_json",  # JSON string for list/dict
    ),
)

VOTES_TABLE: Final[TableSchema] = TableSchema(
    name="votes",
    columns=(
        "run_id",
        "step",
        "area_id",
        "agent_id",
        "candidate_id",
        # Preference signal
        "rank",  # 1..K if ranked (nullable)
        "score",  # float if scored/approval (nullable)
    ),
)


def all_tables() -> Iterable[TableSchema]:
    return (STEPS_TABLE, AREAS_TABLE, VOTES_TABLE)
