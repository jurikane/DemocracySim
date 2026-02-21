from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np

from src.utils.ballots import ordering_from_distribution, score_options_c2


class ParticipationStrategy(Protocol):
    def decide_participation(self, agent: Any, area: Any) -> bool: ...


class VotingStrategy(Protocol):
    """Build a ballot as raw oppose-scores over options.

    Contract:
    - Returns np.ndarray shape (num_options,)
    - Values are in [0,1]
    - Lower = better
    - No normalization (do not force sum==1)

    IMPORTANT: must not mutate/reshuffle agent knowledge sampling.
    `agent.known_cells` is assumed to have been populated by Area._tally_votes().
    """

    def score_options(self, agent: Any, area: Any, options: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DefaultParticipationStrategy:
    """Adaptive probabilistic participation using agent.q_participation."""

    def decide_participation(self, agent: Any, area: Any) -> bool:
        p = agent.participation_probability()
        bias = float(agent.model.bias_toward_participation)
        if bias != 0.0:
            # Simple additive bias in probability space.
            p = float(np.clip(p + bias, 0.0, 1.0))
        # determinism + stream isolation: participation has its own RNG stream.
        return bool(agent.model.participation_rng.random() < p)


@dataclass(frozen=True)
class DefaultVotingStrategy:
    """Two-mode ballot scoring with altruism_factor as mode probability.

    - self-regarding mode: return agent-held precomputed `self_regarding_oppose_scores`
    - altruistic mode: estimate reality -> ordering -> option oppose-scores
    """

    def score_options(self, agent: Any, area: Any, options: np.ndarray) -> np.ndarray:
        altruism_factor = float(np.clip(float(agent.altruism_factor), 0.0, 1.0))
        # Dedicated voting stream (already isolated from participation RNG).
        if float(agent.model.voting_rng.random()) < altruism_factor:
            # Assumes Area._tally_votes already populated agent.known_cells for this election.
            est_real_dist, _conf = agent.estimate_real_distribution(area)
            target_ordering = ordering_from_distribution(
                np.asarray(est_real_dist, dtype=np.float32),
                rng=agent.model.voting_rng,
            )
            agent.voted_altruistically = True
            return score_options_c2(
                target_ordering=target_ordering,
                options=np.asarray(options),
                distance_func=agent.model.distance_func,
                color_search_pairs=agent.model.color_search_pairs,
            )

        agent.voted_altruistically = False
        return np.asarray(agent.self_regarding_oppose_scores, dtype=np.float32)
