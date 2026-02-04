from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np

from src.utils.ballots import mix_distributions, ordering_from_distribution, score_options_c2


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
        p = float(getattr(agent, "participation_probability")())
        bias = float(getattr(agent.model, "bias_toward_participation", 0.0))
        if bias != 0.0:
            # Simple additive bias in probability space.
            p = float(np.clip(p + bias, 0.0, 1.0))
        # determinism: use model-level NumPy RNG
        return bool(float(agent.model.np_random.random()) < p)


@dataclass(frozen=True)
class DefaultVotingStrategy:
    """C2 ballot scoring: mix distributions -> ordering -> ordering distance to options."""

    def score_options(self, agent: Any, area: Any, options: np.ndarray) -> np.ndarray:
        # Assumes Area._tally_votes already populated agent.known_cells for this election.
        est_real_dist, _conf = agent.estimate_real_distribution(area)

        altruism_factor = float(getattr(agent, "altruism_factor", 0.5))
        # - 0.0 => purely self-interest (personal_opt_dist)
        # - 1.0 => purely reality-tracking (est_real_dist)

        target_dist = mix_distributions(
            altruism_factor=altruism_factor,
            est_real_dist=np.asarray(est_real_dist, dtype=np.float32),
            personal_opt_dist=np.asarray(agent.personal_opt_dist, dtype=np.float32),
        )

        target_ordering = ordering_from_distribution(target_dist)

        dist_func = agent.model.distance_func
        search_pairs = agent.model.color_search_pairs

        return score_options_c2(
            target_ordering=target_ordering,
            options=np.asarray(options),
            distance_func=dist_func,
            color_search_pairs=search_pairs,
        )
