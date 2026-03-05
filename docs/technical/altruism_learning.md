# Altruism / Vote-Mode Adaptation

This page documents how `altruism_factor` is set and updated.

## Meaning

`altruism_factor` in `[0,1]` controls vote mode selection:

- with probability `altruism_factor`: altruistic (reality-tracking)
- with probability `1 - altruism_factor`: self-regarding (personality-ordering)

## Runtime Locations

- Vote scoring: `src/agents/strategies.py::DefaultVotingStrategy.score_options`
- Satisfaction-mode mapping: `src/agents/vote_agent.py::VoteAgent.apply_altruism_satisfaction_mode`
- Surprise-learning update: `src/agents/vote_agent.py::VoteAgent.apply_altruism_update`
- Orchestration: `src/agents/area.py::Area.step`, `src/agents/area.py::Area.conduct_election`

## Modes

- `altruism_mode="static"`: fixed `altruism_factor = altruism_static`
- `altruism_mode="satisfaction"` (default): pre-election mapping from dissatisfaction
- `altruism_mode="surprise_learning"`: participant-only post-election update from dissatisfaction signal

## Satisfaction Mode

Given normalized dissatisfaction `d in [0,1]`:

- `s = 1 - d`
- `target = sigmoid(altruism_satisfaction_slope * (s - altruism_satisfaction_theta))`

Response behavior via `altruism_response_gamma`:

- `gamma = 1`: direct mapping to `target`
- `gamma < 1`: smoothed update toward `target`

Final value is clipped to `[altruism_clip_min, altruism_clip_max]`.

## Surprise-Learning Mode

Participant-only update:

- `a <- a - altruism_alpha * dissatisfaction_signal`
- clip to `[altruism_clip_min, altruism_clip_max]`

## Key Knobs

- `altruism_mode`
- `altruism_static`
- `altruism_init`
- `altruism_satisfaction_theta`
- `altruism_satisfaction_slope`
- `altruism_response_gamma`
- `altruism_alpha`
- `altruism_clip_min`
- `altruism_clip_max`
