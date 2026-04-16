# Altruism / Vote-Mode Adaptation

In the current model, `altruism_factor` does not represent a separate strategic planner. 
It controls how likely an agent is to cast a puzzle-aligned rather than a
self-regarding ballot.

## Meaning

`altruism_factor` lies in `[0,1]` and controls vote-mode selection:

- with probability `altruism_factor`: altruistic (puzzle-aligned)
- with probability `1 - altruism_factor`: self-regarding (preference-ordering)

## Implementation Reference

Vote-mode selection is exposed in [Strategies](api/Strategies.md). Agent-level
altruism updates are documented in [VoteAgent](api/VoteAgent.md), and the
surrounding election orchestration is documented in [Area](api/Area.md).

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
