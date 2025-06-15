[![Pages](https://github.com/jurikane/DemocracySim/actions/workflows/ci.yml/badge.svg)](https://jurikane.github.io/DemocracySim/)
[![pytest main](https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/jurikane/DemocracySim/branch/main/graph/badge.svg)](https://codecov.io/gh/jurikane/DemocracySim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![pytest dev]&#40;https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml/badge.svg?branch=dev&#41;]&#40;https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml&#41;)

# DemocracySim: Multi-Agent Simulations to research democratic voting and participation

Codebase of a simulation for the master thesis 
"Influence of different voting rules on participation and welfare in a simulated multi-agent democracy" 
at the group [Swarm Intelligence and Complex Systems](https://siks.informatik.uni-leipzig.de) 
at the [Faculty of Mathematics and Computer Science](https://www.mathcs.uni-leipzig.de/en)
of [Leipzig University](https://www.uni-leipzig.de/en).
This project is kindly supported by [OpenPetition](https://osd.foundation).

### Documentation

For details see the [documentation](https://jurikane.github.io/DemocracySim/) on GitHub-pages.

## Overview

**DemocracySim** is a multi-agent simulation framework designed to study democratic participation 
and group decision-making in a dynamic, evolving environment. 
Agents interact within a grid-based world, form beliefs about their surroundings, 
and vote in elections that influence both their individual outcomes and the state of the system.

The environment consists of a toroidal grid of colored fields, where neighboring cells form territories. 
Each territory holds regular elections in which agents vote on the observed color distribution. 
The results of these elections not only influence how agents are rewarded 
but also shape the environment itself through controlled mutation processes.

Agents have limited resources and face decisions about whether to participate in elections, or remain inactive. 
Each agent belongs to a personality type defined by preferences over the possible field colors, 
with types distributed to create majority and minority dynamics. 
During elections, agents face a strategic trade-off between voting for what benefits them personally 
and voting for what they believe to be the most accurate representation of their territory—decisions 
that impact both immediate rewards and the system’s future state.

The simulation tracks a range of metrics including participation rates, collective accuracy, 
reward inequality (Gini index), and behavioral indicators such as altruism and diversity of expressed opinions. 
**DemocracySim** also allows for the evaluation of group performance under different normative goals—utilitarian, 
egalitarian, or Rawlsian—by comparing actual outcomes to theoretically optimal decisions.

By modeling participation dilemmas, reward mechanisms, and personality-driven behavior, 
**DemocracySim** provides a controlled environment for investigating how democratic systems 
respond to different institutional rules and individual incentives. 
It is intended both as a research tool and as a foundation for future explorations into deliberation, representation, 
and fairness in collective choice.
