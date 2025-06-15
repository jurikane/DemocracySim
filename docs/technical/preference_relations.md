# How preference relations are defined and represented in the system

## Introduction

...

## Definition

A preference relation $\tau\in\mathbb{R}_{\geq 0}^m$ is a numpy vector of length $m$, 
where $m$ is the number of options and each element $\tau[i]$ represents the normalized preference for option $i$,
with $\sum_{\tau}=1$.

### Why using sum normalization?

In computational social choice, **sum normalization** is more common than magnitude normalization. 
This is because sum normalization aligns well with the interpretation of preference vectors as distributions 
or weighted votes, which are prevalent in social choice scenarios.

### Why using non-negative values?

The preference values $\tau[i]$ are non-negative because they represent the strength of preference for each option.
Equvalently, they can be interpreted as the probability of selecting each option 
or the (inverted or negative) distance of an option to the agents' ideal solution.