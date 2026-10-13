# Game theory and Monte Carlo attrition engine (v1.0)

A vectorized and parallelized Monte Carlo combat simulator and game theory tournament engine all built in Python, which models stochastic attrition warfare with unique and custom archetypes, running round robin tournaments and analyzing the results using heat maps.

---

## Overview

Vectorized Simulation: Built using NumPy to run up to 100,000 parallel Monte Carlo combat trials simultaneously without loops.
Game theory analysis: Evaluates round-robin tournaments across different archetypes (including custom ones), generating pairwise win rate matrices.
UI + Analytics: Shown using Streamlit and Plotly for parameter tweaking and visualization

---

## Structure

The project is structured into three main files:

1. simulation.py: NumPy vectorized engine for attrition trials.
2. tournament.py: Runs the tournament + builds the matrix.
3. app.py: Streamlit dashboard showing interactive heatmaps, win-rate distributions, and strategy parameters. The actual frontend.

---

FINALLY my v1.0 release
Just visit https://monte-carlo-attrition-simulator.streamlit.app/

---

## What's next?

- Balancing. This is definitely my 1st priority. These archetypes are heavily unbalanced although I am not able to fix them after pouring hours into this what should be a simple fix. Balancing update will come though!
-   I will add a counter to Greedy and Swarm and adding something that Sniper counters if possible. Those are the most unbalanced archetypes
- Adding more options so users can customize their own archetypes with different healths possibly and sliders
- Evolution? Far out possibility.
