# Game theory and Monte Carlo attrition engine (v1.0)

A vectorized and parallelized Monte Carlo combat simulator and game theory tournament engine built in Python, which models stochastic attrition warfare with unique and custom archetypes, running round robin tournaments and analyzing the results using heat maps.

---

## Overview

- **Vectorized Simulation**: Built using NumPy to run up to 100,000 parallel Monte Carlo combat trials simultaneously without loops.
- **Game Theory Analysis**: Evaluates round-robin tournaments across different archetypes (including custom ones), generating pairwise win rate matrices.
- **UI + Analytics**: Shown using Streamlit and Plotly for parameter tweaking and visualization.

---

## Structure

The project is structured into three main files:

1. **simulation.py**: NumPy vectorized engine for attrition trials.
2. **tournament.py**: Runs the tournament + builds the matrix.
3. **app.py**: Streamlit dashboard showing interactive heatmaps, win-rate distributions, and strategy parameters. The actual frontend.

---
Just visit [monte-carlo-attrition-simulator.streamlit.app](https://monte-carlo-attrition-simulator.streamlit.app/)
---

## What exactly went wrong in v1.1

After building v1.0 and introducing the hard-counter slider in v1.1, I realized why balancing felt impossible:
- **The Equalization Problem**: When archetypes are balanced so their total mass ($\text{Units} \times \text{Quality}$) is roughly equal ($\approx 6.0$), every neutral battle naturally converges to a flat 50% win rate.
- **The Counter Limitation**: Hard counters and sliders add variance, but they do not solve the baseline equalization issue without breaking the underlying principle of all the stratgies being equal.

---

## What's Next (v2.0)

To fix this, I am planning a complete redesign of the engine rather than just tweaking stats:
- Moving to non linear attrition and health/armor mechanics instead of raw scalar quality ratios.
- Adding dynamic unit deployment or reinforcement loops over time.
- Complete app and simulation overhaul to support genuine strategic dominance.
- Yup this is going to take a WHILE. Enjoy v1.1 while you can I guess!
For later: 

- Adding more customization options so users can build custom archetypes with different health pools and live sliders.
- Evolution? Far out possibility. 
