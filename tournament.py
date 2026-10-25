"""Round robin tournament engine for evaluation of archetypes"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simulation import (
    STRATEGY_PRESETS,
    resolve_archetype_matchup,
    run_monte_carlo_game,
)


def run_round_robin_tournament(
    num_simulations_per_matchup: int = 50_000,
    counter_modifier: float = 0.20,
) -> pd.DataFrame:
    """Execute the simulations

    Args:
        num_simulations_per_matchup: Number of Monte Carlo simulations per matchup.
        counter_modifier: Multiplier strength for hard counter relationships (e.g. 0.20 = 20%).
    """
    # Get list of strategy names
    strategies = list(STRATEGY_PRESETS.keys())
    n = len(strategies)
    
    # Initialize empty N x N win-rate matrix
    win_rate_matrix = np.zeros((n, n), dtype=float)
    
    # Execute vectorized pairwise matchups
    for i, attacker_strategy in enumerate(strategies):
        for j, defender_strategy in enumerate(strategies):
            if i == j:
                # Self-matchups are 50/50 by definition
                win_rate_matrix[i, j] = 50.0
                continue
                
            # Resolve archetype matchup to get simulation parameters
            params = resolve_archetype_matchup(
                attacker_strategy,
                defender_strategy,
                counter_modifier=counter_modifier,
            )
            
            # Run Monte Carlo simulation
            result = run_monte_carlo_game(
                num_simulations=num_simulations_per_matchup,
                attacker_units=params["attacker_units"],
                defender_units=params["defender_units"],
                attacker_win_prob=params["attacker_win_prob"],
            )
            
            # Store attacker win rate percentage
            win_rate_matrix[i, j] = result["Attacker Win Rate (%)"]
    
    # Create labeled DataFrame
    df = pd.DataFrame(
        win_rate_matrix,
        index=strategies,
        columns=strategies,
    )
    
    return df


def create_tournament_heatmap(
    win_rate_df: pd.DataFrame,
    title: str = "Round-Robin Tournament Win Rates (%)",
) -> go.Figure:
    """Generate heatmap for tournament results"""
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=win_rate_df.values,
            x=win_rate_df.columns,
            y=win_rate_df.index,
            colorscale="RdBu",
            zmid=50,
            text=np.round(win_rate_df.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Defender Archetype",
        yaxis_title="Attacker Archetype",
        xaxis_tickangle=-45,
        width=800,
        height=600,
        template="plotly_dark",
    )
    
    return fig


if __name__ == "__main__":
    # Run tournament and display results when executed
    tournament_df = run_round_robin_tournament()
    print("Round-Robin Tournament Results:")
    print(tournament_df.round(1))
    
    # Create and show heatmap
    fig = create_tournament_heatmap(tournament_df)
    fig.show()