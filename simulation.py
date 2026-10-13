"""Monte Carlo simulation engine for combat model"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# Combat archetypes (starting units + probabilities)
STRATEGY_PRESETS: dict[str, dict[str, float]] = {
    "Balanced (Standard)": {
        "units": 10,
        "attacker_win_prob": 0.55,
    },
    "Blitzkrieg (Aggressive Strike)": {
        "units": 8,
        "attacker_win_prob": 0.65,
    },
    "Tortoise (Defensive Wall)": {
        "units": 14,
        "attacker_win_prob": 0.45,
    },
    "Swarm (Zerg Rush)": {
        "units": 20,
        "attacker_win_prob": 0.38,
    },
    "Greedy (High Roller)": {
        "units": 25,
        "attacker_win_prob": 0.36,
    },
    "Sniper (Elite Force)": {
        "units": 5,
        "attacker_win_prob": 0.82,
    },
}

DEFAULT_STRATEGY = "Balanced (Standard)"

def strategy_short_name(strategy_name: str) -> str:
    return strategy_name.split(" (")[0]


def clash_win_probability(attacker_quality: float, defender_quality: float) -> float:
    """Calculates normalized win probability"""
    att_q = float(attacker_quality)
    def_q = float(defender_quality)
    if att_q < 0 or def_q < 0:
        raise ValueError("Archetype qualities cannot be negative.")
    if att_q == 0 and def_q == 0:
        return 0.5
    return att_q / (att_q + def_q)


def resolve_archetype_matchup(
    attacker_strategy: str,
    defender_strategy: str,
    presets: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    """Map strategy names to parameters"""
    table = presets if presets is not None else STRATEGY_PRESETS
    if attacker_strategy not in table:
        raise KeyError(f"Unknown attacker strategy: {attacker_strategy!r}")
    if defender_strategy not in table:
        raise KeyError(f"Unknown defender strategy: {defender_strategy!r}")

    attacker = table[attacker_strategy]
    defender = table[defender_strategy]
    att_units = int(attacker["units"])
    def_units = int(defender["units"])
    att_q = float(attacker["attacker_win_prob"])
    def_q = float(defender["attacker_win_prob"])
    clash_p = clash_win_probability(att_q, def_q)

    return {
        "attacker_strategy": attacker_strategy,
        "defender_strategy": defender_strategy,
        "attacker_short_name": strategy_short_name(attacker_strategy),
        "defender_short_name": strategy_short_name(defender_strategy),
        "attacker_units": att_units,
        "defender_units": def_units,
        "attacker_archetype_prob": att_q,
        "defender_archetype_prob": def_q,
        "attacker_win_prob": clash_p,
    }


def run_monte_carlo_game(
    num_simulations: int = 100_000,
    attacker_units: int = 10,
    defender_units: int = 10,
    attacker_win_prob: float = 0.5,
) -> dict:
    """Run vectorized Monte Carlo simulations through NumPy"""
    # Validate inputs
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if attacker_units < 0 or defender_units < 0:
        raise ValueError("Unit counts cannot be negative.")
    if not (0.0 <= attacker_win_prob <= 1.0):
        raise ValueError("attacker_win_prob must be between 0 and 1.")

    n = int(num_simulations)
    a0 = int(attacker_units)
    d0 = int(defender_units)
    p = float(attacker_win_prob)

    # Handle edge cases (0 units)
    if a0 == 0 or d0 == 0:
        remaining_attacker = np.full(n, a0, dtype=np.int64)
        remaining_defender = np.full(n, d0, dtype=np.int64)
        return _summarize(remaining_attacker, remaining_defender)

    # Calculate maximum clashes before elimination
    max_clashes = a0 + d0 - 1

    # Draw each trial in parallel
    attacker_won_clash = np.random.random((n, max_clashes)) < p

    # Compute losses of units
    attacker_losses = np.cumsum(~attacker_won_clash, axis=1)
    defender_losses = np.cumsum(attacker_won_clash, axis=1)

    # Stop when a side has 0 units
    attacker_eliminated = attacker_losses >= a0
    defender_eliminated = defender_losses >= d0
    battle_over = attacker_eliminated | defender_eliminated
    end_clash_index = np.argmax(battle_over, axis=1)

    # Find remaining units when battle ends
    row_ids = np.arange(n)
    remaining_attacker = a0 - attacker_losses[row_ids, end_clash_index]
    remaining_defender = d0 - defender_losses[row_ids, end_clash_index]

    # Make sure no negative units
    remaining_attacker = np.maximum(remaining_attacker, 0)
    remaining_defender = np.maximum(remaining_defender, 0)

    return _summarize(remaining_attacker, remaining_defender)


def _summarize(
    remaining_attacker: np.ndarray,
    remaining_defender: np.ndarray,
) -> dict:
    """Summarize win rates and metrics from simulation"""
    n = remaining_attacker.size

    attacker_won = (remaining_attacker > 0) & (remaining_defender == 0)
    defender_won = (remaining_defender > 0) & (remaining_attacker == 0)

    # Means of boolean arrays are the fraction True (True counts as 1).
    # Multiply by 100 to report a percentage for the UI.
    attacker_win_rate_pct = float(attacker_won.mean() * 100.0)
    defender_win_rate_pct = float(defender_won.mean() * 100.0)

    return {
        "Attacker Win Rate (%)": attacker_win_rate_pct,
        "Defender Win Rate (%)": defender_win_rate_pct,
        "Average remaining Attacker units": float(remaining_attacker.mean()),
        "Average remaining Defender units": float(remaining_defender.mean()),
        # Raw arrays for distribution plots (histograms, box plots, etc.).
        "remaining_attacker_units": remaining_attacker,
        "remaining_defender_units": remaining_defender,
        "num_simulations": n,
    }


if __name__ == "__main__":
    # Local execution
    matchup = resolve_archetype_matchup(
        "Blitzkrieg (Aggressive Strike)",
        "Tortoise (Defensive Wall)",
    )
    print(
        f"Running 50,000 Monte Carlo battles: "
        f"{matchup['attacker_short_name']} vs. {matchup['defender_short_name']} "
        f"({matchup['attacker_units']} vs {matchup['defender_units']}, "
        f"clash p={matchup['attacker_win_prob']:.3f})..."
    )
    result = run_monte_carlo_game(
        num_simulations=50_000,
        attacker_units=matchup["attacker_units"],
        defender_units=matchup["defender_units"],
        attacker_win_prob=matchup["attacker_win_prob"],
    )

    printable = {
        key: (f"array(shape={value.shape}, dtype={value.dtype})"
              if isinstance(value, np.ndarray)
              else value)
        for key, value in result.items()
    }
    for key, value in printable.items():
        print(f"  {key}: {value}")
