"""Streamlit frontend for Monte Carlo simulator"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    DEFAULT_STRATEGY,
    STRATEGY_PRESETS,
    resolve_archetype_matchup,
    run_monte_carlo_game,
    strategy_short_name,
)

from tournament import (
    run_round_robin_tournament,
    create_tournament_heatmap,
)

st.set_page_config(
    page_title="Strategic Game Theory Simulator",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e8eaed;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e8eaed !important;
    }
    h1, h2, h3, p, label {
        color: #e8eaed;
    }
    .hero-sub {
        color: #8b949e;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
    }
    .stButton > button {
        width: 100%;
        background-color: #e85d4c;
        color: #ffffff;
        border: none;
        font-weight: 600;
        padding: 0.6rem 0;
    }
    .stButton > button:hover {
        background-color: #ff7a66;
        color: #ffffff;
        border: none;
    }
    .empty-hint {
        color: #8b949e;
        border: 1px dashed #30363d;
        border-radius: 12px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        background: #161b22;
    }
    .matchup-header {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin: 0.2rem 0 0.35rem 0;
        color: #f0f3f6;
    }
    .matchup-vs {
        color: #e85d4c;
        padding: 0 0.35rem;
    }
    .archetype-chip {
        color: #8b949e;
        font-size: 0.92rem;
        margin-bottom: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None

# Sidebar configuration
STRATEGY_NAMES = list(STRATEGY_PRESETS.keys())
DEFAULT_INDEX = STRATEGY_NAMES.index(DEFAULT_STRATEGY)

with st.sidebar:
    st.header("Combat archetypes")
    st.caption(
        "Pick two archetypes. Each archetype has its own force size and "
        "clash quality. Click Run Simulation to make them fight."
    )

    attacker_strategy = st.selectbox(
        "Attacker Strategy",
        options=STRATEGY_NAMES,
        index=DEFAULT_INDEX,
        help="Archetype for the attacking AI. Loads units and clash quality.",
    )
    defender_strategy = st.selectbox(
        "Defender Strategy",
        options=STRATEGY_NAMES,
        index=DEFAULT_INDEX,
        help="Archetype for the defending AI. Loads units and clash quality.",
    )

    # Hard Counter Bonus Strength slider
    counter_modifier = st.slider(
        "Hard Counter Bonus Strength",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.05,
        help=(
            "Multiplier applied when an archetype faces its hard counter. "
            "1.0 = no bonus, 1.2 = 20% quality boost/penalty (default), "
            "2.0 = 100% boost/penalty."
        ),
    )
    counter_modifier_frac = counter_modifier - 1.0  # Convert to fraction (1.2 -> 0.20)

    matchup = resolve_archetype_matchup(
        attacker_strategy,
        defender_strategy,
        counter_modifier=counter_modifier_frac,
    )
    st.caption(
        f"{matchup['attacker_short_name']}: "
        f"{matchup['attacker_units']} units · quality "
        f"{matchup['attacker_archetype_prob']:.2f}"
    )
    st.caption(
        f"{matchup['defender_short_name']}: "
        f"{matchup['defender_units']} units · quality "
        f"{matchup['defender_archetype_prob']:.2f}"
    )
    st.caption(
        f"Derived clash win probability for the Attacker: "
        f"{matchup['attacker_win_prob']:.3f}  "
        f"(quality_att / (quality_att + quality_def))"
    )

    # How Hard Counters Work info box
    with st.expander("How Hard Counters Work", expanded=False):
        st.markdown(
            """
            **Hard counters create rock-paper-scissors dynamics:**
            
            - When **Archetype A** hard-counters **Archetype B**, A gets a 
              **strategic quality multiplier** (bonus) when attacking B.
            - Conversely, B takes a **penalty** when attacked by its hard counter.
            - The bonus/penalty strength is controlled by the slider above.
            
            **Example:** Sniper (1.59 quality) hard-counters Artillery (1.03).
            With default 1.2× bonus, Sniper's effective quality becomes 1.91
            against Artillery, significantly increasing its clash win probability.
            """
        )

    num_simulations = st.slider(
        "Number of Simulations",
        min_value=1_000,
        max_value=100_000,
        value=50_000,
        step=1_000,
        help="More simulations → tighter estimates, slightly slower runs.",
    )

    # Custom settings 
    advanced_custom = st.toggle(
        "Advanced Custom",
        value=False,
        help="Override archetype presets with manual unit counts and clash probabilities.",
    )

    if advanced_custom:
        attacker_units = st.slider(
            "Attacker Starting Units",
            min_value=5,
            max_value=50,
            value=int(matchup["attacker_units"]),
            step=1,
        )
        defender_units = st.slider(
            "Defender Starting Units",
            min_value=5,
            max_value=50,
            value=int(matchup["defender_units"]),
            step=1,
        )
        attacker_win_prob = st.slider(
            "Attacker Win Probability per Clash",
            min_value=0.10,
            max_value=0.90,
            value=float(round(matchup["attacker_win_prob"], 2)),
            step=0.01,
            help="Chance the Attacker wins a single 1-unit clash.",
        )
        attacker_label = "Custom Sandbox Simulation"
        defender_label = "Custom"
        attacker_full_name = "Custom"
        defender_full_name = "Custom"
    else:
        defender_units = int(matchup["defender_units"])
        attacker_win_prob = float(matchup["attacker_win_prob"])
        attacker_label = matchup["attacker_short_name"]
        defender_label = matchup["defender_short_name"]
        attacker_full_name = attacker_strategy
        defender_full_name = defender_strategy

    # Footer caption
    st.caption("Live convergence plots & deeper parameter settings coming in the next update!")

    run_clicked = st.button("Run Simulation", type="primary")

st.title("Strategic Game Theory & Monte Carlo Simulator")
st.markdown(
    '<p class="hero-sub">Play tens of thousands of independent attrition battles '
    "in parallel, then inspect win rates and remaining-force distributions.</p>",
    unsafe_allow_html=True,
)
if run_clicked:
    with st.spinner("Simulating independent battles..."):
        st.session_state.results = run_monte_carlo_game(
            num_simulations=num_simulations,
            attacker_units=attacker_units,
            defender_units=defender_units,
            attacker_win_prob=attacker_win_prob,
        )
        st.session_state.last_params = {
            "num_simulations": num_simulations,
            "attacker_units": attacker_units,
            "defender_units": defender_units,
            "attacker_win_prob": attacker_win_prob,
            "attacker_label": attacker_label,
            "defender_label": defender_label,
            "attacker_full_name": attacker_full_name,
            "defender_full_name": defender_full_name,
            "advanced_custom": advanced_custom,
        }

results = st.session_state.results
params = st.session_state.last_params

if results is None:
    st.markdown(
        '<div class="empty-hint">'
        "Choose an <b>Attacker Strategy</b> and a <b>Defender Strategy</b> "
        "in the sidebar, then click <b>Run Simulation</b> to pit those "
        "archetypes against each other."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    if params is not None:
        st.markdown(
            f'<div class="matchup-header">'
            f'{params["attacker_label"]}'
            f'<span class="matchup-vs">vs.</span>'
            f'{params["defender_label"]}'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="archetype-chip">'
            f'{params["attacker_full_name"]} ({params["attacker_units"]} units) '
            f"versus "
            f'{params["defender_full_name"]} ({params["defender_units"]} units)'
            f"  ·  {params['num_simulations']:,} battles"
            f"  ·  clash p = {params['attacker_win_prob']:.3f}"
            f"</p>",
            unsafe_allow_html=True,
        )

    st.subheader("Outcome summary")

    win_col, def_col = st.columns(2)
    win_col.metric(
        "Attacker Win Rate",
        f"{results['Attacker Win Rate (%)']:.1f}%",
    )
    def_col.metric(
        "Defender Win Rate",
        f"{results['Defender Win Rate (%)']:.1f}%",
    )
    att_avg_col, def_avg_col = st.columns(2)
    att_avg_col.metric(
        "Avg. Remaining Attacker Units",
        f"{results['Average remaining Attacker units']:.2f}",
    )
    def_avg_col.metric(
        "Avg. Remaining Defender Units",
        f"{results['Average remaining Defender units']:.2f}",
    )

    st.subheader("Remaining units distribution")
    st.caption(
        "Each bar is the share of simulations that finished with that many "
        "units left. A spike at 0 means that side was wiped out often."
    )

    att_series_name = (
        params["attacker_label"] if params is not None else "Attacker"
    )
    def_series_name = (
        params["defender_label"] if params is not None else "Defender"
    )
    remaining_df = pd.DataFrame(
        {
            att_series_name: results["remaining_attacker_units"],
            def_series_name: results["remaining_defender_units"],
        }
    )

    # Make histogram bins
    max_remaining = int(
        max(
            remaining_df[att_series_name].max(),
            remaining_df[def_series_name].max(),
            0,
        )
    )
    bin_size = 1
    xbins = dict(start=-0.5, end=max_remaining + 0.5, size=bin_size)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=remaining_df[att_series_name],
            name=f"{att_series_name} remaining",
            xbins=xbins,
            histnorm="probability",
            marker_color="#e85d4c",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=remaining_df[def_series_name],
            name=f"{def_series_name} remaining",
            xbins=xbins,
            histnorm="probability",
            marker_color="#58a6ff",
            opacity=0.75,
        )
    )
    fig.update_layout(
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e8eaed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Remaining units at battle end",
        yaxis_title="Share of simulations",
        margin=dict(l=40, r=20, t=40, b=40),
        bargap=0.05,
    )
    fig.update_xaxes(gridcolor="#30363d", dtick=1 if max_remaining <= 25 else None)
    fig.update_yaxes(gridcolor="#30363d", tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.header("Round-Robin Tournament Matrix")
st.caption(
    "Every archetype plays every other archetype across pairwise Monte Carlo trials. "
    "The heatmap shows each archetype's attacker win rate against every opponent."
)

# Tournament simulation count slider
tournament_simulations = st.slider(
    "Simulations per matchup",
    min_value=1_000,
    max_value=50_000,
    value=10_000,
    step=1_000,
    help="Number of Monte Carlo simulations per pairwise archetype matchup.",
    key="tournament_sims",
)

# Run tournament button
run_tournament = st.button(
    "Run Round-Robin Tournament",
    help="Execute all 30 pairwise matchups (6 archetypes × 6 archetypes).",
)

if run_tournament:
    with st.spinner(f"Running {len(STRATEGY_PRESETS)}×{len(STRATEGY_PRESETS)} tournament..."):
        tournament_df = run_round_robin_tournament(
            num_simulations_per_matchup=tournament_simulations,
            counter_modifier=counter_modifier_frac,
        )
        st.session_state["tournament_df"] = tournament_df

# Display results
if "tournament_df" in st.session_state:
    tournament_df = st.session_state["tournament_df"]

    # Show the win-rate matrix 
    st.subheader("Win-Rate Matrix (%)")
    st.dataframe(
        tournament_df.round(1),
        use_container_width=True,
    )

    # Render heatmap
    st.subheader("Tournament Heatmap")
    heatmap_fig = create_tournament_heatmap(tournament_df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Show best/worst archetype summary
    st.subheader("Archetype Performance Summary")
    # Compute overall archetype strength (avg win rate across all defenders)
    overall_strength = tournament_df.mean(axis=1).sort_values(ascending=False)
    best_archetype = overall_strength.index[0]
    worst_archetype = overall_strength.index[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Strongest Archetype (Attacker)",
        strategy_short_name(best_archetype),
    )
    col1.caption(f"Avg win rate: {overall_strength.iloc[0]:.1f}%")
    col2.metric(
        "Weakest Archetype (Attacker)",
        strategy_short_name(worst_archetype),
    )
    col2.caption(f"Avg win rate: {overall_strength.iloc[-1]:.1f}%")
    col3.metric(
        "Total Matchups Run",
        f"{len(STRATEGY_PRESETS)}×{len(STRATEGY_PRESETS)}",
    )
    col3.caption(f"{tournament_simulations:,} sims each")
