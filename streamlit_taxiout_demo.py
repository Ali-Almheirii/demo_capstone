
# streamlit_taxiout_demo.py
# --------------------------------------------------------------
# Taxi‑Out Prediction Demo (Refactor: Core-only inputs + Preset Week)
# --------------------------------------------------------------
# - Frontend only captures core fields; engineered features are computed by backend
# - Preset Week mode: browse 30-min bins, select a bin, and predict
# - Manual Scenario mode: set core fields and explicit timestamp and predict
#
# How to run:
#   pip install -r requirements.txt
#   streamlit run streamlit_taxiout_demo.py
# --------------------------------------------------------------

import math
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import altair as alt

st.set_page_config(page_title="Taxi‑Out Prediction Demo", layout="wide")

# -----------------------------
# Constants and helpers
# -----------------------------

AIRPORTS = ["DXB", "AUH", "DOH", "LHR", "JFK", "SIN"]
RUNWAYS = ["05L", "05R", "13L", "13R", "31L", "31R"]
TERMINALS = ["T1", "T2", "T3", "Cargo"]
STANDS = [f"S{i}" for i in range(1, 41)]
AIRCRAFT = ["A320", "B738", "A321", "A333", "A359", "B77W"]

STATE_PALETTE = {
    "Low": "#90caf9",
    "Medium": "#64b5f6",
    "High": "#42a5f5",
    "Surge": "#ef5350",
    "Recovery": "#66bb6a",
}

DEFAULT_API_BASE = "http://localhost:8000"


def safe_to_datetime(val: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(val)
    except Exception:
        return pd.NaT


@st.cache_data(show_spinner=False)
def fetch_week_overview(api_base: str, airport: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"{api_base.rstrip('/')}/traffic/week"
        resp = requests.get(url, params={"airport": airport, "start": start_date, "end": end_date}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def post_predict(api_base: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        url = f"{api_base.rstrip('/')}/predict"
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return None, str(e)


def build_bins_dataframe(week_json: Dict[str, Any]) -> pd.DataFrame:
    bins = week_json.get("bins", [])
    df = pd.DataFrame(bins)
    if df.empty:
        return df
    # Preserve timezone from backend (assumed airport local TZ with offset)
    df["bin_start"] = pd.to_datetime(df["bin_start"])  # keep tz-aware if provided
    df["bin_end"] = pd.to_datetime(df["bin_end"])      # keep tz-aware if provided
    # Maintain readable ordering
    state_cat = pd.CategoricalDtype(categories=["Low", "Medium", "High", "Surge", "Recovery"], ordered=True)
    if "state" in df.columns:
        df["state"] = df["state"].astype(state_cat)
    return df


def bins_label(row: pd.Series) -> str:
    ts = row["bin_end"]
    state = row.get("state", "?")
    dep = row.get("departures_count", "?")
    if isinstance(ts, pd.Timestamp):
        # include timezone offset if available
        try:
            ts_disp = ts.strftime("%Y-%m-%d %H:%M %z")
        except Exception:
            ts_disp = str(ts)
    else:
        ts_disp = str(ts)
    return f"{ts_disp} — {state} — dep {dep}"


def make_week_chart(df_bins: pd.DataFrame, states_to_show: List[str]) -> alt.Chart:
    data = df_bins.copy()
    if states_to_show:
        data = data[data["state"].astype(str).isin(states_to_show)]
    if data.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    base = alt.Chart(data).encode(
        x=alt.X("bin_start:T", title="Time"),
        x2=alt.X2("bin_end:T"),
        y=alt.Y("departures_count:Q", title="Departures (30m)"),
        color=alt.Color("state:N", scale=alt.Scale(domain=list(STATE_PALETTE.keys()), range=list(STATE_PALETTE.values()))),
        tooltip=[
            alt.Tooltip("bin_start:T", title="Start"),
            alt.Tooltip("bin_end:T", title="End"),
            alt.Tooltip("departures_count:Q", title="Departures"),
            alt.Tooltip("state:N", title="State"),
        ],
    )

    chart = base.mark_bar().properties(height=220).interactive()
    return chart


def engineered_feature_chips(engineered: Dict[str, Any]) -> None:
    if not engineered:
        st.caption("No engineered features returned.")
        return
    items = list(engineered.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, items[start:start+3]):
            with col:
                st.metric(k, f"{v}")


# -----------------------------
# Global configuration
# -----------------------------
# Default airport (can be moved to config or made dynamic)
airport = "DXB"

# -----------------------------
# Sidebar: normal distribution
# -----------------------------
with st.sidebar:
    st.header("Taxi-Out Distribution")
    
    # Fetch distribution data from backend
    @st.cache_data(show_spinner=False)
    def fetch_distribution(api_base: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"{api_base.rstrip('/')}/distribution"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None
    
    # Get distribution data
    dist_data = fetch_distribution(st.session_state.get("api_base", DEFAULT_API_BASE))
    
    if dist_data:
        stats = dist_data.get("distribution_stats", {})
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        
        # Create normal distribution plot
        x = np.linspace(max(0, mean - 4*std), mean + 4*std, 100)
        y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')
        ax.fill_between(x, y, alpha=0.3, color='blue')
        
        # Add mean line
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f} min')
        
        # Add percentile lines if available
        percentiles = stats.get("percentiles", {})
        colors = ['green', 'orange', 'purple']
        for i, (pct, val) in enumerate(percentiles.items()):
            if i < len(colors):
                ax.axvline(val, color=colors[i], linestyle=':', alpha=0.7, 
                          label=f'{pct}: {val:.1f} min')
        
        ax.set_xlabel('Taxi-Out Time (minutes)')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution for {airport}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Show key statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean", f"{mean:.1f} min")
            st.metric("Std Dev", f"{std:.1f} min")
        with col2:
            st.metric("Sample Size", f"{stats.get('sample_size', 'N/A'):,}")
        
        # Show percentiles
        if percentiles:
            st.markdown("**Percentiles**")
            for pct, val in percentiles.items():
                st.caption(f"{pct}: {val:.1f} min")
    else:
        st.info("Distribution data not available")
        st.caption("Ensure backend is running and provides /distribution endpoint")

# -----------------------------
# Main layout
# -----------------------------
st.title("Taxi‑Out Prediction Demo")

mode_tabs = st.tabs(["Preset Week", "Manual Scenario", "Config"])

# -----------------------------
# Preset Week Tab
# -----------------------------
with mode_tabs[0]:
    st.subheader("Preset Week")
    c1, c2 = st.columns([1, 1])
    with c1:
        week_start = st.date_input("Week start (local)", value=pd.to_datetime("today").floor("D").date())
    with c2:
        week_end = week_start + timedelta(days=6)
        st.write(f"Week end: {week_end}")

    load_week = st.button("Load week overview", type="primary")
    week_data: Optional[Dict[str, Any]] = None
    if load_week:
        with st.spinner("Loading week overview..."):
            week_data = fetch_week_overview(st.session_state.get("api_base", DEFAULT_API_BASE), airport, start_date=str(week_start), end_date=str(week_end))
            if week_data is None:
                st.error("Failed to load week overview from backend.")
    # Retain in session for interactivity after button press
    if week_data is not None:
        st.session_state["week_data"] = week_data
    else:
        week_data = st.session_state.get("week_data")

    if week_data:
        tz_name = week_data.get("timezone", "UTC")
        df_bins = build_bins_dataframe(week_data)

        st.markdown("**Timeline (30‑minute bins)**")
        states_available = sorted(df_bins["state"].astype(str).dropna().unique().tolist()) if not df_bins.empty else []
        states_to_show = st.multiselect("Filter states", options=states_available, default=states_available)
        chart = make_week_chart(df_bins, states_to_show)
        st.altair_chart(chart, use_container_width=True)

        # Bin selection UI (dropdown coupled to chart)
        if not df_bins.empty:
            df_bins = df_bins.sort_values("bin_end").reset_index(drop=True)
            df_bins["label"] = df_bins.apply(bins_label, axis=1)
            options = df_bins["label"].tolist()
            default_idx = len(options) - 1
            picked = st.selectbox("Select a bin (bin_end)", options, index=default_idx if default_idx >= 0 else 0)
            sel_row = df_bins[df_bins["label"] == picked].iloc[0]

            # Auto-populate core inputs from bin example when selection changes
            core_example = sel_row.get("core_features_example", {}) if "core_features_example" in sel_row else {}
            if picked != st.session_state.get("pw_selected_bin"):
                st.session_state["pw_selected_bin"] = picked
                def _set_if_valid(key, val, options):
                    if isinstance(val, str) and val in options:
                        st.session_state[key] = val
                _set_if_valid("pw_runway", core_example.get("runway"), RUNWAYS)
                _set_if_valid("pw_terminal", core_example.get("terminal"), TERMINALS)
                _set_if_valid("pw_stand", core_example.get("stand"), STANDS)
                _set_if_valid("pw_aircraft", core_example.get("aircraft_type"), AIRCRAFT)
            for _k, _default in [("pw_runway", RUNWAYS[0]), ("pw_terminal", TERMINALS[0]), ("pw_stand", STANDS[0]), ("pw_aircraft", AIRCRAFT[0])]:
                st.session_state.setdefault(_k, _default)

            st.markdown("**Bin details**")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("State", str(sel_row["state"]))
            with d2:
                st.metric("Departures", int(sel_row["departures_count"]))
            with d3:
                st.metric("Prev bin", int(sel_row.get("departures_count_prev_bin", 0)))
            with d4:
                st.caption(f"Start: {sel_row['bin_start']}")
                st.caption(f"End: {sel_row['bin_end']}")

            st.markdown("---")
            st.markdown("**Core Inputs**")
            ci1, ci2, ci3, ci4 = st.columns(4)
            with ci1:
                runway = st.selectbox(
                    "Runway",
                    RUNWAYS,
                    index=(RUNWAYS.index(st.session_state.get("pw_runway", RUNWAYS[0])) if st.session_state.get("pw_runway") in RUNWAYS else 0),
                    key="pw_runway",
                )
            with ci2:
                terminal = st.selectbox(
                    "Terminal",
                    TERMINALS,
                    index=(TERMINALS.index(st.session_state.get("pw_terminal", TERMINALS[0])) if st.session_state.get("pw_terminal") in TERMINALS else 0),
                    key="pw_terminal",
                )
            with ci3:
                stand = st.selectbox(
                    "Stand/Gate",
                    STANDS,
                    index=(STANDS.index(st.session_state.get("pw_stand", STANDS[0])) if st.session_state.get("pw_stand") in STANDS else 0),
                    key="pw_stand",
                )
            with ci4:
                aircraft = st.selectbox(
                    "Aircraft Type",
                    AIRCRAFT,
                    index=(AIRCRAFT.index(st.session_state.get("pw_aircraft", AIRCRAFT[0])) if st.session_state.get("pw_aircraft") in AIRCRAFT else 0),
                    key="pw_aircraft",
                )

            can_predict = all([airport, runway, terminal, stand, aircraft])
            predict_btn = st.button("Predict for selected bin", disabled=not can_predict)

            if predict_btn:
                payload = {
                    "scenario_mode": "historical_preset_week",
                    "airport": airport,
                    "timestamp": pd.to_datetime(sel_row["bin_end"]).isoformat(),
                    "runway": runway,
                    "terminal": terminal,
                    "stand": stand,
                    "aircraft_type": aircraft,
                }
                with st.spinner("Calling backend /predict ..."):
                    result, err = post_predict(st.session_state.get("api_base", DEFAULT_API_BASE), payload)
                if err:
                    st.error(f"Prediction failed: {err}")
                elif result is None:
                    st.error("Prediction failed: unknown error")
                else:
                    st.success("Prediction completed")
                    out_left, out_right = st.columns([1, 1])
                    with out_left:
                        st.subheader("Prediction")
                        m1, m2 = st.columns(2)
                        m1.metric("Predicted taxi‑out (min)", f"{result.get('prediction_min', '—')}")
                        m2.metric("Percentile vs week", f"{result.get('percentile_vs_week', '—')}")
                        if "bin_info" in result:
                            st.caption(f"Bin: {result['bin_info'].get('state', '—')} — departures {result['bin_info'].get('departures_count', '—')}")
                    with out_right:
                        st.subheader("Engineered Features (read‑only)")
                        engineered_feature_chips(result.get("engineered_features", {}))
    else:
        st.info("Load a week to view the timeline and predict.")

# -----------------------------
# Manual Scenario Tab
# -----------------------------
with mode_tabs[1]:
    st.subheader("Manual Scenario (core inputs only)")

    c1, c2, c3 = st.columns(3)
    with c1:
        runway_m = st.selectbox("Runway", RUNWAYS, key="m_runway")
    with c2:
        terminal_m = st.selectbox("Terminal", TERMINALS, key="m_terminal")
    with c3:
        stand_m = st.selectbox("Stand/Gate", STANDS, key="m_stand")

    c4, c5 = st.columns(2)
    with c4:
        aircraft_m = st.selectbox("Aircraft Type", AIRCRAFT, key="m_aircraft")
    with c5:
        tz_input = st.text_input("Timezone (IANA)", value="UTC", help="e.g., Asia/Dubai, Europe/London")

    # Timestamp input (date + time for compatibility)
    date_val = st.date_input("Date", value=datetime.utcnow().date())
    time_val = st.time_input("Time", value=datetime.utcnow().time().replace(second=0, microsecond=0))

    can_predict_manual = all([airport, runway_m, terminal_m, stand_m, aircraft_m, date_val, time_val])
    predict_manual = st.button("Predict", disabled=not can_predict_manual)

    if predict_manual:
        # Build timezone-aware timestamp if possible
        try:
            tz_obj = ZoneInfo(tz_input)
        except Exception:
            tz_obj = timezone.utc
        dt_local = datetime.combine(date_val, time_val).replace(tzinfo=tz_obj)
        ts_iso = dt_local.isoformat()
        payload = {
            "scenario_mode": "manual_core_only",
            "airport": airport,
            "timestamp": ts_iso,
            "runway": runway_m,
            "terminal": terminal_m,
            "stand": stand_m,
            "aircraft_type": aircraft_m,
        }
        with st.spinner("Calling backend /predict ..."):
            result, err = post_predict(st.session_state.get("api_base", DEFAULT_API_BASE), payload)
        if err:
            st.error(f"Prediction failed: {err}")
        elif result is None:
            st.error("Prediction failed: unknown error")
        else:
            st.success("Prediction completed")
            out_left, out_right = st.columns([1, 1])
            with out_left:
                st.subheader("Prediction")
                m1, m2 = st.columns(2)
                m1.metric("Predicted taxi‑out (min)", f"{result.get('prediction_min', '—')}")
                m2.metric("Percentile vs week", f"{result.get('percentile_vs_week', '—')}")
                if "bin_info" in result:
                    st.caption(f"Bin: {result['bin_info'].get('state', '—')} — departures {result['bin_info'].get('departures_count', '—')}")
            with out_right:
                st.subheader("Engineered Features (read‑only)")
                engineered_feature_chips(result.get("engineered_features", {}))


# -----------------------------
# Config Tab
# -----------------------------
with mode_tabs[2]:
    st.subheader("Configuration")
    
    # Use session state for api_base so it can be updated and used throughout the app
    if "api_base" not in st.session_state:
        st.session_state.api_base = DEFAULT_API_BASE
    
    api_base = st.text_input("Backend URL", value=st.session_state.api_base, help="Local backend base URL", key="api_base_input")
    
    # Update the global api_base when user changes it
    if st.session_state.get("api_base_input") != st.session_state.api_base:
        st.session_state.api_base = st.session_state.api_base_input
    
    st.markdown("---")
    st.markdown("### Notes")
    st.write(
        "- UI collects only core inputs. Engineered features are computed by the backend and displayed as read‑only.\n"
        "- Preset Week shows a 30‑minute binned timeline fetched from the backend.\n"
        "- For full train/serve parity, ensure backend uses the same feature pipeline as training."
    )
