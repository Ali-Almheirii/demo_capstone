
# streamlit_taxiout_demo.py
# --------------------------------------------------------------
# Taxi‑Out Prediction Demo (Refactor: Core-only inputs + Preset Week)
# --------------------------------------------------------------
# - Frontend only captures core fields; engineered features are computed by backend
# - Preset Week mode: browse 10-min bins, select a bin, and predict
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

st.set_page_config(
    page_title="Taxi‑Out Prediction Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
def fetch_day_overview(api_base: str, date: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"{api_base.rstrip('/')}/traffic/day"
        resp = requests.get(url, params={"date": date}, timeout=10)
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

    # For line chart, we need to use bin center as x-axis
    data_with_center = data.copy()
    data_with_center['bin_center'] = data_with_center['bin_start'] + (data_with_center['bin_end'] - data_with_center['bin_start']) / 2
    
    base = alt.Chart(data_with_center).encode(
        x=alt.X("bin_center:T", title="Time"),
        y=alt.Y("departures_count:Q", title="Departures (10m)"),
        color=alt.Color("state:N", scale=alt.Scale(domain=list(STATE_PALETTE.keys()), range=list(STATE_PALETTE.values()))),
        tooltip=[
            alt.Tooltip("bin_start:T", title="Start"),
            alt.Tooltip("bin_end:T", title="End"),
            alt.Tooltip("departures_count:Q", title="Departures"),
            alt.Tooltip("state:N", title="State"),
        ],
    )

    # Create trend-focused chart with smoother lines and smaller points
    line_chart = base.mark_line(strokeWidth=2.5, opacity=0.8).properties(height=450)
    point_chart = base.mark_circle(size=40, strokeWidth=1.5, opacity=0.9).properties(height=450)
    
    # Combine line and points for better interaction
    chart = alt.layer(line_chart, point_chart).properties(
        title="Daily Departure Trends"
    ).configure_axis(
        gridColor='#e0e0e0',
        gridOpacity=0.2,
        labelFontSize=11,
        titleFontSize=13
    ).configure_view(
        strokeWidth=0
    ).configure_legend(
        orient='top',
        titleFontSize=12,
        labelFontSize=10
    ).interactive()
    
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
        histogram_data = dist_data.get("histogram_data", [])
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        
        if histogram_data:
            # Create DataFrame from real histogram data
            df_hist = pd.DataFrame(histogram_data)
            df_hist['bin_center'] = (df_hist['bin_start'] + df_hist['bin_end']) / 2
            
            # Create smooth curve using better interpolation and smoothing
            x_points = df_hist['bin_center'].tolist()
            y_points = df_hist['count'].tolist()
            
            if x_points and y_points:
                # Create more natural curve using cubic spline interpolation
                from scipy.interpolate import CubicSpline
                from scipy.ndimage import gaussian_filter1d
                
                # Add padding points for better curve behavior at edges
                x_padded = [0] + x_points + [df_hist['bin_end'].iloc[-1]]
                y_padded = [0] + y_points + [0]
                
                # Create cubic spline interpolation
                cs = CubicSpline(x_padded, y_padded, bc_type='natural')
                
                # Generate smooth curve with more points
                x_smooth = np.linspace(0, df_hist['bin_end'].iloc[-1], 500)  # More points for smoother curve
                y_smooth = cs(x_smooth)
                
                # Apply Gaussian smoothing to remove any remaining artifacts
                y_smooth = gaussian_filter1d(y_smooth, sigma=2)
                
                # Ensure non-negative values
                y_smooth = np.maximum(y_smooth, 0)
                
                # Create DataFrame for smooth curve
                df_smooth = pd.DataFrame({
                    'taxi_time': x_smooth,
                    'count': y_smooth
                })
                
                # Base chart for smooth curve
                base = alt.Chart(df_smooth).encode(
                    x=alt.X('taxi_time:Q', title='Taxi-Out Time (minutes)', 
                           scale=alt.Scale(domain=[0, df_hist['bin_end'].iloc[-1]])),
                    y=alt.Y('count:Q', title='Number of Flights'),
                    tooltip=[
                        alt.Tooltip('taxi_time:Q', title='Time (min)', format='.1f'),
                        alt.Tooltip('count:Q', title='Flights', format='.0f')
                    ]
                )
                
                # Smooth area chart
                area = base.mark_area(
                    fill='#4CAF50',
                    fillOpacity=0.6,
                    stroke='#2E7D32',
                    strokeWidth=2,
                    interpolate='monotone'  # Smooth interpolation
                )
                
                # Line overlay for better definition
                line = base.mark_line(
                    color='#2E7D32',
                    strokeWidth=2,
                    interpolate='monotone'
                )
                
                chart_layers = [area, line]
            else:
                # Fallback if no data
                chart_layers = []
            
            # Mean line
            mean_rule = alt.Chart(pd.DataFrame({'mean': [mean]})).mark_rule(
                color='#FF5722',
                strokeWidth=3,
                strokeDash=[5, 5]
            ).encode(x='mean:Q')
            
            # Percentile lines with legend
            percentiles = stats.get("percentiles", {})
            percentile_rules = []
            percentile_colors = ['#2196F3', '#FF9800', '#9C27B0']
            percentile_names = ['25th', '50th', '75th']
            
            for i, (pct, val) in enumerate(percentiles.items()):
                if i < len(percentile_colors):
                    rule = alt.Chart(pd.DataFrame({'value': [val], 'percentile': [percentile_names[i]]})).mark_rule(
                        color=percentile_colors[i],
                        strokeWidth=2,
                        strokeDash=[3, 3] if i == 0 else [1, 1] if i == 1 else [6, 2]
                    ).encode(
                        x='value:Q',
                        color=alt.Color('percentile:N', scale=alt.Scale(
                            domain=percentile_names,
                            range=percentile_colors
                        ))
                    )
                    percentile_rules.append(rule)
            
            # Combine all layers
            chart = alt.layer(*chart_layers, mean_rule, *percentile_rules).properties(
                title='Taxi-Out Time Distribution (Real Data)',
                width='container',
                height=500
            ).configure_axis(
                gridColor='#e0e0e0',
                gridOpacity=0.3,
                labelFontSize=11,
                titleFontSize=13
            ).configure_axisY(
                format='.0s'  # Short format: 1K, 2K, etc. instead of 1000, 2000
            ).configure_view(
                strokeWidth=0
            ).configure_legend(
                orient='top',
                titleFontSize=12,
                labelFontSize=10
            )
        else:
            # Fallback to theoretical distribution if no histogram data
            x = np.linspace(max(0, mean - 4*std), mean + 4*std, 200)
            y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            df_dist = pd.DataFrame({
                'taxi_time': x,
                'frequency': y
            })
            
            base = alt.Chart(df_dist).encode(
                x=alt.X('taxi_time:Q', title='Taxi-Out Time (minutes)', scale=alt.Scale(domain=[max(0, mean - 4*std), mean + 4*std])),
                y=alt.Y('frequency:Q', title='Frequency', scale=alt.Scale(domain=[0, max(y) * 1.1]))
            )
            
            area = base.mark_area(
                fill='#4CAF50',
                fillOpacity=0.6,
                stroke='#2E7D32',
                strokeWidth=2
            )
            
            line = base.mark_line(
                color='#2E7D32',
                strokeWidth=3
            )
            
            mean_rule = alt.Chart(pd.DataFrame({'mean': [mean]})).mark_rule(
                color='#FF5722',
                strokeWidth=3,
                strokeDash=[5, 5]
            ).encode(x='mean:Q')
            
            percentiles = stats.get("percentiles", {})
            percentile_rules = []
            percentile_colors = ['#2196F3', '#FF9800', '#9C27B0']
            
            for i, (pct, val) in enumerate(percentiles.items()):
                if i < len(percentile_colors):
                    rule = alt.Chart(pd.DataFrame({'value': [val]})).mark_rule(
                        color=percentile_colors[i],
                        strokeWidth=2,
                        strokeDash=[3, 3] if i == 0 else [1, 1] if i == 1 else [6, 2]
                    ).encode(x='value:Q')
                    percentile_rules.append(rule)
            
            chart = alt.layer(area, line, mean_rule, *percentile_rules).properties(
                title='Taxi-Out Time Distribution (Theoretical)',
                width='container',
                height=500
            ).configure_axis(
                gridColor='#e0e0e0',
                gridOpacity=0.3,
                labelFontSize=11,
                titleFontSize=13
            ).configure_axisY(
                format='.0s'  # Short format for consistency
            ).configure_view(
                strokeWidth=0
            )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show key statistics with better styling
        st.markdown("### 📊 Distribution Statistics")
        
        # Main metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="📈 Mean Taxi-Out", 
                value=f"{mean:.1f} min",
                delta=None
            )
        with col2:
            st.metric(
                label="📊 Standard Deviation", 
                value=f"{std:.1f} min",
                delta=None
            )
        with col3:
            sample_size = stats.get('sample_size', 'N/A')
            if sample_size != 'N/A':
                st.metric(
                    label="📋 Sample Size", 
                    value=f"{sample_size:,}",
                    delta=None
                )
            else:
                st.metric(
                    label="📋 Sample Size", 
                    value="N/A",
                    delta=None
                )
        
        # Show percentiles in a more organized way
        if percentiles:
            st.markdown("### 📏 Percentiles")
            pct_cols = st.columns(len(percentiles))
            for i, (pct, val) in enumerate(percentiles.items()):
                with pct_cols[i]:
                    # Use a cleaner format to avoid display issues
                    st.metric(
                        label=f"{pct} Percentile",
                        value=f"{val:.1f}",
                        delta=None
                    )
                    st.caption("minutes")
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
    st.subheader("Daily Overview")
    c1, c2 = st.columns([1, 1])
    with c1:
        selected_date = st.date_input("Select Date (local)", value=pd.to_datetime("today").floor("D").date())
    with c2:
        st.write(f"Viewing: {selected_date}")

    load_day = st.button("Load day overview", type="primary")
    day_data: Optional[Dict[str, Any]] = None
    if load_day:
        with st.spinner("Loading day overview..."):
            day_data = fetch_day_overview(st.session_state.get("api_base", DEFAULT_API_BASE), date=str(selected_date))
            if day_data is None:
                st.error("Failed to load day overview from backend.")
    # Retain in session for interactivity after button press
    if day_data is not None:
        st.session_state["day_data"] = day_data
    else:
        day_data = st.session_state.get("day_data")

    if day_data:
        tz_name = day_data.get("timezone", "UTC")
        df_bins = build_bins_dataframe(day_data)

        st.markdown("**Timeline (10‑minute bins)**")
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
        st.info("Load a day to view the timeline and predict.")

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
    "- Preset Week shows a 10‑minute binned timeline fetched from the backend.\n"
    "- For full train/serve parity, ensure backend uses the same feature pipeline as training."
)
