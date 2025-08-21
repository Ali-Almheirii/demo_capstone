
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
import matplotlib.patches as mpatches
import streamlit as st
import requests
import altair as alt
from streamlit_vega_lite import altair_component
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


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
def fetch_day_overview(api_base: str, date: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        url = f"{api_base.rstrip('/')}/traffic/day"
        resp = requests.get(url, params={"date": date}, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        elif resp.status_code == 404:
            # Try to extract error message from response
            try:
                error_data = resp.json()
                error_msg = error_data.get("detail", "No data available for this date.")
            except:
                error_msg = "No data available for this date."
            return None, error_msg
        else:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return None, str(e)


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


def create_bins_table(df_bins: pd.DataFrame, states_to_show: List[str]) -> pd.DataFrame:
    """Create a simple table for bin selection"""
    data = df_bins.copy()
    
    # Apply state filtering
    if states_to_show and len(states_to_show) > 0:
        data = data[data["state"].astype(str).isin(states_to_show)]
    
    if data.empty:
        return pd.DataFrame()
    
    # Create display table
    display_data = []
    for idx, row in data.iterrows():
        # Format time for display
        time_str = row['bin_end'].strftime('%H:%M')
        
        # Color code the state
        state_color = STATE_PALETTE.get(row['state'], '#cccccc')
        state_display = f"🔵 {row['state']}" if row['state'] == 'Low' else \
                       f"🟡 {row['state']}" if row['state'] == 'Medium' else \
                       f"🟠 {row['state']}" if row['state'] == 'High' else \
                       f"🔴 {row['state']}" if row['state'] == 'Surge' else \
                       f"🟢 {row['state']}"
        
        display_data.append({
            'Time': time_str,
            'State': state_display,
            'Departures': row['departures_count'],
            'Row Index': idx
        })
    
    return pd.DataFrame(display_data)


def create_curve_day_chart(df_bins: pd.DataFrame, states_to_show: List[str]) -> Tuple[plt.Figure, pd.DataFrame]:
    """Create a matplotlib curve chart for daily traffic bins with marked points for lowest, medium, and highest traffic"""
    data = df_bins.copy()

    # Apply state filtering
    if states_to_show and len(states_to_show) > 0:
        data = data[data["state"].astype(str).isin(states_to_show)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    if data.empty:
        ax.text(0.5, 0.5, "No data to display - adjust state filters above",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Departures (10-min bin)")
        ax.set_title("Daily Traffic Pattern")
        return fig, data

    # Sort data by time for smooth curve
    data_sorted = data.sort_values('bin_end').reset_index(drop=True)

    # Extract hour from bin_end for x-axis (0-24) - use local time
    if data_sorted['bin_end'].dt.tz is not None:
        local_times = data_sorted['bin_end'].dt.tz_localize(None)
    else:
        local_times = data_sorted['bin_end']

    hours = local_times.dt.hour + (local_times.dt.minute / 60.0)
    y_values = data_sorted['departures_count'].astype(float).values

    # Create smooth curve
    ax.plot(hours, y_values, color='#2E86AB', linewidth=2, alpha=0.7, label='Traffic Flow')
    ax.fill_between(hours, y_values, alpha=0.2, color='#2E86AB')

    # Find lowest, medium, and highest traffic points
    if len(data_sorted) >= 3:
        sorted_by_traffic = data_sorted.sort_values('departures_count')
        
        lowest_idx = sorted_by_traffic.index[0]
        highest_idx = sorted_by_traffic.index[-1]
        medium_idx = sorted_by_traffic.index[len(sorted_by_traffic)//2]
        
        # Get the corresponding data points
        lowest_point = data_sorted.loc[lowest_idx]
        medium_point = data_sorted.loc[medium_idx]
        highest_point = data_sorted.loc[highest_idx]
        
        # Extract x,y coordinates for markers
        if lowest_point['bin_end'].tz is not None:
            lowest_time = lowest_point['bin_end'].tz_localize(None)
        else:
            lowest_time = lowest_point['bin_end']
        lowest_x = lowest_time.hour + (lowest_time.minute / 60.0)
        lowest_y = lowest_point['departures_count']
        
        if medium_point['bin_end'].tz is not None:
            medium_time = medium_point['bin_end'].tz_localize(None)
        else:
            medium_time = medium_point['bin_end']
        medium_x = medium_time.hour + (medium_time.minute / 60.0)
        medium_y = medium_point['departures_count']
        
        if highest_point['bin_end'].tz is not None:
            highest_time = highest_point['bin_end'].tz_localize(None)
        else:
            highest_time = highest_point['bin_end']
        highest_x = highest_time.hour + (highest_time.minute / 60.0)
        highest_y = highest_point['departures_count']
        
        # Mark the points
        ax.scatter([lowest_x], [lowest_y], color='#28A745', s=150, zorder=5, 
                  label=f'🟢 Lowest ({int(lowest_y)})', edgecolor='white', linewidth=2)
        ax.scatter([medium_x], [medium_y], color='#FFC107', s=150, zorder=5, 
                  label=f'🟡 Medium ({int(medium_y)})', edgecolor='white', linewidth=2)
        ax.scatter([highest_x], [highest_y], color='#DC3545', s=150, zorder=5, 
                  label=f'🔴 Highest ({int(highest_y)})', edgecolor='white', linewidth=2)
        
        # Add annotations for the points
        ax.annotate(f'Lowest\n{lowest_point["bin_end"].strftime("%H:%M")}', 
                   xy=(lowest_x, lowest_y), xytext=(lowest_x, lowest_y + max(y_values) * 0.15),
                   ha='center', fontsize=9, fontweight='bold', color='#28A745',
                   arrowprops=dict(arrowstyle='->', color='#28A745', alpha=0.7))
        
        ax.annotate(f'Medium\n{medium_point["bin_end"].strftime("%H:%M")}', 
                   xy=(medium_x, medium_y), xytext=(medium_x, medium_y + max(y_values) * 0.15),
                   ha='center', fontsize=9, fontweight='bold', color='#FFC107',
                   arrowprops=dict(arrowstyle='->', color='#FFC107', alpha=0.7))
        
        ax.annotate(f'Highest\n{highest_point["bin_end"].strftime("%H:%M")}', 
                   xy=(highest_x, highest_y), xytext=(highest_x, highest_y + max(y_values) * 0.15),
                   ha='center', fontsize=9, fontweight='bold', color='#DC3545',
                   arrowprops=dict(arrowstyle='->', color='#DC3545', alpha=0.7))

    # Customize chart
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Departures (10-min bin)", fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.3)

    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    return fig, data


def populate_session_state_from_core_example(core_example: Dict[str, Any]) -> None:
    """Helper function to populate session state with all core features from backend response"""
    st.session_state["pw_direction"] = core_example.get("flight_direction", st.session_state.get("pw_direction", "D"))
    st.session_state["pw_aobt_utc"] = core_example.get("actual_offblock_time_aobt_utc", st.session_state.get("pw_aobt_utc", ""))
    st.session_state["pw_aircraft_icao"] = core_example.get("aircraft_type_icao", st.session_state.get("pw_aircraft_icao", AIRCRAFT[0] if AIRCRAFT else "A320"))
    st.session_state["pw_terminal"] = core_example.get("terminal", st.session_state.get("pw_terminal", "T1"))
    st.session_state["pw_concourse"] = core_example.get("concourse", st.session_state.get("pw_concourse", "UNKNOWN"))
    st.session_state["pw_service_type"] = core_example.get("service_type", st.session_state.get("pw_service_type", "UNKNOWN"))
    st.session_state["pw_flight_nature"] = core_example.get("flight_nature", st.session_state.get("pw_flight_nature", "UNKNOWN"))
    st.session_state["pw_stand"] = core_example.get("stand", st.session_state.get("pw_stand", ""))
    st.session_state["pw_flight_number"] = core_example.get("flight_number", st.session_state.get("pw_flight_number", "UNKNOWN"))
    st.session_state["pw_aircraft_iata"] = core_example.get("aircraft_type_iata", st.session_state.get("pw_aircraft_iata", ""))
    st.session_state["pw_destination_iata"] = core_example.get("destination_iata", st.session_state.get("pw_destination_iata", ""))
    st.session_state["pw_aircraft_registration"] = core_example.get("aircraft_registration", st.session_state.get("pw_aircraft_registration", ""))
    st.session_state["pw_flight_schedule_time_utc"] = core_example.get("flight_schedule_time_utc", st.session_state.get("pw_flight_schedule_time_utc", ""))
    st.session_state["pw_sobt_utc"] = core_example.get("scheduled_offblock_time_sobt_utc", st.session_state.get("pw_sobt_utc", ""))
    st.session_state["pw_sibt_utc"] = core_example.get("scheduled_inblock_time_sibt_utc", st.session_state.get("pw_sibt_utc", ""))
    st.session_state["pw_airport"] = core_example.get("airport", st.session_state.get("pw_airport", "DXB"))


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
    
    # Fixed x-axis range (0-140 minutes)
    x_max = 140
    
    # Get distribution data
    dist_data = fetch_distribution(st.session_state.get("api_base", DEFAULT_API_BASE))
    
    if dist_data:
        stats = dist_data.get("distribution_stats", {})
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        
        # Create normal distribution plot with dynamic range
        x = np.linspace(0, x_max, 200)
        # Scale to number of flights (assuming sample_size from stats)
        sample_size = stats.get("sample_size", 1000)
        y = (sample_size / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        
        # Create DataFrame for Altair chart
        df_dist = pd.DataFrame({
            'taxi_time': x,
            'flights': y
        })
        
        # Get percentiles for chart lines
        percentiles = stats.get("percentiles", {})
        
        # Create base chart with shared encoding
        base = alt.Chart(df_dist).encode(
            x=alt.X('taxi_time:Q', title='Taxi-Out Time (minutes)', scale=alt.Scale(domain=[0, x_max])),
            y=alt.Y('flights:Q', title='Number of Flights')
        )
        
        # Main distribution area
        area_chart = base.mark_area(
            fill='#4CAF50',
            fillOpacity=0.6,
            stroke='#2E7D32',
            strokeWidth=2
        )
        
        # Mean line only (grey, no legend)
        mean_rule = None
        if mean > 0:
            mean_df = pd.DataFrame({'mean': [mean]})
            mean_rule = alt.Chart(mean_df).mark_rule(
                color='#666666',
                strokeWidth=3,
                strokeDash=[5, 5]
            ).encode(x='mean:Q')
        
        # Combine layers (only area chart and mean line)
        layers = [area_chart]
        if mean_rule:
            layers.append(mean_rule)
        
        chart = alt.layer(*layers).properties(
            title=f'Taxi-Out Time Distribution for {airport}',
            width='container',
            height=450
        ).configure_axis(
            gridColor='#e0e0e0',
            gridOpacity=0.3,
            labelFontSize=11,
            titleFontSize=13
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            orient='top',
            titleFontSize=12,
            labelFontSize=10,
            titleColor='#333333',
            labelColor='#666666'
        )
        
        # Display the combined chart
        st.altair_chart(chart, use_container_width=True, key="dist_chart")
        
        # Show key statistics in a nicely organized format
        st.markdown("---")
        st.markdown("### 📊 Distribution Statistics")
        
        # Create a more organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Central Tendency**")
            st.metric(
                label="Mean",
                value=f"{mean:.1f} min",
                help="Average taxi-out time from real flight data"
            )

            st.metric(
                label="Median",
                value=f"{percentiles.get('50th', 'N/A')} min" if percentiles and '50th' in percentiles else "N/A",
                help="50th percentile (median) taxi-out time from real flight data"
            )
        
        with col2:
            st.markdown("**📏 Variability**")
            st.metric(
                label="Standard Deviation",
                value=f"{std:.1f} min",
                help="Measure of spread around the mean from real flight data"
            )
            st.metric(
                label="Sample Size",
                value=f"{stats.get('sample_size', 'N/A'):,}",
                help="Total number of real flights in the dataset"
            )

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
            day_data, error_msg = fetch_day_overview(st.session_state.get("api_base", DEFAULT_API_BASE), date=str(selected_date))
            if day_data is None:
                if error_msg:
                    st.error(f"Failed to load day overview: {error_msg}")
                else:
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
        
        # Default to showing all states, but allow user to filter
        if not states_available:
            states_to_show = []
        else:
            # Use session state to remember filter selection
            if "states_filter" not in st.session_state:
                st.session_state.states_filter = states_available
            
            states_to_show = st.multiselect(
                "Filter states", 
                options=states_available, 
                default=st.session_state.states_filter,
                help="Select which traffic states to display. Leave all selected to see everything."
            )
            
            # Update session state
            st.session_state.states_filter = states_to_show
        
        # Create and display curve chart with marked points
        st.markdown("**📊 Daily Traffic Pattern - Select traffic level below**")
        fig, chart_data = create_curve_day_chart(df_bins, states_to_show)
        
        # Display the matplotlib chart
        st.pyplot(fig, use_container_width=True)
        
        # Simple bin selection options
        if not chart_data.empty:
            st.markdown("**🎯 Select Traffic Level:**")
            
            # Find lowest, medium, and highest bins by departure count
            sorted_data = chart_data.sort_values('departures_count')
            
            if len(sorted_data) >= 3:
                lowest_bin = sorted_data.iloc[0]
                highest_bin = sorted_data.iloc[-1]
                medium_bin = sorted_data.iloc[len(sorted_data)//2]
                
                # Create selection buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(
                        f"🟢 Lowest Traffic\n{lowest_bin['bin_end'].strftime('%H:%M')}\n({lowest_bin['departures_count']} departures)",
                        key="select_lowest",
                        help=f"Select lowest traffic bin at {lowest_bin['bin_end'].strftime('%H:%M')}"
                    ):
                        # Get the original row from df_bins
                        sel_row = df_bins.iloc[lowest_bin.name]
                        core_example = sel_row.get("core_features_example", {}) if "core_features_example" in sel_row else {}
                        st.session_state["pw_selected_bin"] = bins_label(sel_row)
                        
                        # Populate session state from core example
                        populate_session_state_from_core_example(core_example)
                        
                        # Show success message
                        st.success(f"✅ Selected lowest traffic: {sel_row['state']} at {sel_row['bin_end'].strftime('%H:%M')} with {sel_row['departures_count']} departures")
                        st.rerun()
                
                with col2:
                    if st.button(
                        f"🟡 Medium Traffic\n{medium_bin['bin_end'].strftime('%H:%M')}\n({medium_bin['departures_count']} departures)",
                        key="select_medium",
                        help=f"Select medium traffic bin at {medium_bin['bin_end'].strftime('%H:%M')}"
                    ):
                        # Get the original row from df_bins
                        sel_row = df_bins.iloc[medium_bin.name]
                        core_example = sel_row.get("core_features_example", {}) if "core_features_example" in sel_row else {}
                        st.session_state["pw_selected_bin"] = bins_label(sel_row)
                        
                        # Populate session state from core example
                        populate_session_state_from_core_example(core_example)
                        
                        # Show success message
                        st.success(f"✅ Selected medium traffic: {sel_row['state']} at {sel_row['bin_end'].strftime('%H:%M')} with {sel_row['departures_count']} departures")
                        st.rerun()
                
                with col3:
                    if st.button(
                        f"🔴 Highest Traffic\n{highest_bin['bin_end'].strftime('%H:%M')}\n({highest_bin['departures_count']} departures)",
                        key="select_highest",
                        help=f"Select highest traffic bin at {highest_bin['bin_end'].strftime('%H:%M')}"
                    ):
                        # Get the original row from df_bins
                        sel_row = df_bins.iloc[highest_bin.name]
                        core_example = sel_row.get("core_features_example", {}) if "core_features_example" in sel_row else {}
                        st.session_state["pw_selected_bin"] = bins_label(sel_row)
                        
                        # Populate session state from core example
                        populate_session_state_from_core_example(core_example)
                        
                        # Show success message
                        st.success(f"✅ Selected highest traffic: {sel_row['state']} at {sel_row['bin_end'].strftime('%H:%M')} with {sel_row['departures_count']} departures")
                        st.rerun()
            else:
                st.info("Not enough data points for traffic level selection.")
        


        # Bin selection UI (dropdown coupled to chart)
        if not df_bins.empty:
            df_bins = df_bins.sort_values("bin_end").reset_index(drop=True)
            df_bins["label"] = df_bins.apply(bins_label, axis=1)
            options = df_bins["label"].tolist()
            # If a click set pw_selected_bin, prefer it as default
            preselect_label = st.session_state.get("pw_selected_bin")
            default_idx = options.index(preselect_label) if preselect_label in options else (len(options) - 1)
            picked = st.selectbox("Select a bin", options, index=default_idx if default_idx >= 0 else 0)
            sel_row = df_bins[df_bins["label"] == picked].iloc[0]

            # Auto-populate core inputs from bin example when selection changes (new contract)
            core_example = sel_row.get("core_features_example", {}) if "core_features_example" in sel_row else {}
            if picked != st.session_state.get("pw_selected_bin"):
                st.session_state["pw_selected_bin"] = picked
                populate_session_state_from_core_example(core_example)
            # Ensure defaults exist
            for key, fallback in [
                ("pw_direction", "D"),
                ("pw_aobt_utc", ""),
                ("pw_aircraft_icao", AIRCRAFT[0] if AIRCRAFT else "A320"),
                ("pw_airport", "DXB"),
                ("pw_terminal", "T1"),
                ("pw_concourse", "UNKNOWN"),
                ("pw_service_type", "UNKNOWN"),
                ("pw_flight_nature", "UNKNOWN"),
                ("pw_stand", ""),
                ("pw_flight_number", "UNKNOWN"),
                ("pw_aircraft_iata", ""),
                ("pw_destination_iata", ""),
                ("pw_aircraft_registration", ""),
                ("pw_flight_schedule_time_utc", ""),
                ("pw_sobt_utc", ""),
                ("pw_sibt_utc", ""),
            ]:
                st.session_state.setdefault(key, fallback)

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
            # Row 1 - Basic flight info
            ci1, ci2, ci3 = st.columns(3)
            with ci1:
                direction = st.selectbox(
                    "Flight Direction",
                    ["D", "A"],
                    index=(0 if st.session_state.get("pw_direction", "D") == "D" else 1),
                    key="pw_direction",
                )
            with ci2:
                aircraft_icao = st.selectbox(
                    "Aircraft Type (ICAO)",
                    AIRCRAFT,
                    index=(AIRCRAFT.index(st.session_state.get("pw_aircraft_icao", AIRCRAFT[0])) if st.session_state.get("pw_aircraft_icao") in AIRCRAFT else 0),
                    key="pw_aircraft_icao",
                )
            with ci3:
                airport = st.text_input("Airport", value=st.session_state.get("pw_airport", "DXB"), key="pw_airport")
            
            # Row 1b - Terminal info
            ci4, ci5, ci6 = st.columns(3)
            with ci4:
                terminal = st.text_input("Terminal", value=st.session_state.get("pw_terminal", "T1"), key="pw_terminal")
            with ci5:
                stand = st.text_input("Stand/Gate", value=st.session_state.get("pw_stand", ""), key="pw_stand")
            with ci6:
                concourse = st.text_input("Concourse", value=st.session_state.get("pw_concourse", "UNKNOWN"), key="pw_concourse")
            
            # Row 2 - Flight details
            cj1, cj2, cj3 = st.columns(3)
            with cj1:
                flight_number = st.text_input("Flight Number", value=st.session_state.get("pw_flight_number", "UNKNOWN"), key="pw_flight_number")
            with cj2:
                service_type = st.text_input("Service Type", value=st.session_state.get("pw_service_type", "UNKNOWN"), key="pw_service_type")
            with cj3:
                flight_nature = st.text_input("Flight Nature", value=st.session_state.get("pw_flight_nature", "UNKNOWN"), key="pw_flight_nature")
            
            # Row 3 - Timing details
            ck1, ck2, ck3 = st.columns(3)
            with ck1:
                aobt_utc = st.text_input("AOBT (UTC ISO)", value=st.session_state.get("pw_aobt_utc", ""), key="pw_aobt_utc")
            with ck2:
                flight_schedule_time_utc = st.text_input("Flight Schedule Time (UTC)", value=st.session_state.get("pw_flight_schedule_time_utc", ""), key="pw_flight_schedule_time_utc")
            with ck3:
                sobt_utc = st.text_input("SOBT (UTC)", value=st.session_state.get("pw_sobt_utc", ""), key="pw_sobt_utc")
            
            # Row 4 - Additional aircraft info
            cl1, cl2, cl3 = st.columns(3)
            with cl1:
                aircraft_iata = st.text_input("Aircraft Type (IATA)", value=st.session_state.get("pw_aircraft_iata", ""), key="pw_aircraft_iata")
            with cl2:
                destination_iata = st.text_input("Destination (IATA)", value=st.session_state.get("pw_destination_iata", ""), key="pw_destination_iata")
            with cl3:
                aircraft_registration = st.text_input("Aircraft Registration", value=st.session_state.get("pw_aircraft_registration", ""), key="pw_aircraft_registration")
            
            # Row 5 - Additional timing
            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                sibt_utc = st.text_input("SIBT (UTC)", value=st.session_state.get("pw_sibt_utc", ""), key="pw_sibt_utc")
            with cm2:
                st.empty()  # Empty column for spacing
            with cm3:
                st.empty()  # Empty column for spacing

            can_predict = all([
                direction,
                aobt_utc,
                aircraft_icao,
                terminal,
                stand,
            ])
            predict_btn = st.button("Predict for selected bin", disabled=not can_predict)

            if predict_btn:
                payload = {
                    "flight_direction": direction,
                    "actual_offblock_time_aobt_utc": aobt_utc,
                    "aircraft_type_icao": aircraft_icao,
                    "airport": airport,
                    "terminal": terminal,
                    "concourse": concourse or "UNKNOWN",
                    "service_type": service_type or "UNKNOWN",
                    "flight_nature": flight_nature or "UNKNOWN",
                    "stand": stand,
                    "flight_number": flight_number or "UNKNOWN",
                    "aircraft_type_iata": aircraft_iata or "",
                    "destination_iata": destination_iata or "",
                    "aircraft_registration": aircraft_registration or "",
                    "flight_schedule_time_utc": flight_schedule_time_utc or "",
                    "scheduled_offblock_time_sobt_utc": sobt_utc or "",
                    "scheduled_inblock_time_sibt_utc": sibt_utc or "",
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
                        st.subheader("Prediction Results")
                        
                        # Main prediction metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            predicted = result.get('prediction_min', '—')
                            st.metric(
                                "Predicted Taxi-Out", 
                                f"{predicted} min",
                                help="Model's prediction for taxi-out duration"
                            )
                        with col2:
                            actual_min = result.get('actual_min')
                            if actual_min is not None:
                                st.metric(
                                    "Actual Taxi-Out", 
                                    f"{actual_min} min",
                                    help="Observed taxi-out duration for this exact scenario"
                                )
                            else:
                                st.metric(
                                    "Actual Taxi-Out", 
                                    "No matching scenario found",
                                    help="No historical scenario was found to compare against"
                                )
                        
                        # Additional metrics
                        col3, col4 = st.columns(2)
                        with col3:
                            percentile = result.get('percentile_vs_week', '—')
                            st.metric(
                                "Percentile vs Week", 
                                f"{percentile}",
                                help="How this prediction compares to the week's distribution"
                            )
                        with col4:
                            error_val = result.get('prediction_error')
                            if error_val is None and predicted != '—' and actual_min is not None:
                                try:
                                    error_val = float(predicted) - float(actual_min)
                                except (ValueError, TypeError):
                                    error_val = None
                            if isinstance(error_val, (int, float)):
                                diff_color = "normal" if abs(error_val) <= 5 else "inverse"
                                st.metric(
                                    "Prediction Error", 
                                    f"{error_val:+.1f} min",
                                    delta_color=diff_color,
                                    help="Predicted minus actual (positive = over-prediction)"
                                )
                            else:
                                st.metric("Prediction Error", "—")
                        
                        # Bin information (robust to None)
                        bin_info = result.get("bin_info")
                        if isinstance(bin_info, dict):
                            st.caption(f"📊 Bin: {bin_info.get('state', '—')} — departures {bin_info.get('departures_count', '—')}")
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

    # Manual inputs aligned with new backend contract
    # Row 1 - Basic flight info
    c1, c2, c3 = st.columns(3)
    with c1:
        m_direction = st.selectbox("Flight Direction", ["D", "A"], index=0, key="m_direction")
    with c2:
        m_aircraft_icao = st.selectbox("Aircraft Type (ICAO)", AIRCRAFT, key="m_aircraft_icao")
    with c3:
        m_airport = st.text_input("Airport", value="DXB", key="m_airport")

    # Row 2 - Location details
    c4, c5, c6 = st.columns(3)
    with c4:
        m_terminal = st.text_input("Terminal", value="T1", key="m_terminal")
    with c5:
        m_concourse = st.text_input("Concourse", value="UNKNOWN", key="m_concourse")
    with c6:
        m_service_type = st.text_input("Service Type", value="UNKNOWN", key="m_service_type")

    # Row 3 - Additional details
    c7, c8, c9 = st.columns(3)
    with c7:
        m_flight_nature = st.text_input("Flight Nature", value="UNKNOWN", key="m_flight_nature")
    with c8:
        m_stand = st.text_input("Stand/Gate", value="", key="m_stand")
    with c9:
        m_flight_number = st.text_input("Flight Number", value="UNKNOWN", key="m_flight_number")
    
    # Row 4 - Timing details
    c10, c11, c12 = st.columns(3)
    with c10:
        m_aobt_utc = st.text_input("AOBT (UTC ISO)", value="", key="m_aobt_utc", help="e.g., 2025-01-14T20:00:00+00:00")
    with c11:
        m_flight_schedule_time_utc = st.text_input("Flight Schedule Time (UTC)", value="", key="m_flight_schedule_time_utc")
    with c12:
        m_sobt_utc = st.text_input("SOBT (UTC)", value="", key="m_sobt_utc")
    
    # Row 5 - Additional aircraft info
    c13, c14, c15 = st.columns(3)
    with c13:
        m_aircraft_iata = st.text_input("Aircraft Type (IATA)", value="", key="m_aircraft_iata")
    with c14:
        m_destination_iata = st.text_input("Destination (IATA)", value="", key="m_destination_iata")
    with c15:
        m_aircraft_registration = st.text_input("Aircraft Registration", value="", key="m_aircraft_registration")
    
    # Row 6 - Additional timing
    c16, c17, c18 = st.columns(3)
    with c16:
        m_sibt_utc = st.text_input("SIBT (UTC)", value="", key="m_sibt_utc")
    with c17:
        st.empty()  # Empty column for spacing
    with c18:
        st.empty()  # Empty column for spacing

    can_predict_manual = all([
        m_direction,
        m_aobt_utc,
        m_aircraft_icao,
        m_terminal,
    ])
    predict_manual = st.button("Predict", disabled=not can_predict_manual)

    if predict_manual:
        payload = {
            "flight_direction": m_direction,
            "actual_offblock_time_aobt_utc": m_aobt_utc,
            "aircraft_type_icao": m_aircraft_icao,
            "airport": m_airport,
            "terminal": m_terminal,
            "concourse": m_concourse or "UNKNOWN",
            "service_type": m_service_type or "UNKNOWN",
            "flight_nature": m_flight_nature or "UNKNOWN",
            "stand": m_stand,
            "flight_number": m_flight_number or "UNKNOWN",
            "aircraft_type_iata": m_aircraft_iata or "",
            "destination_iata": m_destination_iata or "",
            "aircraft_registration": m_aircraft_registration or "",
            "flight_schedule_time_utc": m_flight_schedule_time_utc or "",
            "scheduled_offblock_time_sobt_utc": m_sobt_utc or "",
            "scheduled_inblock_time_sibt_utc": m_sibt_utc or "",
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
                st.subheader("Prediction Results")
                
                # Main prediction metrics
                col1, col2 = st.columns(2)
                with col1:
                    predicted = result.get('prediction_min', '—')
                    st.metric(
                        "Predicted Taxi-Out", 
                        f"{predicted} min",
                        help="Model's prediction for taxi-out duration"
                    )
                with col2:
                    actual_min = result.get('actual_min')
                    if actual_min is not None:
                        st.metric(
                            "Actual Taxi-Out", 
                            f"{actual_min} min",
                            help="Observed taxi-out duration for this exact scenario"
                        )
                    else:
                        st.metric(
                            "Actual Taxi-Out", 
                            "No matching scenario found",
                            help="No historical scenario was found to compare against"
                        )
                
                # Additional metrics
                col3, col4 = st.columns(2)
                with col3:
                    percentile = result.get('percentile_vs_week', '—')
                    st.metric(
                        "Percentile vs Week", 
                        f"{percentile}",
                        help="How this prediction compares to the week's distribution"
                    )
                with col4:
                    error_val = result.get('prediction_error')
                    if error_val is None and predicted != '—' and actual_min is not None:
                        try:
                            error_val = float(predicted) - float(actual_min)
                        except (ValueError, TypeError):
                            error_val = None
                    if isinstance(error_val, (int, float)):
                        diff_color = "normal" if abs(error_val) <= 5 else "inverse"
                        st.metric(
                            "Prediction Error", 
                            f"{error_val:+.1f} min",
                            delta_color=diff_color,
                            help="Predicted minus actual (positive = over-prediction)"
                        )
                    else:
                        st.metric("Prediction Error", "—")
                
                # Bin information (robust to None)
                bin_info = result.get("bin_info")
                if isinstance(bin_info, dict):
                    st.caption(f"📊 Bin: {bin_info.get('state', '—')} — departures {bin_info.get('departures_count', '—')}")
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
