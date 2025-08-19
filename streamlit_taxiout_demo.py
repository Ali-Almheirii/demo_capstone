
# streamlit_taxiout_demo.py
# --------------------------------------------------------------
# Taxi‑Out Prediction Demo (Option 2: Sliders + Distribution Overlay)
# --------------------------------------------------------------
# - Mock model: computes a deterministic prediction from user inputs
# - Histogram of actual taxi-out times (upload CSV or use synthetic sample)
# - Overlays a vertical line for the predicted value + percentile annotation
#
# How to run:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run streamlit_taxiout_demo.py
#
# Optional data format (CSV):
#   Must contain a column named: taxi_out_minutes
# --------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Taxi‑Out Prediction Demo", layout="wide")

# -----------------------------
# Utilities
# -----------------------------

RUNWAYS = ["05L", "05R", "13L", "13R", "31L", "31R"]
TERMINALS = ["T1", "T2", "T3", "Cargo"]
STANDS = [f"S{i}" for i in range(1, 41)]
AIRCRAFT = ["A320", "B738", "A321", "A333", "A359", "B77W"]

def idx(arr, val):
    return arr.index(val) if val in arr else 0

def synthetic_distribution(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    # Core mass around 17 ± 3 (clipped at 5)
    core = np.clip(rng.normal(17, 3, int(n * 0.85)), 5, None)
    # Long right tail via Gamma to mimic congestion events
    tail = rng.gamma(shape=3.0, scale=3.0, size=int(n * 0.15)) + 20
    data = np.concatenate([core, tail])
    return pd.DataFrame({"taxi_out_minutes": data})

def percentile_of_value(values: np.ndarray, x: float) -> float:
    # Compute percentile rank of x among values (0-100).
    s = np.sort(values)
    # right-side rank (<= x)
    rank = np.searchsorted(s, x, side="right")
    return (rank / len(s)) * 100.0

def mock_predict(traffic_runway, traffic_terminal, roc_runway, roc_terminal,
                 std_runway, hour, runway, terminal, stand, aircraft):
    # Generate "true" taxi-out time based on inputs
    stand_idx = idx(STANDS, stand)
    runway_idx = idx(RUNWAYS, runway)
    terminal_idx = idx(TERMINALS, terminal)
    aircraft_idx = idx(AIRCRAFT, aircraft)

    # Base taxi-out time
    true_time = 10.0
    
    # Dynamic congestion effects
    true_time += 0.08 * traffic_runway + 0.05 * traffic_terminal
    true_time += 4.0 * roc_runway + 2.0 * roc_terminal
    true_time += 0.04 * std_runway

    # Spatial / categorical effects
    true_time += 0.06 * stand_idx + 0.40 * terminal_idx + 0.30 * runway_idx + 0.20 * aircraft_idx

    # Rush hour bump
    rush = math.sin((hour % 24) / 24 * math.pi * 2)
    true_time += 1.2 * max(0.0, rush)
    
    # Ensure minimum value
    true_time = max(5.0, true_time)
    
    # Generate prediction with controlled error (MAE=2.7, RMSE=4.0)
    # Using normal distribution with mean=0 and std that gives RMSE=4.0
    # For normal distribution: RMSE = std, MAE ≈ 0.8 * std
    # So std ≈ 4.0, which gives MAE ≈ 3.2, but we want MAE=2.7
    # We'll use a mixture to get closer to the target MAE
    
    rng = np.random.default_rng(hash(f"{traffic_runway}{traffic_terminal}{hour}{runway}{terminal}{stand}{aircraft}") % 10000)
    
    # Generate error with target characteristics
    # Using a mixture of normal distributions to achieve MAE=2.7, RMSE=4.0
    if rng.random() < 0.7:
        # 70% of the time: smaller errors (normal distribution)
        error = rng.normal(0, 2.5)
    else:
        # 30% of the time: larger errors (normal distribution)
        error = rng.normal(0, 6.0)
    
    # Add some bias based on conditions to make it more realistic
    if traffic_runway > 100:
        error += 0.5  # Slight overprediction under high traffic
    if hour in [7, 8, 17, 18]:  # Rush hours
        error -= 0.3  # Slight underprediction during rush hours
    
    prediction = true_time + error
    
    # Ensure prediction is reasonable
    prediction = max(3.0, min(60.0, prediction))
    
    return round(true_time, 1), round(prediction, 1)

# -----------------------------
# Sidebar: data source
# -----------------------------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV with column 'taxi_out_minutes' (optional)")
    seed = st.number_input("Synthetic seed", 0, 10000, 42, step=1)
    n_samples = st.slider("Synthetic sample size", 1000, 10000, 5000, step=500)
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            assert "taxi_out_minutes" in df.columns
            data_source = "Uploaded CSV"
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = synthetic_distribution(n=n_samples, seed=seed)
            data_source = "Synthetic (fallback)"
    else:
        df = synthetic_distribution(n=n_samples, seed=seed)
        data_source = "Synthetic"

    st.caption(f"Using data source: **{data_source}**")
    st.markdown("---")
    st.header("Histogram Settings")
    bins = st.slider("Bins", 20, 120, 60, step=5)
    density = st.checkbox("Show density (normalized)", value=True)

# -----------------------------
# Main layout
# -----------------------------
st.title("Taxi‑Out Prediction Demo")
st.caption("Option 2: Keep sliders for feature inputs and overlay the predicted taxi‑out on the actual distribution.")

col_left, col_right = st.columns((1, 1))

with col_left:
    st.subheader("Inputs (mock model)")

    c1, c2 = st.columns(2)
    with c1:
        traffic_runway = st.slider("Traffic to Same Runway (30m)", 0, 200, 40, step=1)
        roc_runway = st.slider("Rate of Change – Runway (per min)", -0.20, 0.20, 0.05, step=0.005)
        std_runway = st.slider("Traffic Std – Runway (30m)", 0, 60, 10, step=1)
        hour = st.slider("Hour of Day", 0, 23, 15, step=1)
    with c2:
        traffic_terminal = st.slider("Traffic in Terminal (30m)", 0, 300, 55, step=1)
        roc_terminal = st.slider("Rate of Change – Terminal (per min)", -0.20, 0.20, 0.02, step=0.005)

        runway = st.selectbox("Runway", RUNWAYS, index=0)
        terminal = st.selectbox("Terminal", TERMINALS, index=0)
        stand = st.selectbox("Stand/Gate", STANDS, index=0)
        aircraft = st.selectbox("Aircraft Type", AIRCRAFT, index=0)

    actual, pred = mock_predict(
        traffic_runway, traffic_terminal, roc_runway, roc_terminal,
        std_runway, hour, runway, terminal, stand, aircraft
    )

    # Summary metrics
    real_vals = df["taxi_out_minutes"].values
    pct_actual = percentile_of_value(real_vals, actual)
    pct_pred = percentile_of_value(real_vals, pred)
    mean, median = float(np.mean(real_vals)), float(np.median(real_vals))
    p90 = float(np.percentile(real_vals, 90))
    p95 = float(np.percentile(real_vals, 95))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual (min)", f"{actual:.1f}")
    m2.metric("Prediction (min)", f"{pred:.1f}")
    m3.metric("Error (min)", f"{pred - actual:.1f}")
    m4.metric("Mean (min)", f"{mean:.1f}")

    st.markdown("**Notes**")
    st.write(
        "- The prediction is mocked for demo purposes. Swap `mock_predict()` with your model API.\\n"
        "- The vertical line on the histogram shows where the prediction lies within real outcomes.\\n"
        "- Percentile is computed against the current dataset (uploaded or synthetic)."
    )

with col_right:
    st.subheader("Distribution of Actual Taxi‑Out Times")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["taxi_out_minutes"].values, bins=bins, density=density)
    
    # Plot both actual and predicted values
    ax.axvline(actual, color='green', linestyle='-', linewidth=2, label='Actual')
    ax.axvline(pred, color='red', linestyle='--', linewidth=2, label='Prediction')
    
    ax.set_xlabel("Taxi‑Out Time (min)")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Actual Distribution with Actual and Predicted Values")

    # Annotate both values
    ymax = ax.get_ylim()[1]
    ax.text(actual, ymax*0.9, f"Actual: {actual:.1f} min\\n{pct_actual:.1f}th pct", 
            rotation=90, va="top", ha="left", color='green', fontweight='bold')
    ax.text(pred, ymax*0.75, f"Pred: {pred:.1f} min\\n{pct_pred:.1f}th pct", 
            rotation=90, va="top", ha="left", color='red', fontweight='bold')

    # Optional reference lines
    ax.axvline(p90, linestyle=":", alpha=0.5)
    ax.axvline(p95, linestyle=":", alpha=0.5)
    
    ax.legend()

    st.pyplot(fig)



st.markdown("---")
st.markdown("### Next Steps") 
st.write(
    "> - Replace `mock_predict()` with a call to your real model (XGBoost/CatBoost/NN).\\n"
    "> - Use the same feature preprocessing pipeline for both training and demo to avoid drift.\\n"
    "> - (Optional) Add SHAP panel to explain single‑prediction contributions after you wire in the model."
)
