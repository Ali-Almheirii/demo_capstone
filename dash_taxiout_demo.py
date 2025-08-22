# dash_taxiout_demo.py
# --------------------------------------------------------------
# Taxi‑Out Prediction Demo (Dash Implementation)
# --------------------------------------------------------------
# - Same functionality as Streamlit version but with better state management
# - Fixed sidebar with distribution chart
# - Instant updates without rerun issues
#
# How to run:
#   pip install dash plotly
#   python dash_taxiout_demo.py
# --------------------------------------------------------------

import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests

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

# -----------------------------
# API Functions
# -----------------------------

def fetch_distribution(api_base: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"{api_base.rstrip('/')}/distribution"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None

def fetch_day_overview(api_base: str, date: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        url = f"{api_base.rstrip('/')}/traffic/day"
        resp = requests.get(url, params={"date": date}, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        elif resp.status_code == 404:
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

# -----------------------------
# Chart Functions
# -----------------------------

def create_distribution_chart(dist_data: Optional[Dict], predicted: Optional[float] = None, actual: Optional[float] = None) -> go.Figure:
    """Create distribution chart with optional prediction markers"""
    fig = go.Figure()
    
    if not dist_data:
        fig.add_annotation(
            text="Distribution data not available<br>Ensure backend is running",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            xaxis_title="Taxi-Out Time (minutes)",
            yaxis_title="Number of Flights",
            height=450
        )
        return fig
    
    stats = dist_data.get("distribution_stats", {})
    mean = stats.get("mean", 0)
    std = stats.get("std", 1)
    sample_size = stats.get("sample_size", 1000)
    
    # Create distribution curve
    x = np.linspace(0, 140, 200)
    y = (sample_size / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # Main distribution area - use solid line with fill like Streamlit version
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.6)',
        line=dict(color='#2E7D32', width=2),
        name='Distribution',
        hovertemplate='Time: %{x:.1f} min<br>Flights: %{y:.0f}<extra></extra>'
    ))
    
    # Mean line - solid grey line like Streamlit version
    if mean > 0:
        fig.add_vline(
            x=mean,
            line_dash="solid",
            line_color="#666666",
            line_width=3,
            annotation_text=f"Mean: {mean:.1f}",
            annotation_position="top"
        )
    
    # Prediction markers with different line styles and vertical offset
    if predicted is not None and predicted >= 0:
        # Determine vertical position based on proximity to actual and mean
        if actual is not None and actual >= 0 and abs(predicted - actual) < 3:
            # Lines are close - offset vertically
            annotation_position = "top left"
        elif mean > 0 and abs(predicted - mean) < 5:
            # Close to mean - offset to avoid overlap
            annotation_position = "top right"
        else:
            annotation_position = "top"
            
        fig.add_vline(
            x=predicted,
            line_dash="dash",  # Dashed line for predicted
            line_color="#1976D2",
            line_width=3,  # Thicker line
            annotation_text=f"<b>Predicted: {predicted:.1f}</b>",  # Bold annotation
            annotation_position=annotation_position,
            annotation=dict(font=dict(size=14))  # Bigger text size
        )
    
    if actual is not None and actual >= 0:
        # Determine vertical position based on proximity to predicted and mean
        if predicted is not None and predicted >= 0 and abs(predicted - actual) < 3:
            # Lines are close - offset vertically
            annotation_position = "bottom right"
        elif mean > 0 and abs(actual - mean) < 5:
            # Close to mean - offset to avoid overlap
            annotation_position = "bottom left"
        else:
            annotation_position = "bottom"
            
        fig.add_vline(
            x=actual,
            line_dash="dot",  # Dotted line for actual
            line_color="#D32F2F",
            line_width=3,  # Thicker line
            annotation_text=f"<b>Actual: {actual:.1f}</b>",  # Bold annotation
            annotation_position=annotation_position,
            annotation=dict(font=dict(size=14))  # Bigger text size
        )
    
    fig.update_layout(
        title=dict(
            text="Taxi-Out Time Distribution for DXB",
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        xaxis_title=dict(
            text="Taxi-Out Time (minutes)",
            font=dict(size=14, color='#495057')
        ),
        yaxis_title=dict(
            text="Number of Flights",
            font=dict(size=14, color='#495057')
        ),
        height=300,
        showlegend=False,
        xaxis=dict(
            range=[0, 140], 
            gridcolor='#e9ecef', 
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#dee2e6',
            linewidth=1
        ),
        yaxis=dict(
            gridcolor='#e9ecef', 
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#dee2e6',
            linewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=20),
        font=dict(family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"),
        autosize=True
    )
    
    return fig

def create_day_chart(day_data: Optional[Dict], states_filter: List[str] = None) -> go.Figure:
    """Create daily traffic curve chart"""
    fig = go.Figure()
    
    if not day_data:
        fig.add_annotation(
            text="No day data loaded",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    bins = day_data.get("bins", [])
    if not bins:
        return fig
    
    df = pd.DataFrame(bins)
    df["bin_end"] = pd.to_datetime(df["bin_end"])
    
    # Apply state filter
    allowed_states = ["Low", "Medium", "High"]
    if states_filter:
        df = df[df["state"].isin([s for s in states_filter if s in allowed_states])]
    else:
        df = df[df["state"].isin(allowed_states)]
    
    if df.empty:
        fig.add_annotation(
            text="No data to display - adjust state filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Sort by time
    df = df.sort_values('bin_end').reset_index(drop=True)
    
    # Extract hours for x-axis
    if df['bin_end'].dt.tz is not None:
        local_times = df['bin_end'].dt.tz_localize(None)
    else:
        local_times = df['bin_end']
    
    hours = local_times.dt.hour + (local_times.dt.minute / 60.0)
    
    # Create curve - make it less bold like Streamlit version
    fig.add_trace(go.Scatter(
        x=hours,
        y=df['departures_count'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
        line=dict(color='#2E86AB', width=1.5),  # Reduced width from 2 to 1.5
        name='Traffic Flow',
        hovertemplate='Hour: %{x:.1f}<br>Departures: %{y}<extra></extra>'
    ))
    
    # Mark lowest, medium, highest points
    if len(df) >= 3:
        sorted_by_traffic = df.sort_values('departures_count')
        lowest_point = sorted_by_traffic.iloc[0]
        highest_point = sorted_by_traffic.iloc[-1]
        medium_point = sorted_by_traffic.iloc[len(sorted_by_traffic)//2]
        
        for point, color, label in [
            (lowest_point, '#28A745', 'Lowest'),
            (medium_point, '#FFC107', 'Medium'),
            (highest_point, '#DC3545', 'Highest')
        ]:
            point_time = point['bin_end']
            if point_time.tz is not None:
                point_time = point_time.tz_localize(None)
            point_hour = point_time.hour + (point_time.minute / 60.0)
            
            fig.add_trace(go.Scatter(
                x=[point_hour],
                y=[point['departures_count']],
                mode='markers',
                marker=dict(color=color, size=12, line=dict(color='white', width=2)),
                name=f"{label} ({int(point['departures_count'])})",
                hovertemplate=f'{label}<br>Time: {point_time.strftime("%H:%M")}<br>Departures: {point["departures_count"]}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text="Daily Traffic Pattern",
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        xaxis_title=dict(
            text="Hour of Day",
            font=dict(size=14, color='#495057')
        ),
        yaxis_title=dict(
            text="Departures (10-min bin)",
            font=dict(size=14, color='#495057')
        ),
        height=400,
        xaxis=dict(
            range=[0, 24], 
            tickmode='linear', 
            tick0=0, 
            dtick=2, 
            gridcolor='#e9ecef', 
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#dee2e6',
            linewidth=1
        ),
        yaxis=dict(
            gridcolor='#e9ecef', 
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#dee2e6',
            linewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60),
        font=dict(family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif")
    )
    
    return fig

# -----------------------------
# App Layout
# -----------------------------

app = dash.Dash(__name__)

# Custom CSS for modern styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Global styles */
            * {
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #ffffff;
                color: #333333;
                line-height: 1.6;
            }
            
            /* Sidebar styling */
            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                width: 33.33%;
                height: 100vh;
                overflow-y: auto;
                background-color: #f0f2f6;
                padding: 1.5rem;
                border-right: 1px solid #d1d3d8;
                z-index: 1000;
                box-shadow: 2px 0 8px rgba(0,0,0,0.12);
            }
            
            /* Main content styling */
            .main-content {
                margin-left: 33.33%;
                padding: 2rem;
                background-color: #ffffff;
                min-height: 100vh;
            }
            
            /* Headers */
            h1 {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1a1a1a;
                margin-bottom: 2rem;
                margin-top: 0;
            }
            
            h3 {
                font-size: 1.5rem;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 1rem;
                margin-top: 0;
            }
            
            h4 {
                font-size: 1.25rem;
                font-weight: 600;
                color: #34495e;
                margin-bottom: 0.75rem;
                margin-top: 0;
            }
            
            h5 {
                font-size: 1.1rem;
                font-weight: 600;
                color: #34495e;
                margin-bottom: 0.5rem;
                margin-top: 0;
            }
            
            /* Metric cards */
            .metric-card {
                background: white;
                padding: 1.25rem;
                border-radius: 0.75rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                margin: 0.75rem 0;
                border: 1px solid #e9ecef;
                transition: box-shadow 0.2s ease;
            }
            
            .metric-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            }
            
            .metric-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: #1f77b4;
                margin-bottom: 0.25rem;
            }
            
            .metric-label {
                font-size: 0.9rem;
                color: #6c757d;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
                         /* Buttons */
             button {
                 background-color: #ff4b4b;
                 color: white;
                 border: none;
                 border-radius: 0.5rem;
                 padding: 0.75rem 1.5rem;
                 font-size: 0.95rem;
                 font-weight: 600;
                 cursor: pointer;
                 transition: all 0.2s ease;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
             }
             
             button:hover {
                 background-color: #e63939;
                 transform: translateY(-1px);
                 box-shadow: 0 4px 8px rgba(0,0,0,0.15);
             }
             
             /* Load day overview button - same blue as stats */
             #load-day-btn {
                 background-color: #1f77b4 !important;
             }
             
             #load-day-btn:hover {
                 background-color: #1a6aa3 !important;
             }
            
            button:disabled {
                background-color: #adb5bd;
                color: #6c757d;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
                         /* Input fields */
             input, select {
                 width: 100%;
                 padding: 0.75rem;
                 border: 2px solid #e9ecef;
                 border-radius: 0.5rem;
                 font-size: 0.95rem;
                 transition: border-color 0.2s ease;
                 background-color: #ffffff;
                 height: 45px;
                 box-sizing: border-box;
                 vertical-align: top;
             }
            
            input:focus, select:focus {
                outline: none;
                border-color: #007bff;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
            }
            
                         /* Labels */
             label {
                 font-weight: 600;
                 color: #495057;
                 margin-bottom: 0.5rem;
                 display: block;
                 font-size: 0.9rem;
             }
             
             /* Form field containers for consistent alignment */
             .form-field-container {
                 display: flex;
                 flex-direction: column;
                 height: 100%;
             }
             
             .form-field-container > div {
                 flex: 1;
                 display: flex;
                 flex-direction: column;
             }
            
                         /* Dropdowns */
             .Select-control {
                 border: 2px solid #e9ecef !important;
                 border-radius: 0.5rem !important;
                 min-height: 45px !important;
                 height: 45px !important;
             }
             
             .Select-control .Select-value {
                 line-height: 43px !important;
             }
            
            .Select-control:hover {
                border-color: #007bff !important;
            }
            
            .Select-control--is-focused {
                border-color: #007bff !important;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
            }
            
            /* Tabs */
            .dash-tab {
                background-color: #f8f9fa !important;
                border: none !important;
                border-radius: 0.5rem 0.5rem 0 0 !important;
                padding: 0.6rem 1.5rem !important;
                font-weight: 600 !important;
                color: #6c757d !important;
                transition: all 0.2s ease !important;
            }
            
            .dash-tab--selected {
                background-color: #ffffff !important;
                color: #ff4b4b !important;
                border-bottom: 3px solid #ff4b4b !important;
            }
            
            .dash-tab:hover {
                background-color: #e9ecef !important;
                color: #495057 !important;
            }
            
                         /* Date picker */
             .DateInput_input {
                 padding: 0.75rem !important;
                 border: 2px solid #e9ecef !important;
                 border-radius: 0.5rem !important;
                 font-size: 0.95rem !important;
                 height: 45px !important;
                 box-sizing: border-box !important;
             }
             
             .DateInput_input:focus {
                 border-color: #007bff !important;
                 box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
             }
             
             /* Date picker container */
             .DateInput {
                 width: 100% !important;
             }
             
             /* Date picker calendar icon */
             .DateInput_input__focused {
                 border-color: #007bff !important;
             }
            
            /* Status messages */
            .status-success {
                background-color: #d4edda;
                color: #155724;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #c3e6cb;
                margin: 1rem 0;
            }
            
            .status-error {
                background-color: #f8d7da;
                color: #721c24;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #f5c6cb;
                margin: 1rem 0;
            }
            
            /* Traffic buttons */
            .traffic-btn-lowest {
                background-color: #28a745 !important;
                margin-right: 0.75rem !important;
            }
            
            .traffic-btn-lowest:hover {
                background-color: #218838 !important;
            }
            
            .traffic-btn-medium {
                background-color: #ffc107 !important;
                color: #212529 !important;
                margin-right: 0.75rem !important;
            }
            
            .traffic-btn-medium:hover {
                background-color: #e0a800 !important;
            }
            
            .traffic-btn-highest {
                background-color: #dc3545 !important;
            }
            
            .traffic-btn-highest:hover {
                background-color: #c82333 !important;
            }
            
            /* Center traffic buttons container */
            #traffic-buttons-container {
                text-align: center !important;
            }
            
            /* JSON display */
            .json-display {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 0.5rem;
                border: 1px solid #e9ecef;
                font-family: 'Courier New', monospace;
                font-size: 0.85rem;
                line-height: 1.4;
                overflow-x: auto;
            }
            
            /* Horizontal rule */
            hr {
                border: none;
                height: 1px;
                background-color: #dee2e6;
                margin: 2rem 0;
            }
            
            /* Distribution chart specific styling */
            #distribution-chart {
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }
            
            #distribution-chart > div {
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }
            
            #distribution-stats {
                margin-top: 10px !important;
                padding-top: 0 !important;
            }
            
            /* Target Plotly chart container specifically */
            .js-plotly-plot {
                margin-bottom: 0 !important;
            }
            
            .plotly {
                margin-bottom: 0 !important;
            }
            
            /* Make stats more compact */
            .metric-card {
                padding: 0.75rem !important;
                margin: 0.25rem 0 !important;
            }
            
            .metric-value {
                font-size: 1.4rem !important;
                margin-bottom: 0.1rem !important;
            }
            
            .metric-label {
                font-size: 0.8rem !important;
            }
            
            /* Force chart container sizing */
            .sidebar .js-plotly-plot {
                width: 100% !important;
                height: 320px !important;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .sidebar {
                    width: 100%;
                    position: relative;
                    height: auto;
                }
                .main-content {
                    margin-left: 0;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Sidebar layout
sidebar = html.Div([
    html.H3("Taxi-Out Distribution", style={'margin-bottom': '0.5rem'}),
    dcc.Graph(id='distribution-chart', 
              style={'margin-bottom': '0rem', 'padding-bottom': '0rem', 'width': '100%', 'height': '320px'},
              config={'displayModeBar': False, 'staticPlot': False, 'responsive': True}),
    html.Div(id='distribution-stats', style={'margin-top': '0.75rem', 'padding-top': '0rem'}),
    
    # Recent Predictions
    html.Div(id='prediction-history', style={'margin-top': '1rem'})
], className='sidebar')

# Main content layout
main_content = html.Div([
    html.H1("Taxi‑Out Prediction Demo"),
    
    dcc.Tabs(id='main-tabs', value='preset-week', children=[
                 dcc.Tab(label='Preset Day', value='preset-week', children=[
            html.Div([
                html.H3("Daily Overview"),
                html.Div([
                    html.Div([
                        html.Label("Select Date (local)"),
                        dcc.DatePickerSingle(
                            id='date-picker',
                            date=datetime.now().date(),
                            display_format='YYYY-MM-DD'
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    html.Div([
                        html.Button('Load day overview', id='load-day-btn')
                    ], style={'width': '45%', 'display': 'inline-block', 'margin-left': '10%', 'margin-top': '1.5rem'})
                ], style={'margin-bottom': '1.5rem', 'display': 'flex', 'alignItems': 'flex-end'}),
                
                html.Div(id='day-load-status'),
                
                                 # Day content area - elements controlled by callbacks
                 html.Div([
                     html.H4("Traffic Timeline", id='traffic-timeline-heading', style={'display': 'none'}),
                     html.Div([
                         html.Label("Filter states:"),
                         dcc.Dropdown(
                             id='states-filter',
                             options=[{'label': s, 'value': s} for s in ['Low', 'Medium', 'High']],
                             value=['Low', 'Medium', 'High'],
                             multi=True
                         )
                     ], id='states-filter-container', style={'display': 'none'}),
                    dcc.Graph(id='day-chart'),
                    
                    html.Div([
                        html.H4("🎯 Select Traffic Level:"),
                        html.Div([
                            html.Button('🟢 Lowest Traffic', id='select-lowest-btn', 
                                      className='traffic-btn-lowest'),
                            html.Button('🟡 Medium Traffic', id='select-medium-btn',
                                      className='traffic-btn-medium'),
                            html.Button('🔴 Highest Traffic', id='select-highest-btn',
                                      className='traffic-btn-highest')
                        ])
                    ], id='traffic-buttons-container', style={'display': 'none', 'margin-bottom': '1rem'}),
                    
                    html.Div(id='bin-selection-status'),
                    
                    html.Hr(),
                    html.H4("Core Inputs"),
                    html.Div(id='preset-core-inputs'),
                    
                    html.Button('Predict for selected bin', id='preset-predict-btn', 
                              disabled=True),
                    
                    html.Div(id='preset-results')
                ], id='day-content')
            ])
        ]),
        
        dcc.Tab(label='Manual Scenario', value='manual', children=[
            html.Div([
                html.H3("Manual Scenario (core inputs only)"),
                html.Div(id='manual-core-inputs'),
                html.Button('Predict', id='manual-predict-btn',
                          disabled=True),
                html.Div(id='manual-results')
            ])
        ]),
        
        dcc.Tab(label='Config', value='config', children=[
            html.Div([
                html.H3("Configuration"),
                html.Label("Backend URL:"),
                dcc.Input(
                    id='api-base-input',
                    value=DEFAULT_API_BASE,
                    type='text',
                    style={'width': '100%', 'padding': '0.5rem', 'margin': '0.5rem 0'}
                )
            ])
        ])
    ])
], className='main-content')

# Complete app layout
app.layout = html.Div([
    sidebar,
    main_content,
    
    # Data stores
    dcc.Store(id='distribution-data'),
    dcc.Store(id='day-data'),
    dcc.Store(id='api-base', data=DEFAULT_API_BASE),
    dcc.Store(id='predicted-min'),
    dcc.Store(id='actual-min'),
    dcc.Store(id='preset-form-data'),
    dcc.Store(id='manual-form-data'),
    dcc.Store(id='prediction-history-data', data=[]),  # Store for prediction history
    dcc.Store(id='current-prediction-data')  # Store for current prediction details
])

# -----------------------------
# Callbacks
# -----------------------------

# Load distribution data on app start
@app.callback(
    Output('distribution-data', 'data'),
    Input('api-base', 'data')
)
def load_distribution_data(api_base):
    return fetch_distribution(api_base)

# Update distribution chart
@app.callback(
    [Output('distribution-chart', 'figure'),
     Output('distribution-stats', 'children')],
    [Input('distribution-data', 'data'),
     Input('predicted-min', 'data'),
     Input('actual-min', 'data')]
)
def update_distribution_chart(dist_data, predicted, actual):
    fig = create_distribution_chart(dist_data, predicted, actual)
    
    stats_children = []
    if dist_data:
        stats = dist_data.get("distribution_stats", {})
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        percentiles = stats.get("percentiles", {})
        sample_size = stats.get("sample_size", 0)
        
        stats_children = [
            html.Div([
                html.Div([
                    html.H5("📈 Central Tendency", style={'margin-top': '0rem', 'margin-bottom': '0.5rem', 'font-size': '1rem'}),
                    html.Div([
                        html.Div(f"{mean:.1f} min", className='metric-value'),
                        html.Div("Mean", className='metric-label')
                    ], className='metric-card'),
                    html.Div([
                        html.Div(f"{percentiles.get('50th', 'N/A')} min" if percentiles and '50th' in percentiles else "N/A", className='metric-value'),
                        html.Div("Median", className='metric-label')
                    ], className='metric-card')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H5("📏 Variability", style={'margin-top': '0rem', 'margin-bottom': '0.5rem', 'font-size': '1rem'}),
                    html.Div([
                        html.Div(f"{std:.1f} min", className='metric-value'),
                        html.Div("Standard Deviation", className='metric-label')
                    ], className='metric-card'),
                    html.Div([
                        html.Div(f"{sample_size:,}", className='metric-value'),
                        html.Div("Sample Size", className='metric-label')
                    ], className='metric-card')
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
            ], style={'margin-top': '0rem', 'padding-top': '0rem'})
        ]
    else:
        stats_children = [html.P("Distribution data not available", style={'color': 'gray', 'margin-top': '1rem'})]
    
    return fig, stats_children

# Load day data and control visibility of day-related elements
@app.callback(
    [Output('day-data', 'data'),
     Output('day-load-status', 'children'),
     Output('states-filter-container', 'style'),
     Output('traffic-buttons-container', 'style'),
     Output('traffic-timeline-heading', 'style')],
    [Input('load-day-btn', 'n_clicks')],
    [State('date-picker', 'date'),
     State('api-base', 'data')]
)
def load_day_data(n_clicks, date, api_base):
    if not n_clicks:
        return None, "", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    day_data, error_msg = fetch_day_overview(api_base, str(date))
    
    if day_data:
        status = html.Div("✅ Day data loaded successfully", className='status-success')
        filter_style = {'display': 'block', 'margin-bottom': '1rem'}
        buttons_style = {'display': 'block', 'margin-bottom': '1rem'}
        heading_style = {'display': 'block'}
    else:
        status = html.Div(f"❌ Failed to load day data: {error_msg}", className='status-error')
        filter_style = {'display': 'none'}
        buttons_style = {'display': 'none'}
        heading_style = {'display': 'none'}
    
    return day_data, status, filter_style, buttons_style, heading_style

# Update day chart - only show when data is loaded
@app.callback(
    [Output('day-chart', 'figure'),
     Output('day-chart', 'style')],
    [Input('day-data', 'data'),
     Input('states-filter', 'value')]
)
def update_day_chart(day_data, states_filter):
    if not day_data:
        # Return empty figure and hide the chart
        fig = go.Figure()
        fig.update_layout(
            height=50,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig, {'display': 'none'}
    
    # Show the chart and return actual data
    return create_day_chart(day_data, states_filter), {'display': 'block'}

# Generate core input forms
@app.callback(
    Output('preset-core-inputs', 'children'),
    Input('preset-form-data', 'data')
)
def generate_preset_core_inputs(form_data):
    if not form_data:
        form_data = {}
    
    return html.Div([
                 # Row 1
         html.Div([
             html.Div([
                 html.Label("Flight Direction"),
                 dcc.Dropdown(
                     id='pw-direction',
                     options=[{'label': 'Departure', 'value': 'D'}, {'label': 'Arrival', 'value': 'A'}],
                     value=form_data.get('flight_direction', 'D')
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Aircraft Type (ICAO)"),
                 dcc.Dropdown(
                     id='pw-aircraft-icao',
                     options=[{'label': a, 'value': a} for a in AIRCRAFT],
                     value=form_data.get('aircraft_type_icao', AIRCRAFT[0])
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Airport"),
                 dcc.Input(
                     id='pw-airport',
                     value=form_data.get('airport', 'DXB'),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
         ], style={'margin-bottom': '1rem'}),
        
                 # Row 2
         html.Div([
             html.Div([
                 html.Label("Terminal"),
                 dcc.Input(
                     id='pw-terminal',
                     value=form_data.get('terminal', 'T1'),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Stand/Gate"),
                 dcc.Input(
                     id='pw-stand',
                     value=form_data.get('stand', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Concourse"),
                 dcc.Input(
                     id='pw-concourse',
                     value=form_data.get('concourse', 'UNKNOWN'),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
         ], style={'margin-bottom': '1rem'}),
        
                 # Row 3 - Additional fields
         html.Div([
             html.Div([
                 html.Label("Service Type"),
                 dcc.Input(
                     id='pw-service-type',
                     value=form_data.get('service_type', 'UNKNOWN'),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Flight Nature"),
                 dcc.Input(
                     id='pw-flight-nature',
                     value=form_data.get('flight_nature', 'UNKNOWN'),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Flight Number"),
                 dcc.Input(
                     id='pw-flight-number',
                     value=form_data.get('flight_number', 'UNKNOWN'),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
         ], style={'margin-bottom': '1rem'}),
        
                 # Row 4 - More fields
         html.Div([
             html.Div([
                 html.Label("Aircraft Type (IATA)"),
                 dcc.Input(
                     id='pw-aircraft-iata',
                     value=form_data.get('aircraft_type_iata', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Destination (IATA)"),
                 dcc.Input(
                     id='pw-destination-iata',
                     value=form_data.get('destination_iata', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Aircraft Registration"),
                 dcc.Input(
                     id='pw-aircraft-registration',
                     value=form_data.get('aircraft_registration', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
         ], style={'margin-bottom': '1rem'}),
        
                 # Row 5 - Timestamps
         html.Div([
             html.Div([
                 html.Label("Flight Schedule Time (UTC)"),
                 dcc.Input(
                     id='pw-flight-schedule-time',
                     value=form_data.get('flight_schedule_time_utc', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Scheduled Offblock Time (UTC)"),
                 dcc.Input(
                     id='pw-scheduled-offblock',
                     value=form_data.get('scheduled_offblock_time_sobt_utc', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
             
             html.Div([
                 html.Label("Scheduled Inblock Time (UTC)"),
                 dcc.Input(
                     id='pw-scheduled-inblock',
                     value=form_data.get('scheduled_inblock_time_sibt_utc', ''),
                     type='text'
                 )
             ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
         ], style={'margin-bottom': '1rem'})
    ])

# Manual scenario prediction callback
@app.callback(
    [Output('manual-results', 'children'),
     Output('predicted-min', 'data', allow_duplicate=True),
     Output('actual-min', 'data', allow_duplicate=True),
     Output('current-prediction-data', 'data', allow_duplicate=True),
     Output('prediction-history-data', 'data', allow_duplicate=True)],
    Input('manual-predict-btn', 'n_clicks'),
    [State('m-direction', 'value'),
     State('m-aircraft-icao', 'value'),
     State('m-airport', 'value'),
     State('m-terminal', 'value'),
     State('m-stand', 'value'),
     State('m-concourse', 'value'),
     State('m-service-type', 'value'),
     State('m-flight-nature', 'value'),
     State('m-flight-number', 'value'),
     State('m-aircraft-iata', 'value'),
     State('m-destination-iata', 'value'),
     State('m-aircraft-registration', 'value'),
     State('m-flight-schedule-time', 'value'),
     State('m-scheduled-offblock', 'value'),
     State('m-scheduled-inblock', 'value'),
     State('m-aobt-utc', 'value'),
     State('prediction-history-data', 'data'),
     State('api-base', 'data')],
    prevent_initial_call=True
)
def handle_manual_prediction(n_clicks, direction, aircraft_icao, airport, terminal, stand, concourse, 
                           service_type, flight_nature, flight_number, aircraft_iata, destination_iata,
                           aircraft_registration, flight_schedule_time, scheduled_offblock, scheduled_inblock,
                           aobt_utc, history_data, api_base):
    if not n_clicks:
        return "", None, None, None, history_data
    
    # Build payload for manual scenario
    payload = {
        "flight_direction": direction or "D",
        "aircraft_type_icao": aircraft_icao or AIRCRAFT[0],
        "airport": airport or "DXB",
        "terminal": terminal or "T1",
        "stand": stand or "",
        "concourse": concourse or "UNKNOWN",
        "service_type": service_type or "UNKNOWN",
        "flight_nature": flight_nature or "UNKNOWN",
        "flight_number": flight_number or "UNKNOWN",
        "aircraft_type_iata": aircraft_iata or "",
        "destination_iata": destination_iata or "",
        "aircraft_registration": aircraft_registration or "",
        "flight_schedule_time_utc": flight_schedule_time or "",
        "scheduled_offblock_time_sobt_utc": scheduled_offblock or "",
        "scheduled_inblock_time_sibt_utc": scheduled_inblock or "",
        "actual_offblock_time_aobt_utc": aobt_utc or "2025-01-14T20:00:00+00:00"
    }
    
    result, error = post_predict(api_base, payload)
    
    if error:
        return html.Div(f"❌ Prediction failed: {error}", className='status-error'), None, None, None, history_data
    
    if not result:
        return html.Div("❌ No prediction result", className='status-error'), None, None, None, history_data
    
    predicted = result.get('prediction_min')
    actual = result.get('actual_min')
    
    # Add to prediction history
    new_prediction = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'predicted': predicted,
        'actual': actual,
        'scenario_type': 'manual',
        'aircraft': aircraft_icao or AIRCRAFT[0]
    }
    
    updated_history = (history_data or [])[-2:] + [new_prediction]  # Keep last 3
    
    # Store current prediction details
    current_prediction = {
        'timestamp': new_prediction['timestamp'],
        'predicted': predicted,
        'actual': actual,
        'scenario_type': 'manual',
        'percentile': result.get('percentile_vs_week', 'N/A')
    }
    
    results_div = html.Div([
        html.H4("Prediction Results"),
        html.Div([
            html.Div([
                html.Div(f"{predicted} min" if predicted else "—", className='metric-value'),
                html.Div("Predicted Taxi-Out", className='metric-label')
            ], className='metric-card', style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(f"{actual} min" if actual else "No match found", className='metric-value'),
                html.Div("Actual Taxi-Out", className='metric-label')
            ], className='metric-card', style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
        ]),
        
        html.H5("Engineered Features (read-only)"),
        html.Div([
            html.Pre(json.dumps(result.get("engineered_features", {}), indent=2))
        ], className='json-display')
    ])
    
    return results_div, predicted, actual, current_prediction, updated_history


@app.callback(
    Output('manual-core-inputs', 'children'),
    Input('manual-form-data', 'data')
)
def generate_manual_core_inputs(form_data):
     if not form_data:
         form_data = {}
     
     return html.Div([
                   # Row 1
          html.Div([
              html.Div([
                  html.Label("Flight Direction"),
                  dcc.Dropdown(
                      id='m-direction',
                      options=[{'label': 'Departure', 'value': 'D'}, {'label': 'Arrival', 'value': 'A'}],
                      value=form_data.get('flight_direction', 'D')
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Aircraft Type (ICAO)"),
                  dcc.Dropdown(
                      id='m-aircraft-icao',
                      options=[{'label': a, 'value': a} for a in AIRCRAFT],
                      value=form_data.get('aircraft_type_icao', AIRCRAFT[0])
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Airport"),
                  dcc.Input(
                      id='m-airport',
                      value=form_data.get('airport', 'DXB'),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
          ], style={'margin-bottom': '1rem'}),
         
                   # Row 2
          html.Div([
              html.Div([
                  html.Label("Terminal"),
                  dcc.Input(
                      id='m-terminal',
                      value=form_data.get('terminal', 'T1'),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Stand/Gate"),
                  dcc.Input(
                      id='m-stand',
                      value=form_data.get('stand', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Concourse"),
                  dcc.Input(
                      id='m-concourse',
                      value=form_data.get('concourse', 'UNKNOWN'),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
          ], style={'margin-bottom': '1rem'}),
          
          # Row 3 - Additional fields
          html.Div([
              html.Div([
                  html.Label("Service Type"),
                  dcc.Input(
                      id='m-service-type',
                      value=form_data.get('service_type', 'UNKNOWN'),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Flight Nature"),
                  dcc.Input(
                      id='m-flight-nature',
                      value=form_data.get('flight_nature', 'UNKNOWN'),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Flight Number"),
                  dcc.Input(
                      id='m-flight-number',
                      value=form_data.get('flight_number', 'UNKNOWN'),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
          ], style={'margin-bottom': '1rem'}),
          
          # Row 4 - More fields
          html.Div([
              html.Div([
                  html.Label("Aircraft Type (IATA)"),
                  dcc.Input(
                      id='m-aircraft-iata',
                      value=form_data.get('aircraft_type_iata', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Destination (IATA)"),
                  dcc.Input(
                      id='m-destination-iata',
                      value=form_data.get('destination_iata', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Aircraft Registration"),
                  dcc.Input(
                      id='m-aircraft-registration',
                      value=form_data.get('aircraft_registration', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
          ], style={'margin-bottom': '1rem'}),
          
          # Row 5 - Timestamps
          html.Div([
              html.Div([
                  html.Label("Flight Schedule Time (UTC)"),
                  dcc.Input(
                      id='m-flight-schedule-time',
                      value=form_data.get('flight_schedule_time_utc', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Scheduled Offblock Time (UTC)"),
                  dcc.Input(
                      id='m-scheduled-offblock',
                      value=form_data.get('scheduled_offblock_time_sobt_utc', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container'),
              
              html.Div([
                  html.Label("Scheduled Inblock Time (UTC)"),
                  dcc.Input(
                      id='m-scheduled-inblock',
                      value=form_data.get('scheduled_inblock_time_sibt_utc', ''),
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
          ], style={'margin-bottom': '1rem'}),
          
          # Row 6 - AOBT (Actual Offblock Time)
          html.Div([
              html.Div([
                  html.Label("Actual Offblock Time (UTC)"),
                  dcc.Input(
                      id='m-aobt-utc',
                      value=form_data.get('actual_offblock_time_aobt_utc', ''),
                      placeholder='e.g., 2025-01-14T20:00:00+00:00',
                      type='text'
                  )
              ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}, className='form-field-container')
          ], style={'margin-bottom': '1rem'})
     ])

# Handle traffic level selection (simplified for demo)
@app.callback(
    [Output('bin-selection-status', 'children'),
     Output('preset-form-data', 'data'),
     Output('preset-predict-btn', 'disabled')],
    [Input('select-lowest-btn', 'n_clicks'),
     Input('select-medium-btn', 'n_clicks'),
     Input('select-highest-btn', 'n_clicks')],
    State('day-data', 'data')
)
def handle_traffic_selection(lowest_clicks, medium_clicks, highest_clicks, day_data):
    if not day_data:
        return "", {}, True
    
    ctx = callback_context
    if not ctx.triggered:
        return "", {}, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Get bins data
    bins = day_data.get("bins", [])
    if not bins:
        return "", {}, True
    
    df = pd.DataFrame(bins)
    allowed_states = ["Low", "Medium", "High"]
    df = df[df["state"].isin(allowed_states)]
    
    if df.empty:
        return "", {}, True
    
    sorted_data = df.sort_values('departures_count')
    
    if button_id == 'select-lowest-btn':
        selected_bin = sorted_data.iloc[0]
        level = "lowest"
    elif button_id == 'select-medium-btn':
        selected_bin = sorted_data.iloc[len(sorted_data)//2]
        level = "medium"
    elif button_id == 'select-highest-btn':
        selected_bin = sorted_data.iloc[-1]
        level = "highest"
    else:
        return "", {}, True
    
    # Get core features example from selected bin
    core_example = selected_bin.get("core_features_example", {})
    
    print(f"Selected {level} bin core_features_example:", core_example)  # Debug
    
    status = html.Div(
        f"✅ Selected {level} traffic: {selected_bin['state']} at {pd.to_datetime(selected_bin['bin_end']).strftime('%H:%M')} with {selected_bin['departures_count']} departures",
        className='status-success'
    )
    
    return status, core_example, False

# Handle predictions
@app.callback(
    [Output('preset-results', 'children'),
     Output('predicted-min', 'data'),
     Output('actual-min', 'data'),
     Output('current-prediction-data', 'data'),
     Output('prediction-history-data', 'data')],
    Input('preset-predict-btn', 'n_clicks'),
    [State('pw-direction', 'value'),
     State('pw-aircraft-icao', 'value'),
     State('pw-airport', 'value'),
     State('pw-terminal', 'value'),
     State('pw-stand', 'value'),
     State('pw-concourse', 'value'),
     State('pw-service-type', 'value'),
     State('pw-flight-nature', 'value'),
     State('pw-flight-number', 'value'),
     State('pw-aircraft-iata', 'value'),
     State('pw-destination-iata', 'value'),
     State('pw-aircraft-registration', 'value'),
     State('pw-flight-schedule-time', 'value'),
     State('pw-scheduled-offblock', 'value'),
     State('pw-scheduled-inblock', 'value'),
     State('preset-form-data', 'data'),
     State('prediction-history-data', 'data'),
     State('api-base', 'data')]
)
def handle_preset_prediction(n_clicks, direction, aircraft_icao, airport, terminal, stand, concourse, 
                           service_type, flight_nature, flight_number, aircraft_iata, destination_iata,
                           aircraft_registration, flight_schedule_time, scheduled_offblock, scheduled_inblock, 
                           preset_form_data, history_data, api_base):
    if not n_clicks:
        return "", None, None, None, history_data
    
    if not preset_form_data:
        return html.Div("❌ No bin selected. Please select a traffic level first.", className='status-error'), None, None, None, history_data
    
    # Build payload from current form values (which may be modified by user)
    # This allows user to either use auto-populated values AS IS or modify them
    payload = {
        "flight_direction": direction if direction is not None else preset_form_data.get("flight_direction", "D"),
        "aircraft_type_icao": aircraft_icao if aircraft_icao is not None else preset_form_data.get("aircraft_type_icao", "A320"),
        "airport": airport if airport is not None else preset_form_data.get("airport", "DXB"),
        "terminal": terminal if terminal is not None else preset_form_data.get("terminal", "T1"),
        "stand": stand if stand is not None else preset_form_data.get("stand", ""),
        "concourse": concourse if concourse is not None else preset_form_data.get("concourse", "UNKNOWN"),
        "service_type": service_type if service_type is not None else preset_form_data.get("service_type", "UNKNOWN"),
        "flight_nature": flight_nature if flight_nature is not None else preset_form_data.get("flight_nature", "UNKNOWN"),
        "flight_number": flight_number if flight_number is not None else preset_form_data.get("flight_number", "UNKNOWN"),
        "aircraft_type_iata": aircraft_iata if aircraft_iata is not None else preset_form_data.get("aircraft_type_iata", ""),
        "destination_iata": destination_iata if destination_iata is not None else preset_form_data.get("destination_iata", ""),
        "aircraft_registration": aircraft_registration if aircraft_registration is not None else preset_form_data.get("aircraft_registration", ""),
        "flight_schedule_time_utc": flight_schedule_time if flight_schedule_time is not None else preset_form_data.get("flight_schedule_time_utc", ""),
        "scheduled_offblock_time_sobt_utc": scheduled_offblock if scheduled_offblock is not None else preset_form_data.get("scheduled_offblock_time_sobt_utc", ""),
        "scheduled_inblock_time_sibt_utc": scheduled_inblock if scheduled_inblock is not None else preset_form_data.get("scheduled_inblock_time_sibt_utc", ""),
        # Add the actual_offblock_time_aobt_utc from the bin data
        "actual_offblock_time_aobt_utc": preset_form_data.get("actual_offblock_time_aobt_utc", "2025-01-14T20:00:00+00:00")
    }
    
    print("Original bin data:", preset_form_data)  # Debug
    print("Final payload (form + bin):", payload)  # Debug
    
    result, error = post_predict(api_base, payload)
    
    if error:
        return html.Div(f"❌ Prediction failed: {error}", className='status-error'), None, None
    
    if not result:
        return html.Div("❌ No prediction result", className='status-error'), None, None
    
    predicted = result.get('prediction_min')
    actual = result.get('actual_min')
    
    results_div = html.Div([
        html.H4("Prediction Results"),
        html.Div([
            html.Div([
                html.Div(f"{predicted} min" if predicted else "—", className='metric-value'),
                html.Div("Predicted Taxi-Out", className='metric-label')
            ], className='metric-card', style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(f"{actual} min" if actual else "No match found", className='metric-value'),
                html.Div("Actual Taxi-Out", className='metric-label')
            ], className='metric-card', style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
        ]),
        
        html.H5("Engineered Features (read-only)"),
        html.Div([
            html.Pre(json.dumps(result.get("engineered_features", {}), indent=2))
        ], className='json-display')
    ])
    
    # Add to prediction history
    new_prediction = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'predicted': predicted,
        'actual': actual,
        'scenario_type': 'preset',
        'aircraft': aircraft_icao or preset_form_data.get("aircraft_type_icao", AIRCRAFT[0])
    }
    
    updated_history = (history_data or [])[-2:] + [new_prediction]  # Keep last 3
    
    # Store current prediction details for insights panel
    current_prediction = {
        'timestamp': new_prediction['timestamp'],
        'predicted': predicted,
        'actual': actual,
        'scenario_type': 'preset',
        'percentile': result.get('percentile_vs_week', 'N/A')
    }
    
    return results_div, predicted, actual, current_prediction, updated_history



# Display prediction history (Option 1)
@app.callback(
    Output('prediction-history', 'children'),
    Input('prediction-history-data', 'data')
)
def display_prediction_history(history_data):
    if not history_data:
        return ""  # Empty when no predictions made
    
    history_items = []
    for i, pred in enumerate(history_data[-3:]):  # Show last 3
        # Sequential numbering (most recent predictions get higher numbers)
        prediction_number = len(history_data) - 2 + i  # This ensures sequential numbering
        
        icon = "📊" if pred['scenario_type'] == 'preset' else "✏️"
        actual_text = f"{pred['actual']} min" if pred['actual'] else "No match"
        
        history_items.append(
            html.Div([
                html.Div([
                    html.Span(f"Prediction {prediction_number} • {pred['timestamp']}", style={'font-weight': '600', 'font-size': '0.85rem'}),
                    html.Br(),
                    html.Span(f"Pred: {pred['predicted']} min | Act: {actual_text}", 
                             style={'font-size': '0.9rem', 'color': '#495057'})  # Made bigger and darker
                ], style={'padding': '0.5rem', 'background': '#f8f9fa', 'border-radius': '0.25rem', 'margin-bottom': '0.25rem'})
            ])
        )
    
    return html.Div([
        html.H5("📈 Recent Predictions", style={'margin-bottom': '0.5rem', 'font-size': '1rem', 'color': '#495057'}),
        html.Div(history_items)
    ])


# Update API base
@app.callback(
    Output('api-base', 'data'),
    Input('api-base-input', 'value')
)
def update_api_base(api_base_input):
    return api_base_input or DEFAULT_API_BASE

# Clientside callback to force chart resize
app.clientside_callback(
    """
    function(figure) {
        if (figure) {
            setTimeout(function() {
                var plotElement = document.getElementById('distribution-chart');
                if (plotElement && window.Plotly) {
                    window.Plotly.Plots.resize(plotElement);
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('distribution-chart', 'style', allow_duplicate=True),
    Input('distribution-chart', 'figure'),
    prevent_initial_call=True
)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
