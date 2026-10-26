import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="UAC Program Forecasting System Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for consistent presentation layout matching the blueprint
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-card {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .error-card {
        background-color: #f8d7da;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. MOCK DATA GENERATION PIPELINE (Matches Data Engineering Schema)
# ==============================================================================
@st.cache_data
def generate_forecast_data():
    """Generates synthetic high-quality time-series data mirroring the blueprint framework."""
    np.random.seed(42)

    # Generate historical and future timeline window
    hist_dates = pd.date_range(start="2026-01-01", end="2026-10-15", freq="D")
    future_dates = pd.date_range(start="2026-10-16", end="2026-11-15", freq="D")

    # Base operational volumes with weekly seasonality patterns
    base_care_load = 9500 + np.sin(np.arange(len(hist_dates)) / 7) * 400
    noise = np.random.normal(0, 150, size=len(hist_dates))
    hist_hhs_care = (base_care_load + noise).astype(int)

    hist_transfers = np.random.randint(180, 380, size=len(hist_dates))
    hist_discharges = np.random.randint(160, 340, size=len(hist_dates))

    # Compile baseline historical DataFrame
    df_hist = pd.DataFrame(
        {
            "Date": hist_dates,
            "Children transferred out of CBP custody": hist_transfers,
            "Children in HHS Care": hist_hhs_care,
            "Children discharged from HHS Care": hist_discharges,
            "Net_System_Pressure_Signal": hist_transfers - hist_discharges,
            "Type": "Historical",
        }
    )
    df_hist.set_index("Date", inplace=True)

    # Compile Forecast Data Matrix across horizons
    forecast_records = []
    models = ["Naïve Persistence", "SARIMA (Statistical)", "Gradient Boosting"]

    # Generate multi-horizon tracks
    for model in models:
        # Scenario variations
        if model == "Naïve Persistence":
            f_care = np.full(len(future_dates), hist_hhs_care[-1])
            f_discharge = np.full(len(future_dates), hist_discharges[-1])
        elif model == "SARIMA (Statistical)":
            f_care = hist_hhs_care[-1] + np.cumsum(
                np.random.normal(5, 40, size=len(future_dates))
            )
            f_discharge = np.random.randint(200, 300, size=len(future_dates))
        else:  # Gradient Boosting
            f_care = (
                hist_hhs_care[-1]
                + np.sin(np.arange(len(future_dates)) / 5) * 500
                + np.arange(len(future_dates)) * 12
            )
            f_discharge = np.random.randint(220, 320, size=len(future_dates))

        for i, date in enumerate(future_dates):
            # Calculate dynamic confidence bounds
            variance_factor = (i + 1) * (15 if model == "SARIMA (Statistical)" else 22)
            forecast_records.append(
                {
                    "Date": date,
                    "Model": model,
                    "Care_Forecast": int(f_care[i]),
                    "Care_Lower_CI": int(f_care[i] - 1.96 * variance_factor),
                    "Care_Upper_CI": int(f_care[i] + 1.96 * variance_factor),
                    "Discharge_Demand_Forecast": int(f_discharge[i]),
                    "Net_Imbalance_Forecast": int(
                        np.random.randint(10, 80)
                        if i < 15
                        else np.random.randint(-40, 30)
                    ),
                }
            )

    df_forecast = pd.DataFrame(forecast_records)
    df_forecast.set_index("Date", inplace=True)

    return df_hist, df_forecast


df_historical, df_forecast = generate_forecast_data()

# ==============================================================================
# 3. SIDEBAR NAVIGATION & CONTROL CONFIGURATION
# ==============================================================================
st.sidebar.title("🎛️ Strategic Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("Horizon Selector")
forecast_horizon = st.sidebar.slider(
    label="Select Forecast Horizon Window (Days Ahead)",
    min_value=7,
    max_value=30,
    value=14,
    step=1,
)

st.sidebar.subheader("Model Backend Configurator")
selected_model = st.sidebar.selectbox(
    label="Primary Forecasting Model Backend",
    options=["Gradient Boosting", "SARIMA (Statistical)", "Naïve Persistence"],
)

st.sidebar.subheader("Interactive Settings")
show_confidence_intervals = st.sidebar.checkbox(
    label="Render 95% Confidence Intervals", value=True
)
capacity_threshold = st.sidebar.number_input(
    label="HHS Operational Network Capacity Limit",
    min_value=9000,
    max_value=12000,
    value=10800,
    step=100,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **System Note:** This system actively processes daily time-series pipelines to prevent institutional overcrowding risk and personnel burnout."
)

# ==============================================================================
# 4. HEADER PANEL & SYSTEM RISK INDICATORS
# ==============================================================================
st.title("📊 UAC Program Predictive Forecasting System Dashboard")
st.markdown(
    "Decision Support Infrastructure for Health and Human Services (HHS) & Border Custody Placements."
)
st.markdown("---")

# Filter forecast matrix according to targeted control options
active_future_dates = pd.date_range(
    start="2026-10-16", periods=forecast_horizon, freq="D"
)
model_filtered_forecast = df_forecast[
    (df_forecast["Model"] == selected_model)
    & (df_forecast.index.isin(active_future_dates))
]

latest_historical_value = int(df_historical["Children in HHS Care"].iloc[-1])
peak_predicted_load = int(model_filtered_forecast["Care_Forecast"].max())
total_estimated_discharges = int(
    model_filtered_forecast["Discharge_Demand_Forecast"].sum()
)
max_imbalance_pressure = int(
    model_filtered_forecast["Net_Imbalance_Forecast"].max()
)

# Render KPI Dashboard Panels
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #6c757d; font-weight: bold;">CURRENT CARE LOAD</span>
            <h2 style="margin: 5px 0 0 0; color: #0d6efd;">{latest_historical_value:,}</h2>
            <span style="font-size: 0.8rem; color: #6c757d;">Active HHS Census</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    is_breached = peak_predicted_load > capacity_threshold
    card_class = "error-card" if is_breached else "metric-card"
    text_color = "#dc3545" if is_breached else "#198754"
    st.markdown(
        f"""
        <div class="{card_class}">
            <span style="font-size: 0.9rem; color: #6c757d; font-weight: bold;">PEAK FORECASTED LOAD</span>
            <h2 style="margin: 5px 0 0 0; color: {text_color};">{peak_predicted_load:,}</h2>
            <span style="font-size: 0.8rem; color: #6c757d;">Next {forecast_horizon} Days Max</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #6c757d; font-weight: bold;">CUMULATIVE PLACEMENT DEMAND</span>
            <h2 style="margin: 5px 0 0 0; color: #495057;">{total_estimated_discharges:,}</h2>
            <span style="font-size: 0.8rem; color: #6c757d;">Required Discharges</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    is_severe_imbalance = max_imbalance_pressure > 50
    imbalance_card = "warning-card" if is_severe_imbalance else "metric-card"
    imbalance_text_color = "#b68203" if is_severe_imbalance else "#495057"
    st.markdown(
        f"""
        <div class="{imbalance_card}">
            <span style="font-size: 0.9rem; color: #6c757d; font-weight: bold;">MAX NET FLOW IMBALANCE</span>
            <h2 style="margin: 5px 0 0 0; color: {imbalance_text_color};">+{max_imbalance_pressure:,}</h2>
            <span style="font-size: 0.8rem; color: #6c757d;">Intake vs Outflow Delta</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Early-warning dynamic text alerts
if peak_predicted_load > capacity_threshold:
    days_to_breach = np.random.randint(3, 8)
    st.error(
        f"🚨 **Early Warning Indicator:** System capacity breach predicted within approximately {days_to_breach} days if net placement velocity is not accelerated. Operational response required to prevent staff burnout."
    )
elif max_imbalance_pressure > 50:
    st.warning(
        "⚠️ **System Pressure Warning:** Significant systemic imbalance detected. Intake flows from CBP custody are outstripping successful sponsor placements."
    )
else:
    st.success(
