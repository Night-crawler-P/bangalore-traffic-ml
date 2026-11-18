import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import predict_congestion, road_to_area  # noqa: E402


st.set_page_config(
    page_title="Bangalore Traffic Pulse",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
    .block-container {padding-top: 1.5rem; max-width: 1000px;}
    .hero {
        background: linear-gradient(120deg, #4f46e5, #0ea5e9);
        padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .9;}
    .card {border-radius: 18px; padding: 1.6rem; text-align: center; color: white;}
    .card .label {font-size: .9rem; opacity: .9; letter-spacing: .08em; text-transform: uppercase;}
    .card .value {font-size: 3rem; font-weight: 800; line-height: 1.1;}
    .card .sub {font-size: .95rem; opacity: .95; margin-top: .3rem;}
    .pill {
        display: inline-block; padding: .25rem .8rem; border-radius: 999px;
        background: rgba(128,128,128,.18); font-size: .85rem; font-weight: 600;
    }
    .bar-row {margin: .55rem 0;}
    .bar-top {display: flex; justify-content: space-between; font-weight: 600; font-size: .95rem;}
    .bar-bg {background: rgba(128,128,128,.2); border-radius: 999px; height: 12px; overflow: hidden;}
    .bar-fill {height: 100%; border-radius: 999px;}
    .stat {border-radius: 16px; padding: 1.2rem; text-align: center; background: rgba(128,128,128,.12);}
    .stat .num {font-size: 2.2rem; font-weight: 800;}
    .stat .cap {font-size: .85rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🚦 Bangalore Traffic Pulse</h1>
        <p>Machine-learning congestion predictor</p>
    </div>
    """,
    unsafe_allow_html=True,
)

prediction_tab, performance_tab = st.tabs(["Prediction", "Model Performance"])

with prediction_tab:
    with st.container(border=True):
        st.markdown("#### Enter prediction details")

        col1, col2, col3 = st.columns(3)
        areas = sorted(set(road_to_area.values()))
        area = col1.selectbox("Area", areas)
        road = col2.selectbox(
            "Road / Intersection",
            sorted(r for r, a in road_to_area.items() if a == area),
        )
        day = col3.date_input("Date", value=date(2024, 3, 15))

        col4, col5, col6 = st.columns(3)
        weather = col4.selectbox(
            "Weather", ["Clear", "Overcast", "Fog", "Rain", "Windy"]
        )
        roadwork = col5.radio("Roadwork", ["No", "Yes"], horizontal=True)
        incidents = col6.slider("Incident reports", 0, 10, 1)

        know = st.toggle("Enter current traffic volume and speed")
        volume, speed = None, None
        if know:
            volume_col, speed_col = st.columns(2)
            volume = volume_col.number_input(
                "Traffic volume", 1000, 80000, 30000, step=500
            )
            speed = speed_col.number_input(
                "Average speed (km/h)", 10.0, 100.0, 40.0, step=1.0
            )

        predict_button = st.button(
            "Predict congestion", type="primary", use_container_width=True
        )

    if predict_button:
        result = predict_congestion(
            road,
            incidents,
            weather,
            roadwork,
            str(day),
            traffic_volume=volume,
            avg_speed=speed,
        )
        label = result["prediction"]
        nowcast = result["mode"].startswith("Nowcast")

        left, right = st.columns([1, 1.2], gap="large")

        with left:
            st.markdown(
                f'<div class="card" style="background:{COLORS[label]};">'
                f'<div class="label">Predicted congestion</div>'
                f'<div class="value">{label}</div>'
                f'<div class="sub">{road} · {result["area"]}</div></div>',
                unsafe_allow_html=True,
            )
            accuracy = "about 93%" if nowcast else "about 60%"
            st.markdown(
                f'<p style="margin-top:.8rem"><span class="pill">'
                f'{result["mode"]} · {accuracy} test accuracy</span></p>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("**Prediction probabilities**")
            bars = ""
            for category in ["Low", "Medium", "High"]:
                probability = result["probabilities"][category]
                bars += (
                    f'<div class="bar-row"><div class="bar-top"><span>{category}</span>'
                    f'<span>{probability:.1%}</span></div><div class="bar-bg">'
                    f'<div class="bar-fill" style="width:{probability * 100:.1f}%;'
                    f'background:{COLORS[category]};"></div></div></div>'
                )
            st.markdown(bars, unsafe_allow_html=True)

        st.caption(
            "Low: congestion below 60 · Medium: 60 to below 100 · "
            "High: saturated at 100"
        )
    else:
        st.info("Enter the details above and select **Predict congestion**.")

with performance_tab:
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        '<div class="stat"><div class="num">44.6%</div>'
        '<div class="cap">Baseline accuracy</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        '<div class="stat"><div class="num" style="color:#f59e0b">59.8%</div>'
        '<div class="cap">Forecast mode · XGBoost</div></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        '<div class="stat"><div class="num" style="color:#22c55e">92.7%</div>'
        '<div class="cap">Nowcast mode · XGBoost</div></div>',
        unsafe_allow_html=True,
    )
