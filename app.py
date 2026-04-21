import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


st.set_page_config(
    page_title="Delhi AQI Prediction",
    page_icon="🌫️",
    layout="centered"
)

st.title("🌫️ Delhi AQI Prediction")
st.markdown("Enter current pollutant and time details to predict next hour AQI")


# ── Prediction History ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Season options ────────────────────────────────────────────────────────────
df = pd.read_csv("Data/delhi_ncr_aqi_dataset.csv")

if "season" in df.columns:
    seasons = sorted(df["season"].dropna().astype(str).unique().tolist())
else:
    seasons = ["Winter", "Summer", "Monsoon", "Post_Monsoon"]


# ── Day of week mapping ───────────────────────────────────────────────────────
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}


# ── Input Layout ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pm25   = st.number_input("PM2.5",  min_value=0.0, value=80.0)
    no2    = st.number_input("NO2",    min_value=0.0, value=35.0)
    co     = st.number_input("CO",     min_value=0.0, value=1.0)
    hour   = st.slider("Hour",  0, 23, 10)
    month  = st.slider("Month", 1, 12, 6)
    season = st.selectbox("Season", seasons)

with col2:
    pm10         = st.number_input("PM10", min_value=0.0, value=140.0)
    so2          = st.number_input("SO2",  min_value=0.0, value=12.0)
    o3           = st.number_input("O3",   min_value=0.0, value=28.0)
    day          = st.slider("Day", 1, 31, 15)
    selected_day = st.selectbox("Day of Week", list(day_map.keys()), index=2)

weekday = day_map[selected_day]


# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict AQI 🌫️"):
    with st.spinner("Generating prediction..."):

        data = CustomData(
            pm25=pm25, pm10=pm10, no2=no2,
            so2=so2, co=co, o3=o3,
            hour=hour, day=day, month=month,
            weekday=weekday, season=season
        )

        pred_df  = data.get_data_as_data_frame()
        pipeline = PredictPipeline()

        # LSTM predict() returns a single float directly
        predicted_aqi = max(0.0, pipeline.predict(pred_df))

    # ── Store history ─────────────────────────────────────────────────────
    st.session_state.history.append({
        "PM2.5": pm25, "PM10": pm10, "NO2": no2,
        "SO2": so2, "CO": co, "O3": o3,
        "Hour": hour, "Month": month, "Day": day,
        "Weekday": selected_day, "Season": season,
        "Predicted AQI": round(predicted_aqi, 2)
    })

    st.success(f"Predicted Next Hour AQI: **{predicted_aqi:.2f}**")

    # ── AQI Category ──────────────────────────────────────────────────────
    if predicted_aqi <= 50:
        st.info("Air Quality: Good")
    elif predicted_aqi <= 100:
        st.success("Air Quality:  Satisfactory")
    elif predicted_aqi <= 200:
        st.warning("Air Quality:  Moderate")
    elif predicted_aqi <= 300:
        st.warning("Air Quality:  Poor")
    elif predicted_aqi <= 400:
        st.error("Air Quality:  Very Poor")
    else:
        st.error("Air Quality:  Severe")


# ── History Section ───────────────────────────────────────────────────────────
if len(st.session_state.history) > 0:
    st.subheader("📜 Previous Predictions")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()