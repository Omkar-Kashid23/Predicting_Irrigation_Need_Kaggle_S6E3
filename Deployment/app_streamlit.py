import streamlit as st
from model_utils import predict

st.set_page_config(page_title="💧 Irrigation Need Predictor", page_icon="💧", layout="wide")
st.title("💧 Agricultural Irrigation Need Predictor")
st.markdown("Enter field & environmental parameters to predict irrigation need.")

col1, col2 = st.columns(2)
with col1:
    soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=1.0, step=0.1)
    ec = st.number_input("Electrical Conductivity", min_value=0.0, value=1.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

with col2:
    sunlight = st.number_input("Sunlight Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
    wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0, step=0.1)
    prev_irr = st.number_input("Previous Irrigation (mm)", min_value=0.0, value=0.0, step=0.1)
    area = st.number_input("Field Area (hectare)", min_value=0.1, value=1.0, step=0.1)

col3, col4, col5 = st.columns(3)
with col3: soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Silt"])
with col4: crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Sugarcane", "Potato", "Cotton"])
with col5: growth_stage = st.selectbox("Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvesting"])

col6, col7, col8 = st.columns(3)
with col6: season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
with col7: irr_type = st.selectbox("Irrigation Type", ["Drip", "Sprinkler", "Rainfed", "Flood"])
with col8: water_src = st.selectbox("Water Source", ["River", "Reservoir", "Well", "Rainwater"])

if st.button("🔍 Predict Irrigation Need", type="primary", use_container_width=True):
    input_data = {
        "Soil_pH": soil_ph, "Soil_Moisture": soil_moisture, "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": ec, "Temperature_C": temp, "Humidity": humidity,
        "Sunlight_Hours": sunlight, "Wind_Speed_kmh": wind, "Rainfall_mm": rainfall,
        "Previous_Irrigation_mm": prev_irr, "Field_Area_hectare": area,
        "Soil_Type": soil_type, "Crop_Type": crop_type, "Crop_Growth_Stage": growth_stage,
        "Season": season, "Irrigation_Type": irr_type, "Water_Source": water_src
    }
    with st.spinner("🌱 Analyzing..."):
        result = predict(input_data)
    
    st.success(f"✅ Predicted Irrigation Need: **{result['prediction']}**")
    st.subheader("📊 Probability Breakdown")
    st.bar_chart(result["probabilities"], horizontal=True)

st.caption("Model trained on Kaggle Playground Series S6E4 dataset. 🌾")
