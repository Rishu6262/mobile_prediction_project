import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="centered"
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------------------
# UI Title
# ------------------------------
st.markdown("<h1 style='text-align: center;'>📱 Mobile Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict mobile price range using machine learning</p>", unsafe_allow_html=True)
st.divider()

# ------------------------------
# Input Section
# ------------------------------
st.subheader("🔢 Enter Mobile Specifications")

col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=6000, step=100)
    clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=4.0, step=0.1)
    fc = st.number_input("Front Camera (MP)", min_value=0, max_value=64)
    int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=512)
    mobile_wt = st.number_input("Mobile Weight (gm)", min_value=80, max_value=300)

with col2:
    ram = st.number_input("RAM (MB)", min_value=256, max_value=16000, step=256)
    pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=200)
    px_height = st.number_input("Pixel Height", min_value=0, max_value=3000)
    px_width = st.number_input("Pixel Width", min_value=0, max_value=4000)
    talk_time = st.number_input("Talk Time (hours)", min_value=1, max_value=24)

# Binary Features
st.subheader("⚙️ Additional Features")

col3, col4, col5 = st.columns(3)

with col3:
    bluetooth = st.selectbox("Bluetooth", [0, 1])
    dual_sim = st.selectbox("Dual SIM", [0, 1])

with col4:
    four_g = st.selectbox("4G", [0, 1])
    three_g = st.selectbox("3G", [0, 1])

with col5:
    touch_screen = st.selectbox("Touch Screen", [0, 1])
    wifi = st.selectbox("WiFi", [0, 1])

# ------------------------------
# Prediction
# ------------------------------
st.divider()

if st.button("🔮 Predict Price Range"):
    input_data = np.array([[
        battery_power, bluetooth, clock_speed, dual_sim, fc,
        four_g, int_memory, mobile_wt, pc, px_height,
        px_width, ram, talk_time, three_g, touch_screen, wifi
    ]])

    prediction = model.predict(input_data)[0]

    price_map = {
        0: "Low Cost 📉",
        1: "Medium Cost 📊",
        2: "High Cost 📈",
        3: "Very High Cost 💎"
    }

    st.success(f"Predicted Price Range: **{price_map[prediction]}**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
