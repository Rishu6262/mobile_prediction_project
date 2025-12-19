import streamlit as st
import pickle
import numpy as np
import pandas as pd

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
# Title
# ------------------------------
st.markdown("<h1 style='text-align:center;'>📱 Mobile Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict mobile price range using Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# ------------------------------
# Input Section
# ------------------------------
st.subheader("🔢 Enter Mobile Specifications")

col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input("Battery Power (mAh)", 500, 6000, 2000)
    blue = st.selectbox("Bluetooth", [0, 1])
    clock_speed = st.number_input("Clock Speed (GHz)", 0.5, 4.0, 1.5)
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    fc = st.number_input("Front Camera (MP)", 0, 64, 5)
    four_g = st.selectbox("4G Support", [0, 1])
    int_memory = st.number_input("Internal Memory (GB)", 2, 512, 64)
    m_dep = st.number_input("Mobile Depth", 0.1, 1.0, 0.5)

with col2:
    mobile_wt = st.number_input("Mobile Weight (gm)", 80, 300, 150)
    n_cores = st.number_input("Number of Cores", 1, 8, 4)
    pc = st.number_input("Primary Camera (MP)", 0, 200, 12)
    px_height = st.number_input("Pixel Height", 0, 3000, 1000)
    px_width = st.number_input("Pixel Width", 0, 4000, 2000)
    ram = st.number_input("RAM (MB)", 256, 16000, 4096)
    sc_h = st.number_input("Screen Height (cm)", 5, 20, 12)
    sc_w = st.number_input("Screen Width (cm)", 5, 20, 7)

talk_time = st.number_input("Talk Time (hours)", 1, 24, 12)
three_g = st.selectbox("3G Support", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi", [0, 1])

# ------------------------------
# Prediction
# ------------------------------
st.divider()

if st.button("🔮 Predict Price Range"):

    # ✅ EXACT SAME ORDER AS TRAINING DATA
    input_data = pd.DataFrame([[
        battery_power, blue, clock_speed, dual_sim, fc,
        four_g, int_memory, m_dep, mobile_wt, n_cores,
        pc, px_height, px_width, ram, sc_h,
        sc_w, talk_time, three_g, touch_screen, wifi
    ]], columns=[
        'battery_power','blue','clock_speed','dual_sim','fc',
        'four_g','int_memory','m_dep','mobile_wt','n_cores',
        'pc','px_height','px_width','ram','sc_h',
        'sc_w','talk_time','three_g','touch_screen','wifi'
    ])

    prediction = model.predict(input_data)[0]

    price_map = {
        0: "Low Cost 📉",
        1: "Medium Cost 📊",
        2: "High Cost 📈",
        3: "Very High Cost 💎"
    }

    st.success(f"### Predicted Price Range: **{price_map[prediction]}**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
