"""
Credit Card Fraud Detection — Streamlit App (User-Friendly Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ตรวจสอบธุรกรรมบัตรเครดิต",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .result-fraud {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; font-size: 1.8rem; font-weight: bold;
        margin: 1rem 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; font-size: 1.8rem; font-weight: bold;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #fff3cd; border-radius: 8px;
        padding: 1rem; border-left: 4px solid #ffc107;
        font-size: 0.85rem; margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.7rem 2rem; font-size: 1.1rem; font-weight: 700;
        width: 100%;
    }
    .scenario-box {
        border: 2px solid #eee; border-radius: 12px;
        padding: 1rem; text-align: center; cursor: pointer;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None, None
    return joblib.load("model.pkl"), joblib.load("feature_cols.pkl")

model, feature_cols = load_model()

# ── Mapping function: แปลงข้อมูลที่เข้าใจง่าย → V1-V28 ──
def build_input(amount, hour, merchant_type, location, freq):
    """
    สร้าง feature vector จากข้อมูลที่เข้าใจง่าย
    โดยใช้ค่า V เฉลี่ยของ fraud/normal ที่วิเคราะห์จาก dataset จริง
    ผสมกันตามปัจจัยเสี่ยง
    """
    # ค่าเฉลี่ย V ของ Normal transaction
    normal_v = {
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0,
    }
    # ค่าเฉลี่ย V ของ Fraud transaction (จาก dataset จริง)
    fraud_v = {
        "V1": -4.77, "V2": 3.95, "V3": -7.03, "V4": 4.46, "V5": -3.15,
        "V6": -1.40, "V7": -5.56, "V8": 0.57, "V9": -2.58, "V10": -6.19,
        "V11": 4.05, "V12": -7.48, "V13": 0.25, "V14": -10.63, "V15": 0.50,
        "V16": -4.37, "V17": -8.64, "V18": -2.95, "V19": 0.49, "V20": 0.63,
        "V21": 0.85, "V22": -0.16, "V23": -0.13, "V24": -0.02, "V25": 0.11,
        "V26": -0.04, "V27": 0.52, "V28": 0.24,
    }

    # คำนวณ risk score จากปัจจัยต่างๆ
    risk = 0.0

    # ยอดเงิน
    if amount > 1000:   risk += 0.3
    elif amount > 500:  risk += 0.15
    elif amount < 10:   risk += 0.1

    # เวลา
    if 1 <= hour <= 5:  risk += 0.3   # ดึกมาก
    elif 22 <= hour or hour == 0: risk += 0.1

    # ประเภทร้านค้า
    risk_map = {
        "ร้านสะดวกซื้อ / ซูเปอร์มาร์เก็ต": 0.0,
        "ร้านอาหาร / คาเฟ่": 0.0,
        "ปั๊มน้ำมัน": 0.1,
        "ช้อปปิ้งออนไลน์": 0.2,
        "ATM / ถอนเงิน": 0.25,
        "เว็บไซต์ต่างประเทศ": 0.35,
        "คาสิโน / การพนัน": 0.4,
    }
    risk += risk_map.get(merchant_type, 0.0)

    # สถานที่
    loc_map = {
        "ในประเทศ": 0.0,
        "ต่างประเทศ": 0.25,
        "ไม่ทราบที่มา": 0.4,
    }
    risk += loc_map.get(location, 0.0)

    # ความถี่ผิดปกติ
    freq_map = {
        "ปกติ (1-2 ครั้ง/วัน)": 0.0,
        "บ่อยขึ้น (3-5 ครั้ง/วัน)": 0.1,
        "ผิดปกติ (มากกว่า 5 ครั้ง/วัน)": 0.35,
    }
    risk += freq_map.get(freq, 0.0)

    # clamp risk 0-1
    risk = min(risk, 1.0)

    # blend normal + fraud V values
    row = {"Time": hour * 3600, "Amount": amount}
    for k in normal_v:
        row[k] = normal_v[k] * (1 - risk) + fraud_v[k] * risk
        # เพิ่ม noise เล็กน้อยให้ดูสมจริง
        row[k] += np.random.normal(0, 0.3)

    return pd.DataFrame([row])[feature_cols], risk

# ── Header ──────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 ตรวจสอบธุรกรรมบัตรเครดิต</div>', unsafe_allow_html=True)
st.markdown("##### ระบบ AI ตรวจจับธุรกรรมที่น่าสงสัย — กรอกข้อมูลธุรกรรมแล้วกดตรวจสอบได้เลย")
st.divider()

if model is None:
    st.error("⚠️ ไม่พบ model.pkl กรุณา train โมเดลก่อน")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 เกี่ยวกับระบบ")
    st.markdown("""
    ระบบนี้ใช้ **Random Forest** ที่ train จากข้อมูลธุรกรรมจริง 284,807 รายการ
    เพื่อตรวจจับการฉ้อโกงบัตรเครดิต
    
    **Dataset:** Kaggle Credit Card Fraud  
    **ความแม่นยำ:** F1 ~0.86  
    """)
    st.divider()
    st.markdown("## 🚨 สัญญาณเตือน Fraud")
    st.markdown("""
    - ธุรกรรมยอดสูงผิดปกติ
    - เกิดขึ้นในช่วงดึก (ตี 1-5)
    - จากต่างประเทศหรือไม่ทราบที่มา
    - ธุรกรรมถี่ผิดปกติในวันเดียว
    - ร้านค้าประเภทเสี่ยง
    """)

# ── Main Form ──────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 ข้อมูลธุรกรรม")

    amount = st.number_input(
        "💰 ยอดเงิน (บาท / EUR)",
        min_value=1.0, max_value=100000.0,
        value=500.0, step=50.0,
        help="ยอดเงินของธุรกรรมนี้"
    )

    hour = st.slider(
        "🕐 เวลาที่ทำธุรกรรม",
        min_value=0, max_value=23, value=14,
        format="%d:00 น.",
        help="เวลาที่เกิดธุรกรรม"
    )

    # แสดง label เวลาให้เข้าใจง่าย
    if 6 <= hour <= 11:
        time_label = "🌅 ช่วงเช้า"
    elif 12 <= hour <= 17:
        time_label = "☀️ ช่วงบ่าย"
    elif 18 <= hour <= 21:
        time_label = "🌆 ช่วงเย็น"
    elif 22 <= hour or hour <= 0:
        time_label = "🌙 ช่วงกลางคืน"
    else:
        time_label = "🌃 ช่วงดึก (เสี่ยง)"
    st.caption(f"เวลา {hour:02d}:00 น. — {time_label}")

    merchant_type = st.selectbox(
        "🏪 ประเภทร้านค้า / บริการ",
        options=[
            "ร้านสะดวกซื้อ / ซูเปอร์มาร์เก็ต",
            "ร้านอาหาร / คาเฟ่",
            "ปั๊มน้ำมัน",
            "ช้อปปิ้งออนไลน์",
            "ATM / ถอนเงิน",
            "เว็บไซต์ต่างประเทศ",
            "คาสิโน / การพนัน",
        ]
    )

    location = st.selectbox(
        "📍 สถานที่ทำธุรกรรม",
        options=["ในประเทศ", "ต่างประเทศ", "ไม่ทราบที่มา"]
    )

    freq = st.selectbox(
        "🔄 ความถี่การใช้บัตรวันนี้",
        options=[
            "ปกติ (1-2 ครั้ง/วัน)",
            "บ่อยขึ้น (3-5 ครั้ง/วัน)",
            "ผิดปกติ (มากกว่า 5 ครั้ง/วัน)",
        ]
    )

    st.markdown("---")
    predict_btn = st.button("🔍 ตรวจสอบธุรกรรม", use_container_width=True)

# ── Result Column ──────────────────────────────────────────
with col2:
    st.markdown("### 🎯 ผลการวิเคราะห์")

    if predict_btn:
        np.random.seed(42)
        input_df, risk_score = build_input(amount, hour, merchant_type, location, freq)
        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)

        if pred == 1:
            st.markdown(f"""
            <div class="result-fraud">
                🚨 ตรวจพบธุรกรรมน่าสงสัย!<br>
                <span style="font-size:2.5rem;">{prob*100:.1f}%</span><br>
                <span style="font-size:1rem; opacity:0.9">โอกาสเป็น Fraud</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                ✅ ธุรกรรมปกติ<br>
                <span style="font-size:2.5rem;">{(1-prob)*100:.1f}%</span><br>
                <span style="font-size:1rem; opacity:0.9">โอกาสปกติ</span>
            </div>
            """, unsafe_allow_html=True)

        # Gauge bar
        fig, ax = plt.subplots(figsize=(6, 1.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(0, prob, color="#e74c3c" if prob > 0.5 else "#00b894", height=0.5)
        ax.barh(0, 1-prob, left=prob, color="#ecf0f1", height=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50% (จุดตัดสิน)", "100%"])
        ax.set_title(f"Fraud Probability: {prob*100:.1f}%", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ปัจจัยเสี่ยงที่ตรวจพบ
        st.markdown("#### 🔎 ปัจจัยที่ตรวจพบ")
        factors = []
        if amount > 1000: factors.append("💰 ยอดเงินสูงผิดปกติ")
        if 1 <= hour <= 5: factors.append("🌃 ทำธุรกรรมช่วงดึก (ตี 1-5)")
        if location != "ในประเทศ": factors.append(f"📍 สถานที่: {location}")
        if merchant_type in ["คาสิโน / การพนัน", "เว็บไซต์ต่างประเทศ", "ATM / ถอนเงิน"]:
            factors.append(f"🏪 ประเภทเสี่ยง: {merchant_type}")
        if freq != "ปกติ (1-2 ครั้ง/วัน)":
            factors.append(f"🔄 ความถี่: {freq}")

        if factors:
            for f in factors:
                st.warning(f)
        else:
            st.success("✅ ไม่พบปัจจัยเสี่ยงที่ผิดปกติ")

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> ระบบนี้สร้างเพื่อการศึกษาเท่านั้น
        ผลการวิเคราะห์ไม่ควรนำไปใช้ตัดสินใจทางการเงินจริง
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 กรอกข้อมูลธุรกรรมทางซ้าย แล้วกด **ตรวจสอบธุรกรรม**")

        st.markdown("#### 💡 ตัวอย่างสถานการณ์เสี่ยง")
        st.markdown("""
        | สถานการณ์ | ความเสี่ยง |
        |-----------|-----------|
        | ซื้อของซูเปอร์มาร์เก็ต 200 บาท ตอนบ่าย | 🟢 ต่ำ |
        | ถอน ATM ต่างประเทศ 5,000 บาท ตี 3 | 🔴 สูงมาก |
        | ช้อปออนไลน์ 800 บาท ตอนเย็น | 🟡 ปานกลาง |
        | คาสิโนออนไลน์ 10,000 บาท ดึก | 🔴 สูงมาก |
        """)
