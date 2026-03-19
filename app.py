"""
Credit Card Fraud Detection — Streamlit App
รัน: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .result-fraud {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; border-radius: 12px; padding: 1.5rem;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .result-normal {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white; border-radius: 12px; padding: 1.5rem;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #667eea;
    }
    .info-box {
        background: #e8f4fd; border-radius: 8px;
        padding: 1rem; border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background: #fff3cd; border-radius: 8px;
        padding: 1rem; border-left: 4px solid #ffc107;
        font-size: 0.9rem; margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None, None, None
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    importances = joblib.load("feature_importances.pkl") if os.path.exists("feature_importances.pkl") else None
    return model, feature_cols, importances

model, feature_cols, importances = load_model()

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Credit Card Fraud Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ระบบตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัย ด้วย Machine Learning</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ ไม่พบไฟล์ model.pkl — กรุณารัน `python train.py` ก่อนเพื่อ train โมเดล")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Sidebar — About
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 เกี่ยวกับโปรเจค")
    st.markdown("""
    **Dataset:** Credit Card Fraud Detection  
    **Source:** Kaggle (MLG-ULB)  
    **จำนวนข้อมูล:** 284,807 ธุรกรรม  
    **Algorithm:** Random Forest  
    """)
    st.divider()
    st.markdown("## 📊 ข้อมูล Dataset")
    st.markdown("""
    - `Time` — เวลาที่ผ่านมา (วินาที) นับจากธุรกรรมแรก  
    - `Amount` — มูลค่าธุรกรรม (EUR)  
    - `V1–V28` — ค่า PCA ที่ซ่อนข้อมูลส่วนตัวไว้  
    - `Class` — 0 = ปกติ, 1 = Fraud  
    """)
    st.divider()
    st.markdown("## 🏆 ประสิทธิภาพโมเดล")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Recall | ~0.85 |
    | Precision | ~0.87 |
    | F1 Score | ~0.86 |
    | AUPRC | ~0.82 |
    """)
    st.divider()
    st.markdown("**สร้างโดย:** ML Deployment Project")

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 ทำนาย", "📊 Feature Importance", "📘 วิธีใช้"])

# ─────────────────────────────────────────────────────────────
# TAB 1 — Prediction
# ─────────────────────────────────────────────────────────────
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### 📝 กรอกข้อมูลธุรกรรม")

        with st.expander("⚙️ ข้อมูลหลัก (Time & Amount)", expanded=True):
            time_val = st.number_input(
                "Time (วินาทีนับจากธุรกรรมแรก)",
                min_value=0.0, max_value=200000.0, value=50000.0, step=100.0,
                help="ช่วงปกติ: 0 – 172,792 วินาที (~2 วัน)"
            )
            amount_val = st.number_input(
                "Amount (EUR)",
                min_value=0.0, max_value=25000.0, value=50.0, step=1.0,
                help="มูลค่าธุรกรรม ในหน่วย EUR"
            )

        with st.expander("🔢 PCA Features (V1 – V14)", expanded=False):
            st.info("ค่า V1–V28 คือผลลัพธ์จาก PCA ที่ซ่อนข้อมูลส่วนตัว ค่าส่วนใหญ่อยู่ระหว่าง -5 ถึง 5")
            v_vals = {}
            cols = st.columns(2)
            for i in range(1, 15):
                with cols[(i - 1) % 2]:
                    v_vals[f"V{i}"] = st.number_input(
                        f"V{i}", value=0.0, step=0.1,
                        min_value=-30.0, max_value=30.0,
                        key=f"v{i}"
                    )

        with st.expander("🔢 PCA Features (V15 – V28)", expanded=False):
            cols2 = st.columns(2)
            for i in range(15, 29):
                with cols2[(i - 15) % 2]:
                    v_vals[f"V{i}"] = st.number_input(
                        f"V{i}", value=0.0, step=0.1,
                        min_value=-30.0, max_value=30.0,
                        key=f"v{i}"
                    )

        st.markdown("---")
        st.markdown("#### 🎲 หรือลองใช้ตัวอย่าง:")
        col_ex1, col_ex2 = st.columns(2)

        preset = None
        with col_ex1:
            if st.button("🟢 ธุรกรรมปกติ"):
                preset = "normal"
        with col_ex2:
            if st.button("🔴 ธุรกรรมน่าสงสัย"):
                preset = "fraud"

        if preset == "normal":
            st.session_state["preset"] = {
                "time": 52000, "amount": 45.0,
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            }
        elif preset == "fraud":
            st.session_state["preset"] = {
                "time": 406, "amount": 149.62,
                "V1": -1.36, "V2": 1.19, "V3": -3.15, "V4": 3.29,
                "V5": -1.54, "V6": -0.97, "V7": -0.19, "V8": 0.20,
                "V9": 0.04, "V10": 0.56, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -5.89, "V15": -0.29, "V16": -0.59,
                "V17": -4.29, "V18": 0.39, "V19": -0.39, "V20": -0.19,
                "V21": -0.45, "V22": -0.26, "V23": -0.10, "V24": -0.20,
                "V25": -0.45, "V26": -0.10, "V27": -0.19, "V28": -0.20,
            }

        predict_btn = st.button("🔍 วิเคราะห์ธุรกรรม", use_container_width=True)

    # ─── Result Column ───
    with col_result:
        st.markdown("### 🎯 ผลการวิเคราะห์")

        if predict_btn:
            # Build input
            preset_data = st.session_state.get("preset", {})
            row = {}
            row["Time"] = preset_data.get("time", time_val)
            for i in range(1, 29):
                key = f"V{i}"
                row[key] = preset_data.get(key, v_vals.get(key, 0.0))
            row["Amount"] = preset_data.get("amount", amount_val)

            # Validate
            if row["Amount"] == 0:
                st.warning("⚠️ Amount เป็น 0 — กรุณาตรวจสอบค่า")

            input_df = pd.DataFrame([row])[feature_cols]
            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= 0.5)

            # Clear preset
            if "preset" in st.session_state:
                del st.session_state["preset"]

            # Show result
            if pred == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    🚨 ตรวจพบธุรกรรมน่าสงสัย<br>
                    <span style="font-size:2rem;">{prob*100:.1f}%</span><br>
                    <span style="font-size:0.9rem;">ความน่าจะเป็น FRAUD</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-normal">
                    ✅ ธุรกรรมปกติ<br>
                    <span style="font-size:2rem;">{(1-prob)*100:.1f}%</span><br>
                    <span style="font-size:0.9rem;">ความน่าจะเป็น ปกติ</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            fig, ax = plt.subplots(figsize=(6, 1.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            bar_color = "#e74c3c" if prob > 0.5 else "#00b894"
            ax.barh(0, prob, color=bar_color, height=0.5)
            ax.barh(0, 1 - prob, left=prob, color="#ecf0f1", height=0.5)
            ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%\n(threshold)", "75%", "100%"])
            ax.set_title(f"Fraud Probability: {prob*100:.1f}%", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Details
            st.markdown(f"""
            <div class="info-box">
            📋 <b>รายละเอียด:</b><br>
            • Amount: <b>{row['Amount']:.2f} EUR</b><br>
            • Time: <b>{row['Time']:.0f} วินาที</b> ({row['Time']/3600:.1f} ชั่วโมง)<br>
            • Fraud Probability: <b>{prob*100:.2f}%</b><br>
            • Prediction: <b>{'🔴 FRAUD' if pred == 1 else '🟢 NORMAL'}</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
            ⚠️ <b>Disclaimer:</b> ระบบนี้เป็นเพียงตัวอย่างทางวิชาการ ผลการทำนายไม่ควรนำไปใช้ตัดสินใจจริง
            ในการดำเนินงานจริงต้องผ่านการตรวจสอบจากผู้เชี่ยวชาญด้านความปลอดภัยทางการเงิน
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 กรอกข้อมูลทางซ้าย แล้วกด **วิเคราะห์ธุรกรรม**")
            st.markdown("""
            **ตัวอย่างการใช้งาน:**
            1. กรอก Amount (มูลค่าธุรกรรม)
            2. ปรับค่า V1–V28 ตามข้อมูลจริง
            3. หรือกดปุ่มตัวอย่างด้านล่าง
            """)

# ─────────────────────────────────────────────────────────────
# TAB 2 — Feature Importance
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Feature Importance — ตัวแปรที่สำคัญในการทำนาย")

    if importances is not None:
        col_chart, col_explain = st.columns([2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#e74c3c" if i < 3 else "#3498db" if i < 7 else "#95a5a6"
                      for i in range(len(importances))]
            importances.plot(kind="barh", ax=ax, color=colors)
            ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance Score")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_explain:
            st.markdown("#### 🔑 ความหมาย")
            st.markdown("""
            **สีแดง = สำคัญมากที่สุด**
            ตัวแปรเหล่านี้มีผลสูงสุดต่อการตัดสินใจของโมเดล

            **สีน้ำเงิน = สำคัญรองลงมา**

            **สีเทา = ผลน้อย**

            ---
            **ข้อสังเกต:**
            - `V14`, `V17`, `V12` มักเป็น features ที่แยก fraud ออกจาก normal ได้ดีที่สุด
            - `Amount` มีความสำคัญเนื่องจาก fraud มักมีมูลค่าต่างจากธุรกรรมปกติ
            - `V1–V28` ทั้งหมดเป็น PCA components ที่ซ่อนข้อมูลส่วนตัวไว้
            """)
    else:
        st.info("Feature importances จะแสดงหลังจากรัน train.py")

    st.divider()
    st.markdown("### 📈 Dataset Statistics")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("จำนวนธุรกรรมทั้งหมด", "284,807")
    with col_s2:
        st.metric("ธุรกรรม Fraud", "492 (0.17%)")
    with col_s3:
        st.metric("Features", "30")
    with col_s4:
        st.metric("Missing Values", "0")

# ─────────────────────────────────────────────────────────────
# TAB 3 — How to Use
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📘 วิธีใช้งานระบบ")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("""
        #### 🖥️ วิธีรันในเครื่อง
        ```bash
        # 1. Clone repo
        git clone <your-repo-url>
        cd credit-fraud-project

        # 2. ติดตั้ง dependencies
        pip install -r requirements.txt

        # 3. วางไฟล์ creditcard.csv
        # ดาวน์โหลดจาก kaggle แล้ววางไว้ในโฟลเดอร์นี้

        # 4. Train โมเดล
        python train.py

        # 5. รัน app
        streamlit run app.py
        ```
        """)

    with col_h2:
        st.markdown("""
        #### 📋 ข้อมูล Input ที่ต้องการ
        | Field | ประเภท | ช่วงค่า |
        |-------|--------|---------|
        | Time | float | 0 – 172,792 |
        | Amount | float | 0 – 25,000 EUR |
        | V1–V28 | float | ประมาณ -30 ถึง 30 |

        #### ⚠️ ข้อจำกัด
        - V1–V28 เป็นค่า PCA ที่ไม่สามารถตีความตรงๆ ได้
        - ระบบนี้ train บนข้อมูลธุรกรรมในยุโรป ปี 2013
        - ใช้เพื่อการศึกษาเท่านั้น
        """)

    st.divider()
    st.markdown("### 🔬 เกี่ยวกับโมเดล")
    st.markdown("""
    | รายการ | รายละเอียด |
    |--------|-----------|
    | Algorithm | Random Forest Classifier |
    | Class Imbalance | SMOTE (sampling_strategy=0.1) |
    | Hyperparameter Tuning | RandomizedSearchCV (5-fold CV) |
    | Primary Metric | AUPRC (Area Under Precision-Recall Curve) |
    | เหตุผลที่ไม่ใช้ Accuracy | Dataset imbalanced 99.8%/0.2% ทำให้ Accuracy ไม่มีความหมาย |
    """)
