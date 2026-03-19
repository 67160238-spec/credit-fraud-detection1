# 🔍 Credit Card Fraud Detection

ระบบตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัย ด้วย Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📋 Problem Statement

การฉ้อโกงบัตรเครดิต (Credit Card Fraud) เป็นปัญหาที่สร้างความเสียหายหลายพันล้านดอลลาร์ต่อปี ระบบนี้ใช้ Machine Learning เพื่อแยกแยะธุรกรรมปกติออกจากธุรกรรมที่น่าสงสัย โดยอัตโนมัติ

**ทำไมถึงใช้ ML?**
- ธุรกรรมเกิดขึ้นหลายล้านรายการต่อวัน ไม่สามารถตรวจด้วยมือได้
- รูปแบบ fraud เปลี่ยนไปตลอดเวลา ML สามารถเรียนรู้รูปแบบใหม่ได้
- Rule-based systems มี false positive สูง ทำให้ลูกค้าไม่สะดวก

---

## 📁 โครงสร้างไฟล์

```
credit-fraud-project/
├── app.py              # Streamlit web application
├── train.py            # EDA + Model training script
├── requirements.txt    # Python dependencies
├── README.md           # เอกสารนี้
├── creditcard.csv      # Dataset (ต้องดาวน์โหลดแยก)
├── model.pkl           # Trained model (สร้างโดย train.py)
├── feature_cols.pkl    # Feature column names
├── feature_importances.pkl  # Feature importance scores
└── plots/              # EDA & evaluation plots
    ├── eda_overview.png
    ├── model_comparison.png
    ├── pr_curve.png
    └── feature_importance.png
```

---

## 🚀 วิธีติดตั้งและรัน

### 1. Clone Repository
```bash
git clone https://github.com/your-username/credit-fraud-project
cd credit-fraud-project
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. ดาวน์โหลด Dataset
ดาวน์โหลด `creditcard.csv` จาก [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) แล้ววางในโฟลเดอร์นี้

### 4. Train โมเดล
```bash
python train.py
```

### 5. รัน Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Dataset

| รายการ | รายละเอียด |
|--------|-----------|
| Source | Kaggle — MLG-ULB |
| จำนวนแถว | 284,807 |
| จำนวน Features | 31 |
| Missing Values | 0 |
| Class Imbalance | 99.83% Normal / 0.17% Fraud |

**Features:**
- `Time` — วินาทีที่ผ่านมาจากธุรกรรมแรก
- `V1–V28` — PCA components (ซ่อนข้อมูลส่วนตัว)
- `Amount` — มูลค่าธุรกรรม (EUR)
- `Class` — 0 = ปกติ, 1 = Fraud

---

## 🤖 Model Development

### Pipeline
```
StandardScaler → SMOTE → Random Forest Classifier
```

### การแก้ปัญหา Class Imbalance
ใช้ **SMOTE** (Synthetic Minority Over-sampling Technique) เพราะ:
- Dataset มี fraud เพียง 0.17% ของข้อมูลทั้งหมด
- การใช้แค่ class_weight อาจไม่เพียงพอ
- SMOTE สร้าง synthetic samples ที่สมจริงกว่า simple oversampling

### Hyperparameter Tuning
ใช้ **RandomizedSearchCV** กับ 5-fold Stratified Cross-Validation
```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}
```

### Evaluation Metrics
ใช้ **AUPRC (Area Under Precision-Recall Curve)** เป็น primary metric เพราะ:
- Accuracy ไม่เหมาะกับ imbalanced dataset (model ทายว่าปกติทุกอย่างก็ได้ 99.83%)
- AUPRC วัดได้ดีกว่าว่าโมเดลตรวจ fraud จริงๆ ได้ดีแค่ไหน

### Model Comparison (Bonus)
| Model | F1 Score | AUPRC |
|-------|----------|-------|
| Logistic Regression | ~0.75 | ~0.72 |
| **Random Forest** | **~0.86** | **~0.82** |
| Gradient Boosting | ~0.83 | ~0.80 |

---

## 🌐 Deploy บน Streamlit Cloud

1. Push code ขึ้น GitHub (ไม่รวม `creditcard.csv` และ `model.pkl`)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. เชื่อมต่อ GitHub repo
4. ใส่ path: `app.py`
5. กด Deploy

> **หมายเหตุ:** ต้อง commit `model.pkl` ขึ้น GitHub ด้วย เนื่องจาก Streamlit Cloud จะใช้ไฟล์นี้

---

## ⚠️ Disclaimer

ระบบนี้สร้างขึ้นเพื่อวัตถุประสงค์ทางการศึกษาเท่านั้น ผลการทำนายไม่ควรนำไปใช้ตัดสินใจในการเงินจริง

---

## 👨‍💻 ผู้พัฒนา

ML Deployment Project — วิชา Machine Learning
