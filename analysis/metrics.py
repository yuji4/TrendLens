import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error
from datetime import datetime

# ===============================
# 성능 지표 계산 함수
# ===============================
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# 모델 성능 기록 함수
# ===============================
def save_model_metrics(model_name, keyword, mape, rmse):
    if "model_metrics" not in st.session_state:
        st.session_state["model_metrics"] = []

    st.session_state["model_metrics"].append({
        "키워드": keyword,
        "모델명": model_name,
        "MAPE(%)": round(mape, 2),
        "RMSE": round(rmse, 4),
        "기록시간": datetime.now().strftime("%H:%M:%S")
    })