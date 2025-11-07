import pandas as pd
import numpy as np
import streamlit as st
import optuna
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.arima.model import ARIMA
from analysis.metrics import root_mean_squared_error 
from statsmodels.tsa.stattools import ccf

# ===============================
# 피처 생성 함수 (Random Forest용)
# ===============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜(ds) 컬럼에서 머신러닝 모델 학습을 위한 시간 피처를 생성합니다."""
    df['dayofweek'] = df['ds'].dt.dayofweek    # 요일
    df['month'] = df['ds'].dt.month            # 월
    df['year'] = df['ds'].dt.year              # 연도
    df['dayofyear'] = df['ds'].dt.dayofyear    # 연도 내 일수 
    
    if 'time_index' not in df.columns:
        df['time_index'] = np.arange(len(df))
        
    return df

# ===============================
# Prophet 모델 함수
# ===============================
@st.cache_data
def run_prophet(df, days):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return model, forecast

# ===============================
# ARIMA 모델 함수
# ===============================
@st.cache_data
def run_arima(df, days):
    """
    ARIMA 모델을 실행하고 예측 결과를 반환합니다.
    
    Args:
        df (pd.DataFrame): 입력 데이터 ('ds'와 'y' 컬럼 필요)
        days (int): 예측할 미래 기간
    
    Returns:
        tuple: (forecast_df, y_true, y_pred_past)
            - forecast_df: 미래 예측값이 담긴 DataFrame
            - y_true: 실제 과거 데이터
            - y_pred_past: 과거 데이터에 대한 예측값
    """
    # ARIMA 모델 학습
    model = ARIMA(df.set_index("ds")["y"], order=(3, 1, 2))
    fitted = model.fit()
    
    # 미래 예측
    future_idx = pd.date_range(df["ds"].iloc[-1], periods=days + 1, freq="D")[1:]
    forecast = fitted.forecast(steps=days)
    
    # 과거 데이터 적합 (전체 기간에 대해 예측)
    y_true = df["y"].values
    y_pred_past = fitted.get_prediction(start=0, end=len(df)-1).predicted_mean
    
    # 예측 결과 DataFrame 생성
    forecast_df = pd.DataFrame({
        "날짜": future_idx,
        "예측값": forecast
    })
    
    return forecast_df, y_true, y_pred_past

# ===============================
# Random Forest 모델 및 튜닝 함수
# ===============================
@st.cache_data
def tune_random_forest_bayesian(X_train, y_train, n_trials=25):
    # Optuna 기반 베이지안 최적화로 RandomForest 하이퍼파라미터 튜닝
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train, 
            scoring="neg_mean_squared_error", cv=3, n_jobs=-1
        )
        return -np.mean(scores)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params, study.best_value

# ===============================
# Random Forest 실행 함수
# ===============================
def run_random_forest(df: pd.DataFrame, days: int, tuned_model=None):
    # 1. 학습 데이터 피처 생성
    train_df = create_features(df.copy())
    
    # 2. 미래 데이터셋 준비 및 피처 생성
    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]
    future_df = pd.DataFrame({'ds': future_dates})
    future_df = create_features(future_df)
    
    # time_index 연속성 유지
    last_index = train_df['time_index'].iloc[-1]
    future_df['time_index'] = np.arange(len(future_df)) + last_index + 1
    
    # 3. 모델 학습
    features = [c for c in train_df.columns if c not in ['ds', 'y']] 
    X_train, y_train = train_df[features], train_df['y']
    
    if tuned_model is not None:
        model = tuned_model
    else: 
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    
    # 4. 예측 (과거 적합도 및 미래 예측)
    y_pred_past = model.predict(X_train) 
    X_future = future_df[features]
    y_pred_future = model.predict(X_future)
    
    # 5. 피처 중요도 추출
    feature_importances = model.feature_importances_
    
    # 결과 통합 (Streamlit 시각화용)
    future_result = future_df[['ds']].rename(columns={'ds': '날짜'})
    future_result['예측값'] = y_pred_future
    
    return future_result, y_train.values, y_pred_past, feature_importances, features

def run_ccf_analysis(series1: np.ndarray, series2: np.ndarray, max_lags: int = 20) -> dict:
    """
    두 시계열 데이터 간의 교차 상관 분석을 수행합니다.
    
    Args:
        series1 (np.ndarray): 첫 번째 시계열 데이터
        series2 (np.ndarray): 두 번째 시계열 데이터
        max_lags (int): 분석할 최대 시차
        
    Returns:
        dict: CCF 분석 결과를 담은 딕셔너리
    """
    # 데이터 정규화
    s1_norm = (series1 - np.mean(series1)) / np.std(series1)
    s2_norm = (series2 - np.mean(series2)) / np.std(series2)
    
    # CCF 계산 (adj 파라미터 제거)
    ccf_values = ccf(s1_norm, s2_norm)
    center_idx = len(ccf_values) // 2
    
    # 시차 배열 생성
    lags = np.arange(-max_lags, max_lags + 1)
    ccf_data = ccf_values[center_idx - max_lags : center_idx + max_lags + 1]
    
    # 결과 데이터프레임 생성
    ccf_df = pd.DataFrame({'Lag': lags, 'CCF': ccf_data})
    
    # 최대 상관계수 및 해당 시차 찾기
    max_row = ccf_df.loc[ccf_df['CCF'].abs().idxmax()]
    optimal_lag = int(max_row['Lag'])
    max_correlation = max_row['CCF']
    
    # 신뢰구간 계산
    conf_level = 1.96 / np.sqrt(len(series1))
    
    # 분석 결과 텍스트 생성
    if abs(max_correlation) > conf_level:
        if optimal_lag > 0:
            analysis_text = f"첫 번째 시계열이 {abs(optimal_lag)}일 선행"
        elif optimal_lag < 0:
            analysis_text = f"두 번째 시계열이 {abs(optimal_lag)}일 선행"
        else:
            analysis_text = "동일 시점에 최대 상관관계"
        status = "success"
    else:
        analysis_text = "유의미한 상관관계 없음"
        status = "info"

    return {
        'ccf_df': ccf_df,
        'optimal_lag': optimal_lag,
        'max_correlation': max_correlation,
        'conf_level': conf_level,
        'analysis_text': analysis_text,
        'status': status
    }