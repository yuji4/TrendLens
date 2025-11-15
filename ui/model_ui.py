import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.modeling import run_prophet, run_arima, run_random_forest, tune_random_forest_bayesian, create_features
from analysis.metrics import mean_absolute_percentage_error, root_mean_squared_error, save_model_metrics


PLOTLY_STYLE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=13, color="#212121"),
    margin=dict(l=40, r=40, t=50, b=40),
    hovermode="x unified",
)

def render_model_info():
    with st.expander("ℹ️ 예측 모델 설명"):
        st.markdown("""
        **🔮 Prophet**  
        - Facebook에서 개발한 시계열 예측 모델  
        - 계절성(weekly/yearly) + 추세(trend) 분석에 강함  
        
        **📈 ARIMA**  
        - 통계 기반 모델  
        - 과거 패턴을 기반으로 안정적인 데이터 예측에 적합  

        **🌲 Random Forest**  
        - 머신러닝 기반 예측  
        - 여러 개의 결정 트리를 결합해 복잡한 패턴을 학습  
        """)

def render_metric_help():
    with st.expander("❓ RMSE / MAPE가 무엇인가요?"):
        st.markdown("""
        **MAPE (Mean Absolute Percentage Error)**  
        실제값 대비 예측 오차를 백분율로 나타낸 값입니다. 낮을수록 정확합니다.  

        **RMSE (Root Mean Squared Error)**  
        예측값과 실제값의 평균 제곱근 오차입니다. 값이 작을수록 예측 정확도가 높습니다.
        """)

def render_prophet_seasonality(forecast):
    """
    Prophet 계절성 분석 (Trend / Yearly / Weekly) 그래프를 3컬럼으로 표시
    expander 내부에서 호출됨.
    """
    st.markdown("### 📉 Prophet 계절성 분석")

    # 1) Trend (장기 추세)
    fig_trend = px.line(
        forecast, x="ds", y="trend",
        title="📈 장기 추세 (Trend)",
        color_discrete_sequence=['#4CAF50']
    )
    fig_trend.update_layout(
        plot_bgcolor="white", paper_bgcolor="#F5F5F5",
        font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    fig_trend.update_yaxes(title_text="영향도")

    # 2) Yearly seasonality
    df_yearly = forecast[["ds", "yearly"]].copy()
    df_yearly = df_yearly.tail(365)

    fig_yearly = px.line(
        df_yearly, x="ds", y="yearly",
        title="📅 연간 계절성 (Yearly Seasonality)",
        color_discrete_sequence=['#2196F3']
    )
    fig_yearly.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="#F5F5F5",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    fig_yearly.update_xaxes(title_text="날짜", tickformat="%m-%d")
    fig_yearly.update_yaxes(title_text="영향도")

    # 3) Weekly seasonality
    df_weekly = forecast[["ds", "weekly"]].tail(7).copy()

    day_names_kr = ['월', '화', '수', '목', '금', '토', '일']
    df_weekly['day_name_kr'] = df_weekly['ds'].dt.day_name(locale='en').map({
        'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 'Thursday': '목', 
        'Friday': '금', 'Saturday': '토', 'Sunday': '일'
    })
    df_weekly['day_name_kr'] = pd.Categorical(df_weekly['day_name_kr'], categories=day_names_kr, ordered=True)
    df_weekly = df_weekly.sort_values('day_name_kr')

    fig_weekly = px.bar(
        df_weekly, x="day_name_kr", y="weekly",
        title="📆 주간 계절성 (Weekly Seasonality)",
        color_discrete_sequence=['#FFC107']
    )
    fig_weekly.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="#F5F5F5",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    fig_weekly.update_xaxes(title_text="요일")
    fig_weekly.update_yaxes(title_text="영향도")

    # 컬럼 3개 배치
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_trend, width='stretch')
    with col2:
        st.plotly_chart(fig_yearly, width='stretch')
    with col3:
        st.plotly_chart(fig_weekly, width='stretch')


# ---------------------------------------------------------
# 📌 Prophet UI
# ---------------------------------------------------------
def render_prophet_ui(df_forecast, keyword, days_ahead):
    st.markdown("### 🔮 Prophet 예측")

    if st.button("🚀 Prophet 예측 실행"):
        with st.spinner("Prophet 모델 예측 중..."):

            model, forecast = run_prophet(df_forecast, days_ahead)

            # 실제값 · 예측값
            y_true = df_forecast["y"].values
            y_pred = forecast["yhat"].head(len(y_true)).values

            # 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_forecast["ds"], y=df_forecast["y"],
                mode="lines+markers", name="실제값",
                line=dict(color="black", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                mode="lines", name="예측값",
                line=dict(color="royalblue", width=2)
            ))
            fig.update_layout(
                title=f"{keyword} {days_ahead}일 예측 (Prophet)",
                **PLOTLY_STYLE
            )
            st.plotly_chart(fig, width='stretch')

            # 성능 metrics
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            save_model_metrics("Prophet", keyword, mape, rmse)

            st.markdown("#### 🌟 모델 성능 지표")
            render_metric_help()

            col_a, col_b = st.columns(2)
            col_a.metric("MAPE", f"{mape:.2f}%")
            col_b.metric("RMSE", f"{rmse:.3f}")

            # 계절성 분석
            with st.expander("📉 Prophet 계절성 분석 보기 (Trend / Yearly / Weekly)"):
                render_prophet_seasonality(forecast)


# ---------------------------------------------------------
# 📌 ARIMA UI
# ---------------------------------------------------------
def render_arima_ui(df_forecast, keyword, days_ahead):
    st.markdown("### 📈 ARIMA 예측")

    if st.button("🚀 ARIMA 예측 실행"):
        with st.spinner("ARIMA 모델 예측 중..."):

            forecast_df, y_true, y_pred_past = run_arima(
                df_forecast, days_ahead
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_forecast["ds"], y=df_forecast["y"],
                mode="lines+markers", name="실제값",
                line=dict(color="black", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["날짜"], y=forecast_df["예측값"],
                mode="lines", name="예측값",
                line=dict(color="royalblue", width=2, dash="dot")
            ))
            fig.update_layout(
                title=f"{keyword} {days_ahead}일 예측 (ARIMA)",
                **PLOTLY_STYLE
            )
            st.plotly_chart(fig, width='stretch')

            mape = mean_absolute_percentage_error(y_true, y_pred_past)
            rmse = root_mean_squared_error(y_true, y_pred_past)
            save_model_metrics("ARIMA", keyword, mape, rmse)

            st.markdown("#### 🌟 모델 성능 지표")
            render_metric_help()

            col_a, col_b = st.columns(2)
            col_a.metric("MAPE", f"{mape:.2f}%")
            col_b.metric("RMSE", f"{rmse:.3f}")


# ---------------------------------------------------------
# 📌 Random Forest UI
# ---------------------------------------------------------
def render_random_forest_ui(df_forecast, keyword, days_ahead):
    st.markdown("### 🌲 Random Forest 예측")

    tune = st.checkbox("Bayesian Optimization(Optuna) 튜닝 실행", value=False)

    if tune:
        n_trials = st.slider("탐색 횟수", 10, 50, 25, 5)
    else:
        n_trials = None

    if st.button("🚀 Random Forest 예측 실행"):
        with st.spinner("Random Forest 예측 중..."):

            tuned_model = None

            # 튜닝 로직
            if tune:
                train_df = create_features(df_forecast.copy())
                features_x = [
                    c for c in train_df.columns if c not in ["ds", "y"]
                ]
                X_train, y_train = train_df[features_x], train_df["y"]
                tuned_model, _, _ = tune_random_forest_bayesian(
                    X_train, y_train, n_trials=n_trials
                )

            # 예측 실행
            forecast_df, y_true, y_pred_past, feature_imp, features = run_random_forest(
                df_forecast, days_ahead, tuned_model=tuned_model
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_forecast["ds"], y=df_forecast["y"],
                mode="lines+markers", name="실제값",
                line=dict(color="black", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["날짜"], y=forecast_df["예측값"],
                mode="lines", name="예측값",
                line=dict(color="#FF5722", width=2.5, dash="dot")
            ))
            fig.update_layout(
                title=f"{keyword} {days_ahead}일 예측 (Random Forest)",
                **PLOTLY_STYLE
            )
            st.plotly_chart(fig, width='stretch')

            mape = mean_absolute_percentage_error(y_true, y_pred_past)
            rmse = root_mean_squared_error(y_true, y_pred_past)
            save_model_metrics("Random Forest", keyword, mape, rmse)

            st.markdown("#### 🌟 모델 성능 지표")
            render_metric_help()

            col_a, col_b = st.columns(2)
            col_a.metric("MAPE", f"{mape:.2f}%")
            col_b.metric("RMSE", f"{rmse:.3f}")

            st.subheader("💡 피처 중요도 분석")
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": feature_imp
            }).sort_values(by="Importance", ascending=True)

            fig_imp = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                title="검색량 예측에 기여한 피처 중요도",
                orientation="h",
                color="Importance",
                color_continuous_scale=px.colors.sequential.Teal
            )
            fig_imp.update_layout(**PLOTLY_STYLE)
            st.plotly_chart(fig_imp, width='stretch')
