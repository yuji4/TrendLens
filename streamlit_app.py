import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
from datetime import date, timedelta, datetime
import atexit, os, glob, warnings
from io import BytesIO 

warnings.filterwarnings("ignore")

# ===============================
# 내부 모듈 Import
# ===============================
from analysis.api_manager import get_naver_trend_data
from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv
from analysis.metrics import mean_absolute_percentage_error, root_mean_squared_error, save_model_metrics
from analysis.modeling import run_prophet, run_arima, run_random_forest, tune_random_forest_bayesian, create_features, run_ccf_analysis
from components.ui_components import render_sidebar, setup_scheduler
from report.pdf_generator import generate_trend_report

# ===============================
# 전역 시각화 스타일
# ===============================
PLOTLY_STYLE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=14, color="#212121"),
    hovermode="x unified",
    margin=dict(l=40, r=30, t=60, b=40),
    legend=dict(orientation="h", y=-0.2)
)

# ===============================
# 기본 설정 및 스타일
# ===============================
st.set_page_config(page_title="TrendLens - 네이버 트렌드 분석", layout="wide")

st.markdown(
    """
    <style>
    h1 {
        text-align: center;
        color: #0D47A1;
        font-size: 36px !important;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976D2 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👀 TrendLens: 네이버 검색 트렌드 분석")


# ===============================s
# ⚙️ 사이드바 렌더링 및 설정 값 로드
# ===============================
keywords, time_unit, start_date, end_date, align_option, update_btn, merge_btn = render_sidebar()

if not keywords:
    st.warning("검색어를 1개 이상 입력하세요.")
    st.stop()


# ===============================
# 📦 데이터 로드 및 전처리
# ===============================
df = None

if update_btn:
    # 키워트 세트 변경 시, 이전 예측 기록 초기화
    if "model_metrics" in st.session_state:
        st.session_state["model_metrics"].clear()
        st.info("🔄 키워드 세트 변경 감지: 기존 모델 성능 데이터 초기화 완료")

    with st.spinner("데이터를 가져오는 중..."):
        try:
            data = get_naver_trend_data(
                keywords=keywords,
                start_date=str(start_date),
                end_date=str(end_date),
                time_unit=time_unit,
            )
            if not data or "results" not in data:
                st.error("선택한 조건에 데이터가 없습니다.")
            else:
                file_path = save_data_to_csv(data)
                st.success(f"✅ 최신 데이터 저장 완료: {file_path}")
                df = load_latest_csv() 
        except Exception as e:
            st.error(f"데이터 수집 중 오류: {e}")

df = load_latest_csv() if df is None else df

if merge_btn:
    merged = merge_all_csv()
    if merged.empty:
        st.warning("병합할 CSV 파일이 없습니다.")
    else:
        df = merged
        st.success(f"🗂 CSV 병합 완료")

if df is not None and not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if align_option == "공통 날짜":
        df = df.dropna(subset=[c for c in df.columns if c != "date"])


# ===============================
# 📊 메인 탭 
# ===============================
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 트렌드 비교",
        "📈 상세 분석",
        "🔗 상관 분석",
        "🔮 트렌드 예측",
        "📊 예측 모델 성능 비교",
        "⬇️ 다운로드"
    ])

    # --- 탭 1: 트렌드 비교 ---
    with tab1:
        st.caption("선택한 키워드별 검색량 추이를 이동평균을 적용하여 부드럽게 비교합니다.")
        st.subheader("📊 키워드별 트렌드 변화")
        smooth_window = st.slider("이동평균 기간", 1, 14, 1, 1)

        df_vis = df.copy()
        if smooth_window > 1:
            value_cols = [c for c in df.columns if c != "date"]
            df_vis[value_cols] = df_vis[value_cols].rolling(window=smooth_window, min_periods=1).mean()

        df_long = df_vis.melt(id_vars="date", var_name="keyword", value_name="ratio")
        fig = px.line(df_long, x="date", y="ratio", color="keyword", markers=True)
        fig.update_layout(**PLOTLY_STYLE)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_vis, use_container_width=True)

    # --- 탭 2: 상세 분석 ---
    with tab2:
        st.caption("급등·급락 변화율을 분석합니다.")
        st.subheader("📈 급상승·급하락 분석")

        df2 = df.copy().set_index("date")
        pct = df2.pct_change().reset_index()
        pct.columns = ["date"] + [f"{c}_증감률(%)" for c in df2.columns]
        
        for c in pct.columns[1:]:
            pct[c] = (pct[c] * 100).round(2)

        threshold = st.slider("급변 기준(%)", 10, 200, 50, 10)

        alerts = []
        for col in pct.columns[1:]:
            spikes = pct.loc[pct[col].abs() >= threshold, ["date", col]]
            for _, r in spikes.iterrows():
                alerts.append({
                    "키워드": col.replace("_증감률(%)", ""),
                    "날짜": r["date"].date(),
                    "유형": "급등" if r[col] > 0 else "급락",
                    "변동률(%)": round(r[col], 1)
                })

        alert_df = pd.DataFrame(alerts)

        if alert_df.empty:
            st.info("✅ 급변 변화 없음.")
        else:
            selected_kw = st.selectbox("🔍 키워드 선택", sorted(df2.columns))
            kw_alerts = alert_df[alert_df["키워드"] == selected_kw]
            if kw_alerts.empty:
                st.info(f"{selected_kw} 키워드에서 급변 없음.")
            else:
                st.dataframe(kw_alerts, use_container_width=True)
                
                fig_kw = px.line(df2.reset_index(), x="date", y=selected_kw, title=f"{selected_kw} 급등·급락 구간")
                for _, r in kw_alerts.iterrows():
                    color = "red" if r["유형"] == "급등" else "blue"
                    fig_kw.add_vline(x=r["날짜"], line_dash="dot", line_color=color)
                
                fig_kw.update_layout(**PLOTLY_STYLE)
                st.plotly_chart(fig_kw, use_container_width=True)


    # --- 탭 3: 상관 분석 ---
    with tab3:
        st.caption("키워드 간 검색 패턴 유사도를 상관계수 및 네트워크로 분석합니다.")
        st.subheader("🔗 상관관계 분석")

        # 기본 상관 분석
        corr = df.set_index("date").corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".3f",  # 소수점 셋째 자리까지 표시
            aspect="auto",
            title="키워드 간 검색 패턴 유사도 (상관 히트맵)",
            color_continuous_scale="RdBu_r"
        )

        # 레이아웃 업데이트 (PLOTLY_STYLE은 외부에서 정의되었다고 가정)
        fig_corr.update_layout(**PLOTLY_STYLE)
    
        # x축과 y축의 레이블을 중앙에 배치하여 가독성 개선
        fig_corr.update_xaxes(side="top", tickangle=0)
        fig_corr.update_yaxes(tickangle=0)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()
        st.markdown("### 🕸️ 네트워크 상관 그래프")
        threshold_net = st.slider("상관계수 임계값", 0.0, 1.0, 0.6, 0.05)
        G = nx.Graph()
        for i in corr.columns:
            for j in corr.columns:
                if i != j and abs(corr.loc[i, j]) >= threshold_net:
                    G.add_edge(i, j, weight=corr.loc[i, j])

        if len(G.edges) == 0:
            st.info(f"임계값 {threshold_net} 이상인 상관 없음.")
        else:
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y, edge_color = [], [], []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                color = "rgba(255,0,0,0.3)" if d["weight"] > 0 else "rgba(0,0,255,0.3)"
                edge_color.append(color)

            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="lightgray"))
            node_x, node_y = zip(*[pos[n] for n in G.nodes()])
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()),
                textposition="top center", marker=dict(size=25, color="#90CAF9", line=dict(width=2, color="#1565C0"))
            )
            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(title=f"키워드 네트워크 (|r| ≥ {threshold_net})", **PLOTLY_STYLE)
            st.plotly_chart(fig_net, use_container_width=True)

        # 키워드 간 교차 상관 분석
        st.divider()
        st.subheader("🔬 키워드 간 교차 상관 분석 (Cross-Correlation)")
        st.caption("두 키워드 검색량의 시간 지연(Lag)에 따른 상관관계를 분석하여 선행/후행 관계를 파악합니다.")

        # 키워드 선택
        kw_list = [c for c in df.columns if c != "date"]
        col_ccf_select = st.columns(2)
        with col_ccf_select[0]:
            kw_a = st.selectbox("키워드 A (X축)", kw_list, index=0)
        with col_ccf_select[1]:
            # 기본적으로 A와 다른 키워드를 선택하도록 설정
            default_index = 1 if len(kw_list) > 1 and kw_list[0] == kw_a else 0
            kw_b = st.selectbox("키워드 B (Y축)", kw_list, index=default_index)

        max_lag = st.slider("최대 지연 기간 (Lag, 일)", 7, min(30, len(df)//2 - 1), 14, 1)

        if kw_a == kw_b:
            st.warning("⚠️ 교차 상관 분석을 위해서는 서로 다른 두 키워드를 선택해야 합니다.")
        else:
            df_ccf = df.set_index("date").dropna()
            try:
                ccf_results = run_ccf_analysis(df_ccf[kw_a].values, df_ccf[kw_b].values, max_lags=max_lag)
                
                # Plotly 시각화
                fig_ccf = go.Figure(data=[
                    go.Bar(x=ccf_results['ccf_df']['Lag'], 
                          y=ccf_results['ccf_df']['CCF'], 
                          marker_color='#E91E63')
                ])

                fig_ccf.add_vline(x=ccf_results['optimal_lag'], 
                                 line_width=2, 
                                 line_dash="dash", 
                                 line_color="#FFC107")
                fig_ccf.add_hline(y=ccf_results['conf_level'], 
                                 line_dash="dot", 
                                 line_color="#4CAF50")
                fig_ccf.add_hline(y=-ccf_results['conf_level'], 
                                 line_dash="dot", 
                                 line_color="#4CAF50")
                
                fig_ccf.update_layout(
                    title=f"{kw_a} ↔ {kw_b} 교차 상관 함수 (CCF)",
                    xaxis_title=f"지연 (Lag, 일) | +Lag: {kw_a}가 {kw_b}를 선행",
                    yaxis_title="교차 상관 계수",
                    **PLOTLY_STYLE,
                )

                st.plotly_chart(fig_ccf, use_container_width=True)

                st.markdown("#### 🔍 분석 결과")
                if abs(ccf_results['max_correlation']) > ccf_results['conf_level']:
                    st.success(f"**최적 지연: {ccf_results['optimal_lag']}일** (상관 계수: {ccf_results['max_correlation']:.3f})")
                    st.markdown(ccf_results['analysis_text'])
                else:
                    st.info("선택한 두 키워드 간에 통계적으로 유의미한 교차 상관 관계는 발견되지 않았습니다.")

            except Exception as e:
                st.error(f"CCF 분석 중 오류가 발생했습니다: {str(e)}")

    # --- 탭 4: 예측 ---
    with tab4:
        st.caption("Prophet / ARIMA / Random Forest 기반 미래 검색 트렌드 예측 및 비교 분석을 제공합니다.")
        st.subheader("🔮 트렌드 예측")
        model_type = st.radio("모델 선택", ["Prophet", "ARIMA", "Random Forest"], horizontal=True)
        selected_kw = st.selectbox("예측할 키워드", [c for c in df.columns if c != "date"])
        days_ahead = st.slider("예측 기간 (일)", 7, 180, 30, 7)
        df_forecast = df[["date", selected_kw]].rename(columns={"date": "ds", selected_kw: "y"})

        if model_type == "Random Forest":
            st.markdown("#### 🌲 Random Forest 하이퍼파라미터 튜닝 설정")
            tune = st.checkbox("Bayesian Optimizatin 기반 하이퍼파라미터 튜닝 실행", value=False)

            if tune: 
                n_trials = st.slider("탐색 시도 횟수", 10, 50, 25, 5)
            else:
                n_trials = None

        if st.button("🚀 예측 실행", type="primary"):
            with st.spinner("예측 중..."):
                try:
                    if model_type == "Prophet":
                        model, forecast = run_prophet(df_forecast, days_ahead)
                    
                        y_true = df_forecast['y'].values
                        y_pred = forecast['yhat'].head(len(y_true)).values
                    
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="예측값",
                                                 line=dict(color="royalblue", width=2)))
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0),
                                                 fill=None, showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                                                 fill="tonexty", fillcolor="rgba(135,206,250,0.2)", line=dict(width=0),
                                                 name="신뢰구간"))
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], mode="lines+markers",
                                                 name="실제값", line=dict(color="black", width=3)))
                        fig.update_layout(title=f"{selected_kw} {days_ahead}일 예측 (Prophet)", **PLOTLY_STYLE)
                        st.plotly_chart(fig, use_container_width=True)

                        mape = mean_absolute_percentage_error(y_true, y_pred)
                        rmse = root_mean_squared_error(y_true, y_pred)
                        save_model_metrics("Prophet", selected_kw, mape, rmse)

                        st.markdown("#### 🌟 모델 성능 지표")
                        col_metrics = st.columns(2)
                        col_metrics[0].metric(label="MAPE (Mean Absolute Percentage Error)", value=f"{mape:.2f}%")
                        col_metrics[1].metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")
                        st.caption("MAPE와 RMSE는 예측 기간을 제외한 과거 데이터에 대한 모델의 적합도를 나타냅니다.")
                    
                        # 트렌드 분해 시각화
                        st.divider()
                        st.subheader("✨ 트렌드 분해 분석 (Prophet)")
                        st.caption("검색량 데이터에서 장기 추세, 연간 계절성, 주간 계절성을 분리하여 보여줍니다.")
                        
                        fig_trend = px.line(forecast, x="ds", y="trend", title="장기 추세 (Trend)", color_discrete_sequence=['#4CAF50'])
                        fig_trend.update_layout(plot_bgcolor="white", paper_bgcolor="#F5F5F5", font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        fig_trend.update_yaxes(title_text="영향도")
                        
                        df_yearly_pattern = forecast[['ds', 'yearly']].tail(365).copy() 
                        fig_yearly = go.Figure()
                        fig_yearly.add_trace(go.Scatter(x=df_yearly_pattern["ds"], y=df_yearly_pattern["yearly"], mode="lines", name="연간 계절성", line=dict(color="#2196F3")))
                        fig_yearly.update_layout(title="연간 계절성 (Yearly Seasonality)", plot_bgcolor="white", paper_bgcolor="#F5F5F5", font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        fig_yearly.update_xaxes(title_text="날짜", tickformat="%m-%d") 
                        fig_yearly.update_yaxes(title_text="영향도")
                        
                        df_weekly = forecast[["ds", "weekly"]].tail(7).copy()
                        day_names_kr = ['월', '화', '수', '목', '금', '토', '일']
                        df_weekly['day_name_kr'] = df_weekly['ds'].dt.day_name(locale='en').map({
                            'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 'Thursday': '목', 
                            'Friday': '금', 'Saturday': '토', 'Sunday': '일'
                        })
                        df_weekly['day_name_kr'] = pd.Categorical(df_weekly['day_name_kr'], categories=day_names_kr, ordered=True)
                        df_weekly = df_weekly.sort_values('day_name_kr')

                        fig_weekly = px.bar(df_weekly, x="day_name_kr", y="weekly", title="주간 계절성 (Weekly Seasonality)",
                                                color_discrete_sequence=['#FFC107'])
                        fig_weekly.update_layout(plot_bgcolor="white", paper_bgcolor="#F5F5F5", font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        fig_weekly.update_xaxes(title_text="요일", categoryorder='array', categoryarray=day_names_kr)
                        fig_weekly.update_yaxes(title_text="영향도")
                        
                        cols_comp = st.columns(3)
                        with cols_comp[0]:
                            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
                        with cols_comp[1]:
                            st.plotly_chart(fig_yearly, use_container_width=True, config={'displayModeBar': False})
                        with cols_comp[2]:
                            st.plotly_chart(fig_weekly, use_container_width=True, config={'displayModeBar': False})

                    elif model_type == "ARIMA":
                        with st.spinner("ARIMA 모델 예측 중..."):
                            try:
                                # 모든 ARIMA 관련 로직을 modeling.py의 함수로 대체
                                forecast_df, y_true, y_pred_past = run_arima(df_forecast, days_ahead)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], 
                                                       mode="lines+markers",
                                                       name="실제값", 
                                                       line=dict(color="black", width=3)))
                                fig.add_trace(go.Scatter(x=forecast_df["날짜"], 
                                                       y=forecast_df["예측값"], 
                                                       mode="lines",
                                                       name="예측값", 
                                                       line=dict(color="royalblue", width=2.5, dash="dot")))
                                fig.update_layout(title=f"ARIMA 기반 {selected_kw} {days_ahead}일 예측", **PLOTLY_STYLE)
                                st.plotly_chart(fig, use_container_width=True)

                                mape = mean_absolute_percentage_error(y_true, y_pred_past)
                                rmse = root_mean_squared_error(y_true, y_pred_past)
                                save_model_metrics("ARIMA", selected_kw, mape, rmse)
                                
                                # 성능 지표 표시
                                st.markdown("#### 🌟 모델 성능 지표")
                                col_metrics = st.columns(2)
                                col_metrics[0].metric(label="MAPE", value=f"{mape:.2f}%")
                                col_metrics[1].metric(label="RMSE", value=f"{rmse:.2f}")
                                st.caption("MAPE와 RMSE는 훈련 데이터에 대한 모델의 적합도를 나타냅니다.")
                                
                            except Exception as e:
                                st.error(f"❌ ARIMA 예측 오류: {str(e)}")

                    elif model_type == "Random Forest":
                        tuned_model = None
                        st.subheader("🌲 Random Forest 예측 및 Bayesian 튜닝")
                        
                        if tune:
                            with st.spinner("Optuna Bayesian Optimization 튜닝 중... ⏳"):
                                train_df_rf = create_features(df_forecast.copy())
                                features_x_rf = [c for c in train_df_rf.columns if c not in ['ds', 'y']]
                                X_train_rf, y_train_rf = train_df_rf[features_x_rf], train_df_rf['y']
                                
                                best_model, best_params, best_score = tune_random_forest_bayesian(X_train_rf, y_train_rf, n_trials=n_trials)
                            
                            st.success("🎯 Bayesian Optimization 완료!")
                            st.json(best_params)
                            st.caption(f"최적 MSE: {best_score:.4f}")
                            tuned_model = best_model

                        forecast_df, y_true, y_pred_past, feature_importances, features = run_random_forest(df_forecast, days_ahead, tuned_model=tuned_model)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], mode="lines+markers",
                                                 name="실제값", line=dict(color="black", width=3)))
                        fig.add_trace(go.Scatter(x=forecast_df["날짜"], y=forecast_df["예측값"], mode="lines",
                                                 name="예측값", line=dict(color="#FF5722", width=2.5, dash="dot")))
                        fig.update_layout(title=f"Random Forest 기반 {selected_kw} {days_ahead}일 예측", **PLOTLY_STYLE)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        mape = mean_absolute_percentage_error(y_true, y_pred_past) 
                        rmse = root_mean_squared_error(y_true, y_pred_past)
                        save_model_metrics("Random Forest", selected_kw, mape, rmse)
        
                        st.markdown("#### 🌟 모델 성능 지표")
                        col_metrics = st.columns(2)
                        col_metrics[0].metric(label="MAPE (Mean Absolute Percentage Error)", value=f"{mape:.2f}%")
                        col_metrics[1].metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")
                        st.caption("MAPE와 RMSE는 훈련 데이터에 대한 모델의 적합도를 나타냅니다.")

                        st.divider()
                        st.subheader("💡 피처 중요도 분석 (Random Forest)")
                        st.caption("모델 예측에 가장 큰 영향을 미친 시간 피처의 기여도를 보여줍니다.")
                        
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=True)
                        
                        fig_import = px.bar(
                            importance_df, x='Importance', y='Feature', orientation='h',
                            title='검색량 예측에 기여한 시간 요인', color='Importance',
                            color_continuous_scale=px.colors.sequential.Teal
                        )
                        fig_import.update_layout(
                            plot_bgcolor='white', paper_bgcolor='#F5F5F5',
                            margin=dict(l=20, r=20, t=30, b=20), font=dict(size=12)
                        )
                        st.plotly_chart(fig_import, use_container_width=True, config={'displayModeBar': False})
                        
                except Exception as e:
                    st.error(f"❌ 예측 오류: {e}")

    # --- 탭 5: 모델 성능 비교 ---
    with tab5:
        st.subheader("📊 모델별 성능 비교 대시보드")

        if "model_metrics" not in st.session_state or len(st.session_state["model_metrics"]) == 0:
            st.info("아직 저장된 모델 성능 데이터가 없습니다. 예측을 먼저 실행하세요.")
        else:
            df_metrics = pd.DataFrame(st.session_state["model_metrics"])
            
            available_keywords = df_metrics["키워드"].unique()
            try:
                default_index = list(available_keywords).index(selected_kw)
            except ValueError:
                default_index = 0
            selected_comparison_kw = st.selectbox("키워드 선택 (비교 대상)", available_keywords, index=default_index)

            df_filtered = df_metrics[df_metrics["키워드"] == selected_comparison_kw]
            st.dataframe(df_filtered, use_container_width=True)
            
            if not df_filtered.empty:
                best_row = df_filtered.loc[df_filtered["RMSE"].idxmin()]
                st.success(f"🏆 키워드 **'{selected_comparison_kw}'**에 대한 최적 모델: **{best_row['모델명']}** (RMSE {best_row['RMSE']:.4f})")

                st.markdown("#### RMSE 비교")
                fig_rmse = px.bar(df_filtered, x="모델명", y="RMSE", color="모델명",
                                    text="RMSE", title=f"'{selected_comparison_kw}' 모델별 RMSE 비교", color_discrete_sequence=px.colors.qualitative.Set2)
                fig_rmse.update_layout(**PLOTLY_STYLE)
                st.plotly_chart(fig_rmse, use_container_width=True)

                st.markdown("### MAPE 비교")
                fig_mape = px.bar(df_filtered, x="모델명", y="MAPE(%)", color="모델명",
                                    text="MAPE(%)", title=f"'{selected_comparison_kw}' 모델별 MAPE 비교", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_mape.update_layout(**PLOTLY_STYLE)
                st.plotly_chart(fig_mape, use_container_width=True)
            else:
                st.info(f"키워드 '{selected_comparison_kw}'에 대해 저장된 측정값이 없습니다. 예측을 실행하여 저장하세요.")

    # --- 탭 6: 다운로드 ---
    with tab6:
        st.subheader("⬇️ 데이터 및 리포트 다운로드")

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 최신 데이터 다운로드", csv, "trend_data_latest.csv", "text/csv")

        st.divider()
        st.markdown("💾 모델 성능 리포트 (PDF 생성)")

        if "model_metrics" not in st.session_state or len(st.session_state["model_metrics"]) == 0:
            st.info("모델 성능 데이터가 없습니다. 예측을 먼저 실행하세요.")
        else:
            if st.button("🧾 PDF 리포트 생성", type="primary"):
                try:
                    buffer = generate_trend_report(
                        df=df,
                        keywords=keywords,
                        start_date=start_date,
                        end_date=end_date,
                        time_unit=time_unit,
                        model_metrics=st.session_state.get("model_metrics", [])
                    )
                    
                    st.download_button(
                        label="📥 리포트 다운로드 (PDF)",
                        data=buffer,
                        file_name=f"TrendLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF 리포트 생성 완료!")
                    
                except Exception as e:
                    st.error(f"PDF 생성 중 오류 발생: {str(e)}")

else:
    st.info("좌측에서 검색어를 입력하고 '업데이트'를 눌러주세요.")

# ===============================
# ⏰ 자동 업데이트 스케줄러
# ===============================
# 분리된 함수를 메인 앱의 import된 함수와 연결하여 호출
setup_scheduler()
