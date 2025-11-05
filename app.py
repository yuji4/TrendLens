import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit, os, glob, warnings
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 내부 모듈 (이 모듈들은 사용자의 환경에 맞게 존재해야 합니다)
# from analysis.api_manager import get_naver_trend_data
# from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv

# 로컬 환경에서 테스트를 위해 더미 함수 정의 (원본 모듈이 없는 경우)
def get_naver_trend_data(keywords, start_date, end_date, time_unit, gender):
    # 실제 API 호출 로직 대신 더미 데이터 반환
    return {"results": [{"data": [{"period": str(date.today() - timedelta(days=i)), "ratio": 100 - i * 0.5, "group": keywords[0]} for i in range(90)]}]}

def save_data_to_csv(data):
    # 실제 CSV 저장 로직 대신 더미 파일 경로 반환
    return "data/trend_data_latest.csv"

def load_latest_csv():
    # 실제 CSV 로드 로직 대신 더미 데이터프레임 반환
    dates = pd.date_range(end=date.today(), periods=90)
    data = {
        "date": dates,
        "Python": [70 + i % 10 + (i % 7) * 2 for i in range(90)],
        "AI": [80 + (90 - i) % 10 + (i % 5) * 3 for i in range(90)],
        "Study": [50 + (i % 20) for i in range(90)],
    }
    df = pd.DataFrame(data)
    # Prophet을 위한 최소 데이터 요구사항 충족
    if len(df) < 14:
        return pd.DataFrame()
    return df

def merge_all_csv():
    # 실제 CSV 병합 로직 대신 빈 데이터프레임 반환
    return pd.DataFrame()


# ===============================
# 🔁 자동 업데이트 함수
# ===============================
def auto_update_job():
    try:
        keywords = ["Python", "AI", "Study"]
        today = date.today()
        start = today - timedelta(days=7)
        data = get_naver_trend_data(
            keywords=keywords,
            start_date=str(start),
            end_date=str(today),
            time_unit="date",
            gender="",
        )
        if data and "results" in data:
            file_path = save_data_to_csv(data)
            st.session_state["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ [자동 수집 완료] {file_path}")
        else:
            print("⚠️ [자동 수집 실패] 응답 없음")
    except Exception as e:
        print(f"❌ 자동 업데이트 오류: {e}")


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


# ===============================
# ⚙️ 사이드바
# ===============================
with st.sidebar:
    st.markdown("### ⚙️ 기본 설정")
    raw_keywords = st.text_input("검색어 입력 (쉼표로 구분)", "Python, AI, Study")
    time_unit = st.selectbox("데이터 단위", ["date", "week", "month"])

    today = date.today()
    default_start = today - timedelta(days=90)
    start_date, end_date = st.date_input("조회 기간 선택", (default_start, today))

    gender_display = st.radio("성별 선택", ["전체", "남성", "여성"], horizontal=True)
    gender = {"전체": "", "남성": "m", "여성": "f"}[gender_display]

    st.divider()
    st.markdown("### 📊 데이터 옵션")
    align_option = st.radio("날짜 정렬 기준", ["모든 날짜", "공통 날짜"], index=0)

    st.divider()
    st.markdown("### 🪄 데이터 관리")
    colA, colB = st.columns(2)
    with colA:
        update_btn = st.button("🔄 업데이트", use_container_width=True)
    with colB:
        merge_btn = st.button("🗂 CSV 병합", use_container_width=True)

    st.divider()
    st.markdown("### 🕒 자동 수집 상태")
    if st.session_state.get("last_update_time"):
        st.success(f"마지막 수집: {st.session_state['last_update_time']}")
    else:
        st.info("자동 수집 기록이 없습니다.")

    st.markdown("#### 📈 최근 자동 수집 로그")
    # glob은 실제 파일 시스템에 의존하므로, 로컬에서 실행 시 경로를 확인해야 합니다.
    csv_files = sorted(glob.glob("data/trend_data_*.csv"), key=os.path.getctime, reverse=True) if os.path.exists("data") else []
    log_df = pd.DataFrame([
        {"파일": os.path.basename(f), "생성시각": datetime.fromtimestamp(os.path.getctime(f))}
        for f in csv_files
    ])
    if not log_df.empty:
        log_df = log_df[log_df["생성시각"] > datetime.now() - timedelta(days=7)]
        for _, row in log_df.head(3).iterrows():
            st.markdown(
                f"<div style='font-size:13px; padding:4px 0;'>"
                f"📂 <b>{row['파일']}</b><br>"
                f"⏰ {row['생성시각'].strftime('%Y-%m-%d %H:%M:%S')}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("최근 로그 없음.")


# ===============================
# 📦 데이터 로드 및 전처리
# ===============================
keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
if not keywords:
    st.warning("검색어를 1개 이상 입력하세요.")
    st.stop()

df = None

if update_btn:
    with st.spinner("데이터를 가져오는 중..."):
        try:
            data = get_naver_trend_data(
                keywords=keywords,
                start_date=str(start_date),
                end_date=str(end_date),
                time_unit=time_unit,
                gender=gender,
            )
            if not data or "results" not in data:
                st.error("선택한 조건에 데이터가 없습니다.")
            else:
                file_path = save_data_to_csv(data)
                st.success(f"✅ 최신 데이터 저장 완료: {file_path}")
                df = load_latest_csv() # 더미 함수 사용
        except Exception as e:
            st.error(f"데이터 수집 중 오류: {e}")

if df is None:
    df = load_latest_csv()

if merge_btn:
    merged = merge_all_csv()
    if merged.empty:
        st.warning("병합할 CSV 파일이 없습니다.")
    else:
        # 실제 파일 경로 대신 더미 처리
        merged_path = f"data/merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # merged.to_csv(merged_path, index=False, encoding="utf-8-sig") # 실제 저장 로직 주석 처리
        df = merged
        st.success(f"🗂 CSV 병합 완료 → (파일 경로 생략)")

if df is not None and not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if align_option == "공통 날짜":
        df = df.dropna(subset=[c for c in df.columns if c != "date"])


# ===============================
# 📊 메인 탭
# ===============================
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 트렌드 비교",
        "📈 상세 분석",
        "🔗 상관 분석",
        "🔮 트렌드 예측",
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
        st.caption("급등·급락 변화율과 정규화 데이터를 분석합니다.")
        st.subheader("📈 급상승·급하락 분석")

        view_mode = st.radio("분석 보기 모드", ["전체 요약 보기", "키워드별 상세 보기"], horizontal=True)
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
            if view_mode == "전체 요약 보기":
                st.warning(f"⚠️ 감지된 급변 이벤트 {len(alert_df)}건")
                st.dataframe(alert_df, use_container_width=True)
                summary = alert_df.groupby(["키워드", "유형"]).size().unstack(fill_value=0)
                st.markdown("#### 📊 키워드별 급등/급락 요약")
                st.dataframe(summary, use_container_width=True)
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

        st.divider()
        scaled = df2.copy()
        for col in df2.columns:
            minv, maxv = scaled[col].min(), scaled[col].max()
            scaled[col] = (scaled[col] - minv) / (maxv - minv) if maxv != minv else 0
        scaled = scaled.reset_index()
        df_scaled_long = scaled.melt(id_vars="date", var_name="metric", value_name="value")
        fig_scaled = px.line(df_scaled_long, x="date", y="value", color="metric", title="정규화(0~1) 추세")
        fig_scaled.update_layout(**PLOTLY_STYLE)
        st.plotly_chart(fig_scaled, use_container_width=True)

    # --- 탭 3: 상관 분석 ---
    with tab3:
        st.caption("키워드 간 검색 패턴 유사도를 상관계수 및 네트워크로 분석합니다.")
        st.subheader("🔗 상관관계 분석")
        corr = df.set_index("date").corr()
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(**PLOTLY_STYLE)
        st.plotly_chart(fig_corr, use_container_width=True)

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

    # --- 탭 4: 예측 ---
    with tab4:
        st.caption("Prophet / ARIMA 기반 미래 검색 트렌드 예측")
        st.subheader("🔮 트렌드 예측")
        model_type = st.radio("모델 선택", ["Prophet", "ARIMA"], horizontal=True)
        selected_kw = st.selectbox("예측할 키워드", [c for c in df.columns if c != "date"])
        days_ahead = st.slider("예측 기간 (일)", 7, 180, 30, 7)
        df_forecast = df[["date", selected_kw]].rename(columns={"date": "ds", selected_kw: "y"})

        @st.cache_data
        def run_prophet(df, days):
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            return model, forecast

        @st.cache_data
        def run_arima(df, days):
            model = ARIMA(df.set_index("ds"), order=(3, 1, 2))
            fitted = model.fit()
            future_idx = pd.date_range(df["ds"].iloc[-1], periods=days + 1, freq="D")[1:]
            forecast = fitted.forecast(steps=days)
            return pd.DataFrame({"날짜": future_idx, "예측값": forecast})

        if st.button("🚀 예측 실행", type="primary"):
            with st.spinner("예측 중..."):
                try:
                    if model_type == "Prophet":
                        model, forecast = run_prophet(df_forecast, days_ahead)
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

                        # =========================================================
                        # ✨ 1. Prophet 기반 계절성 및 추세 분해 시각화 (수정 및 개선)
                        # =========================================================
                        st.divider()
                        st.subheader("✨ 트렌드 분해 분석 (Prophet)")
                        st.caption("검색량 데이터에서 장기 추세, 연간 계절성, 주간 계절성을 분리하여 보여줍니다.")

                        # -------------------- 1. 장기 추세 (Trend) --------------------
                        fig_trend = px.line(forecast, x="ds", y="trend", title="장기 추세 (Trend)",
                                            color_discrete_sequence=['#4CAF50'])
                        fig_trend.update_layout(plot_bgcolor="white", paper_bgcolor="#F5F5F5", font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        fig_trend.update_yaxes(title_text="영향도")
                        
                        # -------------------- 2. 연간 계절성 (Yearly) --------------------
                        # Prophet의 연간 계절성 패턴만 추출 (1년치 데이터)
                        # Prophet은 예측 기간이 짧아도 전체 패턴을 보여주므로, 전체 forecast를 사용하거나,
                        # 시계열의 패턴을 보여주는 관점에서 1년치 패턴을 추출하여 시각화합니다.
                        df_yearly_pattern = forecast[['ds', 'yearly']].tail(365).copy() 
                        fig_yearly = go.Figure()
                        fig_yearly.add_trace(go.Scatter(x=df_yearly_pattern["ds"], y=df_yearly_pattern["yearly"], mode="lines", name="연간 계절성", line=dict(color="#2196F3")))
                        fig_yearly.update_layout(title="연간 계절성 (Yearly Seasonality)", plot_bgcolor="white", paper_bgcolor="#F5F5F5", font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        fig_yearly.update_xaxes(title_text="날짜", tickformat="%m-%d") 
                        fig_yearly.update_yaxes(title_text="영향도")
                        
                        # -------------------- 3. 주간 계절성 (Weekly) --------------------
                        # Prophet의 주간 계절성 패턴만 추출 (7일 데이터)
                        df_weekly = forecast[["ds", "weekly"]].tail(7).copy()
                        
                        # 요일별 정렬을 위해 요일 이름 및 순서 정의 (한국어)
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
                        
                        # -------------------- 4. 3분할 컬럼에 차트 표시 --------------------
                        cols_comp = st.columns(3)
                        with cols_comp[0]:
                            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
                        with cols_comp[1]:
                            st.plotly_chart(fig_yearly, use_container_width=True, config={'displayModeBar': False})
                        with cols_comp[2]:
                            st.plotly_chart(fig_weekly, use_container_width=True, config={'displayModeBar': False})
                        # =========================================================

                    else:
                        forecast_df = run_arima(df_forecast, days_ahead)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], mode="lines+markers",
                                                 name="실제값", line=dict(color="black", width=3)))
                        fig.add_trace(go.Scatter(x=forecast_df["날짜"], y=forecast_df["예측값"], mode="lines",
                                                 name="예측값", line=dict(color="royalblue", width=2.5, dash="dot")))
                        fig.update_layout(title=f"ARIMA 기반 {selected_kw} {days_ahead}일 예측", **PLOTLY_STYLE)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 예측 오류: {e}")

    # --- 탭 5: 다운로드 ---
    with tab5:
        st.subheader("⬇️ CSV 다운로드")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 최신 데이터 다운로드", csv, "trend_data_latest.csv", "text/csv")

else:
    st.info("좌측에서 검색어를 입력하고 '업데이트'를 눌러주세요.")


# ===============================
# ⏰ 자동 업데이트 스케줄러
# ===============================
scheduler = BackgroundScheduler()
scheduler.add_job(auto_update_job, "interval", hours=24)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())