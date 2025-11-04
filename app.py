import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit, os, glob, warnings
# Prophet, ARIMA, networkx는 모두 유지
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx 

# matplotlib.pyplot은 예측 탭에서 사용하므로 전역으로 import
import matplotlib.pyplot as plt 

warnings.filterwarnings("ignore")

# 내부 모듈 (두 코드 모두 동일하게 필요)
from analysis.api_manager import get_naver_trend_data
from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv


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
            # session_state에 저장
            st.session_state["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ [자동 수집 완료] {file_path}")
        else:
            print("⚠️ [자동 수집 실패] 응답 없음")
    except Exception as e:
        print(f"❌ 자동 업데이트 오류: {e}")


# ===============================
# 전역 스타일 및 기본 설정
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
    st.markdown("### 📊 데이터 옵션") # 시각화 옵션에서 데이터 옵션으로 변경

    align_option = st.radio(
        "날짜 정렬 기준",
        ["모든 날짜", "공통 날짜"],
        index=0,
        help="모든 날짜를 표시하거나, 모든 키워드에 값이 존재하는 날짜만 표시할 수 있습니다."
    )
    
    # smooth_window 슬라이더 제거 -----------------------------------
    # smooth_window = st.slider(...)
    # -------------------------------------------------------------

    st.divider()
    st.markdown("### 🪄 데이터 관리")
    colA, colB = st.columns(2)
    with colA:
        update_btn = st.button("🔄 업데이트", width='stretch')
    with colB:
        merge_btn = st.button("🗂 CSV 병합", width='stretch')

    st.divider()
    st.markdown("### 🕒 자동 수집 상태")

    if st.session_state.get("last_update_time"):
        st.success(f"마지막 수집: {st.session_state['last_update_time']}")
    else:
        st.info("자동 수집 기록이 없습니다.")

    # 최근 자동 수집 로그 요약 (7일치)
    st.markdown("#### 📈 최근 자동 수집 로그")
    csv_files = sorted(glob.glob("data/trend_data_*.csv"), key=os.path.getctime, reverse=True)
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

# 데이터 업데이트
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
                df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"데이터 수집 중 오류: {e}")

# 최근 CSV 불러오기
if df is None:
    df = load_latest_csv()

# 데이터 병합
if merge_btn:
    merged = merge_all_csv()
    if merged.empty:
        st.warning("병합할 CSV 파일이 없습니다.")
    else:
        merged_path = f"data/merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        df = merged
        st.success(f"🗂 CSV 병합 완료 → {merged_path}")

# 공통 전처리 및 옵션 적용
if df is not None and not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 공통 날짜 필터링 적용 (이동평균은 Tab 1으로 이동)
    if align_option == "공통 날짜":
         df = df.dropna(subset=[c for c in df.columns if c != "date"])


# ===============================
# 📊 메인 탭 (기능 통합)
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
        
        # 이동평균 슬라이더를 Tab 1 내부로 이동
        smooth_window = st.slider(
            "이동평균 적용 기간 (그래프 부드럽게)",
            min_value=1, max_value=14, value=1, step=1,
            help="값을 1보다 크게 하면 트렌드 그래프가 부드럽게 표시됩니다."
        )
        
        df_vis = df.copy()
        
        # Tab 1에서만 이동평균 적용
        if smooth_window > 1:
            value_cols = [c for c in df.columns if c != "date"]
            df_vis[value_cols] = df_vis[value_cols].rolling(window=smooth_window, min_periods=1).mean()
        
        df_long = df_vis.melt(id_vars="date", var_name="keyword", value_name="ratio")
        fig = px.line(df_long, x="date", y="ratio", color="keyword", markers=True)
        fig.update_layout(plot_bgcolor="white", font=dict(size=14))
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("#### 원본/이동평균 적용 데이터")
        st.dataframe(df_vis, width='stretch')

    # --- 탭 2: 상세 분석 (급등/급락, 정규화, 증감률) ---
    with tab2:
        st.caption("일별 증감률, 정규화된 데이터, 그리고 급변 지점을 감지합니다.")
        st.subheader("📈 트렌드 급상승·급하락 감지 및 상세 분석")
        
        # df2는 이동평균이 적용되지 않은 df를 사용해야 정확한 증감률 계산 가능
        df2 = df.copy().set_index("date") 
        
        # 1. 증감률 계산
        pct = df2.pct_change(fill_method=None).reset_index()
        pct.columns = ["date" if c == "date" else f"{c}_증감률(%)" for c in pct.columns]
        for c in pct.columns:
            if c != "date":
                pct[c] = (pct[c] * 100).round(2)

        # 2. 급변 감지
        threshold = st.slider("급변 기준(%)", 10, 200, 50, step=10, key="tab2_threshold")
        alerts = []
        for col in pct.columns:
            if col != "date":
                spikes = pct.loc[pct[col].abs() >= threshold, ["date", col]]
                for _, row in spikes.iterrows():
                    change = row[col]
                    direction = "급등" if change > 0 else "급락"
                    alerts.append({
                        "키워드": col.replace("_증감률(%)", ""),
                        "날짜": row["date"].date(),
                        "유형": direction,
                        "변동률(%)": round(change, 1)
                    })

        if alerts:
            alert_df = pd.DataFrame(alerts).sort_values(["키워드", "날짜"])
            st.warning(f"⚠️ 감지된 급상승/급하락 이벤트: {len(alert_df)}건")
            st.dataframe(alert_df, width='stretch', height=200)
        else:
            st.info("✅ 설정된 임계값 내 급변 변화 없음.")
            
        st.divider()
        
        # 3. 정규화 계산
        scaled = df2.copy()
        for col in [c for c in df2.columns if c != "date"]:
            minv, maxv = scaled[col].min(), scaled[col].max()
            scaled[col] = (scaled[col] - minv) / (maxv - minv) if (maxv - minv) != 0 else 0
        scaled = scaled.reset_index()
        scaled.columns = ["date"] + [f"{c}_정규화(0~1)" for c in df2.columns]
        
        # 4. 정규화 그래프 (두 번째 코드 기능)
        df_scaled_long = scaled.melt(id_vars="date", var_name="metric", value_name="value")
        fig_scaled = px.line(
            df_scaled_long, x="date", y="value", color="metric", title="정규화(0~1) 추세"
        )
        fig_scaled.update_layout(plot_bgcolor='white', font=dict(size=14))
        st.plotly_chart(fig_scaled, width='stretch')


    # --- 탭 3: 상관 분석 (히트맵 + 네트워크) ---
    with tab3:
        st.caption("키워드 간의 검색량 패턴 유사도를 상관계수를 통해 분석하고, 네트워크 형태로 시각화합니다.")
        st.subheader("🔗 키워드 상관관계 분석")
        
        # 이동평균이 적용되지 않은 df 사용
        corr = df.set_index("date").corr() 
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), width='stretch')
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig_corr, width='stretch')
        
        st.divider()
        st.markdown("### 🕸️ 네트워크 상관관계 그래프")
        
        threshold_net = st.slider("상관계수 임계값 (네트워크)", 0.0, 1.0, 0.6, 0.05, key="net_threshold")
        G = nx.Graph()
        
        for i in corr.columns:
            G.add_node(i)
        for i in corr.columns:
            for j in corr.columns:
                if i != j and abs(corr.loc[i, j]) >= threshold_net:
                    G.add_edge(i, j, weight=corr.loc[i, j])

        if len(G.edges) == 0:
            st.info(f"임계값 {threshold_net} 이상인 상관관계가 없습니다.")
        else:
            pos = nx.spring_layout(G, seed=42, k=0.5)

            edge_x, edge_y, edge_text = [], [], []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_text.append(f"{u} ↔ {v}: {data['weight']:.2f}")

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode='lines', line=dict(width=1.5, color='lightgray'),
                hoverinfo='text', text=edge_text, hoverlabel=dict(bgcolor='white')
            )

            node_x, node_y, node_size, node_text = [], [], [], []
            for node in G.nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                deg = len(list(G.neighbors(node)))
                avg_weight = (
                    sum(abs(G[node][nbr]['weight']) for nbr in G.neighbors(node)) / deg
                    if deg > 0 else 0
                )
                node_size.append(15 + avg_weight * 30)
                node_text.append(f"{node}<br>연결 {deg}개<br>평균 상관도 {avg_weight:.2f}")

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', text=list(G.nodes),
                textposition="top center", hoverinfo="text", hovertext=node_text,
                marker=dict(size=node_size, color='skyblue', line=dict(width=2, color='DarkSlateGrey'))
            )

            fig_network = go.Figure(data=[edge_trace, node_trace])
            fig_network.update_layout(
                title=f"키워드 네트워크 (|r| ≥ {threshold_net})",
                showlegend=False, hovermode='closest', height=650,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                plot_bgcolor='white', font=dict(size=14)
            )
            st.plotly_chart(fig_network, width='stretch')


    # --- 탭 4: 트렌드 예측 (Prophet / ARIMA) ---
    with tab4:
        st.caption("과거 데이터를 기반으로 향후 트렌드를 예측하고, 신뢰구간을 시각적으로 표시합니다.")
        st.subheader("🔮 미래 트렌드 예측 (Prophet / ARIMA)")
        
        model_type = st.radio("예측 모델 선택", ["Prophet", "ARIMA"], horizontal=True)
        selected_kw = st.selectbox("예측할 키워드 선택", [c for c in df.columns if c != "date"])
        days_ahead = st.slider("예측 기간 (일)", 7, 180, 30, step=7)
        # 예측 데이터는 항상 원본 데이터를 기반으로 해야 함 (이동평균 미적용)
        df_forecast = df[["date", selected_kw]].rename(columns={"date": "ds", selected_kw: "y"})

        # 캐싱된 예측 함수 (Prophet)
        @st.cache_data(show_spinner=False)
        def run_prophet_forecast(df, days):
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            return model, forecast

        # 캐싱된 예측 함수 (ARIMA)
        @st.cache_data(show_spinner=False)
        def run_arima_forecast(df, days):
            model = ARIMA(df.set_index("ds"), order=(3, 1, 2))
            fitted = model.fit()
            future_index = pd.date_range(df["ds"].iloc[-1], periods=days + 1, freq="D")[1:]
            forecast = fitted.forecast(steps=days)
            forecast_df = pd.DataFrame({"날짜": future_index, "예측값": forecast})
            return forecast_df

        if st.button("🚀 예측 실행", type="primary"):
            with st.spinner("예측 중..."):
                try:
                    if model_type == "Prophet":
                        model, forecast = run_prophet_forecast(df_forecast, days_ahead)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                                                     line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                                                     fill="tonexty", fillcolor="rgba(135,206,250,0.25)",
                                                     line=dict(width=0), name="신뢰구간"))
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                                     mode="lines", name="예측값",
                                                     line=dict(color="royalblue", width=2.5, dash="dot")))
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"],
                                                     mode="lines+markers", name="실제값",
                                                     line=dict(color="black", width=3), marker=dict(size=4)))
                        fig.update_layout(title=f"{selected_kw} {days_ahead}일 예측 (Prophet)",
                                                 plot_bgcolor="white", hovermode="x unified", font=dict(size=14))
                        st.plotly_chart(fig, width='stretch')
                        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days_ahead), width='stretch')

                        with st.expander("📉 트렌드 및 계절성 분해 보기"):
                            comp_fig = model.plot_components(forecast)
                            st.pyplot(comp_fig)
                            plt.close(comp_fig) # Streamlit 경고 방지

                    else: # ARIMA
                        forecast_df = run_arima_forecast(df_forecast, days_ahead)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"],
                                                     mode="lines+markers", name="실제값", line=dict(color="black", width=3)))
                        fig.add_trace(go.Scatter(x=forecast_df["날짜"], y=forecast_df["예측값"],
                                                     mode="lines", name="예측값", line=dict(color="royalblue", width=2.5, dash="dot")))
                        fig.update_layout(title=f"ARIMA 기반 {selected_kw} {days_ahead}일 예측",
                                                 plot_bgcolor="white", hovermode="x unified", font=dict(size=14))
                        st.plotly_chart(fig, width='stretch')
                        st.dataframe(forecast_df, width='stretch')

                except Exception as e:
                    st.error(f"❌ 예측 중 오류 발생: {e}")

    # --- 탭 5: CSV 다운로드 ---
    with tab5:
        st.caption("현재 로드된 분석 데이터를 CSV 파일로 다운로드합니다.")
        st.subheader("⬇️ CSV 다운로드")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 최신 데이터 다운로드", csv, "trend_data_latest.csv", "text/csv")

else:
    st.info("좌측에서 검색어를 입력하고 '데이터 업데이트'를 눌러주세요.")


# ===============================
# ⏰ 자동 업데이트 스케줄러
# ===============================
scheduler = BackgroundScheduler()
scheduler.add_job(auto_update_job, "interval", hours=24)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())