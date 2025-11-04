import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import os, glob
import networkx as nx
import plotly.graph_objects as go

# 내부 모듈
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
            st.session_state["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ [자동 수집 완료] {file_path}")
        else:
            print("⚠️ [자동 수집 실패] 응답 없음")
    except Exception as e:
        print(f"❌ 자동 업데이트 오류: {e}")


# ===============================
# 전역 스타일
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

    # 최근 자동 수집 로그 요약 (3개만)
    csv_files = sorted(glob.glob("data/trend_data_*.csv"), key=os.path.getctime, reverse=True)
    if csv_files:
        log_df = pd.DataFrame([
            {"파일명": os.path.basename(f), "시간": datetime.fromtimestamp(os.path.getctime(f))}
            for f in csv_files
        ])
        st.markdown("#### 📈 최근 자동 수집 (최신 3건)")
        for _, row in log_df.head(3).iterrows():
            st.markdown(
                f"<div style='font-size:13px; padding:4px 0;'>"
                f"📂 <b>{row['파일명']}</b><br>"
                f"⏰ {row['시간'].strftime('%Y-%m-%d %H:%M:%S')}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("최근 로그 없음.")


# ===============================
# 📦 데이터 로드 및 병합
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
                df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"데이터 수집 중 오류: {e}")

if df is None:
    df = load_latest_csv()

if merge_btn:
    merged = merge_all_csv()
    if merged.empty:
        st.warning("병합할 CSV 파일이 없습니다.")
    else:
        merged_path = f"data/merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        df = merged
        st.success(f"🗂 CSV 병합 완료 → {merged_path}")

if df is not None and not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")


# ===============================
# 📊 메인 탭
# ===============================
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 트렌드 비교", "📈 급등/급락 감지", "🔗 상관 분석", "⬇️ 다운로드"])

    # 📊 탭 1: 트렌드 비교
    with tab1:
        st.subheader("📊 키워드별 트렌드 변화")

        smooth_window = st.slider("이동평균 적용 (그래프 부드럽게)", 1, 14, 1)
        df_vis = df.copy()
        if smooth_window > 1:
            value_cols = [c for c in df.columns if c != "date"]
            df_vis[value_cols] = df_vis[value_cols].rolling(window=smooth_window, min_periods=1).mean()

        df_long = df_vis.melt(id_vars="date", var_name="keyword", value_name="ratio")
        fig = px.line(df_long, x="date", y="ratio", color="keyword", markers=True)
        fig.update_layout(
            plot_bgcolor="white",
            font=dict(size=14),
            margin=dict(l=10, r=10, t=40, b=10),
            legend_title_text="키워드",
        )
        st.plotly_chart(fig, width='stretch')
        st.dataframe(df, width='stretch')

    # 📈 탭 2: 급등/급락 감지
    with tab2:
        st.subheader("📈 트렌드 급상승·급하락 감지")

        df2 = df.copy().set_index("date")
        pct = df2.pct_change(fill_method=None).reset_index()
        pct.columns = ["date" if c == "date" else f"{c}_증감률(%)" for c in pct.columns]
        for c in pct.columns:
            if c != "date":
                pct[c] = (pct[c] * 100).round(2)

        threshold = st.slider("급변 기준(%)", 10, 200, 50, step=10)
        alerts = []
        for col in pct.columns:
            if col != "date":
                spikes = pct.loc[pct[col].abs() >= threshold, ["date", col]]
                for _, row in spikes.iterrows():
                    alerts.append({
                        "키워드": col.replace("_증감률(%)", ""),
                        "날짜": row["date"].date(),
                        "유형": "급등" if row[col] > 0 else "급락",
                        "변동률(%)": round(row[col], 1)
                    })

        if alerts:
            alert_df = pd.DataFrame(alerts).sort_values(["키워드", "날짜"])
            st.warning(f"⚠️ 감지된 급상승/급하락 이벤트: {len(alert_df)}건")

            selected_kw = st.selectbox("🔍 키워드 선택", ["전체 보기"] + sorted(alert_df["키워드"].unique()))
            filtered = alert_df if selected_kw == "전체 보기" else alert_df[alert_df["키워드"] == selected_kw]

            def highlight_row(row):
                color = "#FFEBEE" if row["유형"] == "급등" else "#E3F2FD"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(filtered.style.apply(highlight_row, axis=1), width='stretch', height=350)

            pct_long = pct.melt(id_vars="date", var_name="keyword", value_name="change")
            fig_change = px.bar(
                pct_long,
                x="date",
                y="change",
                color='keyword',
                barmode="group",
                title="📊 일간 증감률(%) 변화"
            )

            # 급등/급락 포인트 분리
            spikes_up = pct_long[pct_long["change"] >= threshold]
            spikes_down = pct_long[pct_long["change"] <= -threshold]

            fig_change.add_scatter(
                x=spikes_up["date"],
                y=spikes_up["change"],
                mode="markers",
                name="급등 포인트",
                marker=dict(size=9, color="crimson", symbol="triangle-up", opacity=0.8, line=dict(width=1, color="darkred")),
                hovertext=[f"{r['keyword']} (+{r['change']:.1f}%)" for _, r in spikes_up.iterrows()],
                hoverinfo="text"
            )
            fig_change.add_scatter(
                x=spikes_down["date"],
                y=spikes_down["change"],
                mode="markers",
                name="급락 포인트",
                marker=dict(size=9, color="#1976D2", symbol="triangle-down", opacity=0.8, line=dict(width=1, color="navy")),
                hovertext=[f"{r['keyword']} ({r['change']:.1f}%)" for _, r in spikes_down.iterrows()],
                hoverinfo="text"
            )

            # 그래프 정돈
            fig_change.update_layout(
                plot_bgcolor="white",
                font=dict(size=14),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1
                ),
                margin=dict(l=10, r=10, t=60, b=10)
            )

            st.plotly_chart(fig_change, width='stretch')
        else:
            st.info("✅ 설정된 임계값 내 급변 변화 없음.")

    # 🔗 탭 3: 상관 분석
    with tab3:
        st.subheader("🔗 키워드 상관관계 분석")
        corr = df.set_index("date").corr()
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), width='stretch')

        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig_corr, width='stretch')

        threshold = st.slider("상관계수 임계값", 0.0, 1.0, 0.6, 0.05)
        G = nx.Graph()
        for i in corr.columns:
            for j in corr.columns:
                if i != j and abs(corr.loc[i, j]) >= threshold:
                    G.add_edge(i, j, weight=corr.loc[i, j])

        if len(G.edges) == 0:
            st.info(f"임계값 {threshold} 이상인 상관관계 없음.")
        else:
            pos = nx.spring_layout(G, seed=42, k=0.5)
            edge_x, edge_y, edge_text = [], [], []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_text.append(f"{u} ↔ {v}: {data['weight']:.2f}")

            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                    line=dict(width=1.5, color='lightgray'),
                                    hoverinfo='text', text=edge_text)
            node_trace = go.Scatter(
                x=[pos[n][0] for n in G.nodes],
                y=[pos[n][1] for n in G.nodes],
                mode='markers+text', text=list(G.nodes),
                textposition="top center",
                marker=dict(size=20, color="#90CAF9", line=dict(width=2, color="#0D47A1"))
            )
            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(title=f"키워드 네트워크 (|r| ≥ {threshold})",
                                  plot_bgcolor="white", showlegend=False, height=600)
            st.plotly_chart(fig_net, width='stretch')

    # ⬇️ 탭 4: CSV 다운로드
    with tab4:
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
