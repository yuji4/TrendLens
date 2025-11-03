import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# 내부 모듈
from analysis.api_manager import get_naver_trend_data
from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv

last_update_time = st.session_state.get("last_update_time", None)

def auto_update_job():
    global last_update_time
    try:
        keywords = ["Python", "AI", "Study"]  # 기본 키워드
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
            last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["last_update_time"] = last_update_time
            print(f"✅ [자동 수집 완료] {file_path}")
        else:
            print("⚠️ [자동 수집 실패] 응답 없음")
    except Exception as e:
        print(f"❌ 자동 업데이트 중 오류: {e}")


# Streamlit 기본 설정
st.set_page_config(page_title="네이버 검색 트렌드 분석", layout="wide")
st.title("👀 TrendLens: 네이버 검색 트렌드")

# 사이드바 입력 영역
with st.sidebar:
    st.header("⚙️ 설정")

    raw_keywords = st.text_input("검색어 입력 (쉼표로 구분)", "Python, AI, Study")
    time_unit = st.selectbox("데이터 단위", ["date", 'week', "month"])

    today = date.today()
    default_start = today - timedelta(days=90)
    start_date, end_date = st.date_input(
        "조회 기간 선택",
        (default_start, today),
        format="YYYY-MM-DD",
    )

    # 성별 선택
    gender_display = st.selectbox("성별 선택", ["전체", "남성", "여성"])
    if gender_display == "남성": gender = "m"
    elif gender_display == "여성": gender = "f"
    else: gender = ""

    st.divider()
    st.subheader("📅 데이터 병합 및 정렬 옵션")

    # 병합 옵션
    align_option = st.radio(
        "날짜 정렬 기준",
        ["모든 날짜", "공통 날짜"],
        index=0,
        help="모든 날짜를 표시하거나, 모든 키워드에 값이 존재하는 날짜만 표시할 수 있습니다."
    )

    smooth_window = st.slider(
        "이동평균(부드럽게)",
        min_value=1, max_value=14, value=1, step=1,
        help="값을 1보다 크게 하면 그래프가 부드럽게 표시됩니다."
    )

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        update_btn = st.button("🔄 데이터 업데이트", type="primary")
    with colB:
        merge_btn = st.button("🗂 CSV 전체 병합")

    st.divider()
    st.subheader("🕒 자동 업데이트 상태")

    if "last_update_time" in st.session_state and st.session_state["last_update_time"]:
        st.success(f"마지막 자동 수집: {st.session_state['last_update_time']}")
    else:
        st.info("자동 수집 기록이 아직 없습니다.")

# 키워드 처리
keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
if not keywords:
    st.warning("검색어를 1개 이상 입력하세요.")
    st.stop()

# 데이터 수집 / 불러오기
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
                st.error("선택한 성별 조건에 대한 데이터가 없습니다.")
            else:
                file_path = save_data_to_csv(data)
                st.success(f"✅ 최신 데이터 저장 완료: {file_path}")
                df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"데이터 수집 중 오류 발생: {e}")

# 최근 CSV 불러오기
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
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if align_option == "공통 날짜":
         df = df.dropna(subset=[c for c in df.columns if c != "date"])
    if smooth_window > 1:
        value_cols = [c for c in df.columns if c != "date"]
        df[value_cols] = (
            df[value_cols]
            .rolling(window=smooth_window, min_periods=1)
            .mean()
        )

# 대시보드 출력
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 트렌드 비교", "📈 정규화/증감", "🔗 상관 분석", "⬇️ 다운로드"])

    # 📊 탭 1: 트렌드 비교
    with tab1:
        st.subheader("트렌드 비교 그래프")
        df_long = df.melt(id_vars="date", var_name="keyword", value_name="ratio")
        fig = px.line(df_long, x="date", y="ratio", color="keyword", markers=True,
                      title="📈 키워드별 검색 트렌드 변화")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(df, width='stretch')

    # 📈 탭 2: 정규화/증감
    with tab2:
        st.subheader("정규화 및 증감률 분석")

        # 원본 데이터 준비
        df2 = df.copy()
        df2.set_index("date", inplace=True)

        # 증감률 계산 (백분율 변환)
        pct = df2.pct_change(fill_method=None).reset_index()
        pct.columns = [
            "date" if c == "date" else f"{c}_증감률(%)" for c in pct.columns
        ]
        for c in pct.columns:
            if c != "date":
                pct[c] = (pct[c] * 100).round(2)

        threshold = st.slider("이상치 감지 임계값(%)", 10, 200, 50, step=10)
        alerts = []
        for col in pct.columns:
            if col != "date":
                spikes = pct.loc[pct[col].abs() >= threshold, ["date", col]]
                for _, row in spikes.iterrows():
                    change = row[col]
                    direction = "📈 급등" if change > 0 else "📉 급락"
                    alerts.append(f"- [{col.replace('_증감률(%)','')}] {row['date'].date()} : {direction} ({change:+.1f}%)")

        if alerts:
            st.warning("⚠️ 이상치 감지 결과:\n" + "\n".join(alerts))
        else:
            st.info("✅ 설정된 임계값 내에서는 이상치가 없습니다.")

        # 정규화
        scaled = df2.copy()
        for col in [c for c in df2.columns if c != "date"]:
            minv, maxv = scaled[col].min(), scaled[col].max()
            scaled[col] = (scaled[col] - minv) / (maxv - minv) if (maxv - minv) != 0 else 0
        scaled = scaled.reset_index()
        scaled.columns = ["date"] + [f"{c}_정규화(0~1)" for c in df2.columns]

        # 통합 테이블
        df_combined = df.merge(pct, on="date", how="left").merge(scaled, on="date", how="left")
        
        styled_df = df_combined.style.set_table_styles([
            {'selector': 'th',
            'props': [('font-size', '15px'),
                      ('font-weight', 'bold'),
                      ('background-color','#E3F2FD'),
                      ('color', '#0D47A1')]},
            {'selector': 'td',
             'props': [('font-size', '13px'),
                       ('color', '#212121')]}
        ]).highlight_max(axis=0, color='#C5E1A5')

        st.dataframe(styled_df, width='stretch', height=500)

        # 증감률 그래프
        pct_long = pct.melt(id_vars="date", var_name="keyword", value_name="change")
        fig_change = px.bar(
            pct_long,
            x="date", y="change", color="keyword",
            title="📊 일간 증감률(%) 변화",
            barmode="group"
        )
        fig_change.update_layout(
            plot_bgcolor='white',
            font=dict(size=14),
            xaxis_tickangle=-45,
            legend_title_text="키워드"
        )
        st.plotly_chart(fig_change, width='stretch')

        # 정규화 그래프
        df_scaled_long = scaled.melt(id_vars="date", var_name="metric", value_name="value")
        fig_scaled = px.line(
            df_scaled_long,
            x="date", y="value", color="metric",
            title="정규화(0~1) 추세"  
        )
        fig_scaled.update_traces(line=dict(width=2.5))
        fig_scaled.update_layout(
            plot_bgcolor='white',
            font=dict(size=14),
            legend_title_text="정규화 키워드"
        )
        st.plotly_chart(fig_scaled, width='stretch')

    # 🔗 탭 3: 상관 분석
    with tab3:
        st.subheader("키워드 간 상관관계")

        corr = df.set_index("date").corr()
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), width='stretch')

        fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig3, width='stretch')

        # 네트워크 그래프 추가
        st.markdown("### 🕸️ 네트워크 상관관계 그래프")
        import networkx as nx
        import plotly.graph_objects as go

        threshold = st.slider("상관계수 임계값", 0.0, 1.0, 0.6, 0.05)
        G = nx.Graph()
        
        for i in corr.columns:
            G.add_node(i)
        for i in corr.columns:
            for j in corr.columns:
                if i != j and abs(corr.loc[i, j]) >= threshold:
                    G.add_edge(i, j, weight=corr.loc[i, j])

        if len(G.edges) == 0:
            st.info(f"임계값 {threshold} 이상인 상관관계가 없습니다.")
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
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.5, color='lightgray'),
            hoverinfo='text',
            text=edge_text,
            hoverlabel=dict(bgcolor='white')
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
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes),
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color='skyblue',
                line=dict(width=2, color='DarkSlateGrey')
            )
        )

        fig_network = go.Figure(data=[edge_trace, node_trace])
        fig_network.update_layout(
            title=f"키워드 네트워크 (|r| ≥ {threshold})",
            showlegend=False,
            hovermode='closest',
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=650,
            plot_bgcolor='white',
            font=dict(size=14)
        )
        st.plotly_chart(fig_network, width='stretch')
        

    # ⬇️ 탭 4: CSV 다운로드
    with tab4:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 CSV 다운로드", csv, "trend_data_latest.csv", "text/csv")
else:
    st.info("좌측에서 검색어를 입력하고 '데이터 업데이트'를 눌러주세요.")

# 자동 업데이트 스케줄러 등록
scheduler = BackgroundScheduler()
scheduler.add_job(auto_update_job, 'interval', hours=24)  # 하루 한 번
scheduler.start()
atexit.register(lambda: scheduler.shutdown())