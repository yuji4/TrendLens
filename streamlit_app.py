import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ===============================
# 내부 모듈 Import
# ===============================
from analysis.api_manager import get_naver_trend_data
from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv
from analysis.modeling import run_ccf_analysis
from components.ui_components import render_sidebar, setup_scheduler
from report.pdf_generator import generate_trend_report
from analysis.trend_events import detect_surge_events
from analysis.news_fetcher import fetch_news_articles
from analysis.ai.ai_cause_analysis import analyze_news_articles
from ui.model_ui import render_prophet_ui, render_arima_ui, render_random_forest_ui, render_model_info
from ui.metrics_ui import render_metrics_comparison
from ui.correlation_ui import render_correlation_ui


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
        st.plotly_chart(fig, width='stretch')
        st.dataframe(df_vis, width='stretch')

    # --- 탭 2: 상세 분석 ---
    with tab2:
        st.caption("검색량 급등 이벤트를 자동 감지하고, 키워드 관련 뉴스 기반으로 AI가 원인을 분석합니다.")
        st.subheader("📈 급등 이벤트 분석")

        # 1) 급등 이벤트 감지
        events = detect_surge_events(df, threshold_percent=50)

        if events.empty:
            st.info("📉 급등 이벤트가 감지되지 않았습니다.")
        else:
            st.success(f"총 {len(events)}개의 급등 이벤트 감지됨")
            st.dataframe(events, width='stretch')

            # 선택박스 만들기
            max_change_indices = events.groupby('keyword')['change'].idxmax()
            max_events = events.loc[max_change_indices]
            event_key_list = max_events.apply(
                lambda r: f"{r['keyword']} | +{r['change']}%",
                axis=1
            )
            selected = st.selectbox("분석할 이벤트 선택", event_key_list)

            # 선택된 데이터 찾기
            keyword_to_find = selected.split(' | ')[0].strip()
            ev = max_events[max_events['keyword'] == keyword_to_find].iloc[0]

            keyword = ev["keyword"]
            change = ev["change"]

            st.info(f"🔍 선택한 이벤트: **{keyword}** (증가율 +{change}%)")

            if st.button("📡 뉴스 수집 + AI 원인 분석 실행"):
                with st.spinner("뉴스 수집 중..."):
                    articles = fetch_news_articles(
                        keyword, 
                        max_articles=100 
                    )

                if len(articles) == 0:
                    st.warning("관련 뉴스가 부족해 AI 분석을 수행할 수 없습니다.")
                else:
                    st.success(f"{len(articles)}개 뉴스 수집됨")

                    # 3) AI 분석
                    with st.spinner("AI가 급등 원인을 분석하는 중..."):
                        cause_text = analyze_news_articles(keyword, articles)

                    st.warning("⚠️ **분석 결과 안내:** 네이버 API 정책상 과거 급등 시점의 뉴스를 수집할 수 없습니다. 따라서 아래 AI 분석은 **선택된 키워드의 급등 원인**이 아닌, **현재 시점에서 가장 최근 발행된 뉴스들**을 기반으로 해당 키워드가 어떻게 활용되고 있는지에 대한 **최신 논점을 요약**한 것입니다.")
                    st.markdown("### 🔥 급등 원인 분석 결과")
                    st.write(cause_text)

                    st.markdown("### 📰 참조된 뉴스")
                    for a in articles:
                        st.markdown(f"""
                        **{a['title']}**  
                        {a['desc']}  
                        🔗 [기사 보기]({a['link']})
                        """)
                        st.divider()

    # --- 탭 3: 상관 분석 ---
    with tab3:
        st.caption("키워드 간 검색 패턴 유사도를 상관계수 및 네트워크로 분석합니다.")
        st.subheader("🔗 상관관계 분석")

        render_correlation_ui(df, PLOTLY_STYLE)

    # --- 탭 4: 예측 ---
    with tab4:
        st.caption("Prophet / ARIMA / Random Forest 기반 미래 검색 트렌드 예측 및 비교 분석을 제공합니다.")
        st.subheader("🔮 트렌드 예측")

        model_type = st.radio("모델 선택", ["Prophet", "ARIMA", "Random Forest"], horizontal=True)
        render_model_info()

        selected_kw = st.selectbox(
            "예측할 키워드", [c for c in df.columns if c != "date"]
        )
        days_ahead = st.slider("예측 기간 (일)", 7, 180, 30, 7)

        # Prophet/ARIMA/RF 공통 데이터 포맷(ds, y)
        df_forecast = df[["date", selected_kw]].rename(
            columns={"date": "ds", selected_kw: "y"}
        )

        # 각 모델의 UI 처리 함수 호출
        if model_type == "Prophet":
            render_prophet_ui(df_forecast, selected_kw, days_ahead)

        elif model_type == "ARIMA":
            render_arima_ui(df_forecast, selected_kw, days_ahead)

        elif model_type == "Random Forest":
            render_random_forest_ui(df_forecast, selected_kw, days_ahead)
        

    # --- 탭 5: 모델 성능 비교 ---
    with tab5:
        st.caption("예측 모델별 정확도(MAPE, RMSE)를 비교하여 최적 모델을 확인합니다.")
        st.subheader("📊 모델별 성능 비교 대시보드")
        
        df_metrics = pd.DataFrame(st.session_state.get("model_metrics", []))
        render_metrics_comparison(df_metrics, selected_kw, PLOTLY_STYLE)

    # --- 탭 6: 다운로드 ---
    with tab6:
        st.caption("검색 데이터 및 모델 성능 리포트를 다운로드할 수 있습니다.")
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
setup_scheduler()
