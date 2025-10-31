import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# 내부 모듈
from analysis.api_manager import get_naver_trend_data
from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv

# Streamlit 기본 설정
st.set_page_config(page_title="네이버 검색 트렌드 분석", layout="wide")
st.title("📈 네이버 검색 트렌드 분석 대시보드")

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

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        update_btn = st.button("🔄 데이터 업데이트", type="primary")
    with colB:
        merge_btn = st.button("🗂 CSV 전체 병합")

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
            )
            if not data or "results" not in data:
                st.error("데이터를 불러오지 못했습니다. 키워드 또는 API 키를 확인하세요.")
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
        path = save_data_to_csv({"results": []}, folder_path="data")  # 더미 저장용
        merged.to_csv(path, index=False)
        df = merged
        st.success(f"🗂 CSV 병합 완료 → {path}")

# 대시보드 출력
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 트렌드 비교", "📈 정규화/증감", "🔗 상관 분석", "⬇️ 다운로드"])

    # 📊 탭 1: 트렌드 비교
    with tab1:
        st.subheader("트렌드 비교 그래프")
        df_long = df.melt(id_vars="date", var_name="keyword", value_name="ratio")
        fig = px.line(df_long, x="date", y="ratio", color="keyword", markers=True,
                      title="📈 키워드별 검색 트렌드 변화")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

    # 📈 탭 2: 정규화/증감
    with tab2:
        st.subheader("정규화 및 증감률 분석")

        df2 = df.copy()
        df2.set_index("date", inplace=True)

        pct = df2.pct_change().reset_index().rename(
            columns=lambda c: f"{c}_pct" if c != "date" else c
        )

        scaled = df2.copy()
        for col in [c for c in df2.columns if c != "date"]:
            minv, maxv = scaled[col].min(), scaled[col].max()
            scaled[col] = (scaled[col] - minv) / (maxv - minv) if (maxv - minv) != 0 else 0
        scaled = scaled.reset_index().rename(
            columns=lambda c: f"{c}_scaled" if c != "date" else c
        )

        df_combined = df.merge(pct, on="date", how="left").merge(scaled, on="date", how="left")
        st.dataframe(df_combined, use_container_width=True)

        df_scaled_long = scaled.melt(id_vars="date", var_name="metric", value_name="value")
        fig2 = px.line(df_scaled_long, x="date", y="value", color="metric", title="정규화(0~1) 추세")
        st.plotly_chart(fig2, use_container_width=True)

    # 🔗 탭 3: 상관 분석
    with tab3:
        st.subheader("키워드 간 상관관계")
        corr = df.set_index("date").corr()
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
        fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig3, use_container_width=True)

    # ⬇️ 탭 4: CSV 다운로드
    with tab4:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 CSV 다운로드", csv, "trend_data_latest.csv", "text/csv")
else:
    st.info("좌측에서 검색어를 입력하고 '데이터 업데이트'를 눌러주세요.")