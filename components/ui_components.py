import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit, glob, os
from analysis.api_manager import get_naver_trend_data  # ✅ 클래스 → 함수로 변경
from analysis.data_manager import save_data_to_csv

# ===============================
# ⚙️ 사이드바 렌더링 함수
# ===============================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ 기본 설정")
        raw_keywords = st.text_input("검색어 입력 (쉼표로 구분)", "봄, 여름, 가을, 겨울")
        time_unit = st.selectbox("데이터 단위", ["date", "week", "month"])

        today = date.today()
        default_start = today - timedelta(days=90)
        start_date, end_date = st.date_input("조회 기간 선택", (default_start, today))

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

    keywords_list = [k.strip() for k in raw_keywords.split(",") if k.strip()]
    
    return keywords_list, time_unit, start_date, end_date, align_option, update_btn, merge_btn

# ===============================
# ⏰ 자동 업데이트 스케줄러 설정 함수
# ===============================
def setup_scheduler():
    """자동 업데이트 스케줄러를 설정합니다."""
    
    try:
        def auto_update_job():
            try:
                keywords = ["봄", "여름", "가을", "겨울"]
                today = date.today()
                start = today - timedelta(days=7)
                
                # ✅ 함수 직접 호출로 변경
                data = get_naver_trend_data(
                    keywords=keywords,
                    start_date=str(start),
                    end_date=str(today),
                    time_unit="date",
                )
                
                if data and "results" in data:
                    file_path = save_data_to_csv(data)
                    print(f"✅ [자동 수집 완료] {file_path} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print("⚠️ [자동 수집 실패] 응답 없음")
            except Exception as e:
                print(f"❌ 자동 업데이트 오류: {e}")

        scheduler = BackgroundScheduler()
        if not scheduler.running:
            scheduler.add_job(auto_update_job, "interval", hours=24)
            scheduler.start()
            atexit.register(lambda: scheduler.shutdown())

    except ValueError as e:
        st.error(f"스케줄러 초기화 실패: {str(e)}")
