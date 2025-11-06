import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import optuna
from datetime import date, timedelta, datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit, os, glob, warnings
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import ccf
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

warnings.filterwarnings("ignore")
pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))

# 내부 모듈 
from analysis.api_manager import get_naver_trend_data
from analysis.data_manager import save_data_to_csv, load_latest_csv, merge_all_csv

# ===============================
# 성능 지표 계산 함수
# ===============================
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# 머신러닝 모델 함수 (Random Forest)
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
        return -np.mean(scores)  # 최소화된 MSE
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params, study.best_value

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
    
    # 반환값 변경: future_result, y_true, y_pred_past, feature_importances, features 목록 반환
    return future_result, y_train.values, y_pred_past, feature_importances, features

# ===============================
# 모델 성능 기록 함수
# ===============================
def save_model_metrics(model_name, keyword, mape, rmse):
    if "model_metrics" not in st.session_state:
        st.session_state["model_metrics"] = []

    st.session_state["model_metrics"].append({
        "키워드": keyword,
        "모델명": model_name,
        "MAPE(%)": round(mape, 2),
        "RMSE": round(rmse, 4),
        "기록시간": datetime.now().strftime("%H:%M:%S")
    })

# ===============================
# 자동 업데이트 함수
# ===============================
def auto_update_job():
    try:
        keywords = ["봄", "여름", "가을", "겨울"]
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
# 실시간 자동 새로고침 옵션
# ===============================
st.sidebar.markdown("### ⚡ 실시간 모드 설정")

# 새로고침 간격(초 단위)
refresh_interval = st.sidebar.slider("자동 새로고침 주기 (초)", 30, 600, 60, step=30)
enable_live = st.sidebar.toggle("실시간 데이터 갱신 활성화", value=False, help="네이버 트렌드 데이터를 주기적으로 갱신합니다.")

if enable_live:
    st.sidebar.success(f"✅ 실시간 모드 ON ({refresh_interval}초 간격)")
    st.sidebar.caption(f"마지막 새로고침: {datetime.now().strftime('%H:%M:%S')}")

    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )

else:
    st.sidebar.info("⏸ 실시간 모드 비활성화 중")

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
    raw_keywords = st.text_input("검색어 입력 (쉼표로 구분)", "봄, 여름, 가을, 겨울")
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
        update_btn = st.button("🔄 업데이트", width='stretch')
    with colB:
        merge_btn = st.button("🗂 CSV 병합", width='stretch')

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
        merged_path = f"data/merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
                st.dataframe(alert_df, width='stretch')
                summary = alert_df.groupby(["키워드", "유형"]).size().unstack(fill_value=0)
                st.markdown("#### 📊 키워드별 급등/급락 요약")
                st.dataframe(summary, width='stretch')
            else:
                selected_kw = st.selectbox("🔍 키워드 선택", sorted(df2.columns))
                kw_alerts = alert_df[alert_df["키워드"] == selected_kw]
                if kw_alerts.empty:
                    st.info(f"{selected_kw} 키워드에서 급변 없음.")
                else:
                    st.dataframe(kw_alerts, width='stretch')
                    fig_kw = px.line(df2.reset_index(), x="date", y=selected_kw, title=f"{selected_kw} 급등·급락 구간")
                    for _, r in kw_alerts.iterrows():
                        color = "red" if r["유형"] == "급등" else "blue"
                        fig_kw.add_vline(x=r["날짜"], line_dash="dot", line_color=color)
                    fig_kw.update_layout(**PLOTLY_STYLE)
                    st.plotly_chart(fig_kw, width='stretch')

        st.divider()
        scaled = df2.copy()
        for col in df2.columns:
            minv, maxv = scaled[col].min(), scaled[col].max()
            scaled[col] = (scaled[col] - minv) / (maxv - minv) if maxv != minv else 0
        scaled = scaled.reset_index()
        df_scaled_long = scaled.melt(id_vars="date", var_name="metric", value_name="value")
        fig_scaled = px.line(df_scaled_long, x="date", y="value", color="metric", title="정규화(0~1) 추세")
        fig_scaled.update_layout(**PLOTLY_STYLE)
        st.plotly_chart(fig_scaled, width='stretch')

    # --- 탭 3: 상관 분석 ---
    with tab3:
        st.caption("키워드 간 검색 패턴 유사도를 상관계수 및 네트워크로 분석합니다.")
        st.subheader("🔗 상관관계 분석")

        # 기본 상관 분석
        corr = df.set_index("date").corr()
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn"), width='stretch')
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(**PLOTLY_STYLE)
        st.plotly_chart(fig_corr, width='stretch')

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
            st.plotly_chart(fig_net, width='stretch')

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
            if len(df_ccf) > max_lag * 2:
                # CCF 계산
                ccf_values = ccf(df_ccf[kw_a], df_ccf[kw_b], adjusted=False)
                
                # 지연 값 배열 생성 및 최대 지연 기간에 맞게 필터링
                full_lags = [i - (len(df_ccf) - 1) // 2 for i in range(len(ccf_values))]
                center_idx = len(ccf_values) // 2
                
                lags = full_lags[center_idx - max_lag : center_idx + max_lag + 1]
                ccf_data = ccf_values[center_idx - max_lag : center_idx + max_lag + 1]

                ccf_df = pd.DataFrame({'Lag': lags, 'CCF': ccf_data})

                # 최대 상관 계수 찾기
                max_ccf_abs = ccf_df['CCF'].abs().max()
                max_row = ccf_df.loc[ccf_df['CCF'].abs().idxmax()]
                optimal_lag = int(max_row['Lag'])
                
                # Plotly 시각화
                fig_ccf = go.Figure(data=[
                    go.Bar(x=ccf_df['Lag'], y=ccf_df['CCF'], marker_color='#E91E63')
                ])

                # 최적 지연에 수직선 추가
                fig_ccf.add_vline(x=optimal_lag, line_width=2, line_dash="dash", line_color="#FFC107")
                
                # 유의성 경계선 (대략적인 95% 신뢰 구간) 추가
                conf_level = 1.96 / (len(df_ccf) ** 0.5)
                fig_ccf.add_hline(y=conf_level, line_dash="dot", line_color="#4CAF50")
                fig_ccf.add_hline(y=-conf_level, line_dash="dot", line_color="#4CAF50")
                
                fig_ccf.update_layout(
                    title=f"{kw_a} ↔ {kw_b} 교차 상관 함수 (CCF)",
                    xaxis_title=f"지연 (Lag, 일) | +Lag: {kw_a}가 {kw_b}를 선행",
                    yaxis_title="교차 상관 계수",
                    **PLOTLY_STYLE,
                )

                st.plotly_chart(fig_ccf, width='stretch')

                st.markdown("#### 🔍 분석 결과")
                if abs(max_row['CCF']) > conf_level:
                    analysis_result = ""
                    if optimal_lag > 0:
                        analysis_result = f"**{kw_a}**의 검색량 패턴이 **{abs(optimal_lag)}일** **먼저** 발생한 후, **{kw_b}**의 검색 패턴과 가장 높은 상관관계를 가집니다. (선행 지표: **{kw_a}**)"
                    elif optimal_lag < 0:
                        analysis_result = f"**{kw_b}**의 검색량 패턴이 **{abs(optimal_lag)}일** **먼저** 발생한 후, **{kw_a}**의 검색 패턴과 가장 높은 상관관계를 가집니다. (선행 지표: **{kw_b}**)"
                    else:
                        analysis_result = f"**{kw_a}**와 **{kw_b}**는 **동일 시점(Lag 0)**에 가장 높은 상관관계를 가집니다."
                    
                    st.success(f"**최적 지연: {optimal_lag}일** (상관 계수: {max_row['CCF']:.3f})")
                    st.markdown(analysis_result)

                else:
                    st.info("선택한 두 키워드 간에 통계적으로 유의미한 교차 상관 관계는 발견되지 않았습니다.")
            else:
                st.warning("데이터 길이가 충분하지 않거나, 최대 지연 기간이 너무 길어 CCF를 계산할 수 없습니다. 기간을 줄여주세요.")

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
                    
                        # MAPE/RMSE 계산을 위한 실제값/예측값 추출
                        y_true = df_forecast['y'].values
                        y_pred = forecast['yhat'].head(len(y_true)).values
                    
                        # 예측 차트 표시 (width='stretch' -> width='stretch'로 최적화)
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
                        st.plotly_chart(fig, width='stretch') # 최적화 적용

                        # -------------------- 🌟 3. 모델 성능 지표 표시 (Prophet) --------------------
                        mape = mean_absolute_percentage_error(y_true, y_pred)
                        rmse = root_mean_squared_error(y_true, y_pred)
                        save_model_metrics("Prophet", selected_kw, mape, rmse)

                        st.markdown("#### 🌟 모델 성능 지표")
                        col_metrics = st.columns(2)
                        col_metrics[0].metric(label="MAPE (Mean Absolute Percentage Error)", value=f"{mape:.2f}%")
                        col_metrics[1].metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")
                        st.caption("MAPE와 RMSE는 예측 기간을 제외한 과거 데이터에 대한 모델의 적합도를 나타냅니다.")
                    
                        # =========================================================
                        # ✨ 1. Prophet 기반 계절성 및 추세 분해 시각화 
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
                        df_yearly_pattern = forecast[['ds', 'yearly']].tail(365).copy() 
                        fig_yearly = go.Figure()
                        fig_yearly.add_trace(go.Scatter(x=df_yearly_pattern["ds"], y=df_yearly_pattern["yearly"], mode="lines", name="연간 계절성", line=dict(color="#2196F3")))
                        fig_yearly.update_layout(title="연간 계절성 (Yearly Seasonality)", plot_bgcolor="white", paper_bgcolor="#F5F5F5", font=dict(size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                        fig_yearly.update_xaxes(title_text="날짜", tickformat="%m-%d") 
                        fig_yearly.update_yaxes(title_text="영향도")
                    
                        # -------------------- 3. 주간 계절성 (Weekly) --------------------
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
                    
                        # -------------------- 4. 3분할 컬럼에 차트 표시 --------------------
                        cols_comp = st.columns(3)
                        with cols_comp[0]:
                            st.plotly_chart(fig_trend, width='stretch', config={'displayModeBar': False})
                        with cols_comp[1]:
                            st.plotly_chart(fig_yearly, width='stretch', config={'displayModeBar': False})
                        with cols_comp[2]:
                            st.plotly_chart(fig_weekly, width='stretch', config={'displayModeBar': False})

                    elif model_type == "ARIMA":
                        forecast_df = run_arima(df_forecast, days_ahead)
                    
                        # ARIMA 모델 성능 측정을 위한 예측치 추출
                        model_arima = ARIMA(df_forecast.set_index("ds"), order=(3, 1, 2))
                        fitted_arima = model_arima.fit()
                    
                        y_true = df_forecast['y'].iloc[1:].values
                        y_pred_past = fitted_arima.predict(start=1, end=len(df_forecast) - 1, dynamic=False).values
                    
                        # 예측 차트 표시 (width='stretch' -> width='stretch'로 최적화)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], mode="lines+markers",
                                             name="실제값", line=dict(color="black", width=3)))
                        fig.add_trace(go.Scatter(x=forecast_df["날짜"], y=forecast_df["예측값"], mode="lines",
                                             name="예측값", line=dict(color="royalblue", width=2.5, dash="dot")))
                        fig.update_layout(title=f"ARIMA 기반 {selected_kw} {days_ahead}일 예측", **PLOTLY_STYLE)
                        st.plotly_chart(fig, width='stretch') # 최적화 적용
                    
                        # -------------------- 🌟 3. 모델 성능 지표 표시 (ARIMA) --------------------
                        mape = mean_absolute_percentage_error(y_true, y_pred_past)
                        rmse = root_mean_squared_error(y_true, y_pred_past)
                        save_model_metrics("ARIMA", selected_kw, mape, rmse)


                        st.markdown("#### 🌟 모델 성능 지표")
                        col_metrics = st.columns(2)
                        col_metrics[0].metric(label="MAPE (Mean Absolute Percentage Error)", value=f"{mape:.2f}%")
                        col_metrics[1].metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")
                        st.caption("MAPE와 RMSE는 훈련 데이터에 대한 모델의 적합도를 나타냅니다.")
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
                            
                            # 최적 모델을 사용했으므로, X_train_rf로 과거 예측값 재계산
                            y_pred_past_rf = tuned_model.predict(X_train_rf)
                        else:
                            # 튜닝 안 할 경우 기본 모델로 과거 예측값 계산
                            model_default = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                            train_df_rf = create_features(df_forecast.copy())
                            features_x_rf = [c for c in train_df_rf.columns if c not in ['ds', 'y']]
                            X_train_rf, y_train_rf = train_df_rf[features_x_rf], train_df_rf['y']
                            model_default.fit(X_train_rf, y_train_rf)
                            y_pred_past_rf = model_default.predict(X_train_rf)
                            tuned_model = model_default

                        # ⭐ 예측 실행 (run_random_forest 함수에 튜닝된 모델 전달)
                        forecast_df, y_true, y_pred_past, feature_importances, features = run_random_forest(df_forecast, days_ahead, tuned_model=tuned_model)

                        # 2. 예측 차트 표시
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_forecast["ds"], y=df_forecast["y"], mode="lines+markers",
                                                 name="실제값", line=dict(color="black", width=3)))
                        fig.add_trace(go.Scatter(x=forecast_df["날짜"], y=forecast_df["예측값"], mode="lines",
                                                 name="예측값", line=dict(color="#FF5722", width=2.5, dash="dot"))) # 주황색 계열
                        fig.update_layout(title=f"Random Forest 기반 {selected_kw} {days_ahead}일 예측", **PLOTLY_STYLE)
                        st.plotly_chart(fig, width='stretch')
                        
                        # 3. 모델 성능 지표 표시
                        mape = mean_absolute_percentage_error(y_true, y_pred_past) # y_pred_past는 튜닝 결과 반영
                        rmse = root_mean_squared_error(y_true, y_pred_past)
                        save_model_metrics("Random Forest", selected_kw, mape, rmse) # ⭐ 키워드 인자 추가
        
                        st.markdown("#### 🌟 모델 성능 지표")
                        col_metrics = st.columns(2)
                        col_metrics[0].metric(label="MAPE (Mean Absolute Percentage Error)", value=f"{mape:.2f}%")
                        col_metrics[1].metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")
                        st.caption("MAPE와 RMSE는 훈련 데이터에 대한 모델의 적합도를 나타냅니다.")

                        # -------------------- 💡 피처 중요도 분석 시각화 --------------------
                        st.divider()
                        st.subheader("💡 피처 중요도 분석 (Random Forest)")
                        st.caption("모델 예측에 가장 큰 영향을 미친 시간 피처의 기여도를 보여줍니다.")
                        
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=True)
                        
                        # Plotly 막대 그래프로 시각화
                        fig_import = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='검색량 예측에 기여한 시간 요인',
                            color='Importance',
                            color_continuous_scale=px.colors.sequential.Teal
                        )
                        fig_import.update_layout(
                            plot_bgcolor='white', paper_bgcolor='#F5F5F5',
                            margin=dict(l=20, r=20, t=30, b=20),
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig_import, width='stretch', config={'displayModeBar': False})
                       
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
            st.dataframe(df_filtered, width='stretch')
            
            if not df_filtered.empty:
                # 최적 모델 찾기
                best_row = df_filtered.loc[df_filtered["RMSE"].idxmin()]
                st.success(f"🏆 키워드 **'{selected_comparison_kw}'**에 대한 최적 모델: **{best_row['모델명']}** (RMSE {best_row['RMSE']:.4f})")

                # 시각화(RMSE / MAPE 비교)
                st.markdown("#### RMSE 비교")
                fig_rmse = px.bar(df_filtered, x="모델명", y="RMSE", color="모델명",
                                    text="RMSE", title=f"'{selected_comparison_kw}' 모델별 RMSE 비교", color_discrete_sequence=px.colors.qualitative.Set2)
                fig_rmse.update_layout(**PLOTLY_STYLE)
                st.plotly_chart(fig_rmse, width='stretch')

                st.markdown("### MAPE 비교")
                fig_mape = px.bar(df_filtered, x="모델명", y="MAPE(%)", color="모델명",
                                    text="MAPE(%)", title=f"'{selected_comparison_kw}' 모델별 MAPE 비교", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_mape.update_layout(**PLOTLY_STYLE)
                st.plotly_chart(fig_mape, width='stretch')
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
                    from io import BytesIO
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.units import cm
                    from reportlab.lib import colors

                    buffer = BytesIO()
                    c = canvas.Canvas(buffer, pagesize=A4)
                    width, height = A4

                    c.setFont("HYSMyeongJo-Medium", 18)
                    c.setFillColor(colors.HexColor("#0D47A1"))
                    c.drawCentredString(width / 2, height - 2 * cm, "TrendLens 모델 성능 리포트")

                    c.setFont("HYSMyeongJo-Medium", 11)
                    c.setFillColor(colors.black)
                    c.drawString(2 * cm, height - 3 * cm, f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    data = pd.DataFrame(st.session_state["model_metrics"])
                    start_y = height - 4 * cm
                    c.setFont("HYSMyeongJo-Medium", 12)
                    c.drawString(2 * cm, start_y, "모델별 성능 요약:")

                    start_y -= 0.7 * cm
                    c.setFont("HYSMyeongJo-Medium", 10)
                    for i, row in data.iterrows():
                        line = f"- [{row['키워드']}] {row['모델명']} | MAPE: {row['MAPE(%)']}% | RMSE: {row['RMSE']} | {row['기록시간']}"
                        c.drawString(2.2 * cm, start_y, line)
                        start_y -= 0.5 * cm
                        if start_y < 2 * cm:  # 페이지 넘김 처리
                            c.showPage()
                            c.setFont("HYSMyeongJo-Medium", 10)
                            start_y = height - 3 * cm

                    c.setFont("HYSMyeongJo-Medium", 9)
                    c.setFillColor(colors.gray)
                    c.drawString(2 * cm, 1.5 * cm, "Generated by TrendLens | Naver Trend Analysis Dashboard")

                    c.save()
                    buffer.seek(0)

                    st.download_button(
                        label="📥 리포트 다운로드 (PDF)",
                        data=buffer,
                        file_name=f"TrendLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF 리포트 생성 완료!")
                except Exception as e:
                    st.error(f"PDF 생성 중 오류 발생: {e}")

else:
    st.info("좌측에서 검색어s를 입력하고 '업데이트'를 눌러주세요.")

# ===============================
# ⏰ 자동 업데이트 스케줄러
# ===============================
scheduler = BackgroundScheduler()
scheduler.add_job(auto_update_job, "interval", hours=24)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())