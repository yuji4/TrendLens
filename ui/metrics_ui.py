import streamlit as st
import plotly.express as px

def render_metrics_comparison(df_metrics, selected_kw, plot_style):
    """
    탭5: 모델 성능 비교 UI 렌더링
    """

    if df_metrics is None or df_metrics.empty:
        st.info("아직 저장된 모델 성능 데이터가 없습니다. 예측을 먼저 실행하세요.")
        return

    # 비교할 키워드 선택
    available_keywords = df_metrics["키워드"].unique().tolist()

    try:
        default_index = available_keywords.index(selected_kw)
    except ValueError:
        default_index = 0

    selected_comparison_kw = st.selectbox(
        "키워드 선택 (비교 대상)",
        available_keywords,
        index=default_index
    )

    df_filtered = df_metrics[df_metrics["키워드"] == selected_comparison_kw]
    st.dataframe(df_filtered, use_container_width=True)

    if df_filtered.empty:
        st.info(f"키워드 '{selected_comparison_kw}'에 대한 저장된 데이터가 없습니다.")
        return

    # 최적 모델 표시
    best_row = df_filtered.loc[df_filtered["RMSE"].idxmin()]
    st.success(
        f"🏆 키워드 **'{selected_comparison_kw}'**의 최적 모델: "
        f"**{best_row['모델명']}** (RMSE {best_row['RMSE']:.4f})"
    )

    # RMSE Bar Chart
    st.markdown("#### RMSE 비교")
    fig_rmse = px.bar(
        df_filtered,
        x="모델명",
        y="RMSE",
        color="모델명",
        text="RMSE",
        title=f"'{selected_comparison_kw}' 모델별 RMSE 비교",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_rmse.update_layout(**plot_style)
    st.plotly_chart(fig_rmse, use_container_width=True)

    # MAPE Bar Chart
    st.markdown("### MAPE 비교")
    fig_mape = px.bar(
        df_filtered,
        x="모델명",
        y="MAPE(%)",
        color="모델명",
        text="MAPE(%)",
        title=f"'{selected_comparison_kw}' 모델별 MAPE 비교",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_mape.update_layout(**plot_style)
    st.plotly_chart(fig_mape, use_container_width=True)
