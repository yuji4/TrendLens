import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from analysis.modeling import run_ccf_analysis


def render_correlation_ui(df, plot_style):
    """
    탭3: 상관 분석 UI (히트맵 + 네트워크 그래프 + CCF)
    """

    # ----------------------
    # 📌 상관 히트맵
    # ----------------------
    corr = df.set_index("date").corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".3f",
        aspect="auto",
        title="키워드 간 검색 패턴 유사도 (상관 히트맵)",
        color_continuous_scale="RdBu_r",
    )
    fig_corr.update_layout(**plot_style)
    fig_corr.update_xaxes(side="top", tickangle=0)
    fig_corr.update_yaxes(tickangle=0)

    st.plotly_chart(fig_corr, width='stretch')

    # ----------------------
    # 📌 네트워크 그래프
    # ----------------------
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

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2, color="lightgray"),
        )

        node_x, node_y = zip(*[pos[n] for n in G.nodes()])
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                size=25,
                color="#90CAF9",
                line=dict(width=2, color="#1565C0"),
            ),
        )

        fig_net = go.Figure([edge_trace, node_trace])
        fig_net.update_layout(
            title=f"키워드 네트워크 (|r| ≥ {threshold_net})",
            **plot_style,
        )
        st.plotly_chart(fig_net, width='stretch')

    # ----------------------
    # 📌 교차 상관(Cross-Correlation)
    # ----------------------
    st.divider()
    st.subheader("🔬 키워드 간 교차 상관 분석 (Cross-Correlation)")
    st.caption("두 키워드 검색량의 시간 지연(Lag)에 따른 상관관계를 분석하여 선행/후행 관계를 파악합니다.")

    kw_list = [c for c in df.columns if c != "date"]

    col_select = st.columns(2)
    with col_select[0]:
        kw_a = st.selectbox("키워드 A (X축)", kw_list)
    with col_select[1]:
        kw_b = st.selectbox(
            "키워드 B (Y축)",
            kw_list,
            index=1 if len(kw_list) > 1 and kw_list[0] == kw_a else 0,
        )

    if kw_a == kw_b:
        st.warning("⚠️ 서로 다른 두 키워드를 선택하세요.")
        return

    max_lag = st.slider(
        "최대 지연 기간 (Lag, 일)",
        7,
        min(30, len(df) // 2 - 1),
        14,
        1,
    )

    # 데이터 준비
    df_ccf = df.set_index("date").dropna()

    try:
        ccf_results = run_ccf_analysis(
            df_ccf[kw_a].values,
            df_ccf[kw_b].values,
            max_lags=max_lag,
        )

        # 그래프
        fig_ccf = go.Figure(
            go.Bar(
                x=ccf_results["ccf_df"]["Lag"],
                y=ccf_results["ccf_df"]["CCF"],
                marker_color="#E91E63",
            )
        )

        fig_ccf.add_vline(
            x=ccf_results["optimal_lag"],
            line_width=2,
            line_dash="dash",
            line_color="#FFC107",
        )
        fig_ccf.add_hline(
            y=ccf_results["conf_level"],
            line_dash="dot",
            line_color="#4CAF50",
        )
        fig_ccf.add_hline(
            y=-ccf_results["conf_level"],
            line_dash="dot",
            line_color="#4CAF50",
        )

        fig_ccf.update_layout(
            title=f"{kw_a} ↔ {kw_b} 교차 상관 함수 (CCF)",
            xaxis_title=f"지연 (Lag, 일) | +Lag: {kw_a}가 {kw_b}를 선행",
            yaxis_title="교차 상관 계수",
            **plot_style,
        )

        st.plotly_chart(fig_ccf, width='stretch')

        st.markdown("#### 🔍 분석 결과")
        if abs(ccf_results["max_correlation"]) > ccf_results["conf_level"]:
            st.success(
                f"**최적 지연: {ccf_results['optimal_lag']}일** (상관 계수: {ccf_results['max_correlation']:.3f})"
            )
            st.markdown(ccf_results["analysis_text"])
        else:
            st.info("통계적으로 유의미한 교차 상관 관계 없음.")

    except Exception as e:
        st.error(f"CCF 분석 중 오류 발생: {e}")
