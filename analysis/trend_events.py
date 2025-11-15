import pandas as pd

def detect_surge_events(df: pd.DataFrame, threshold_percent: float = 50):
    """
    검색량 급등 이벤트 자동 감지
    - threshold_percent: 몇 % 이상 상승하면 '급등' 이벤트로 판단
    반환값: 이벤트 DataFrame (keyword, date, change)
    """

    if "date" not in df.columns:
        raise ValueError("df에는 'date' 컬럼이 있어야 합니다.")

    # 'date' 제외 컬럼 = 키워드들
    kw_cols = [c for c in df.columns if c != "date"]

    df2 = df.copy().set_index("date")
    pct = df2.pct_change() * 100  # 변화율 계산 (%)

    events = []

    for kw in kw_cols:
        surges = pct[pct[kw] >= threshold_percent]
        for date, value in surges[kw].items():
            events.append({
                "keyword": kw,
                "date": date,
                "change": round(value, 2)
            })

    return pd.DataFrame(events)
