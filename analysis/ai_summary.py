import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_trend_summary(keyword: str, df: pd.DataFrame, spike_events: list, forecast_info=None) -> str:
    """
    검색량 변화 df + 급등·급락 이벤트 + (선택) 예측 정보를 종합해
    자연어 트렌드 분석 요약을 생성
    
    Parameters:
        keyword (str)
        df (pd.DataFrame): "date", "ratio" 형태로 된 검색량 데이터
        spike_events (list[dict]): 급등/급락 이벤트 리스트
        forecast_info (dict): 예측 모델 결과(선택)
    """

    # DataFrame → JSON (LLM 입력용)
    df_json = df.to_dict(orient="records")

    prompt = f"""
당신은 데이터 분석 전문 AI입니다.
아래는 '{keyword}'에 대한 네이버 검색량 데이터입니다.

[검색량 데이터]
{df_json}

[급등/급락 이벤트]
{spike_events}

[예측 정보]
{forecast_info}

위 데이터를 기반으로 아래 기준을 만족하는 '자연어 요약 분석'을 작성하세요.

요약 작성 기준:
- 전체 4~6문장
- 비즈니스 리포트 톤 (전문적·객관적·정제된 문장)
- 최근 검색량 흐름 1~2문장
- 급등 또는 급락 발생 시점과 원인 추정 1~2문장
- 향후 전망 1문장
- 마케팅 조언/광고 유도 문구 절대 금지
- 문장 길이는 너무 길지 않게
- 독자가 바로 이해할 수 있도록 자연스럽고 명확하게 작성
- 한 문장이 끝나면 띄어쓰기를 해 가독성을 향상

이 기준에 따라 '검색 트렌드 분석 요약'을 작성하세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # lightweight 고속 모델
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    summary_text = response.choices[0].message.content
    return summary_text
