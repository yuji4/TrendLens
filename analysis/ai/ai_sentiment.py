import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_sentiment(text: str) -> str:
    """
    뉴스 요약 텍스트 감성 분석: 긍정 / 부정 / 중립 중 하나 반환
    """
    if not text:
        return "중립"

    prompt = f"""
    다음 문장의 감성을 평가하세요.
    '긍정', '부정', '중립' 중 하나로만 답하세요.

    문장:
    {text}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()
