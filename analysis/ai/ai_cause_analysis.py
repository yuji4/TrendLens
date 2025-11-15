from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

def analyze_news_articles(keyword: str, articles: list):
    """
    뉴스 기사 여러개 기반으로 '급등 원인'을 생성하는 AI 모듈
    """
    if not articles:
        return "관련 뉴스가 부족하여 원인 분석을 수행할 수 없습니다."

    blocks = []
    for a in articles[:10]:
        blocks.append(f"[제목] {a['title']}\n[내용] {a['desc']}")

    combined = "\n\n".join(blocks)

    prompt = f"""
다음은 '{keyword}' 검색량이 급등한 날짜 주변의 뉴스들이다.
이 뉴스 내용들을 기반으로 검색량 급등의 원인을 분석해줘.

뉴스 데이터:
{combined}

결과 형식:
- 급등 원인 3~4줄 요약
- 관련 주요 키워드 3~5개
    """

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content
