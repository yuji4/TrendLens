import os
import requests
from dotenv import load_dotenv

# .env에서 환경 변수 로드
load_dotenv()

API_URL = "https://openapi.naver.com/v1/datalab/search"

def get_naver_api_keys():
    """네이버 DataLab API 키 가져오기"""
    client_id = os.environ.get('NAVER_DATALAB_CLIENT_ID')
    client_secret = os.environ.get('NAVER_DATALAB_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("오류: .env 파일에서 NAVER_DATALAB_CLIENT_ID 또는 NAVER_DATALAB_CLIENT_SECRET 를 찾을 수 없습니다.")
        return None, None
    
    return client_id, client_secret


def get_naver_news_api_keys():
    """네이버 뉴스 API 키 가져오기"""
    client_id = os.environ.get('NAVER_NEWS_CLIENT_ID')
    client_secret = os.environ.get('NAVER_NEWS_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("오류: .env 파일에서 NAVER_NEWS_CLIENT_ID 또는 NAVER_NEWS_CLIENT_SECRET 를 찾을 수 없습니다.")
        return None, None
    
    return client_id, client_secret


def get_naver_trend_data(
    keywords: list, 
    start_date: str, 
    end_date: str, 
    time_unit: str = 'date'
) -> dict:
    """네이버 DataLab 검색량 데이터 가져오기"""

    client_id, client_secret = get_naver_api_keys()
    if not client_id or not client_secret:
        return {}

    if not (1 <= len(keywords) <= 5):
        print("ERROR: 키워드는 1~5개여야 합니다.")
        return {}

    keywords = [k.strip() for k in keywords]

    # 요청 body 구성
    keyword_groups = [
        {
            "groupName": kw,
            "keywords": [kw]
        }
        for kw in keywords
    ]

    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": keyword_groups,
        "device": ""
    }

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json"
    }

    try:
        print(f"INFO: {keywords} 트렌드 요청 중…")
        res = requests.post(API_URL, headers=headers, json=body)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"ERROR: 네이버 트렌드 API 요청 실패: {e}")
        return {}
