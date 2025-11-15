import requests
from datetime import datetime
from analysis.api_manager import get_naver_news_api_keys

def expand_query(keyword: str) -> list:
    base = keyword.strip()
    return [
        base,
        f"{base} 트렌드",
        f"{base} 증가 이유",
        f"{base} 검색량 급등",
        f"{base} 관련 뉴스",
        f"{base} 이슈",
        f"{base} 원인",
        f"{base} 분석",
        f"{base} 변화",
    ]

def fetch_news_articles(keyword: str, start_date: str, end_date: str, max_articles: int = 40):
    client_id, client_secret = get_naver_news_api_keys()
    if not client_id or not client_secret:
        return []

    search_url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    queries = expand_query(keyword)
    articles = []

    for q in queries:
        params = {
            "query": q,
            "display": 100,
            "start": 1,
            "sort": "date"
        }

        try:
            res = requests.get(search_url, headers=headers, params=params)
            res.raise_for_status()
            data = res.json().get("items", [])

            for item in data:
                try:
                    pub_date = datetime.strptime(item["pubDate"], "%a, %d %b %Y %H:%M:%S %z").date()
                except:
                    continue

                # 날짜 필터링
                if not (datetime.strptime(start_date, "%Y-%m-%d").date() <= pub_date <= datetime.strptime(end_date, "%Y-%m-%d").date()):
                    continue

                articles.append({
                    "title": item["title"],
                    "desc": item["description"],
                    "link": item["link"],
                    "pub_date": pub_date
                })

                if len(articles) >= max_articles:
                    return articles

        except:
            continue

    return articles
