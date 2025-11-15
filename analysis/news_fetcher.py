import requests
from analysis.api_manager import get_naver_news_api_keys

def _clean_html(text: str) -> str:
    return (
        text.replace("<b>", "")
            .replace("</b>", "")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&")
    )

def fetch_news_articles(keyword: str, max_articles: int = 40):

    client_id, client_secret = get_naver_news_api_keys()
    if not client_id:
        return []

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    params = {
        "query": keyword,
        "display": 100,
        "start": 1,
        "sort": "date",
    }

    res = requests.get(url, headers=headers, params=params)
    items = res.json().get("items", [])
    
    collected = []
    for item in items:
        collected.append({
            "title": _clean_html(item["title"]),
            "desc": _clean_html(item["description"]),
            "link": item["link"],
            "pub_date": item["pubDate"]
        })

        if len(collected) >= max_articles:
            break

    return collected

