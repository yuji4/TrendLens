import os
import requests
import json
import re
from dotenv import load_dotenv
from datetime import datetime

# .env에서 환경 변수 로드
load_dotenv()

API_URL = "https://openapi.naver.com/v1/datalab/search"

def get_naver_api_keys():
    # .env 파일에서 네이버 DataLab API 키(Client ID, Client Secret) 불러오기
    client_id = os.environ.get('NAVER_DATALAB_CLIENT_ID')
    client_secret = os.environ.get('NAVER_DATALAB_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("오류: .env 파일에서 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET를 찾을 수 없습니다.")
        return None, None
    
    return client_id, client_secret

def get_naver_news_api_keys():
    client_id = os.environ.get('NAVER_NEWS_CLIENT_ID')
    client_secret = os.environ.get('NAVER_NEWS_CLIENT_SECRET')
    if not client_id or not client_secret:
        print("오류: .env 파일에서 NAVER_NEWS_CLIENT_ID 또는 NAVER_NEWS_CLIENT_SECRET를 찾을 수 없습니다.")
        return None, None
    
    return client_id, client_secret

def get_naver_trend_data(
    keywords: list, 
    start_date: str, 
    end_date: str, 
    time_unit: str = 'date',
    gender: str = '',
) -> dict:
    # 네이버 DataLab 검색 트렌드 API를 호출하여 데이터 가져오기
    '''
    매개변수:
        keywords(list): 검색할 키워드 목록 (최대 5개). 예: ["여행", "주식"]
        start_date(str): 검색 시작 날짜. 포맷: "YYYY-MM-DD"
        end_date(str): 검색 종료 날짜. 포맷: "YYYY-MM-DD"
        time_unit(str): 데이터 단위. 'date', 'week', 'month' 중 하나 (기본값은 'date')
        gender: 'm' (남성), 'f' (여성), 또는 '' (전체)
        ages: ['10', '20', '30', ...] 형태의 문자열 리스트

    반환값:
        dict: API 응답 JSON 데이터. 오류 발생 시 딕셔너리 반환
    '''
    client_id, client_secret = get_naver_api_keys()

    if not client_id or not client_secret:
        return {}
    
    if not (1 <= len(keywords) <= 5):
        print("ERROR: 키워드는 1개에서 5개 사이여야 합니다.")
        return {}
    keywords = [k.strip() for k in keywords]
    
    if gender not in ['', 'm', 'f']:
        print("WARNING: gender 값이 올바르지 않습니다.")
        gender = ''

    
    # API 리퀘스트 본문 구성
    # 각 키워드를 별도의 그룹으로 처리하여 여러 트렌드를 동시 요청
    keyword_groups = [{
        'groupName': keyword,
        'keywords': [keyword]
    } for keyword in keywords ]

    request_body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": keyword_groups,
        "device": "",
        "gender": gender
    }

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json"
    }

    try: 
        print(f"INFO: {len(keywords)}개의 키워드에 대한 데이터 요청 중: {keywords}")
        print("요청 본문:", json.dumps(request_body, ensure_ascii=False, indent=2))
        print("요청 헤더:", headers)

        response = requests.post(API_URL, headers=headers, json=request_body)
        response.raise_for_status()  # 200 이외의 상태 코드 시 예외 발생
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API 요청 실패 - {e}")
        return {}
    
def get_naver_news_count(keyword: str, target_date_str: str) -> int:
    # 네이버 뉴스 검색 API를 이용해 특정 키워드의 일자별 뉴스 개수 반환
    client_id, client_secret = get_naver_news_api_keys()
    if not client_id or not client_secret:
        return 0
    
    search_url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-naver-Client-Id": client_id,
        "X-naver-Client-Secret": client_secret
    }

    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: 잘못된 날짜 형식입니다: {target_date_str}")
        return 
    
    target_count = 0
    start = 1
    display = 100
    max_api_limit = 1000

    while start <= max_api_limit:
        params = {
            "query": keyword,
            "display": display,
            "start": start,
            "sort": "date" # 최신순 정렬
        }

        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if not items:
                break

            found_on_target_date = 0
            
            for item in items:
                # pubDate 포맷: 'Mon, 01 Jan 2024 09:00:00 +0900'
                pub_date_str = item.get("pubDate")
                
                # pubDate 파싱 (요일, 일 월 연도 시:분:초 타임존)
                try:
                    # RFC 2822 형식 파싱
                    item_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z").date()
                except ValueError:
                    # 파싱 실패 시 예외 처리
                    print(f"경고: 날짜 파싱 실패 - {pub_date_str}")
                    continue

                if item_date == target_date:
                    found_on_target_date += 1
                elif item_date < target_date:
                    # 최신순 정렬이므로, 목표 날짜보다 이전 날짜가 나오면 검색 중단
                    start = max_api_limit + 1 # 루프 강제 종료
                    break

            target_count += found_on_target_date

            if found_on_target_date < display:
                # 현재 페이지에서 목표 날짜의 뉴스가 100건 미만이면 다음 페이지에도 없을 확률이 높음
                # 하지만 정확도를 위해 API 한도까지는 계속 확인하는 것이 좋음.
                pass 
            
            if start > max_api_limit: # 루프가 이미 종료되도록 설정된 경우
                break

            start += display
        except requests.exceptions.RequestException as e:
            print(f"ERROR: 뉴스 API 요청 실패 - {e}")
            break
            
    print(f"INFO: [{keyword}] {target_date_str} 뉴스 언급량: {target_count}건")
    return target_count


if __name__ == "__main__":
    print("--- Naver API Key Load Test ---")

    # 키 로드 테스트
    client_id, client_secret = get_naver_api_keys()
    if client_id and client_secret:
        print(f"Client ID 로드 성공: {client_id[:5]}...{client_id[-5:]}")
        
        # API 호출 테스트
        test_keywords = ["아이폰", "겨울"]
        test_start = "2024-01-01"
        test_end = "2024-01-31"
        test_gender = "f"

        test_data = get_naver_trend_data(
            test_keywords, 
            test_start, 
            test_end, 
            gender=test_gender,
        )

        if test_data:
            print("\n✅ API 호출 성공 및 데이터 획득")
            print("데이터 미리보기:")
            print(json.dumps(test_data, indent=2)[:500] + "...") 
        else:
            print("\n❌ API 호출 실패. 키, 날짜, 키워드를 확인하세요.")  
    
        test_keyword_news = "반도체"
        test_date_news = "2024-01-15"
        print(f"\n--- 뉴스 언급량 테스트: [{test_keyword_news}] {test_date_news} ---")
        
        # get_naver_news_count_for_date 함수가 정의되어 있다고 가정하고 호출
        try:
            news_count = get_naver_news_count(test_keyword_news, test_date_news)
            
            if news_count > 0:
                print(f"✅ [뉴스] 카운트 성공. {test_date_news} 언급량: {news_count}건")
            elif news_count == 0:
                 print(f"✅ [뉴스] 카운트 성공. {test_date_news} 언급량: 0건 (정상)")
            else: # 에러가 발생했으나 함수 내에서 0이 아닌 다른 음수 값 등을 리턴한 경우 방지
                print("❌ [뉴스] 카운트 실패. API 로직 또는 키 확인 필요.")

        except NameError:
            print("⚠️ [뉴스] 테스트 실패: 'get_naver_news_count_for_date' 함수가 정의되지 않았습니다.")
        except Exception as e:
            print(f"❌ [뉴스] 테스트 중 예외 발생: {e}")

    else:
        print("❌ API 키 로드 실패. .env 파일을 확인하세요.")