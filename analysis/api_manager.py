import os
import requests
import json
from dotenv import load_dotenv

# .env에서 환경 변수 로드
load_dotenv()

API_URL = "https://openapi.naver.com/v1/datalab/search"

def get_naver_api_keys():
    # .env 파일에서 네이버 DataLab API 키(Client ID, Client Secret) 불러오기
    client_id = os.environ.get('NAVER_CLIENT_ID')
    client_secret = os.environ.get('NAVER_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("오류: .env 파일에서 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET를 찾을 수 없습니다.")
        return None, None
    
    return client_id, client_secret

def get_naver_trend_data(keywords: list, start_date: str, end_date: str, time_unit: str = 'date') -> dict:
    # 네이버 DataLab 검색 트렌드 API를 호출하여 데이터 가져오기
    '''
    매개변수:
        keywords(list): 검색할 키워드 목록 (최대 5개). 예: ["여행", "주식"]
        start_date(str): 검색 시작 날짜. 포맷: "YYYY-MM-DD"
        end_date(str): 검색 종료 날짜. 포맷: "YYYY-MM-DD"
        time_unit(str): 데이터 단위. 'date', 'week', 'month' 중 하나 (기본값은 'date')

    반환값:
        dict: API 응답 JSON 데이터. 오류 발생 시 딕셔너리 반환
    '''
    client_id, client_secret = get_naver_api_keys()

    if not client_id or not client_secret:
        return {}
    
    if not (1 <= len(keywords) <= 5):
        print("ERROR: 키워드는 1개에서 5개 사이여야 합니다.")
        return {}
    
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
        "gender": "",
        "ages": []
    }

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json"
    }

    try: 
        print(f"INFO: {len(keywords)}개의 키워드에 대한 데이터 요청 중: {keywords}")
        response = requests.post(API_URL, headers=headers, json=request_body)
        response.raise_for_status()  # 200 이외의 상태 코드 시 예외 발생

        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API 요청 실패 - {e}")
        return {}

if __name__ == "__main__":
    print("--- Naver API Key Load Test ---")

    # 키 로드 테스트
    client_id, client_secret = get_naver_api_keys()
    if client_id and client_secret:
        print(f"Client ID 로드 성공: {client_id[:5]}...{client_id[-5:]}")
        
        # API 호출 테스트
        test_keywords = ["커피", "차"]
        test_start = "2024-01-01"
        test_end = "2024-01-31"

        test_data = get_naver_trend_data(test_keywords, test_start, test_end)

        if test_data:
            print("\n✅ API 호출 성공 및 데이터 획득")
            # 응답 일부 출력
            print("데이터 미리보기:")
            print(json.dumps(test_data, indent=2)[:500] + "...") 
        else:
            print("\n❌ API 호출 실패. 키, 날짜, 키워드를 확인하세요.")  
    else:
        print("❌ API 키 로드 실패. .env 파일을 확인하세요.")