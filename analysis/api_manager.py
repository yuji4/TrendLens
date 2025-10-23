import os
from dotenv import load_dotenv

load_dotenv()

def get_naver_api_keys():
    # .env 파일에서 네이버 DataLab API 키(Client ID, Client Secret) 불러오기
    client_id = os.environ.get('NAVER_CLIENT_ID')
    client_secret = os.environ.get('NAVER_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("오류: .env 파일에서 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET를 찾을 수 없습니다.")
        return None, None
    
    return client_id, client_secret

if __name__ == "__main__":
    print("--- Naver API Key Load Test ---")

    client_id, client_secret = get_naver_api_keys()

    if client_id and client_secret:
        print(f"Client ID 로드 성공: {client_id[:5]}...{client_id[-5:]}")
        print(f"Client Secret 로드 성공: {client_secret[:5]}...{client_secret[-5:]}")
    else:
        print("API 키 로드 실패")