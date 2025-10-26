import pandas as pd
import os
import glob
from datetime import datetime


def save_data_to_csv(data: dict, folder_path: str = 'data') -> str:
    '''
    네이버 API 응답 JSON을 DataFrame으로 변환 후 타임스탬프를 붙여 CSV로 저장
    DataFrame은 날짜(index)별로 키워드별 검색 비율을 컬럼으로 갖는 피벗 형식

    매개변수:
        data(dict): Naver API 응답 JSON 딕셔너리
        folder_path(str): CSV 파이을 저장할 폴더 경로 (기본값: 'data')

    반환값:
        str: 저장할 파일의 전체 경로(성공) or 빈 문자열(실패) 
    '''
    if not data or 'results' not in data:
        print("ERROR: 유효하지 않거나 비어 있는 데이터입니다.")
        return ""
    
    # 'data/' 폴더 생성 확인 (없으면 생성)
    os.makedirs(folder_path, exist_ok=True)

    # JSON 응답을 분석하여 단일 DataFrame으로 변환
    results = data.get('results', {})
    all_data = []

    # 각 키워드 그룹(resluts 항목)의 데이터를 추출하여 all_date 리스트에 통합
    for result in results:
        keyword_group = result['title']

        # 'data' 필드의 각 날짜/비율 항목을 처리
        for item in result['data']:
            row = {
                'date': item['period'],
                'keyword': keyword_group,
                'ratio': item['ratio']
            }
            all_data.append(row)

    if not all_data:
        print("ERROR: 추출된 데이터가 없습니다. JSON 응답 구조를 확인하세요.")
        return ""

    # DataFrame 생성 및 타입 변환
    df_raw = pd.DataFrame(all_data, columns=['date', 'keyword', 'ratio'])
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['ratio'] = df_raw['ratio'].astype(float)  

    # 데이터를 피벗하여 날짜를 인덱스, 키워드를 컬럼으로 만들기
    df_pivot = df_raw.pivot_table(
        index='date',
        columns='keyword',
        values='ratio'
    ).reset_index()

    # 타임스탬프 기반 파일명 생성 및 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"trend_data_{timestamp}.csv"
    file_path = os.path.join(folder_path, file_name)

    try:
        df_pivot.to_csv(file_path, index=False)
        print(f"INFO: 데이터가 성공적으로 저장되었습니다. -> {file_path}")
        return file_path
    except Exception as e:
        print(f"ERROR: CSV 파일 저장 실패: {e}")
        return ""
        

def load_latest_csv(folder_path: str = 'data') -> pd.DataFrame:
    '''
    저장된 폴더에서 가장 최근에 저장된 CSV 파일을 DateFrame으로 로드

    매개변수:
        folder_paht(str): CSV 파일이 저장된 폴더 경로 (기본값은 'data')

    반환값:
        pd.DataFrame: 최신 CSV 파일의 데이터. 파일을 찾지 못하면 빈 DataFrame 반환
    '''

    # 폴더 내 모든 'trend_data_*.csv' 패턴 파일 검색
    search_pattern = os.path.join(folder_path, "trend_data_*.csv")
    list_of_files = glob.glob(search_pattern)

    if not list_of_files:
        print(f"WARNING: 폴더 내에서 CSV 파일(패턴: {search_pattern})을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    # 파일 이름 기준으로 정렬
    latest_file = max(list_of_files)

    try:
        print(f"INFO: 최신 데이터를 로드합니다. -> {latest_file}")
        df = pd.read_csv(latest_file)
        # 분석을 위해 'date' 컬럼을 datetime 객체로 변환
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"ERROR: 최신 CSV 파일 로드 실패: {e}")
        return pd.DataFrame()
    
if __name__ == '__main__':
    print("--- Data Manager Module Test (더미 데이터 사용) ---")

    # API 응답 JSON을 모방한 더미 데이터
    dummy_data = {
        'startDate': '2024-01-01', 'endDate': '2024-01-03', 'timeUnit': 'date',
        'results': [
            {'title': '키워드A', 'data': [
                {'period': '2024-01-01', 'ratio': 5.0},
                {'period': '2024-01-02', 'ratio': 15.0},
                {'period': '2024-01-03', 'ratio': 25.0}
            ]},
            {'title': '키워드B', 'data': [
                {'period': '2024-01-01', 'ratio': 10.0},
                {'period': '2024-01-02', 'ratio': 20.0},
                {'period': '2024-01-03', 'ratio': 30.0}
            ]}
        ]
    }

    # 1. 저장 테스트
    saved_path = save_data_to_csv(dummy_data, folder_path='data_test')

    # 2. 로드 테스트
    if saved_path:
        df_loaded = load_latest_csv(folder_path='data_test')
        if not df_loaded.empty:
            print("\n✅ 로드 성공. 로드된 데이터 구조 (상위 5개 행):")
            print(df_loaded.head().to_string(index=False))

        # 테스트 후 생성된 파일과 폴더 정리
        try: 
            os.remove(saved_path)
            if os.path.exists('data_test') and not os.listdir('data_test'):
                os.rmdir('data_test')
        except OSError:
            pass
