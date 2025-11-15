import pandas as pd
import os
import glob
from datetime import datetime
import re

def save_data_to_csv(data: dict, folder_path: str = 'data', auto: bool= False) -> str:
    '''
    네이버 API 응답 JSON을 DataFrame으로 변환 후 타임스탬프를 붙여 CSV로 저장
    DataFrame은 날짜(index)별로 키워드별 검색 비율을 컬럼으로 갖는 피벗 형식
    '''
    os.makedirs(folder_path, exist_ok=True)

    if not data or 'results' not in data or not data['results']:
        print("ERROR: 유효하지 않거나 비어 있는 데이터입니다. 저장 건너뜀.")
        return ""

    # 타임스탬프 기반 파일명 생성 및 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "auto_trend" if auto else "trend_data"
    file_name = f"{prefix}_{timestamp}.csv"
    file_path = os.path.join(folder_path, file_name)

    # 유효성 검사
    if not data or 'results' not in data:
        print("ERROR: 유효하지 않거나 비어 있는 데이터입니다.")
        return file_path

    # JSON 응답을 분석하여 단일 DataFrame으로 변환
    results = data['results']
    all_data = []

    # 각 키워드 그룹(results 항목)의 데이터를 추출하여 all_data 리스트에 통합
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
        return file_path

    # DataFrame 생성 및 타입 변환
    df_raw = pd.DataFrame(all_data, columns=['date', 'keyword', 'ratio'])
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['ratio'] = df_raw['ratio'].astype(float)  

    # 데이터를 피벗하여 날짜를 인덱스, 키워드를 컬럼으로 만들기
    df_pivot = df_raw.pivot_table(
        index='date', columns='keyword', values='ratio'
    ).reset_index()
    df_pivot = df_pivot.sort_values("date")
    df_pivot = collapse_dup_columns(df_pivot)
 
    try:
        df_pivot.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"INFO: 데이터가 성공적으로 저장되었습니다. -> {file_path}")
    except Exception as e:
        print(f"ERROR: CSV 파일 저장 실패: {e}")
    
    return file_path
        

def load_latest_csv(folder_path: str = 'data') -> pd.DataFrame:
    '''
    저장된 폴더에서 가장 최근에 저장된 CSV 파일을 DateFrame으로 로드
    '''

    # 폴더 내 모든 'trend_data_*.csv' 패턴 파일 검색
    search_pattern = os.path.join(folder_path, "trend_data_*.csv")
    list_of_files = glob.glob(search_pattern)

    if not list_of_files:
        print(f"WARNING: 폴더 내에서 CSV 파일(패턴: {search_pattern})을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    # 파일 생성 시간 기준으로 정렬
    latest_file = max(list_of_files, key=os.path.getctime)

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
    

def collapse_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 접미사 제거 후 같은 이름 컬럼은 평균으로 통합
    new_cols = [re.sub(r'_dup(\.\d+)?$', '', c) for c in df.columns]
    df = df.copy()
    df.columns = new_cols 
    if len(set(new_cols)) != len(new_cols): 
        df = df.T.groupby(level=0).mean(numeric_only=True).T
    return df


def merge_all_csv(folder_path: str = 'data') -> pd.DataFrame:
    '''
    폴더 내의 trend_data_*.csv 파일을 모두 병합하여 하나의 DataFrame으로 반환
    중복된 날짜는 평균값으로 처리, 동일 키워드는 평균 후 1개만 유지
    '''
    search_pattern = os.path.join(folder_path, "trend_data_*.csv")
    file_list = glob.glob(search_pattern)

    if not file_list:
        print("WARNING: 병합할 CSV 파일이 없습니다.")
        return pd.DataFrame()
    
    frames = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            frames.append(df)
        except Exception as e:
            print(f"ERROR: {file} 로드 실패 - {e}")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.groupby('date', as_index=False).mean(numeric_only=True)
    merged = merged.dropna(axis=1, how='all')
    merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
    merged = merged.dropna(subset=['date']).sort_values('date')

    merged = collapse_dup_columns(merged)
    
    print(f"INFO: {len(file_list)}개의 CSV 병합 완료 (총 {len(merged)}행)")
    return merged
    

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    기본적인 데이터 클리닝 수행
    (1) 결측치 제거
    (2) 이상치 처리
    (3) 컬럼명 정리 
    '''
    if df.empty:
        print("WARNING: 클리닝할 데이터가 없습니다.")
        return df
    
    df = df.copy()
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip().str.lower()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    print(f"INFO: 클리닝 완료 (남은 행 수: {len(df)})")
    return df


def summarize_data(df: pd.DataFrame):
    '''
    키워드별 평균, 최대, 최소값 요약 통계 출력
    '''
    if df.empty:
        print("WARNING: 통계 계산할 데이터가 없습니다.")
        return
    
    numeric_cols = [c for c in df.columns if c != 'date']
    summary = df[numeric_cols].agg(['mean', 'max', 'min']).T.round(2)
    return summary
    



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
    '''
        # 테스트 후 생성된 파일과 폴더 정리
        try: 
            os.remove(saved_path)
            if os.path.exists('data_test') and not os.listdir('data_test'):
                os.rmdir('data_test')
        except OSError:
            pass
    '''
