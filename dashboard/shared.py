from pathlib import Path
import time
import threading
import pandas as pd
import joblib
from processing import load_data, load_bound_data
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from shiny import reactive
from datetime import timedelta, datetime
from pathlib import Path
import matplotlib.font_manager as fm

# 앱 디렉터리 설정
app_dir = Path(__file__).parent

# 한글 폰트 설정: MaruBuri-Regular.ttf 직접 로드
font_path = app_dir / "MaruBuri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)


app_dir = Path(__file__).parent
DATA_PATH_SE = 'asset/data/a_shap_applied_no_data.csv'
DATA = load_data('asset/data/a_shap_applied_no_data.csv')
DATA_ANSWER = load_data('asset/data/a_shap_applied_data.csv')
TEST_DATA = load_data('./asset/data/test.csv')
BOUND_DATA = load_bound_data('asset/data/bounds.csv')
NUM_COLS = DATA.select_dtypes(include='number').columns.tolist()
CAT_COLS = DATA.select_dtypes(include='object').columns.tolist()
MOLD_CODE = DATA['mold_code'].unique().tolist()
CURRENT_MOLD_CODE = reactive.value(None)

# mask_17 = DATA["registration_time"].dt.day == 17
# DATA.loc[mask_17, "passorfail"] = np.nan

# DATA.to_csv('asset/data/soeun4_final.csv', index=False)

# DATA.drop(columns='Unnamed: 0.1', inplace=True)
# DATA[(DATA['registration_time'].dt.day == 17)& (DATA['registration_time'].dt.hour==00)& (DATA['registration_time'].dt.minute==00)]['registration_time'].sort_values()



def plot_shap_summary(shap_values_path, sample_df_path):
    # Load SHAP values and sample
    shap_values = np.load(shap_values_path)
    with open(sample_df_path, "rb") as f:
        sample_df = pickle.load(f)

    # Make plot
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, sample_df, show=False)
    
    return plt.gcf()  # current figure 반환


# add probability
# DATA['probability'] = np.nan
# DATA.to_csv('./asset/data/soeun3.csv')
# temp = pd.read_csv('./asset/data/soeun3.csv')
# temp['passorfail'] = np.nan
# temp['anomalyornot'] = np.nan
# temp.to_csv('./asset/data/soeun3_empty.csv')
# EMPTY_DATA = pd.read_csv('./asset/data/soeun3_empty.csv')

# DATA['passorfail'] = pd.to_numeric(DATA['passorfail'], errors='coerce')  # NaN 인식되게
# result = DATA.groupby("mold_code")["passorfail"].value_counts(dropna=False).unstack(fill_value=0)
# print(result)


# val_df = pd.read_csv('../data/validation.csv')
# test_df = pd.read_csv('../data/test.csv')

# val_df['passorfail'] = pd.to_numeric(val_df['passorfail'], errors='coerce')  # NaN 인식되게
# result = val_df.groupby("mold_code")["passorfail"].value_counts(dropna=False).unstack(fill_value=0)
# print(result)
# hj_df = hj_df[hj_df['passorfail'] == 1]
# soeun_df = pd.read_csv('asset/data/soeun3_empty.csv')
# soeun_df = pd.read_csv('asset/data/soeun4_final.csv')
# soeun_df[soeun_df['passorfail'].isna()]
# # # 원하는 날짜 범위
# base_date = pd.to_datetime("2019-03-17")
# start_ts = pd.Timestamp("2019-03-17 00:00:00")
# end_ts = pd.Timestamp("2019-03-17 01:00:00")

# # 초 단위 범위로 변환
# total_seconds = int((end_ts - start_ts).total_seconds())

# # 무작위 초 생성
# random_seconds = np.random.randint(0, total_seconds, size=len(soeun_df[soeun_df['passorfail'].isna()]))
# soeun_df[soeun_df['passorfail'].isna()]['registration_time']
# # 랜덤 시간 생성
# # soeun_df[soeun_df['passorfail'].isna()]["registration_time"] = start_ts + pd.to_timedelta(random_seconds, unit="s")
# mask = soeun_df["passorfail"].isna()
# soeun_df.loc[mask, "registration_time"] = start_ts + pd.to_timedelta(random_seconds, unit="s")

# soeun_df.to_csv('asset/data/soeun2.csv')
# soeun_df[(soeun_df['mold_code'] == 8412) & (soeun_df['passorfail'].isna())]['registration_time']


# hj_df['passorfail'] = np.nan
# hj_df.to_csv('./asset/data/hj.csv')



# # # -----값 채우기
# val_df = pd.read_csv('../data/validation.csv')

# # val_df.drop(columns=['Unnamed: 0.1'], inplace=True)
# val_df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

# merged = test_df.merge(val_df[['id', 'passorfail']], on='id', how='left')
# merged.info()
# hj_df = merged[merged['mold_code'] == 8917]
# hj_df.info()
# # soeun_df['anomalyornot'] = np.nan
# hj_df['anomalyornot'] = np.nan
# # # 2. passorfail 컬럼 초기화 (전부 NaN)
# merged['passorfail_final'] = np.nan

# # # 3. 0~3599까지만 passorfail 값 채우기
# merged.loc[:, 'passorfail_final'] = merged.loc[:, 'passorfail_y']
# # merged.info()
# # # 4. 불필요한 passorfail 컬럼 삭제하고 새 컬럼으로 대체
# merged.drop(columns=['passorfail_x', 'passorfail_y'], inplace=True)
# merged.rename(columns={'passorfail_final': 'passorfail'}, inplace=True)
# # merged.info()

# # merged.to_csv('asset/data/to3600.csv')

# data1 = merged[merged['mold_code'] == 8917]
# data2 = merged[merged['mold_code'] == 8722]
# data3 = merged[merged['mold_code'] == 8412]

# # data1['passorfail'].value_counts()
# # data2['passorfail'].value_counts()
# # data3['passorfail'].value_counts()

# # # 1) data1, data2, data3를 한 번에 합치기
# merged_restored = pd.concat([data1, data2, data3], axis=0, ignore_index=True)

# # # 2) 'id' 컬럼 기준으로 오름차순 정렬
# merged_restored = merged_restored.sort_values("id").reset_index(drop=True)

# merged_restored['registration_time'] = pd.to_datetime(merged_restored['registration_time'])
# len(merged_restored[merged_restored['registration_time'].dt.day == 16])
# merged_restored.to_csv('./asset/data/soeun3_random_time.csv')
# merged_restored = pd.read_csv('./asset/data/soeun3_random_time.csv')
# mask = merged_restored['registration_time'].dt.day == 17
# merged_restored.loc[mask, 'passorfail'] = np.nan
# merged_restored.to_csv('./asset/data/soeun3_random_time_test.csv')

# # merged_restored['registration_time'].info()


# # =============================================================================
# # 1) 두 개의 시간 구간 정의
# # =============================================================================

# # 1/3 구간: 2019-03-16 23:59:59 ~ 2019-03-17 00:00:00
# start_ts1 = pd.Timestamp("2019-03-16 00:00:00")
# end_ts1   = pd.Timestamp("2019-03-17 00:00:00")
# total_seconds_1 = int((end_ts1 - start_ts1).total_seconds())  # 보통 1초

# # 2/3 구간: 2019-03-17 00:00:00 ~ 2019-03-17 01:00:00
# start_ts2 = pd.Timestamp("2019-03-17 00:00:00")
# end_ts2   = pd.Timestamp("2019-03-17 01:00:00")
# total_seconds_2 = int((end_ts2 - start_ts2).total_seconds())  # 보통 3600초

# # =============================================================================
# # 2) “하나의 df, 하나의 그룹(passorfail)” 조합마다 다음을 수행하는 함수 정의
# # =============================================================================
# def reassign_times_one_group(df: pd.DataFrame, group_value: int) -> None:
#     """
#     df 내에서 df['passorfail'] == group_value인 행들을
#     전체의 1/3 vs 2/3 비율로 나누어,
#     1/3에는 첫 번째 시간 구간, 2/3에는 두 번째 시간 구간에서 무작위로 registration_time 재할당.
#     (원본 df의 registration_time 컬럼을 in-place 수정)
#     """

#     # 1) 해당 그룹(passorfail==group_value)에 해당하는 모든 인덱스 리스트
#     idx_all = df.index[df["passorfail"] == group_value].to_numpy()
#     if len(idx_all) == 0:
#         return

#     # 2) 섞어서 순서를 랜덤으로 섞음
#     np.random.shuffle(idx_all)

#     # 3) 1/3 분할 지점 계산
#     n_total = len(idx_all)
#     n_one_third = n_total // 3  # 몫 부분이 1/3의 개수
#     # 나머지(2/3)는 n_total - n_one_third로 자동 계산

#     # 4) 첫 번째 1/3 인덱스, 나머지 2/3 인덱스로 나눔
#     idx_first  = idx_all[:n_one_third]
#     idx_second = idx_all[n_one_third:]

#     # 5) 첫 번째 1/3 그룹에 대해서 무작위 초를 뽑아서 registration_time 갱신
#     if len(idx_first) > 0:
#         # 0 <= random_seconds < total_seconds_1
#         # size=len(idx_first) 만큼 0~(total_seconds_1-1) 범위의 정수 생성
#         random_secs_1 = np.random.randint(0, total_seconds_1, size=len(idx_first))
#         # 각 랜덤 초를 start_ts1에 더해서 새로운 타임스탬프 생성
#         new_times_1 = start_ts1 + pd.to_timedelta(random_secs_1, unit="s")
#         # df.loc[idx_first, "registration_time"]을 해당 값들로 덮어쓰기
#         df.loc[idx_first, "registration_time"] = new_times_1

#     # 6) 두 번째 2/3 그룹에 대해서 무작위 초를 뽑아서 registration_time 갱신
#     if len(idx_second) > 0:
#         # 0 <= random_seconds < total_seconds_2
#         random_secs_2 = np.random.randint(0, total_seconds_2, size=len(idx_second))
#         new_times_2 = start_ts2 + pd.to_timedelta(random_secs_2, unit="s")
#         df.loc[idx_second, "registration_time"] = new_times_2

#     # 끝. in-place로 df["registration_time"]이 수정됨.
#     return


# # =============================================================================
# # 3) data1, data2, data3에 대해 “passorfail==0”과 “passorfail==1” 그룹 각각 호출
# # =============================================================================

# # --- data1 ---
# reassign_times_one_group(data1, 0)
# reassign_times_one_group(data1, 1)

# # --- data2 ---
# reassign_times_one_group(data2, 0)
# reassign_times_one_group(data2, 1)

# # --- data3 ---
# reassign_times_one_group(data3, 0)
# reassign_times_one_group(data3, 1)

# # =============================================================================
# # 4) 결과 확인 (예시 출력)
# # =============================================================================
# print("=== data1 불량0 중 3분의1 개수:", len(data1[(data1["passorfail"]==0) & 
#        (data1["registration_time"] >= start_ts1) & 
#        (data1["registration_time"] < end_ts1)]))
# print("=== data1 불량0 중 3분의2 개수:", len(data1[(data1["passorfail"]==0) & 
#        (data1["registration_time"] >= start_ts2) & 
#        (data1["registration_time"] < end_ts2)]))
# print("=== data1 불량1 중 3분의1 개수:", len(data1[(data1["passorfail"]==1) & 
#        (data1["registration_time"] >= start_ts1) & 
#        (data1["registration_time"] < end_ts1)]))
# print("=== data1 불량1 중 3분의2 개수:", len(data1[(data1["passorfail"]==1) & 
#        (data1["registration_time"] >= start_ts2) & 
#        (data1["registration_time"] < end_ts2)]))

# # data2, data3도 같은 방식으로 검증할 수 있습니다.

# temp_db = pd.read_csv('./asset/data/soeun3_random_time_test.csv')
# temp_db = pd.read_csv('./asset/data/soeun3.csv')
# temp_db[]
# temp_db[(temp_db['registration_time'].dt.day == 17) & (temp_db['passorfail'] == 1)]

# # 초 시간 수정
# temp_df = pd.read_csv('./asset/data/soeun3_random_time_test2.csv')
# temp_df = DATA
# temp_df['registration_time'] = pd.to_datetime(temp_df['registration_time'])
# boundary = pd.Timestamp("2019-03-17 00:00:00")
# # temp_df[temp_df["registration_time"] == boundary]
# mask = temp_df["registration_time"] == boundary
# # temp_df.loc[mask, 'registration_time'] = temp_df.loc[mask, "registration_time"] + pd.Timedelta(seconds=1)
# temp_df.drop(columns=['Unnamed: 0.5', 'Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
# temp_df.to_csv('./asset/data/soeun4_time_edit.csv')

# # temp_df[temp_df['registration_time'].dt.day == 16]['registration_time']