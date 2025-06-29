import pandas as pd
import numpy as np

test_df = pd.read_csv('../data/test.csv')
submit_df = pd.read_csv('../data/submit.csv')
train_df = pd.read_csv('../data/train.csv')


train_df.shape
submit_df.shape
test_df.shape


train_df.info()

# 1. date, time 이름 바꾸기
train_df = train_df.rename(columns={'date': 'time', 'time': 'date'})


train_df.info()
# [1] line
train_df['line'].info()
# 결측치 없음
train_df['line'].value_counts()
# 모든 값 동일

# [2] name
train_df['name'].unique()
train_df['name'].value_counts()
train_df['name'].info()
# 결측치 없음 / 모든 값 동일

# [3] mold_name
train_df['mold_name'].unique()
train_df['mold_name'].value_counts()
train_df['mold_name'].info()
# 결측치 없음, 모든 값 동일



# [6] count 
train_df['count'].info()
# 결측치 없음
train_df['count'].value_counts()
# 10자리수가 탑 5 (19~32)
train_df['count'].sort_values().unique()
# 1~334

# [7] working 
train_df['working'].info()
train_df[train_df['working'].isna()]
# 결측치 1개
# id=19327
train_df['working'].value_counts()
# 가동 73573 정지 38 
# 범주형으로

# [8] emergency_stop
train_df['emergency_stop'].info()
train_df[train_df['emergency_stop'].isna()]
# 결측 1
# id=19327
train_df['emergency_stop'].value_counts()
# ON 73611, 결측 1

# [9] **molten_temp**
# hist 그리기
# 나머지 조건에 맞춰서 600대로 맞춰서 처리할듯 > 지피티한테 물어보기
train_df['molten_temp'].info()
train_df['molten_temp'].isna().sum()
# null값 2261개
train_df['molten_temp'].value_counts()
# 보통 700대, 600대도 있음, 72도 있는데 이상치 같음
train_df['molten_temp'].isnull().sum()
train_df['molten_temp'].sort_values().unique()
# 0, 7, 70, 71, 72, 73, 626~ 쭈우우욱 ~735

# [10] facility_operation_cycleTime
train_df['facility_operation_cycleTime'].info()
# 결측치 없음
train_df['facility_operation_cycleTime'].value_counts()
# 100대가 대부분
train_df['facility_operation_cycleTime'].sort_values().unique()
# 0, 69~100, 101~348, 457

# [11] production_cycletime
train_df['production_cycletime'].info()
# 결측치 없음
train_df['production_cycletime'].value_counts()
# 대부분 100대
train_df['production_cycletime'].sort_values().unique()
# 0, 73 ~ 393, 462, 482, 485


# [12] *low_section_speed*
# 이상치 확인~~~~
train_df['low_section_speed'].info()
train_df[train_df['low_section_speed'].isna()]
# 1개 결측치
# id=19327
train_df['low_section_speed'].value_counts()
# 대부분 100대로 추정
train_df['low_section_speed'].sort_values()
# 대부분 100인데 65535같은 이상한 값이 하나 껴잇음
# 0~ 200, 65535

# [13] high_section_speed
train_df['high_section_speed'].info()
# 1개 결측치
train_df['high_section_speed'].value_counts()
# 상위 5개 100대
train_df['high_section_speed'].sort_values()
# 0 ~ 388

# [14] **molten_volume**
# 어렵농
train_df['molten_volume'].info()
train_df['molten_volume'].isnull().sum()
# 결측치 34992개 -> 결측치 전략 세우기 필요
train_df['molten_volume'].value_counts()
# 2767 -> 49 -> 72 -> 94 -> 70 상위 5
train_df['molten_volume'].sort_values()
train_df['molten_volume'].max()
# 0 ~ 2767

# [15] cast_pressure
train_df['cast_pressure'].info()
# 결측치 1개
train_df['cast_pressure'].value_counts()
# 탑5 300대
train_df['cast_pressure'].sort_values()
# 41 ~ 344


# [16] *biscuit_thickness*
# 이상치 확인
train_df['biscuit_thickness'].info()
# 결측치 1개
train_df['biscuit_thickness'].value_counts()
# 50대가 top 5
train_df['biscuit_thickness'].sort_values().unique()
# 1 ~ 88,  **420., 422**


# [17] *upper_mold_temp1*
train_df['upper_mold_temp1'].info()
# 결측치 1개
train_df['upper_mold_temp1'].value_counts()
# 100대가 탑 5
train_df['upper_mold_temp1'].sort_values()
# 18 ~ 300후반까지가 보통이고 마지막 이상치인지 1449가 하나 있음



# [18] *upper_mold_temp2*
train_df['upper_mold_temp2'].info()
# 결측치 1개
train_df['upper_mold_temp2'].value_counts()
# 180대가 탑 5
train_df['upper_mold_temp2'].sort_values()
# 15에서 389까지 있고 4232가 마지막인데 이상치인지 있음

# [19] **upper_mold_temp3**
# ** ems_operation_time 값이 전부 25임 **
train_df['upper_mold_temp3'].info()
train_df['upper_mold_temp3'].isnull().sum()
# 결측치 313개
train_df['upper_mold_temp3'].value_counts()
# 1449가 제일 많고 나머지 top4는 110대 
train_df['upper_mold_temp3'].sort_values()
train_df['upper_mold_temp3'].max()
# 42 ~ 1449


# [20] lower_mold_temp1
train_df['lower_mold_temp1'].info()
# 결측치 1개
train_df['lower_mold_temp1'].value_counts()
# 100 ~ 200대까지가 탑 5
train_df['lower_mold_temp1'].sort_values()
# 20~369까지 있음


# [21] lower_mold_temp2
train_df['lower_mold_temp2'].info()
train_df[train_df['lower_mold_temp2'].isnull()]
# 결측치 1개 
# id = 19327
train_df['lower_mold_temp2'].value_counts()
# 100번대가 탑 5
train_df['lower_mold_temp2'].sort_values()
# 20~500까지


# [22] lower_mold_temp3
train_df['lower_mold_temp3'].info()
train_df[train_df['lower_mold_temp3'].isnull()]
# 313개의 결측치(아마 upper_mold_temp3이랑 동일한 개수이므로 같은 행일거라 추정)
train_df['lower_mold_temp3'].value_counts()
# 1000도 이상이 top4이며 대부분이 1449도 임
train_df['lower_mold_temp3'].sort_values()
train_df['lower_mold_temp3'].max()
# 299 ~ 65503

# [23] sleeve_temperature
train_df['sleeve_temperature'].info()
train_df[train_df['sleeve_temperature'].isnull()]
# 1개 결측치
# id=19327
train_df['sleeve_temperature'].value_counts()
# 470번대가 top5
train_df['sleeve_temperature'].sort_values()
# train_df[train_df['sleeve_temperature'] == 0]
# 24~1449까지

# [24] physical_strength
train_df['physical_strength'].info()
train_df[train_df['physical_strength'].isnull()]
# 1개 결측치
# id=19327
train_df['physical_strength'].value_counts()
# 700번대가 top5
train_df['physical_strength'].sort_values()
# 0부터 65535까지, 근데 보통 737까지고 65535가 세개임 

# [25] Coolant_temperature
train_df['Coolant_temperature'].info()
train_df[train_df['Coolant_temperature'].isnull()]
# 1개 결측치
# id=19327
train_df['Coolant_temperature'].value_counts()
# 30대가 탑 10
train_df['Coolant_temperature'].sort_values()
# 16 ~ 1449


# [26] EMS_operation_time
# 0, 25인게 temp3 결측값 + id = 19327
train_df['EMS_operation_time'].info()
# 결측치 없음
train_df['EMS_operation_time'].value_counts()
# 23 -> 6 -> 3 -> 25
train_df['EMS_operation_time'].sort_values()
# 0, 3, 6, 23, 25

# [27] registration_time
train_df['registration_time'].info()
# 결측치 없음
train_df['registration_time'].value_counts()
train_df['registration_time'].sort_values()
# 20190102 16시 45분 ~ 20190312 06시 10분

# [28] passorfail
train_df['passorfail'].info()
# 결측치 없음
train_df['passorfail'].value_counts()
# 0 or 1
train_df['passorfail'].sort_values()

# [29] **tryshot_signal**
# 다 뺄 것 -> temp3같은거나 19327 확인해볼 필요 잇음
train_df['tryshot_signal'].info()
train_df['tryshot_signal'].isnull().sum()
# 72368개의 결측치
train_df['tryshot_signal'].value_counts()
# D 값 1244개
train_df['tryshot_signal'].sort_values()


# [30] mold_code
# 컬럼 추가 할수도?
train_df['mold_code'].info()
# 결측치 없음
train_df['mold_code'].value_counts()
# 8917 -> 8722 -> 8412 -> 8573 -> 8600
train_df['mold_code'].sort_values()


# [31] heating_furnace
train_df['heating_furnace'].info()
train_df['heating_furnace'].isna().sum()
# 40881개의 결측값
train_df['heating_furnace'].value_counts()
# A 16413, B 16318
train_df['heating_furnace'].sort_values()

train_df.info()

pd.to_datetime(train_df['registration_time'])

(train_df['datetime'] == train_df['registration_time']).all()


mismatch = train_df[train_df['datetime'] != train_df['registration_time']]
print(mismatch[['datetime', 'registration_time']].head())

(train_df['datetime'] != train_df['registration_time']).sum()

train_df.drop(columns='datetime', inplace=True)
train_df['registration_time'].to_datetime

train_df['registration_time'] = pd.to_datetime(train_df['registration_time'])

train_df.columns

train_df
