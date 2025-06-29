import pandas as pd
import numpy as np



#### merge를 위해 train data 전처리
train_df = pd.read_csv('../data/train.csv')
# 결측치 하나 있는거 제거
train_df = train_df.dropna(subset=['working'])
# tryshot제거
train_df = train_df.drop(['tryshot_signal'],axis=1)

# =====
# [1] 컬럼 드랍
# =====
train_df.drop(columns=['count','line', 'name', 'mold_name', 'time', 'date','heating_furnace','molten_volume','passorfail'],
               inplace=True)

train_df.info()
# =====
# [2] 형변환
# =====
# int
train_df['id'] = train_df['id'].astype(object)

# datetime
train_df['registration_time'] = pd.to_datetime(train_df['registration_time'])

# category
train_df['working'] = train_df['working'].astype('category')
train_df['emergency_stop'] = train_df['emergency_stop'].astype('category')

# mold code 원핫 인코딩 -> int값으로 변환
train_df = pd.get_dummies(train_df, columns=['mold_code'], drop_first=False)
train_df['mold_code_8412'] = train_df['mold_code_8412'].astype('int')
train_df['mold_code_8573'] = train_df['mold_code_8573'].astype('int')
train_df['mold_code_8600'] = train_df['mold_code_8600'].astype('int')
train_df['mold_code_8722'] = train_df['mold_code_8722'].astype('int')
train_df['mold_code_8917'] = train_df['mold_code_8917'].astype('int')


# KNN보간법은 각 변수간 값의 차이를 기준으로 값을 찾는데 더미화하면 첫번째 범주가 없어지니까
# 값이 으로 간주되서 정확한 값을 산출하지 못할 것이라 생각

from sklearn.impute import KNNImputer
import pandas as pd
train_df.info()
train_df.isnull().sum()
# 결측치가 있는 온도 컬럼들
temp_cols_with_missing = [
    'molten_temp',
    'upper_mold_temp3',
    'lower_mold_temp3'
]

# 보간 시 사용할 주변 변수들 (숫자형, 온도/운전 관련 변수 위주)
base_numeric_cols = [
    'facility_operation_cycleTime',
    'production_cycletime', 'EMS_operation_time',
    'mold_code', 'cast_pressure', 'biscuit_thickness',
    'upper_mold_temp1', 'upper_mold_temp2',
    'lower_mold_temp1', 'lower_mold_temp2',
    'Coolant_temperature', 'sleeve_temperature','low_section_speed',
    'high_section_speed','physical_strength','mold_code_8412',
      'mold_code_8573', 'mold_code_8600', 'mold_code_8722',
       'mold_code_8917']


# 보간 수행
for temp_col in temp_cols_with_missing:
    if temp_col in train_df.columns:
        missing_count = train_df[temp_col].isna().sum()
        if missing_count > 0:
            # 현재 temp_col을 포함한 컬럼 조합
            available_cols = [col for col in base_numeric_cols + [temp_col] if col in train_df.columns]

            df_temp = train_df[available_cols].copy()

            # KNN 보간기 초기화 및 적용
            knn_imputer = KNNImputer(n_neighbors=min(5, len(train_df)), weights='uniform')
            df_imputed = pd.DataFrame(
                knn_imputer.fit_transform(df_temp),
                columns=available_cols,
                index=df_temp.index
            )

            # 원본에 보간된 값 반영
            train_df[temp_col] = df_imputed[temp_col]
            print(f"✅ KNN 보간으로 '{temp_col}'의 결측치 {missing_count}개 처리 완료")


train_df.keys()
train_df.isnull().sum()

# 현실적으로 일어날 수 없는 극단적인 이상치만 제거
# 정지 데이터가 있는데 이걸 뺄까 말까 데이터가 별로 업어서 빼도 괜찮음 
# 이상치 극단값 삭제 기준 입력값오류 물리적으로 불가능한 값, 10개 이하 개수

# low_section_speed 65535값 제거
train_df = train_df[train_df['low_section_speed'] != 65535]

# biscuit_thickness 420 422 값 제거 입력 오류라고 생각 두자리수는 두께가 몰드 코드별로 다를수도 있다고 생각
train_df['biscuit_thickness'].unique()
train_df['biscuit_thickness'].value_counts()
train_df = train_df[train_df['biscuit_thickness'] != 422]
train_df = train_df[train_df['biscuit_thickness'] != 420]
train_df = train_df[train_df['biscuit_thickness'] >= 10]

# upper_mold_temp1 1449 18~300 후반 1449 말안되는값
train_df['upper_mold_temp1'].sort_values()
train_df = train_df[train_df['upper_mold_temp1'] != 1449]

# upper_mold_temp2  15~389 4232 말안되는 값
train_df['upper_mold_temp2'].sort_values()
train_df = train_df[train_df['upper_mold_temp2'] != 4232]

# lower_mold_temp3 65503 말안되는 값
train_df['lower_mold_temp3'].sort_values()
train_df['lower_mold_temp3'].value_counts()
train_df = train_df[train_df['lower_mold_temp3'] != 65503]

# physical_strength 보통 600 ~ 700 (65535원래있었는데 다른거 제외하다 없어진듯) 0,2 말안되는 값 
train_df['physical_strength'].sort_values()
train_df['physical_strength'].value_counts()
train_df['physical_strength'].unique()
train_df = train_df[train_df['physical_strength'] != 0]
train_df = train_df[train_df['physical_strength'] != 2]

# Coolant_temperature 냉각수온도가 1449도? 말안되지
train_df['Coolant_temperature'].value_counts()
train_df['Coolant_temperature'].unique()
train_df = train_df[train_df['Coolant_temperature'] != 1449]

train_df_E = pd.read_csv('../data/train_ES.csv')
#########################################################

# IQR 확인
# 이상치 상한 하한 기준 기본적으로 IQR 1.5로 하되 
# 한쪽으로 치우쳐진 그래프일 때 상한 하한 조정하는 방식
# IQR 1.5로 했을 때 이상치가 많이 찍혀도 다음 단계에 DBSCAN이랑 isolation forest로 이상치 후보군을 
# 다시 탐지할거기 때문에 일괄적으로 잡아도 될것 같다고 생각


import matplotlib.pyplot as plt
import seaborn as sns

# 수치형 변수 리스트 (원하는 변수들만 선택 가능)
numeric_vars = [
    'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 
    'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
    'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 
    'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3', 
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature','EMS_operation_time'
]

for var in numeric_vars:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='mold_code', y=var, data=train_df_E)
    plt.title(f'Mold Code별 {var} 분포')
    plt.xlabel('Mold Code')
    plt.ylabel(var)
    plt.grid(True)
    plt.show()


# mold code 별 분석
train_df_E = pd.read_csv('../data/train_ES.csv')

# mold code 8412  18114개
# IQR 
train_8412 = train_df_E[train_df_E['mold_code']==8412]

# molten_temp
Q1 = train_8412['molten_temp'].quantile(0.25)
Q3 = train_8412['molten_temp'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

normal = ((train_8412['molten_temp'] > lower_bound) & (train_8412['molten_temp'] < upper_bound)).sum()
outliers = ((train_8412['molten_temp'] <= lower_bound) | (train_8412['molten_temp'] >= upper_bound)).sum()




# facility_operation_cycleTime 코드 별로 IQR 1.5로 끊기
Q1 = train_8412['facility_operation_cycleTime'].quantile(0.25)
Q3 = train_8412['facility_operation_cycleTime'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

normal = ((train_8412['facility_operation_cycleTime'] > lower_bound) & (train_8412['facility_operation_cycleTime'] < upper_bound)).sum()
outliers = ((train_8412['facility_operation_cycleTime'] <= lower_bound) | (train_8412['facility_operation_cycleTime'] >= upper_bound)).sum()





###################################### 이상치###################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# IQR
# train_df_E = pd.read_csv('../data/train_ES.csv')
# # IQR 이상치 1.5 상한 하한 하고 그랬을 때 변수 별 이상치 비율
# def code_IQR():
#     numeric_vars = [
#         'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 
#         'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
#         'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 
#         'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3', 
#         'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time'
#     ]

#     result = []

#     for i in train_df_E['mold_code'].unique():
#         mold_data = train_df_E[train_df_E['mold_code'] == i]
#         for var in numeric_vars:
#             Q1 = mold_data[var].quantile(0.25)
#             Q3 = mold_data[var].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - IQR * 1.5
#             upper_bound = Q3 + IQR * 1.5
#             outliers = (mold_data[var] < lower_bound) | (mold_data[var] > upper_bound)
#             outlier_ratio = outliers.sum() / len(mold_data)

#             result.append({
#                 'mold_code': i,
#                 'variable': var,
#                 'outlier_ratio(%)': round(outlier_ratio * 100, 2)
#             })

#     return result



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# DBSCAN 이상치 후보 중 이상치 (군집있는건 패턴이 있는 이상치(확인필요), 군집없는 이상치는 한번씩 튀는 불규칙적인 이상치라고 생각)
# IQR 1.5로하고  PCA -> IQR -> DBSCAN -> 시각화
# IQR로 이상치 후보군을 한번 걸러내고 또 DBSCAN으로 이상치 
train_df_E = pd.read_csv('../data/train_ES.csv') # 몰드 코드 원핫인코딩
train_df_E = train_df_E.drop(columns=['Unnamed: 0'])
train_df_E = train_df_E.drop(columns=['id'])
train_df_E['mold_code'] = train_df_E['mold_code'].astype('object')
train_df_E2 = train_df_E.copy()
train_df_E2.info()
# 1. 수치형 데이터만 추출 (float64, int64)
numeric_cols = train_df_E2.select_dtypes(include=['float64', 'int64']).columns
X_numeric = train_df_E2[numeric_cols]

# 2. 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 3. PCA (2개 주성분)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# 5. PCA 결과 DataFrame 생성
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# 6. IQR 기준 이상치 탐지 함수
def find_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)].index
    return outliers

outliers_PC1 = find_outliers(df_pca['PC1'])
outliers_PC2 = find_outliers(df_pca['PC2'])

# 7. 이상치 후보 인덱스 합집합
outlier_indices = list(set(outliers_PC1) | set(outliers_PC2))
print(f"Detected outliers count: {len(outlier_indices)}")

# 8. 이상치 후보 데이터만 추출
X_outliers = df_pca.loc[outlier_indices]


# 9. DBSCAN 클러스터링
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_outliers)
print("Estimated number of clusters: " ,len(np.unique(dbscan_labels[dbscan_labels!=-1])))
print("Estimated number of noise points: ",len(dbscan_labels[dbscan_labels==-1]))

# 10. DBSCAN 결과 시각화
plt.figure(figsize=(8, 6))
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = (dbscan_labels == k)
    xy = X_outliers[class_member_mask]

    if k == -1:
        col = [0, 0, 0, 1]  # Noise = 검정색

    plt.scatter(xy['PC1'], xy['PC2'], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise', alpha=0.6)

plt.title('DBSCAN Clustering on PCA Outliers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


#################################3isolation forest로 뽑은 정상치의 IQR상한 하한으로 이상치 뽑아서
# DBSCAN돌린거       isolation -> IQR(isolation) -> PCA -> DBSCAN -> 시각화
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 데이터 불러오기
train_df_E = pd.read_csv('../data/train_ES.csv')
train_df_E = train_df_E.drop(columns=['Unnamed: 0'])
train_df_E = train_df_E.drop(columns=['id'])
train_df_E['mold_code'] = train_df_E['mold_code'].astype('object')
train_df_E3 = train_df_E.copy()

# 수치형 변수 선택
numeric_cols = train_df_E3.select_dtypes(include=['float64', 'int64']).columns
X_numeric = train_df_E3[numeric_cols]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Isolation Forest로 정상 데이터 선별
iso = IsolationForest(contamination=0.06, random_state=42)
iso_labels = iso.fit_predict(X_scaled)  # 정상: 1, 이상치: -1

normal_data = train_df_E3[iso_labels == 1]  # 정상 데이터만 추출

# ✅ IQR 상하한 계산 함수 (mold_code 기준)
def normal_IQR(normal_data, mold_col='mold_code'):
    numeric_vars = [
        'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 
        'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
        'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 
        'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3', 
        'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time'
    ]
    
    bounds = []

    for code in normal_data[mold_col].unique():
        mold_data = normal_data[normal_data[mold_col] == code]
        for var in numeric_vars:
            Q1 = mold_data[var].quantile(0.25)
            Q3 = mold_data[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            bounds.append({
                'mold_code': code,
                'variable': var,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    return pd.DataFrame(bounds)

# IQR 상하한 기준 얻기
iqr_bounds_df = normal_IQR(normal_data)

# ✅ IQR 기준 이상치 판별 함수
def is_iqr_outlier(row, bounds_df, mold_col='mold_code'):
    mold = row[mold_col]
    for _, bound in bounds_df[bounds_df['mold_code'] == mold].iterrows():
        var = bound['variable']
        val = row[var]
        if val < bound['lower_bound'] or val > bound['upper_bound']:
            return True  # 하나라도 벗어나면 이상치로 간주
    return False

# 전체 데이터 중 IQR 벗어난 이상치만 필터링
iqr_outliers = train_df_E3[train_df_E3.apply(lambda row: is_iqr_outlier(row, iqr_bounds_df), axis=1)]
print(f"IQR 기반 이상치 수: {len(iqr_outliers)}")

# PCA를 위한 수치형 변수 추출 및 표준화
X_iqr_outliers = iqr_outliers[numeric_cols]
X_iqr_scaled = scaler.transform(X_iqr_outliers)  # 같은 scaler 사용

# PCA (2개 주성분)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iqr_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# ✅ DBSCAN 클러스터링
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_pca)

print("Estimated number of clusters:", len(set(dbscan_labels) - {-1}))
print("Estimated number of noise points:", list(dbscan_labels).count(-1))

# ✅ 시각화
plt.figure(figsize=(8, 6))
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_mask = (dbscan_labels == k)
    cluster_points = df_pca[class_mask]

    if k == -1:
        col = [0, 0, 0, 1]  # Noise = 검정색

    plt.scatter(cluster_points['PC1'], cluster_points['PC2'], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise', alpha=0.6)

plt.title('DBSCAN Clustering on IQR Outliers (based on ISO-normal data)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()




#####################  PCA -> DBSCAN -> 시각화
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 데이터 불러오기 및 전처리
train_df_E = pd.read_csv('../data/train_ES.csv')
train_df_E = train_df_E.drop(columns=['Unnamed: 0'])
train_df_E = train_df_E.drop(columns=['id'])
train_df_E['mold_code'] = train_df_E['mold_code'].astype('object')

# 수치형 변수만 추출
numeric_cols = train_df_E.select_dtypes(include=['float64', 'int64']).columns
X_numeric = train_df_E[numeric_cols]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# PCA 2차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# DBSCAN 클러스터링 (IQR 제거)
dbscan = DBSCAN(eps=0.5, min_samples=11)
dbscan_labels = dbscan.fit_predict(df_pca)

# 클러스터 수 및 노이즈 포인트 수 출력
print("Estimated number of clusters:", len(set(dbscan_labels) - {-1}))
print("Estimated number of noise points:", list(dbscan_labels).count(-1))

# 시각화
plt.figure(figsize=(8, 6))
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_mask = (dbscan_labels == k)
    cluster_points = df_pca[class_mask]
    
    if k == -1:
        col = [0, 0, 0, 1]  # Noise = 검정색

    plt.scatter(cluster_points['PC1'], cluster_points['PC2'], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise', alpha=0.6)

plt.title('DBSCAN Clustering on PCA Data (No IQR)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()



##########################3전체 데이터 isolation forest#####
# 불량률 전체 약4.4% 실제 불량률 보다 약간의 여유를 두고자 6%로 설정

train_df_E = pd.read_csv('../data/train_ES.csv')
train_df_E = train_df_E.drop(columns=['Unnamed: 0','id'])
train_df_E['mold_code'] = train_df_E['mold_code'].astype('object')

from sklearn.ensemble import IsolationForest 

train_df_E3 = train_df_E.copy()
X = train_df_E3.select_dtypes(include=np.number) 
X.keys()
# 스케일링
X_scaled = StandardScaler().fit_transform(X)

iso_forest = IsolationForest(n_estimators=150, contamination=0.06, random_state=42)
iso_labels = iso_forest.fit_predict(X_scaled)

print("Isolation Forest 이상치 수:", sum(iso_labels == -1))
print("Isolation Forest 정상 수 :", sum(iso_labels == 1))
# 이상치 수: 4385
# 정상 수: 68696

# PCA 2D 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['anomaly'] = iso_labels



# 결과 시각화
plt.figure(figsize=(8, 6))
colors = ['red' if label == -1 else 'green' for label in iso_labels]
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=colors, alpha=0.6)
plt.title('Isolation Forest on PCA Inliers (IQR 안쪽 데이터)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



 
 ## isolation forest로 이상치가 아닌 샘플로 IQR 상한하한 정하기

normal_data = train_df_E3[iso_labels==1] # traindata와 스케일 한 X_scaled랑 배열이 같기때문에 이런식을 필터링 가능

normal_data


def normal_IQR(normal_data, mold_col='mold_code'):
    numeric_vars = [
        'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 
        'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
        'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 
        'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3', 
        'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time'
    ]
    
    bounds = []

    for code in normal_data[mold_col].unique():
        mold_data = normal_data[normal_data[mold_col] == code]
        for var in numeric_vars:
            Q1 = mold_data[var].quantile(0.25)
            Q3 = mold_data[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            bounds.append({
                'mold_code': code,
                'variable': var,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    return pd.DataFrame(bounds)


################# isolation forest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib

# 1. 데이터 로드 및 전처리
train_df_E = pd.read_csv('../data/train_ES.csv')
train_df_E = train_df_E.drop(columns=['Unnamed: 0', 'id'])
train_df_E['mold_code'] = train_df_E['mold_code'].astype('object')
train_df_E.keys()
# 2. 수치형 데이터 선택
X = train_df_E.select_dtypes(include=np.number)

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Isolation Forest 모델 학습
iso_forest = IsolationForest(n_estimators=150, contamination=0.06, random_state=42)
iso_forest.fit(X_scaled)

# 5. PCA 2차원 학습
pca = PCA(n_components=2)
pca.fit(X_scaled)

# 6. 모델 저장
joblib.dump(scaler, "scaler.pkl")
joblib.dump(iso_forest, "isolation_model.pkl")
joblib.dump(pca, "pca_model.pkl")

print("✅ Scaler, IsolationForest, PCA 모델 저장 완료")

#
import pandas as pd
train_df_E

import pandas as pd

# 사용할 센서 컬럼들
sensor_columns = [
    'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 
    'low_section_speed', 'high_section_speed', 'cast_pressure', 
    'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2', 
    'upper_mold_temp3', 'lower_mold_temp1', 'lower_mold_temp2', 
    'lower_mold_temp3', 'sleeve_temperature', 'physical_strength', 
    'Coolant_temperature'
]

# 양품 데이터만 필터링 (passorfail == 0)
good_data = train_df_E[train_df_E['passorfail'] == 0]

print(f"전체 데이터: {len(train_df_E)}건")
print(f"양품 데이터: {len(good_data)}건 ({len(good_data)/len(train_df_E)*100:.1f}%)")
print()

# 몰드별 양품 평균 계산
mold_good_averages = {}
mold_codes = sorted(train_df_E['mold_code'].unique())

for mold_code in mold_codes:
    mold_good_data = good_data[good_data['mold_code'] == mold_code]
    
    print(f"🔧 몰드 {mold_code}: 양품 {len(mold_good_data)}건")
    
    if len(mold_good_data) > 0:
        mold_averages = {}
        for sensor in sensor_columns:
            if sensor in mold_good_data.columns:
                avg_value = mold_good_data[sensor].mean()
                mold_averages[sensor] = round(avg_value, 2)
                print(f"  {sensor}: {avg_value:.2f}")
        
        mold_good_averages[str(mold_code)] = mold_averages
    else:
        print(f"  ⚠️ 양품 데이터 없음")
    print()

# 결과를 딕셔너리 형태로 출력 (코드에 직접 복사 가능)
print("="*60)
print("📋 코드에 직접 사용할 수 있는 형태:")
print("="*60)
print("DEFECT_NORMAL_PATTERNS = {")

for mold_code, averages in mold_good_averages.items():
    print(f'    "{mold_code}": {{')
    for sensor, value in averages.items():
        print(f'        "{sensor}": {value},')
    print("    },")

print("}")

# 각 몰드별 통계 요약
print("\n" + "="*60)
print("📊 몰드별 양품 데이터 통계:")
print("="*60)
for mold_code in mold_codes:
    mold_good_count = len(good_data[good_data['mold_code'] == mold_code])
    mold_total_count = len(train_df_E[train_df_E['mold_code'] == mold_code])
    good_rate = mold_good_count / mold_total_count * 100 if mold_total_count > 0 else 0
    
    print(f"몰드 {mold_code}: 양품 {mold_good_count}/{mold_total_count}건 ({good_rate:.1f}%)")

# 함수 형태로도 제공
print("\n" + "="*60)
print("🔧 함수 형태 (paste-4.txt에 추가):")
print("="*60)

function_code = '''
def get_defect_normal_pattern(mold_code):
    """불량 예측용 양품 기준값 - 실제 양품 데이터 기반"""
    
    DEFECT_NORMAL_PATTERNS = {
'''

for mold_code, averages in mold_good_averages.items():
    function_code += f'        "{mold_code}": {{\n'
    for sensor, value in averages.items():
        function_code += f'            "{sensor}": {value},\n'
    function_code += "        },\n"

function_code += '''    }
    
    pattern = DEFECT_NORMAL_PATTERNS.get(str(mold_code))
    
    if pattern:
        return pattern
    else:
        print(f"🚨 [WARNING] 몰드 {mold_code} 양품 패턴 없음, 기본값 사용")
        # 전체 몰드 평균 반환
        return DEFECT_NORMAL_PATTERNS.get("8412", {})  # 기본값
'''

print(function_code)