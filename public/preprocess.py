import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('../data/train.csv')


# =====
# [1] 컬럼 드랍
# =====
train_df.drop(columns=['line', 'name', 'mold_name', 'time', 'date', 'tryshot_signal'], inplace=True)

# =====
# [2] 형변환
# =====
# int
train_df['id'] = train_df['id'].astype(int)

# datetime
train_df['registration_time'] = pd.to_datetime(train_df['registration_time'])

# category
train_df['working'] = train_df['working'].astype('category')
train_df['emergency_stop'] = train_df['emergency_stop'].astype('category')
train_df['mold_code'] = train_df['mold_code'].astype('category')
train_df['heating_furnace'] = train_df['heating_furnace'].astype('category')

# bool
train_df['passorfail'] = train_df['passorfail'].astype('bool')

train_df['hour'] = train_df['registration_time'].dt.hour
train_df['day'] = train_df['registration_time'].dt.day
train_df['weekday'] = train_df['registration_time'].dt.weekday


# ===============================================
