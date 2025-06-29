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

DATA = load_data('asset/data/a_shap_applied_data.csv')
submit_df = pd.read_csv('asset/data/submit.csv')

# 2) pick only the columns you need from DATA
passorfail_only = DATA[['id', 'passorfail']]

# 3) left-merge onto submit_df
merged = submit_df.merge(
    passorfail_only,
    on='id',
    how='left'
)

merged['passorfail_y'].value_counts()
merged.to_csv('asset/data/submit_4.csv')

merged.drop(columns=['passorfail_x'], inplace=True)
merged.rename(columns={'passorfail_y': 'passorfail'}, inplace=True)

mask = merged['passorfail'].isna()
merged[mask]['']
DATA['']