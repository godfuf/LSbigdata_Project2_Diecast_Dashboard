import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # working 컬럼이 있으면 dropna, 없으면 스킵
        if 'working' in X.columns:
            X = X.dropna(subset=['working'])
        
        drop_cols = ['tryshot_signal', 'working', 'count', 'EMS_operation_time', 'line', 'name',
                     'mold_name', 'time', 'date', 'heating_furnace', 'molten_volume',
                     'emergency_stop', 'registration_time', 'passorfail']
        X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)
        
        if 'id' in X.columns:
            X['id'] = X['id'].astype(object)
            
        temp_cols_with_missing = [
            'molten_temp', 'upper_mold_temp3', 'lower_mold_temp3'
        ]
        base_cols = [col for col in [
            'facility_operation_cycleTime', 'production_cycletime', 'cast_pressure',
            'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2',
            'lower_mold_temp1', 'lower_mold_temp2', 'Coolant_temperature',
            'sleeve_temperature', 'low_section_speed', 'high_section_speed',
            'physical_strength'
        ] if col in X.columns]

        for temp_col in temp_cols_with_missing:
            if temp_col in X.columns and X[temp_col].isna().sum() > 0:
                impute_cols = [col for col in base_cols + [temp_col] if col in X.columns]
                df_temp = X[impute_cols].copy()
                imputer = KNNImputer(n_neighbors=5)
                df_imputed = pd.DataFrame(imputer.fit_transform(df_temp), columns=impute_cols, index=df_temp.index)
                X[temp_col] = df_imputed[temp_col]

        return X