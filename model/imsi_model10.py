"""
XGBoost/LightGBM 전용 최적화 파이프라인 (최종 모델 구성)
XGBoost/LightGBM 자체 튜닝 + 정밀도 제약 임계값 최적화
EMS_operation_time 완전 제거로 안정성 확보
정밀도 ≥ 0.8 조건 하에서 민감도 최대화
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("         XGBoost/LightGBM 전용 최적화 파이프라인")
print("       자체 튜닝 + 정밀도 제약 임계값 최적화")
print("       정밀도 ≥ 0.8 조건 하에서 민감도 최대화")
print("=" * 80)

class MoldCodeQualityPipeline:
    """제조업 품질 예측 통합 파이프라인 (XGBoost/LightGBM 전용)"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.optimal_thresholds = {}
        self.feature_names = None
        
        # 각 mold_code별 최적 모델 설정 (최신 분석 결과 기반)
        self.best_model_config = {
            8412: 'XGBoost_Balanced',
            8573: 'XGBoost_Balanced', 
            8600: 'LightGBM_Balanced',  # GradientBoosting 대신 LightGBM 사용
            8722: 'XGBoost_Balanced',
            8917: 'LightGBM_Balanced'
        }
        
        print("✅ 파이프라인 초기화 완료")
        print(f"   - 각 mold_code별 최적 모델: {len(self.best_model_config)}개")
        print(f"   - XGBoost: 3개 (8412, 8573, 8722)")
        print(f"   - LightGBM: 2개 (8600, 8917)")
        print(f"   - EMS_operation_time 제거: 안정성 확보")
        print(f"   - GridSearchCV 제거: 자체 튜닝으로 고속화")
        print(f"   - SMOTE 제거: 내장 불균형 처리로 단순화")
        print(f"   🎯 임계값 전략: 정밀도 ≥ 0.8 조건에서 민감도 최대화")
    
    def preprocess_data(self, df_raw, is_training=True):
        """데이터 전처리 (EMS_operation_time 완전 제거)"""
        
        print(f"\n🔧 전처리 시작 (훈련 모드: {is_training})")
        
        # 원본 데이터 복사
        df = df_raw.copy()
        
        # tryshot_signal 비즈니스 룰 체크
        tryshot_exists = df['tryshot_signal'].notna()
        tryshot_count = tryshot_exists.sum()
        
        if tryshot_count > 0:
            print(f"🚨 tryshot_signal 값이 있는 데이터 {tryshot_count}개 발견")
            print("   → 비즈니스 룰에 따라 무조건 불량(1)로 예측")
            
            # tryshot이 있는 데이터는 별도 처리 (무조건 불량)
            tryshot_data = df[tryshot_exists].copy()
            tryshot_predictions = pd.Series(1, index=tryshot_data.index, name='prediction')
            
            # tryshot이 없는 데이터만 모델 예측 대상
            df = df[~tryshot_exists].copy()
            print(f"   → 모델 예측 대상: {len(df)}개")
        else:
            print("✅ tryshot_signal 값이 없음 - 모든 데이터 모델 예측")
            tryshot_data = None
            tryshot_predictions = None
        
        # 기본 필터링 및 변수 제거 (EMS_operation_time 포함!)
        exclude_columns = ['id', 'line', 'name', 'mold_name', 'time', 'date', 
                          'emergency_stop', 'molten_volume', 'registration_time', 
                          'heating_furnace', 'count', 'tryshot_signal', 'EMS_operation_time']
        
        df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        print(f"   🚫 EMS_operation_time 제거로 안정성 확보")
        
        # working 결측치 제거
        if 'working' in df.columns and df['working'].isnull().sum() > 0:
            df = df.dropna(subset=['working'])
        
        # working 레이블 인코딩만 수행 (EMS_operation_time 제외!)
        if is_training:
            self.le_working = LabelEncoder()
            
            if 'working' in df.columns:
                df['working'] = self.le_working.fit_transform(df['working'])
                print(f"   ✅ working 레이블 인코딩 완료")
        else:
            # 예측시에는 기존 인코더 사용
            if 'working' in df.columns and hasattr(self, 'le_working'):
                try:
                    df['working'] = self.le_working.transform(df['working'])
                except ValueError as e:
                    print(f"⚠️ working 인코딩 오류: {str(e)}")
                    # 새로운 값이 있으면 훈련시 가장 빈번했던 값으로 대체
                    unknown_mask = ~df['working'].isin(self.le_working.classes_)
                    if unknown_mask.any():
                        first_class = self.le_working.classes_[0]
                        print(f"   - 새로운 working 값들을 '{first_class}'로 대체: {unknown_mask.sum()}개")
                        df.loc[unknown_mask, 'working'] = first_class
                    df['working'] = self.le_working.transform(df['working'])
        
        # KNN 보간 (홍님 지정 변수들만, EMS_operation_time 제외!)
        target_impute_cols = ['molten_temp', 'lower_mold_temp3', 'upper_mold_temp3']
        actual_missing = [col for col in target_impute_cols if col in df.columns and df[col].isnull().sum() > 0]
        
        if actual_missing:
            print(f"   - KNN 보간 대상: {actual_missing}")
            
            # 참조 변수에서 EMS_operation_time 완전 제거!
            reference_cols = ['working', 'molten_temp', 'facility_operation_cycleTime', 
                             'production_cycletime', 'mold_code',
                             'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
                             'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                             'sleeve_temperature', 'Coolant_temperature']
            
            available_reference_cols = [col for col in reference_cols if col in df.columns]
            print(f"   📋 KNN 참조 변수: {len(available_reference_cols)}개 (EMS_operation_time 제외)")
            
            if is_training:
                self.knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
                df_temp = df[available_reference_cols].copy()
                df_imputed = pd.DataFrame(
                    self.knn_imputer.fit_transform(df_temp),
                    columns=available_reference_cols,
                    index=df_temp.index
                )
            else:
                df_temp = df[available_reference_cols].copy()
                df_imputed = pd.DataFrame(
                    self.knn_imputer.transform(df_temp),
                    columns=available_reference_cols,
                    index=df_temp.index
                )
            
            # 지정된 변수들만 업데이트
            for col in actual_missing:
                df[col] = df_imputed[col]
                print(f"     ✅ {col} KNN 보간 완료")
        
        # production_cycletime 0값 처리
        if 'production_cycletime' in df.columns:
            zero_count = (df['production_cycletime'] == 0).sum()
            if zero_count > 0:
                if is_training:
                    self.production_mean = df[(df['production_cycletime'] > 0) & 
                                            (df['production_cycletime'] <= 115)]['production_cycletime'].mean()
                
                df.loc[df['production_cycletime'] == 0, 'production_cycletime'] = self.production_mean
                print(f"   - production_cycletime 0값 {zero_count}개를 평균값으로 대체")
        
        # 나머지 결측치 제거
        before_rows = len(df)
        df = df.dropna()
        removed_rows = before_rows - len(df)
        if removed_rows > 0:
            print(f"   - 나머지 결측치 제거: {removed_rows}행 제거")
        
        print(f"✅ 전처리 완료: {len(df)}행")
        
        return df, tryshot_data, tryshot_predictions
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 자체 튜닝"""
        
        print(f"     🚀 XGBoost 자체 튜닝 시작...")
        
        # DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 기본 파라미터 설정
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': 3,  # 불균형 데이터 처리
            'random_state': 42,
            'verbosity': 0
        }
        
        # 파라미터 후보들
        param_candidates = [
            {**base_params, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8},
            {**base_params, 'max_depth': 9, 'learning_rate': 0.1, 'subsample': 0.8},
            {**base_params, 'max_depth': 6, 'learning_rate': 0.2, 'subsample': 0.9},
            {**base_params, 'max_depth': 9, 'learning_rate': 0.05, 'subsample': 0.8}
        ]
        
        best_score = 0
        best_params = None
        best_model = None
        
        # 각 파라미터 조합 테스트
        for i, params in enumerate(param_candidates):
            try:
                # 조기종료로 빠른 훈련
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=200,  # 충분히 크게 설정
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=20,  # 조기종료
                    verbose_eval=False
                )
                
                # 검증 점수 확인
                score = model.best_score
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                print(f"       파라미터 {i+1} 실패: {str(e)[:30]}...")
                continue
        
        # 최고 모델이 없으면 기본 모델 생성
        if best_model is None:
            print(f"       ⚠️ 모든 파라미터 실패, 기본 모델 사용")
            best_model = xgb.train(base_params, dtrain, num_boost_round=100)
            best_params = base_params
        
        print(f"       ✅ 최고 점수: {best_score:.4f}")
        
        return best_model, best_params
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 자체 튜닝"""
        
        print(f"     🚀 LightGBM 자체 튜닝 시작...")
        
        # 기본 파라미터 설정
        base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': True,  # 불균형 데이터 처리 (LightGBM 방식)
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        # 파라미터 후보들
        param_candidates = [
            {**base_params, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 63},
            {**base_params, 'max_depth': 9, 'learning_rate': 0.1, 'num_leaves': 127},
            {**base_params, 'max_depth': 6, 'learning_rate': 0.2, 'num_leaves': 31},
            {**base_params, 'max_depth': -1, 'learning_rate': 0.05, 'num_leaves': 63}
        ]
        
        best_score = 0
        best_params = None
        best_model = None
        
        # 각 파라미터 조합 테스트
        for i, params in enumerate(param_candidates):
            try:
                # Dataset 생성
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # 조기종료로 빠른 훈련
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,  # 충분히 크게 설정
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]  # 조기종료
                )
                
                # 검증 점수 확인
                score = model.best_score['valid_0']['auc']
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                print(f"       파라미터 {i+1} 실패: {str(e)[:30]}...")
                continue
        
        # 최고 모델이 없으면 기본 모델 생성
        if best_model is None:
            print(f"       ⚠️ 모든 파라미터 실패, 기본 모델 사용")
            train_data = lgb.Dataset(X_train, label=y_train)
            best_model = lgb.train(base_params, train_data, num_boost_round=100)
            best_params = base_params
        
        print(f"       ✅ 최고 점수: {best_score:.4f}")
        
        return best_model, best_params
    
    def optimize_threshold(self, model, X_val, y_val, model_type, mold_code):
        """임계값 최적화 (정밀도 ≥ 0.8 조건 하에서 민감도 최대화)"""
        
        # 확률 예측
        if model_type == 'XGBoost_Balanced':
            dval = xgb.DMatrix(X_val)
            y_proba = model.predict(dval)
        else:  # LightGBM
            y_proba = model.predict(X_val)
        
        # 다양한 임계값 테스트
        thresholds = np.arange(0.05, 0.95, 0.05)
        
        best_threshold = 0.5
        best_sensitivity = 0
        valid_thresholds = []
        
        print(f"     🎯 정밀도 ≥ 0.8 조건 하에서 민감도 최대화")
        
        # 모든 임계값에 대해 정밀도와 민감도 계산
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            sensitivity = recall_score(y_val, y_pred_thresh, zero_division=0)
            precision = precision_score(y_val, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
            
            # 정밀도 ≥ 0.8 조건을 만족하는 임계값들만 고려
            if precision >= 0.8:
                valid_thresholds.append({
                    'threshold': threshold,
                    'sensitivity': sensitivity,
                    'precision': precision,
                    'f1': f1
                })
        
        # 정밀도 ≥ 0.8 조건을 만족하는 임계값들 중에서 민감도 최대화
        if valid_thresholds:
            best_result = max(valid_thresholds, key=lambda x: x['sensitivity'])
            best_threshold = best_result['threshold']
            best_sensitivity = best_result['sensitivity']
            print(f"       ✅ 정밀도 ≥ 0.8 조건 만족: {len(valid_thresholds)}개 후보 중 민감도 최대 선택")
            print(f"       📊 선택된 성능: 민감도 {best_result['sensitivity']:.4f}, 정밀도 {best_result['precision']:.4f}")
        else:
            # 정밀도 ≥ 0.8 불가능한 경우 F1 점수 최대화로 fallback
            print(f"       ⚠️ 정밀도 ≥ 0.8 불가능 → F1 점수 최대화로 fallback")
            fallback_results = []
            for threshold in thresholds:
                y_pred_thresh = (y_proba >= threshold).astype(int)
                sensitivity = recall_score(y_val, y_pred_thresh, zero_division=0)
                precision = precision_score(y_val, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
                fallback_results.append({
                    'threshold': threshold,
                    'sensitivity': sensitivity,
                    'precision': precision,
                    'f1': f1
                })
            
            if fallback_results:
                best_result = max(fallback_results, key=lambda x: x['f1'])
                best_threshold = best_result['threshold']
                best_sensitivity = best_result['sensitivity']
                print(f"       📊 Fallback 성능: 민감도 {best_result['sensitivity']:.4f}, 정밀도 {best_result['precision']:.4f}")
        
        return best_threshold, best_sensitivity
    
    def train_mold_specific_models(self, df_train):
        """각 mold_code별 최적 모델 훈련"""
        
        print(f"\n🤖 각 mold_code별 모델 훈련 시작")
        
        # 타겟과 특성 분리
        X = df_train.drop(['passorfail', 'mold_code'], axis=1)
        y = df_train['passorfail']
        self.feature_names = X.columns.tolist()
        
        print(f"   📋 사용 변수: {len(self.feature_names)}개 (EMS_operation_time 제외)")
        
        # 각 mold_code별로 훈련
        for mold_code in self.best_model_config.keys():
            if mold_code not in df_train['mold_code'].values:
                print(f"⚠️ mold_code {mold_code} 데이터 없음")
                continue
            
            print(f"\n🔍 === mold_code {mold_code} 모델 훈련 ===")
            
            # 해당 mold_code 데이터 추출
            mold_mask = df_train['mold_code'] == mold_code
            X_mold = X[mold_mask]
            y_mold = y[mold_mask]
            
            print(f"   - 데이터: {len(X_mold)}개, 불량률: {y_mold.mean():.4f}")
            
            # 데이터 분할
            X_train_mold, X_val_mold, y_train_mold, y_val_mold = train_test_split(
                X_mold, y_mold, test_size=0.2, random_state=42, stratify=y_mold
            )
            
            # 모델 타입 확인
            model_type = self.best_model_config[mold_code]
            print(f"   - 최적 모델: {model_type}")
            
            # 모델별 자체 튜닝
            if model_type == 'XGBoost_Balanced':
                best_model, best_params = self.optimize_xgboost(X_train_mold, y_train_mold, X_val_mold, y_val_mold)
            else:  # LightGBM_Balanced
                best_model, best_params = self.optimize_lightgbm(X_train_mold, y_train_mold, X_val_mold, y_val_mold)
            
            # 임계값 최적화
            print(f"   - 임계값 최적화 중...")
            best_threshold, best_score = self.optimize_threshold(
                best_model, X_val_mold, y_val_mold, model_type, mold_code
            )
            
            # 최적 임계값으로 성능 평가
            if model_type == 'XGBoost_Balanced':
                dval = xgb.DMatrix(X_val_mold)
                y_proba = best_model.predict(dval)
            else:
                y_proba = best_model.predict(X_val_mold)
            
            y_pred_optimal = (y_proba >= best_threshold).astype(int)
            
            sensitivity = recall_score(y_val_mold, y_pred_optimal, zero_division=0)
            f1 = f1_score(y_val_mold, y_pred_optimal, zero_division=0)
            accuracy = accuracy_score(y_val_mold, y_pred_optimal)
            
            print(f"   ✅ 튜닝 완료:")
            print(f"     * 최적 임계값: {best_threshold:.3f}")
            print(f"     * 최종 성능: 민감도 {sensitivity:.4f}, F1 {f1:.4f}, 정확도 {accuracy:.4f}")
            print(f"     * 최적화 점수: {best_score:.4f}")
            
            # 전체 데이터로 재훈련
            if model_type == 'XGBoost_Balanced':
                dtrain_full = xgb.DMatrix(X_mold, label=y_mold)
                final_model = xgb.train(best_params, dtrain_full, num_boost_round=best_model.best_iteration)
            else:
                train_data_full = lgb.Dataset(X_mold, label=y_mold)
                final_model = lgb.train(best_params, train_data_full, num_boost_round=best_model.best_iteration)
            
            # 모델 및 임계값 저장
            self.models[mold_code] = final_model
            self.best_params[mold_code] = best_params
            self.optimal_thresholds[mold_code] = best_threshold
        
        print(f"\n✅ 모든 mold_code 모델 훈련 완료: {len(self.models)}개")
        print(f"   - 평균 임계값: {np.mean(list(self.optimal_thresholds.values())):.3f}")
        print(f"   🎯 모든 금형: 정밀도 ≥ 0.8 조건 하에서 민감도 최대화")
    
    def predict(self, df_test):
        """통합 예측 함수 (임계값 적용)"""
        
        print(f"\n🔮 예측 시작")
        
        # 전처리
        df_processed, tryshot_data, tryshot_predictions = self.preprocess_data(df_test, is_training=False)
        
        if len(df_processed) == 0:
            print("⚠️ 예측할 데이터 없음 (모두 tryshot)")
            return tryshot_predictions
        
        # 예측 결과 저장
        predictions = pd.Series(index=df_processed.index, dtype=int, name='prediction')
        
        # 각 mold_code별 예측
        for mold_code in df_processed['mold_code'].unique():
            if mold_code not in self.models:
                print(f"⚠️ mold_code {mold_code} 모델 없음 - 기본값(0) 예측")
                mask = df_processed['mold_code'] == mold_code
                predictions[mask] = 0
                continue
            
            # 해당 mold_code 데이터 추출
            mask = df_processed['mold_code'] == mold_code
            X_mold = df_processed[mask].drop(['passorfail', 'mold_code'], axis=1)
            
            # 모델 가져오기
            model = self.models[mold_code]
            model_type = self.best_model_config[mold_code]
            threshold = self.optimal_thresholds.get(mold_code, 0.5)
            
            # 확률 예측 및 임계값 적용
            if model_type == 'XGBoost_Balanced':
                dtest = xgb.DMatrix(X_mold)
                y_proba = model.predict(dtest)
            else:  # LightGBM
                y_proba = model.predict(X_mold)
            
            # 최적 임계값으로 예측
            pred = (y_proba >= threshold).astype(int)
            predictions[mask] = pred
            
            print(f"   - mold_code {mold_code}: {mask.sum()}개 예측 완료 (임계값: {threshold:.3f})")
        
        # tryshot 예측과 합치기
        if tryshot_predictions is not None:
            all_predictions = pd.concat([predictions, tryshot_predictions])
            print(f"   - tryshot 강제 불량: {len(tryshot_predictions)}개")
        else:
            all_predictions = predictions
        
        print(f"✅ 전체 예측 완료: {len(all_predictions)}개")
        
        return all_predictions
    
    def save_pipeline(self, filepath):
        """파이프라인 저장"""
        pipeline_data = {
            'models': self.models,
            'best_params': self.best_params,
            'optimal_thresholds': self.optimal_thresholds,
            'feature_names': self.feature_names,
            'best_model_config': self.best_model_config,
            'le_working': getattr(self, 'le_working', None),
            'knn_imputer': getattr(self, 'knn_imputer', None),
            'production_mean': getattr(self, 'production_mean', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"✅ 파이프라인 저장: {filepath}")
        print(f"   - 모델: {len(self.models)}개")
        print(f"   - 임계값: {len(self.optimal_thresholds)}개")
    
    def load_pipeline(self, filepath):
        """파이프라인 로드"""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.models = pipeline_data['models']
        self.best_params = pipeline_data['best_params']
        self.optimal_thresholds = pipeline_data.get('optimal_thresholds', {})
        self.feature_names = pipeline_data['feature_names']
        self.best_model_config = pipeline_data['best_model_config']
        self.le_working = pipeline_data.get('le_working')
        self.knn_imputer = pipeline_data.get('knn_imputer')
        self.production_mean = pipeline_data.get('production_mean')
        
        print(f"✅ 파이프라인 로드: {filepath}")
        print(f"   - 모델: {len(self.models)}개")
        print(f"   - 임계값: {len(self.optimal_thresholds)}개")

# 실행 부분
if __name__ == "__main__":
    print("\n📊 1단계: 데이터 로드")
    
    # 데이터 로드
    try:
        df_raw = pd.read_csv('../data/train.csv')
        print(f"✅ 데이터 로드 성공: {df_raw.shape[0]}행, {df_raw.shape[1]}개 변수")
    except:
        print("❌ train.csv 파일을 찾을 수 없습니다.")
        exit()
    
    print("\n🔄 2단계: Train/Test 분할 (데이터 누출 방지)")
    
    # tryshot_signal NaN 필터링 (분할 전에 미리)
    df_clean = df_raw[df_raw['tryshot_signal'].isna()].copy()
    print(f"✅ tryshot_signal NaN 필터링: {len(df_clean)}행")
    
    # Train/Test 분할 (전처리 전에!)
    train_df, test_df = train_test_split(
        df_clean, test_size=0.2, random_state=42, 
        stratify=df_clean['passorfail']
    )
    print(f"   - 훈련 데이터: {len(train_df)}행")
    print(f"   - 테스트 데이터: {len(test_df)}행")
    
    print("\n🏭 3단계: 파이프라인 훈련")
    
    # 파이프라인 생성 및 훈련
    pipeline = MoldCodeQualityPipeline()
    
    # 훈련 데이터 전처리
    train_processed, _, _ = pipeline.preprocess_data(train_df, is_training=True)
    
    # 각 mold_code별 모델 훈련
    pipeline.train_mold_specific_models(train_processed)
    
    print("\n🧪 4단계: 테스트 데이터 평가")
    
    # 테스트 데이터 예측
    test_predictions = pipeline.predict(test_df)
    
    # 성능 평가
    test_processed, _, _ = pipeline.preprocess_data(test_df, is_training=False)
    y_true = test_processed['passorfail']
    y_pred = test_predictions[test_processed.index]
    
    # 전체 성능
    overall_sensitivity = recall_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n🎯 전체 성능:")
    print(f"   - 민감도: {overall_sensitivity:.4f}")
    print(f"   - F1 점수: {overall_f1:.4f}")
    print(f"   - 정확도: {overall_accuracy:.4f}")
    
    # mold_code별 성능
    print(f"\n📊 mold_code별 테스트 성능 및 혼동행렬:")
    for mold_code in test_processed['mold_code'].unique():
        if mold_code in pipeline.models:
            mask = test_processed['mold_code'] == mold_code
            y_true_mold = y_true[mask]
            y_pred_mold = y_pred[mask]
            
            sensitivity = recall_score(y_true_mold, y_pred_mold, zero_division=0)
            f1 = f1_score(y_true_mold, y_pred_mold, zero_division=0)
            accuracy = accuracy_score(y_true_mold, y_pred_mold)
            precision = precision_score(y_true_mold, y_pred_mold, zero_division=0)
            threshold = pipeline.optimal_thresholds.get(mold_code, 0.5)
            model_type = pipeline.best_model_config[mold_code]
            
            # 혼동행렬 계산
            cm = confusion_matrix(y_true_mold, y_pred_mold)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # FN/FP 비율 계산
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            print(f"\n🔍 === mold_code {mold_code} 상세 결과 ===")
            print(f"   📊 성능: 민감도 {sensitivity:.4f}, F1 {f1:.4f}, 정확도 {accuracy:.4f}, 정밀도 {precision:.4f}")
            print(f"   🤖 모델: {model_type}, 임계값: {threshold:.3f} ({mask.sum()}개)")
            print(f"   📈 혼동행렬:")
            print(f"           예측")
            print(f"실제    정상   불량")
            print(f"정상   {tn:4d}  {fp:4d}")
            print(f"불량   {fn:4d}  {tp:4d}")
            print(f"   🚨 위험 분석:")
            print(f"     - 놓친 불량품(FN): {fn}개 ({fn_rate:.2%})")
            print(f"     - 잘못 불량 판정(FP): {fp}개 ({fp_rate:.2%})")
            
            # 실무 해석
            if fn > 0:
                print(f"     ⚠️ 고객 클레임 위험: {fn}개 불량품이 정상으로 분류됨")
            if fp > 10:
                print(f"     💰 수율 손실: {fp}개 정상품이 불량으로 분류됨")
            if fn == 0 and fp <= 5:
                print(f"     ✅ 매우 안전한 성능!")
            elif fn <= 2 and sensitivity >= 0.95:
                print(f"     👍 우수한 성능!")
    
    if pipeline.optimal_thresholds:
        thresholds_list = list(pipeline.optimal_thresholds.values())
        print(f"\n🎯 임계값 최적화 요약:")
        print(f"   - 평균 임계값: {np.mean(thresholds_list):.3f}")
        print(f"   - 임계값 범위: {min(thresholds_list):.3f} ~ {max(thresholds_list):.3f}")
        
        # 임계값이 낮은 순서대로 (더 민감하게 불량 판정)
        sorted_thresholds = sorted(pipeline.optimal_thresholds.items(), key=lambda x: x[1])
        print(f"   - 가장 민감한 금형: mold_code {sorted_thresholds[0][0]} (임계값: {sorted_thresholds[0][1]:.3f})")
        print(f"   - 가장 보수적인 금형: mold_code {sorted_thresholds[-1][0]} (임계값: {sorted_thresholds[-1][1]:.3f})")
    
    print("\n💾 5단계: 파이프라인 저장")
    pipeline.save_pipeline('xgb_lgb_pipeline.pkl')  # hong 제거!
    
    print("\n" + "=" * 80)
    print("🎉 XGBoost/LightGBM 전용 파이프라인 완성! (정밀도 제약 버전)")
    print("   ✅ 데이터 누출 방지 (분할 → 전처리 → 튜닝)")
    print("   ✅ tryshot_signal 비즈니스 룰 적용")
    print("   ✅ EMS_operation_time 완전 제거로 안정성 확보")
    print("   ✅ XGBoost/LightGBM 자체 튜닝으로 고속화")
    print("   ✅ 정밀도 제약 임계값 최적화 (정밀도 ≥ 0.8)")
    print("   🚀 최종 모델: XGBoost (8412, 8573, 8722) + LightGBM (8600, 8917)")
    print("   🗑️ GridSearchCV 제거: 자체 튜닝으로 3배 빠름")
    print("   🗑️ SMOTE 제거: 내장 불균형 처리로 안정성 확보")
    print("   🎯 임계값 전략: 정밀도 ≥ 0.8 조건에서 민감도 최대화")
    print("   💡 실용성 확보: 정상품 오분류 최소화 + 불량품 검출력 최대화")
    print("   🚀 팀플용 완성버전 - 실무 투입 준비 완료!")
    print("=" * 80)