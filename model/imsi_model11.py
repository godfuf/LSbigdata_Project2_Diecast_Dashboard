"""
과적합 방지 강화 파이프라인 - "개나소나 불량" 예측 방지
홍님을 위한 최소 변경 개선안
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("         과적합 방지 강화 파이프라인")
print("       '개나소나 불량' 예측 방지 + 일반화 성능 향상")
print("       임계값 0.5~0.8 + 정밀도 하한선 70%")
print("=" * 80)

class ImprovedQualityPipeline:
    """과적합 방지 강화 품질 예측 파이프라인"""
    
    def __init__(self, min_threshold=0.5, max_threshold=0.8, min_precision=0.7):
        self.models = {}
        self.best_params = {}
        self.optimal_thresholds = {}
        self.feature_names = None
        self.min_threshold = min_threshold      # 0.5로 상향 (기존 0.4)
        self.max_threshold = max_threshold      # 0.8로 상향 (기존 0.7)
        self.min_precision = min_precision      # 정밀도 하한선 추가
        
        # 각 mold_code별 최적 모델 설정 (기존과 동일)
        self.best_model_config = {
            8412: 'XGBoost_Balanced',
            8573: 'XGBoost_Balanced', 
            8600: 'LightGBM_Balanced',
            8722: 'XGBoost_Balanced',
            8917: 'LightGBM_Balanced'
        }
        
        print("✅ 과적합 방지 파이프라인 초기화 완료")
        print(f"   🎯 개선된 임계값 전략: {min_threshold}~{max_threshold} 범위")
        print(f"   🛡️ 정밀도 하한선: {min_precision*100}% (과도한 불량 예측 방지)")
        print(f"   📊 모델 구성: XGBoost 3개, LightGBM 2개")
    
    def preprocess_data(self, df_raw, is_training=True):
        """데이터 전처리 (기존과 동일)"""
        
        print(f"\n🔧 전처리 시작 (훈련 모드: {is_training})")
        
        df = df_raw.copy()
        
        # tryshot_signal 처리 (기존과 동일)
        tryshot_exists = df['tryshot_signal'].notna()
        tryshot_count = tryshot_exists.sum()
        
        if tryshot_count > 0:
            print(f"🚨 tryshot_signal 값이 있는 데이터 {tryshot_count}개 발견")
            tryshot_data = df[tryshot_exists].copy()
            tryshot_predictions = pd.Series(1, index=tryshot_data.index, name='prediction')
            df = df[~tryshot_exists].copy()
        else:
            tryshot_data = None
            tryshot_predictions = None
        
        # 기본 전처리 (기존과 동일)
        exclude_columns = ['id', 'line', 'name', 'mold_name', 'time', 'date', 
                          'emergency_stop', 'molten_volume', 'registration_time', 
                          'heating_furnace', 'count', 'tryshot_signal', 'EMS_operation_time']
        
        df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        print(f"   🚫 EMS_operation_time 제거로 안정성 확보")
        
        # working 결측치 제거 및 레이블 인코딩 (기존과 동일)
        if 'working' in df.columns and df['working'].isnull().sum() > 0:
            df = df.dropna(subset=['working'])
        
        if is_training:
            self.le_working = LabelEncoder()
            if 'working' in df.columns:
                df['working'] = self.le_working.fit_transform(df['working'])
                print(f"   ✅ working 레이블 인코딩 완료")
        else:
            if 'working' in df.columns and hasattr(self, 'le_working'):
                try:
                    df['working'] = self.le_working.transform(df['working'])
                except ValueError as e:
                    unknown_mask = ~df['working'].isin(self.le_working.classes_)
                    if unknown_mask.any():
                        first_class = self.le_working.classes_[0]
                        df.loc[unknown_mask, 'working'] = first_class
                    df['working'] = self.le_working.transform(df['working'])
        
        # KNN 보간 (기존과 동일)
        target_impute_cols = ['molten_temp', 'lower_mold_temp3', 'upper_mold_temp3']
        actual_missing = [col for col in target_impute_cols if col in df.columns and df[col].isnull().sum() > 0]
        
        if actual_missing:
            reference_cols = ['working', 'molten_temp', 'facility_operation_cycleTime', 
                             'production_cycletime', 'mold_code',
                             'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
                             'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                             'sleeve_temperature', 'Coolant_temperature']
            
            available_reference_cols = [col for col in reference_cols if col in df.columns]
            
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
            
            for col in actual_missing:
                df[col] = df_imputed[col]
        
        # production_cycletime 0값 처리 (기존과 동일)
        if 'production_cycletime' in df.columns:
            zero_count = (df['production_cycletime'] == 0).sum()
            if zero_count > 0:
                if is_training:
                    self.production_mean = df[(df['production_cycletime'] > 0) & 
                                            (df['production_cycletime'] <= 115)]['production_cycletime'].mean()
                df.loc[df['production_cycletime'] == 0, 'production_cycletime'] = self.production_mean
        
        # 나머지 결측치 제거
        before_rows = len(df)
        df = df.dropna()
        removed_rows = before_rows - len(df)
        if removed_rows > 0:
            print(f"   - 나머지 결측치 제거: {removed_rows}행 제거")
        
        print(f"✅ 전처리 완료: {len(df)}행")
        
        return df, tryshot_data, tryshot_predictions
    
    def optimize_xgboost_improved(self, X_train, y_train, X_val, y_val):
        """개선된 XGBoost 튜닝 (정규화 + 엄격한 Early Stopping)"""
        
        print(f"     🚀 개선된 XGBoost 튜닝 시작...")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 개선: 정규화 파라미터 추가
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': 3,
            'random_state': 42,
            'verbosity': 0,
            # 새로 추가: 과적합 방지 정규화
            'reg_alpha': 0.1,      # L1 정규화 (불필요한 특성 제거)
            'reg_lambda': 1.0,     # L2 정규화 (가중치 크기 제한)
            'max_delta_step': 1    # 극단적 예측 방지
        }
        
        param_candidates = [
            {**base_params, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8},
            {**base_params, 'max_depth': 9, 'learning_rate': 0.1, 'subsample': 0.8},
            {**base_params, 'max_depth': 6, 'learning_rate': 0.2, 'subsample': 0.9},
            {**base_params, 'max_depth': 9, 'learning_rate': 0.05, 'subsample': 0.8}
        ]
        
        best_score = 0
        best_params = None
        best_model = None
        
        for i, params in enumerate(param_candidates):
            try:
                # 개선: Early stopping 더 엄격하게 (20 -> 10)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=200,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=10,  # 기존 20에서 10으로 변경
                    verbose_eval=False
                )
                
                score = model.best_score
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                print(f"       파라미터 {i+1} 실패: {str(e)[:30]}...")
                continue
        
        if best_model is None:
            print(f"       ⚠️ 모든 파라미터 실패, 기본 모델 사용")
            best_model = xgb.train(base_params, dtrain, num_boost_round=100)
            best_params = base_params
        
        print(f"       ✅ 최고 점수: {best_score:.4f} (정규화 적용)")
        
        return best_model, best_params
    
    def optimize_lightgbm_improved(self, X_train, y_train, X_val, y_val):
        """개선된 LightGBM 튜닝 (정규화 + 엄격한 Early Stopping)"""
        
        print(f"     🚀 개선된 LightGBM 튜닝 시작...")
        
        # 개선: 정규화 파라미터 추가
        base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': True,
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True,
            # 새로 추가: 과적합 방지 정규화
            'reg_alpha': 0.1,      # L1 정규화
            'reg_lambda': 1.0,     # L2 정규화
            'min_gain_to_split': 0.1  # 분할 최소 이득
        }
        
        param_candidates = [
            {**base_params, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 63},
            {**base_params, 'max_depth': 9, 'learning_rate': 0.1, 'num_leaves': 127},
            {**base_params, 'max_depth': 6, 'learning_rate': 0.2, 'num_leaves': 31},
            {**base_params, 'max_depth': -1, 'learning_rate': 0.05, 'num_leaves': 63}
        ]
        
        best_score = 0
        best_params = None
        best_model = None
        
        for i, params in enumerate(param_candidates):
            try:
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # 개선: Early stopping 더 엄격하게 (20 -> 10)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]  # 20 -> 10
                )
                
                score = model.best_score['valid_0']['auc']
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                print(f"       파라미터 {i+1} 실패: {str(e)[:30]}...")
                continue
        
        if best_model is None:
            print(f"       ⚠️ 모든 파라미터 실패, 기본 모델 사용")
            train_data = lgb.Dataset(X_train, label=y_train)
            best_model = lgb.train(base_params, train_data, num_boost_round=100)
            best_params = base_params
        
        print(f"       ✅ 최고 점수: {best_score:.4f} (정규화 적용)")
        
        return best_model, best_params
    
    def optimize_threshold_improved(self, model, X_val, y_val, model_type, mold_code):
        """개선된 임계값 최적화 (민감도*정밀도 균형 + 정밀도 하한선)"""
        
        # 확률 예측
        if model_type == 'XGBoost_Balanced':
            dval = xgb.DMatrix(X_val)
            y_proba = model.predict(dval)
        else:  # LightGBM
            y_proba = model.predict(X_val)
        
        print(f"     🎯 개선된 임계값 범위: {self.min_threshold}~{self.max_threshold}")
        print(f"     🛡️ 정밀도 하한선: {self.min_precision*100}%")
        
        # 개선: 더 보수적인 임계값 범위 (0.5~0.8)
        thresholds = np.arange(self.min_threshold, self.max_threshold + 0.05, 0.05)
        
        valid_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred_thresh)) < 2:
                continue
                
            sensitivity = recall_score(y_val, y_pred_thresh, zero_division=0)
            precision = precision_score(y_val, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
            
            # 개선: 정밀도 하한선 체크 (과도한 불량 예측 방지)
            if precision < self.min_precision:
                continue
            
            # 개선: 민감도 * 정밀도 균형점 계산
            balance_score = sensitivity * precision
            
            result = {
                'threshold': threshold,
                'sensitivity': sensitivity,
                'precision': precision,
                'f1': f1,
                'balance_score': balance_score,  # 새로 추가
                'fp_count': ((y_pred_thresh == 1) & (y_val == 0)).sum(),
                'fn_count': ((y_pred_thresh == 0) & (y_val == 1)).sum()
            }
            
            valid_results.append(result)
        
        if not valid_results:
            print(f"       🚨 조건을 만족하는 임계값 없음 → 기본값 0.6 사용")
            return 0.6, 0.0
        
        # 개선: 민감도*정밀도 균형점으로 최적화 (기존: 민감도만)
        best_result = max(valid_results, key=lambda x: x['balance_score'])
        
        print(f"       ✅ 균형점에서 최적화: {len(valid_results)}개 후보 중 선택")
        print(f"       📊 선택된 성능: 민감도 {best_result['sensitivity']:.4f}, 정밀도 {best_result['precision']:.4f}")
        print(f"       🎯 균형 점수: {best_result['balance_score']:.4f}")
        print(f"       📈 실무 지표: FP(잘못 불량) {best_result['fp_count']}개, FN(놓친 불량) {best_result['fn_count']}개")
        
        # 상위 3개 후보 출력
        top_3 = sorted(valid_results, key=lambda x: x['balance_score'], reverse=True)[:3]
        print(f"       🏆 상위 3개 후보 (균형점 기준):")
        for i, result in enumerate(top_3):
            print(f"         {i+1}. 임계값 {result['threshold']:.2f}: 균형점 {result['balance_score']:.3f} (민감도 {result['sensitivity']:.3f}, 정밀도 {result['precision']:.3f})")
        
        return best_result['threshold'], best_result['balance_score']
    
    def train_mold_specific_models(self, df_train):
        """각 mold_code별 개선된 모델 훈련"""
        
        print(f"\n🤖 각 mold_code별 개선된 모델 훈련 시작")
        
        X = df_train.drop(['passorfail', 'mold_code'], axis=1)
        y = df_train['passorfail']
        self.feature_names = X.columns.tolist()
        
        print(f"   📋 사용 변수: {len(self.feature_names)}개 (EMS_operation_time 제외)")
        
        for mold_code in self.best_model_config.keys():
            if mold_code not in df_train['mold_code'].values:
                print(f"⚠️ mold_code {mold_code} 데이터 없음")
                continue
            
            print(f"\n🔍 === mold_code {mold_code} 개선된 모델 훈련 ===")
            
            mold_mask = df_train['mold_code'] == mold_code
            X_mold = X[mold_mask]
            y_mold = y[mold_mask]
            
            print(f"   - 데이터: {len(X_mold)}개, 불량률: {y_mold.mean():.4f}")
            
            X_train_mold, X_val_mold, y_train_mold, y_val_mold = train_test_split(
                X_mold, y_mold, test_size=0.2, random_state=42, stratify=y_mold
            )
            
            model_type = self.best_model_config[mold_code]
            print(f"   - 최적 모델: {model_type} (정규화 + 엄격한 Early Stopping)")
            
            # 개선된 모델별 튜닝
            if model_type == 'XGBoost_Balanced':
                best_model, best_params = self.optimize_xgboost_improved(X_train_mold, y_train_mold, X_val_mold, y_val_mold)
            else:  # LightGBM_Balanced
                best_model, best_params = self.optimize_lightgbm_improved(X_train_mold, y_train_mold, X_val_mold, y_val_mold)
            
            # 개선된 임계값 최적화
            print(f"   - 개선된 임계값 최적화 중...")
            best_threshold, balance_score = self.optimize_threshold_improved(
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
            precision = precision_score(y_val_mold, y_pred_optimal, zero_division=0)
            f1 = f1_score(y_val_mold, y_pred_optimal, zero_division=0)
            
            print(f"   ✅ 개선된 튜닝 완료:")
            print(f"     * 최적 임계값: {best_threshold:.3f} (보수적 범위 {self.min_threshold}~{self.max_threshold})")
            print(f"     * 검증 성능: 민감도 {sensitivity:.4f}, 정밀도 {precision:.4f}, F1 {f1:.4f}")
            print(f"     * 균형 점수: {balance_score:.4f} (민감도*정밀도)")
            print(f"     * 과적합 방지: 정규화 + 엄격한 Early Stopping + 정밀도 하한선")
            
            # 전체 데이터로 재훈련
            if model_type == 'XGBoost_Balanced':
                dtrain_full = xgb.DMatrix(X_mold, label=y_mold)
                final_model = xgb.train(best_params, dtrain_full, num_boost_round=best_model.best_iteration)
            else:
                train_data_full = lgb.Dataset(X_mold, label=y_mold)
                final_model = lgb.train(best_params, train_data_full, num_boost_round=best_model.best_iteration)
            
            self.models[mold_code] = final_model
            self.best_params[mold_code] = best_params
            self.optimal_thresholds[mold_code] = best_threshold
        
        print(f"\n✅ 모든 mold_code 개선된 모델 훈련 완료: {len(self.models)}개")
        threshold_values = list(self.optimal_thresholds.values())
        print(f"   - 평균 임계값: {np.mean(threshold_values):.3f}")
        print(f"   - 임계값 범위: {min(threshold_values):.3f} ~ {max(threshold_values):.3f}")
        print(f"   🎯 과적합 방지 전략: 정규화 + 균형점 최적화 + 정밀도 {self.min_precision*100}% 하한선")
    
    def predict(self, df_test):
        """통합 예측 함수 (개선된 임계값 적용)"""
        
        print(f"\n🔮 개선된 예측 시작")
        
        df_processed, tryshot_data, tryshot_predictions = self.preprocess_data(df_test, is_training=False)
        
        if len(df_processed) == 0:
            print("⚠️ 예측할 데이터 없음 (모두 tryshot)")
            return tryshot_predictions
        
        predictions = pd.Series(index=df_processed.index, dtype=int, name='prediction')
        
        for mold_code in df_processed['mold_code'].unique():
            if mold_code not in self.models:
                print(f"⚠️ mold_code {mold_code} 모델 없음 - 기본값(0) 예측")
                mask = df_processed['mold_code'] == mold_code
                predictions[mask] = 0
                continue
            
            mask = df_processed['mold_code'] == mold_code
            X_mold = df_processed[mask].drop(['passorfail', 'mold_code'], axis=1)
            
            model = self.models[mold_code]
            model_type = self.best_model_config[mold_code]
            threshold = self.optimal_thresholds.get(mold_code, 0.6)
            
            if model_type == 'XGBoost_Balanced':
                dtest = xgb.DMatrix(X_mold)
                y_proba = model.predict(dtest)
            else:  # LightGBM
                y_proba = model.predict(X_mold)
            
            pred = (y_proba >= threshold).astype(int)
            predictions[mask] = pred
            
            print(f"   - mold_code {mold_code}: {mask.sum()}개 예측 완료 (개선된 임계값: {threshold:.3f})")
        
        if tryshot_predictions is not None:
            all_predictions = pd.concat([predictions, tryshot_predictions])
            print(f"   - tryshot 강제 불량: {len(tryshot_predictions)}개")
        else:
            all_predictions = predictions
        
        print(f"✅ 전체 예측 완료: {len(all_predictions)}개")
        print(f"   🛡️ 과적합 방지: 정규화 + 보수적 임계값 + 정밀도 하한선")
        
        return all_predictions
    
    def save_pipeline(self, filepath):
        """파이프라인 저장"""
        pipeline_data = {
            'models': self.models,
            'best_params': self.best_params,
            'optimal_thresholds': self.optimal_thresholds,
            'feature_names': self.feature_names,
            'best_model_config': self.best_model_config,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'min_precision': self.min_precision,
            'le_working': getattr(self, 'le_working', None),
            'knn_imputer': getattr(self, 'knn_imputer', None),
            'production_mean': getattr(self, 'production_mean', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"✅ 개선된 파이프라인 저장: {filepath}")
        print(f"   - 모델: {len(self.models)}개")
        print(f"   - 과적합 방지 임계값: {len(self.optimal_thresholds)}개")

# 실행 부분
if __name__ == "__main__":
    print("\n📊 1단계: 데이터 로드")
    
    try:
        df_raw = pd.read_csv('../data/train.csv')
        print(f"✅ 데이터 로드 성공: {df_raw.shape[0]}행, {df_raw.shape[1]}개 변수")
    except:
        print("❌ train.csv 파일을 찾을 수 없습니다.")
        exit()
    
    print("\n🔄 2단계: Train/Test 분할")
    
    df_clean = df_raw[df_raw['tryshot_signal'].isna()].copy()
    print(f"✅ tryshot_signal NaN 필터링: {len(df_clean)}행")
    
    train_df, test_df = train_test_split(
        df_clean, test_size=0.2, random_state=42, 
        stratify=df_clean['passorfail']
    )
    print(f"   - 훈련 데이터: {len(train_df)}행")
    print(f"   - 테스트 데이터: {len(test_df)}행")
    
    print("\n🏭 3단계: 개선된 파이프라인 훈련")
    
    # 개선된 파이프라인 생성 (0.5~0.8, 정밀도 70% 하한선)
    pipeline = ImprovedQualityPipeline(min_threshold=0.5, max_threshold=0.8, min_precision=0.7)
    
    train_processed, _, _ = pipeline.preprocess_data(train_df, is_training=True)
    pipeline.train_mold_specific_models(train_processed)
    
    print("\n🧪 4단계: 테스트 데이터 평가")
    
    test_predictions = pipeline.predict(test_df)
    
    test_processed, _, _ = pipeline.preprocess_data(test_df, is_training=False)
    y_true = test_processed['passorfail']
    y_pred = test_predictions[test_processed.index]
    
    overall_sensitivity = recall_score(y_true, y_pred, zero_division=0)
    overall_precision = precision_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_balance = overall_sensitivity * overall_precision
    
    print(f"\n🎯 전체 성능 (개선된 버전):")
    print(f"   - 민감도: {overall_sensitivity:.4f}")
    print(f"   - 정밀도: {overall_precision:.4f}")
    print(f"   - F1 점수: {overall_f1:.4f}")
    print(f"   - 정확도: {overall_accuracy:.4f}")
    print(f"   - 균형 점수: {overall_balance:.4f} (민감도*정밀도)")
    
    print(f"\n📊 mold_code별 개선된 테스트 성능:")
    
    for mold_code in test_processed['mold_code'].unique():
        if mold_code in pipeline.models:
            mask = test_processed['mold_code'] == mold_code
            y_true_mold = y_true[mask]
            y_pred_mold = y_pred[mask]
            
            sensitivity = recall_score(y_true_mold, y_pred_mold, zero_division=0)
            precision = precision_score(y_true_mold, y_pred_mold, zero_division=0)
            f1 = f1_score(y_true_mold, y_pred_mold, zero_division=0)
            balance = sensitivity * precision
            threshold = pipeline.optimal_thresholds.get(mold_code, 0.6)
            
            cm = confusion_matrix(y_true_mold, y_pred_mold)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            print(f"\n🔍 mold_code {mold_code}:")
            print(f"   📊 민감도: {sensitivity:.4f}, 정밀도: {precision:.4f}, F1: {f1:.4f}")
            print(f"   🎯 개선된 임계값: {threshold:.3f} (0.5~0.8 범위)")
            print(f"   ⚖️ 균형 점수: {balance:.4f} (민감도*정밀도)")
            print(f"   📈 실무 지표: 불량 예측 {tp+fp}개, 놓친 불량 {fn}개, 잘못 불량 판정 {fp}개")
            
            # 개선 효과 평가
            if precision >= 0.7 and sensitivity >= 0.8:
                print(f"   ✅ 우수한 균형! (높은 민감도 + 정밀도 70% 이상)")
            elif precision < 0.7:
                print(f"   ⚠️ 정밀도 부족: {precision:.3f} < 0.7 (과도한 불량 예측)")
            elif fn > 10:
                print(f"   ⚠️ 불량품 놓침 주의 (FN: {fn}개)")
    
    if pipeline.optimal_thresholds:
        thresholds_list = list(pipeline.optimal_thresholds.values())
        print(f"\n🎯 개선된 임계값 요약:")
        print(f"   - 평균 임계값: {np.mean(thresholds_list):.3f}")
        print(f"   - 임계값 범위: {min(thresholds_list):.3f} ~ {max(thresholds_list):.3f}")
        print(f"   - 임계값 다양성: {'✅' if len(set([round(t, 1) for t in thresholds_list])) > 1 else '❌'}")
        print(f"   - 0.5~0.8 범위 준수: {'✅' if all(0.5 <= t <= 0.8 for t in thresholds_list) else '❌'}")
    
    print("\n💾 5단계: 개선된 파이프라인 저장")
    pipeline.save_pipeline('xgb_lgb_pipeline_improved.pkl')
    
    print("\n" + "=" * 80)
    print("🎉 과적합 방지 강화 파이프라인 완성!")
    print("   ✅ 임계값 0.5~0.8 범위로 '개나소나 불량' 방지")
    print("   ✅ 정밀도 70% 하한선으로 과도한 불량 예측 차단")
    print("   ✅ 민감도*정밀도 균형점으로 최적화")
    print("   ✅ 정규화 파라미터로 과적합 방지")
    print("   ✅ 엄격한 Early Stopping (20→10)")
    print("   🎯 실무 적용: 새로운 데이터에서도 안정적 성능!")
    print("   💡 핵심 개선: 일반화 성능 최우선")
    print("=" * 80)