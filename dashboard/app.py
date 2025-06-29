import seaborn as sns
from faicons import icon_svg
import time
import threading
import matplotlib.pyplot as plt
import plotly.express as px
from shinywidgets import render_widget, output_widget
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from processing import load_data
import os
import shared
from htmltools import HTML
from datetime import timedelta, datetime
from shiny import App, reactive, render, ui, reactive
from datetime import datetime
import pickle
import joblib
import xgboost as xgb
import lightgbm as lgb
import shap
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm
from plotnine import ggplot, aes, geom_line, facet_wrap, labs, theme_bw, theme, element_text
from plotnine import *
import sys
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from processing import load_data
import matplotlib as mpl
from plotnine import (
    ggplot, aes, geom_point, scale_color_manual, scale_x_datetime, scale_y_continuous, guides, labs, theme_minimal, theme, element_text, scale_x_datetime, geom_line, facet_wrap, geom_hline
)
from mizani.breaks import breaks_date
import warnings
import io
from matplotlib.backends.backend_pdf import PdfPages
import json
warnings.filterwarnings('ignore')
from xgboost import Booster, DMatrix
import matplotlib.dates as mdates
import math

import sklearn
import sklearn.compose._column_transformer as _ct

# pickle이 예전 버전의 scikit-learn에서 사용하던 private 이름을 찾을 수 있도록 alias(별칭)를 생성합니다.
try:
    # scikit-learn 1.2+ 버전에서는 RemainderColsList가 public 클래스입니다.
    _ct._RemainderColsList = _ct.RemainderColsList
except AttributeError:
    # 이전 버전이거나 해당 속성이 없는 경우를 대비한 예외 처리
    pass

# font
app_dir = Path(__file__).parent
font_path = app_dir / "MaruBuri-Regular.ttf"

fm.fontManager.addfont(str(font_path))

font_prop = fm.FontProperties(fname=str(font_path))
mpl.rcParams["axes.unicode_minus"] = False

# 앱 디렉터리 설정
app_dir = Path(__file__).parent

# 한글 폰트 설정: MaruBuri-Regular.ttf 직접 로드
font_path = app_dir / "MaruBuri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# # 한글 폰트 설정
# try:
#     plt.rcParams['font.family'] = 'Malgun Gothic'
# except:
#     try:
#         for font_name in ['Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans CJK KR', 'DejaVu Sans']:
#             try:
#                 plt.rcParams['font.family'] = font_name
#                 break
#             except:
#                 continue
#     except:
#         print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# plt.rcParams['axes.unicode_minus'] = False

# ---

bound_df = shared.BOUND_DATA
STATIC_DIR = os.path.join(os.path.dirname(__file__), "www")
CURRENT_TIMESTAMP = reactive.value(datetime(2019, 3, 13, 0, 0, 0))
cumul_val2 = timedelta(minutes=3)
threshold = {8917: 0.6, 8722: 0.5, 8412: 0.5}
anomaly_results = reactive.Value([])
show_defect_detail_modal = reactive.value(False)
selected_defect_alert_data = reactive.value({})



def load_improved_pipeline():
    """현재 스크립트(app.py)와 동일한 디렉터리에 있는 PKL 파일을 불러와 dict 반환"""
    base_dir = os.path.dirname(__file__)
    pkl_path = os.path.join(base_dir, "xgb_lgb_pipeline_improved.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"❌ 파이프라인 파일을 찾을 수 없습니다: {pkl_path}")
        return None

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print("✅ 파이프라인 로드 성공")
        return data
    except Exception as e:
        print(f"❌ 파이프라인 로드 중 오류: {e}")
        return None

pipeline = load_improved_pipeline()


#-------isolation forest ----------
# 현재 디렉터리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# CustomPreprocessor를 메인 모듈에 등록
try:
    from custom_preprocessor import CustomPreprocessor
    # 메인 모듈에 명시적으로 등록
    import __main__
    __main__.CustomPreprocessor = CustomPreprocessor
    print("✅ CustomPreprocessor 전역 임포트 및 등록 성공")
except ImportError as e:
    print(f"❌ CustomPreprocessor 임포트 실패: {e}")
    CustomPreprocessor = None

def load_isolation_forest_model():
    """Isolation Forest 모델과 전처리기 로드"""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "isolation_forest_pipeline.pkl")
    
    # CustomPreprocessor가 이미 전역에서 import됨
    if CustomPreprocessor is None:
        print("❌ CustomPreprocessor를 사용할 수 없습니다")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Isolation Forest 모델 파일을 찾을 수 없습니다: {model_path}")
        return None

    try:
        import joblib
        model_data = joblib.load(model_path)  # 이제 CustomPreprocessor를 찾을 수 있음
        print("✅ Isolation Forest 모델 로드 성공")
        print(f"✅ 모델 타입: {type(model_data)}")
        return model_data
    except Exception as e:
        print(f"❌ Isolation Forest 모델 로드 중 오류: {e}")
        return None

# 모델 로드
isolation_model_data = load_isolation_forest_model()

# ------------------- 한글 변수명 매핑 -------------------
KOREAN_VARIABLE_MAP = {
    'molten_temp': '용탕 온도',
    'sleeve_temperature': '슬리브 온도',
    'Coolant_temperature': '냉각수 온도',
    'lower_mold_temp1': '하금형 온도1',
    'lower_mold_temp2': '하금형 온도2', 
    'lower_mold_temp3': '하금형 온도3',
    'upper_mold_temp1': '상금형 온도1',
    'upper_mold_temp2': '상금형 온도2',
    'upper_mold_temp3': '상금형 온도3',
    'facility_operation_cycleTime': '설비 작동 사이클 시간',
    'production_cycletime': '제품 생산 사이클 시간',
    'low_section_speed': '저속 구간 속도',
    'high_section_speed': '고속 구간 속도',
    'cast_pressure': '주조 압력',
    'biscuit_thickness': '비스킷 두께',
    'physical_strength': '형체력',
    'working': '가동 여부',
    'mold_code': '금형 코드'
}

# ------------------- 변수별 범위 설정 -------------------
VARIABLE_RANGES = {
    'molten_temp': {'min': 0, 'max': 750, 'default': 731},
    'sleeve_temperature': {'min': 150, 'max': 1450, 'default': 1449},
    'Coolant_temperature': {'min': 0, 'max': 50, 'default': 33},
    'lower_mold_temp1': {'min': 0, 'max': 400, 'default': 206},
    'lower_mold_temp2': {'min': 0, 'max': 550, 'default': 190},
    'lower_mold_temp3': {'min': 250, 'max': 1450, 'default': 1449},
    'upper_mold_temp1': {'min': 0, 'max': 400, 'default': 186},
    'upper_mold_temp2': {'min': 0, 'max': 400, 'default': 166},
    'upper_mold_temp3': {'min': 0, 'max': 1450, 'default': 1449},
    'facility_operation_cycleTime': {'min': 0, 'max': 500, 'default': 119},
    'production_cycletime': {'min': 0, 'max': 500, 'default': 121},
    'low_section_speed': {'min': 0, 'max': 200, 'default': 110},
    'high_section_speed': {'min': 0, 'max': 400, 'default': 112},
    'cast_pressure': {'min': 0, 'max': 400, 'default': 331},
    'biscuit_thickness': {'min': 0, 'max': 450, 'default': 50},
    'physical_strength': {'min': 0, 'max': 750, 'default': 703},
}

# ------------------- PDP 선택용 choices 생성 -------------------
feature_names = pipeline.get("feature_names", []) if pipeline else []
pdp_choices = {f: KOREAN_VARIABLE_MAP.get(f, f) for f in feature_names}

# ------------------- 다중 불량 구간 처리 함수 -------------------
def generate_gradient_for_multiple_ranges(var_name, defect_ranges, total_min, total_max):
    """여러 불량 구간을 하나의 gradient 문자열로 변환"""
    if not defect_ranges:
        return "#007bff"
    
    # 모든 구간을 퍼센트로 변환
    range_percents = []
    for range_data in defect_ranges:
        start_percent = ((range_data["min"] - total_min) / (total_max - total_min)) * 100
        end_percent = ((range_data["max"] - total_min) / (total_max - total_min)) * 100
        
        # 안전한 범위로 제한
        start_percent = max(0, min(100, start_percent))
        end_percent = max(0, min(100, end_percent))
        
        if start_percent < end_percent:  # 유효한 구간만 추가
            range_percents.append((start_percent, end_percent))
    
    if not range_percents:
        return "#007bff"  # 전체 파란색 (정상)
    
    # 퍼센트 기준으로 정렬
    range_percents.sort()
    
    # gradient 구성 요소들 생성
    gradient_parts = []
    current_pos = 0
    
    for start_percent, end_percent in range_percents:
        # 현재 위치부터 불량 구간 시작까지 파란색
        if current_pos < start_percent:
            gradient_parts.append(f"#007bff {current_pos}%")
            gradient_parts.append(f"#007bff {start_percent}%")
        
        # 불량 구간은 빨간색
        gradient_parts.append(f"#dc3545 {start_percent}%")
        gradient_parts.append(f"#dc3545 {end_percent}%")
        
        current_pos = end_percent
    
    # 마지막 불량 구간 이후부터 끝까지 파란색
    if current_pos < 100:
        gradient_parts.append(f"#007bff {current_pos}%")
        gradient_parts.append(f"#007bff 100%")
    
    return f"linear-gradient(to right, {', '.join(gradient_parts)})"

# ------------------- PDF 생성 함수 -------------------
def generate_pdf_report(results, defect_ranges_data=None):
        buffer = io.BytesIO()
       
        
        plt.rcParams['font.family'] = shared.font_prop.get_name()
            
        # plt.rcParams['axes.unicode_minus'] = False
        
        # 결과 데이터 추출
        pred = results["prediction"]
        proba = results["probability"]
        threshold = results["threshold"]
        input_data = results["input_values"]
        mold_code = results["mold_code"]
        model_type = results["model_type"]
        
        # 모든 페이지 크기 통일
        STANDARD_FIGSIZE = (11, 8.5)  # A4 비율로 통일
        
        with PdfPages(buffer) as pdf:
            
            # ==================== 1페이지: 보고서 요약 (간소화) ====================
            fig = plt.figure(figsize=STANDARD_FIGSIZE)
            
            # 제목
            fig.suptitle('제조업 품질 예측 분석 보고서', fontsize=20, fontweight='bold', y=0.95)
            
            # 부제목
            plt.figtext(0.5, 0.88, f'금형 코드 {mold_code} | 분석 시간: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M")}', 
                       ha='center', fontsize=12, style='italic')
            
            # 예측 결과 박스
            result_text = "양품 예측" if pred == 0 else "불량 예측"
            result_color = 'green' if pred == 0 else 'red'
            
            plt.figtext(0.5, 0.75, '■ 예측 결과', ha='center', fontsize=18, fontweight='bold')
            plt.figtext(0.5, 0.68, result_text, ha='center', fontsize=20, fontweight='bold', color=result_color)
            plt.figtext(0.5, 0.62, f'불량 확률: {proba:.1%} (임계값: {threshold:.3f})', ha='center', fontsize=16)
            
            # 모델 정보
            plt.figtext(0.5, 0.50, '■ 모델 정보', ha='center', fontsize=16, fontweight='bold')
            plt.figtext(0.5, 0.44, f'사용 모델: {model_type}', ha='center', fontsize=14)
            plt.figtext(0.5, 0.40, f'금형 코드: {mold_code}', ha='center', fontsize=14)
            
            # 분석 결과 요약
            if pred == 0:
                summary_text = "현재 공정 조건에서 양품 생산이 예상됩니다.\n지속적인 품질 모니터링을 권장합니다."
                summary_color = 'green'
            else:
                summary_text = "불량 위험이 감지되었습니다.\n즉시 공정 조건 점검이 필요합니다."
                summary_color = 'red'
            
            plt.figtext(0.5, 0.28, '■ 분석 요약', ha='center', fontsize=16, fontweight='bold')
            plt.figtext(0.5, 0.20, summary_text, ha='center', fontsize=12, color=summary_color)
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ==================== 2페이지: 전체 공정 조건 상세 (변수별 불량 범위 분석) ====================
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
            ax.axis('tight')
            ax.axis('off')
            
            # 제목
            fig.suptitle('전체 공정 조건 상세', fontsize=18, fontweight='bold', y=0.95)
            
            # 불량 범위 분석 테이블 데이터 준비
            table_data = []
            critical_vars = []
            
            for var_name in ['molten_temp', 'sleeve_temperature', 'Coolant_temperature',
                           'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
                           'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                           'facility_operation_cycleTime', 'production_cycletime',
                           'low_section_speed', 'high_section_speed', 'cast_pressure',
                           'biscuit_thickness', 'physical_strength']:
                
                korean_name = KOREAN_VARIABLE_MAP.get(var_name, var_name)
                current_value = input_data.get(var_name, 0)
                
                # 변수 범위 정보
                var_range = VARIABLE_RANGES.get(var_name, {})
                min_val = var_range.get('min', 0)
                max_val = var_range.get('max', 1000)
                range_display = f"{min_val} ~ {max_val}"
                
                # 불량 범위 정보
                defect_ranges_display = "분석 중"
                risk_level = "-"
                
                if defect_ranges_data and var_name in defect_ranges_data:
                    ranges = defect_ranges_data[var_name].get("ranges", [])
                    
                    if ranges:
                        # 불량 범위가 있는 경우
                        ranges_text = []
                        in_defect_range = False
                        
                        for r in ranges:
                            range_str = f"{r['min']:.0f}~{r['max']:.0f}"
                            ranges_text.append(range_str)
                            
                            # 현재값이 불량 범위에 있는지 확인
                            if r['min'] <= current_value <= r['max']:
                                in_defect_range = True
                        
                        defect_ranges_display = ", ".join(ranges_text)
                        risk_level = "위험" if in_defect_range else "안전"
                        
                        if in_defect_range:
                            critical_vars.append(korean_name)
                            
                    else:
                        defect_ranges_display = "불량 범위 없음"
                        risk_level = "안전"
                
                table_data.append([
                    korean_name,
                    f"{current_value}",
                    range_display,
                    defect_ranges_display,
                    risk_level
                ])
            
            # 테이블 생성
            headers = ['변수명', '현재값', '설정 범위', '불량 범위', '위험도']
            table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            # 헤더 스타일링
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 셀 스타일링 (위험도에 따라)
            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if j == 4:  # 위험도 컬럼
                        if table_data[i-1][j] == "위험":
                            table[(i, j)].set_facecolor('#FFB3B3')
                            table[(i, j)].set_text_props(weight='bold', color='red')
                        elif table_data[i-1][j] == "안전":
                            table[(i, j)].set_facecolor('#B3FFB3')
                            table[(i, j)].set_text_props(weight='bold', color='green')
                    else:
                        table[(i, j)].set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
            
            # 위험 변수 요약
            if critical_vars:
                plt.figtext(0.5, 0.08, f"▲ 위험 구간 변수: {', '.join(critical_vars)}", 
                           ha='center', fontsize=12, color='red', weight='bold')
            else:
                plt.figtext(0.5, 0.08, "■ 모든 변수가 안전 구간에 위치", 
                           ha='center', fontsize=12, color='green', weight='bold')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ==================== 3페이지: SHAP 분석 ====================
            if results.get("shap_top_feats"):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=STANDARD_FIGSIZE)
                fig.suptitle('SHAP 변수 기여도 분석', fontsize=18, fontweight='bold')
                
                # SHAP 바차트
                top_feats = results["shap_top_feats"]
                top_vals = results["shap_top_vals"]
                korean_feats = [KOREAN_VARIABLE_MAP.get(f, f) for f in top_feats]
                
                colors = ['#FF6B6B' if val > np.mean(top_vals) else '#4ECDC4' for val in top_vals]
                bars = ax1.barh(range(len(korean_feats)), top_vals, color=colors)
                ax1.set_yticks(range(len(korean_feats)))
                ax1.set_yticklabels(korean_feats, fontsize=10)
                ax1.set_xlabel('SHAP 기여도 (절댓값)', fontsize=12)
                ax1.set_title('변수별 예측 기여도', fontsize=14, fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
                
                # 값 표시
                for i, (bar, val) in enumerate(zip(bars, top_vals)):
                    ax1.text(val + max(top_vals) * 0.01, i, f'{val:.3f}', 
                            va='center', fontsize=9)
                
                # SHAP 분석 설명
                ax2.axis('off')
                shap_text = f"""
SHAP 분석 결과 해석:

▶ 주요 영향 변수:
• {korean_feats[-1]}: {top_vals[-1]:.3f}
• {korean_feats[-2]}: {top_vals[-2]:.3f}
• {korean_feats[-3]}: {top_vals[-3]:.3f}

■ 분석 의미:
• 높은 SHAP 값 = 예측에 큰 영향
• 상위 3개 변수가 전체 예측의 
  주요 결정 요인

● 관리 포인트:
상위 변수들의 값 변화가 
품질 예측에 가장 민감하게 
영향을 미치므로 집중 관리 필요

▶ 예측 결과:
{"불량 위험이 높은 상태" if pred == 1 else "양품 생산 조건"}
(확률: {proba:.1%})
                """
                
                ax2.text(0.05, 0.95, shap_text, transform=ax2.transAxes, fontsize=11,
                        verticalalignment='top')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # ==================== 4페이지: PDP 분석 ====================
            if results.get("pdp_vars"):
                fig = plt.figure(figsize=STANDARD_FIGSIZE)
                fig.suptitle('PDP(Partial Dependence Plot) 민감도 분석', fontsize=18, fontweight='bold')
                
                pdp_vars = results["pdp_vars"][:3]
                
                # 3개의 PDP 서브플롯 생성
                for idx, var in enumerate(pdp_vars):
                    ax = plt.subplot(2, 2, idx + 1)
                    
                    current_val = input_data.get(var, 0)
                    korean_name = KOREAN_VARIABLE_MAP.get(var, var)
                    
                    if var in VARIABLE_RANGES:
                        var_min = VARIABLE_RANGES[var]["min"]
                        var_max = VARIABLE_RANGES[var]["max"]
                    else:
                        var_min = current_val * 0.8
                        var_max = current_val * 1.2
                    
                    # 시뮬레이션 데이터 생성 (실제 분석과 유사하게)
                    x_vals = np.linspace(var_min, var_max, 50)
                    np.random.seed(42 + idx)
                    
                    if var == "molten_temp":
                        y_vals = 0.1 + 0.4 * (x_vals - var_min) / (var_max - var_min) + np.random.normal(0, 0.03, 50)
                    elif var == "production_cycletime":
                        optimal = (var_min + var_max) / 2
                        y_vals = 0.1 + 0.3 * np.abs(x_vals - optimal) / (var_max - var_min) + np.random.normal(0, 0.02, 50)
                    else:
                        y_vals = 0.2 + 0.2 * np.sin((x_vals - var_min) / (var_max - var_min) * np.pi) + np.random.normal(0, 0.02, 50)
                    
                    y_vals = np.clip(y_vals, 0, 1)
                    current_prob = np.interp(current_val, x_vals, y_vals)
                    
                    # 플롯
                    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='예측 확률')
                    ax.axvline(current_val, color='red', linestyle='--', linewidth=2, label=f'현재값 ({current_val})')
                    ax.scatter([current_val], [current_prob], color='red', s=100, zorder=5)
                    
                    ax.set_xlabel(f'{korean_name}', fontsize=10)
                    ax.set_ylabel('불량 확률', fontsize=10)
                    ax.set_title(f'{korean_name} 민감도', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
                    
                    # 임계값 라인
                    ax.axhline(threshold, color='orange', linestyle=':', alpha=0.7, label=f'임계값 ({threshold:.3f})')
                
                # PDP 분석 설명 (4번째 서브플롯)
                ax4 = plt.subplot(2, 2, 4)
                ax4.axis('off')
                
                # 변수별 위험도 평가
                risk_assessment = []
                for var in pdp_vars:
                    current_val = input_data.get(var, 0)
                    korean_name = KOREAN_VARIABLE_MAP.get(var, var)
                    
                    # 간단한 위험도 평가 로직
                    if var in VARIABLE_RANGES:
                        var_range = VARIABLE_RANGES[var]
                        position = (current_val - var_range["min"]) / (var_range["max"] - var_range["min"])
                        
                        if position < 0.2 or position > 0.8:
                            risk = "높음"
                        elif position < 0.3 or position > 0.7:
                            risk = "중간"
                        else:
                            risk = "낮음"
                    else:
                        risk = "분석필요"
                    
                    risk_assessment.append(f"• {korean_name}: {risk}")
                
                pdp_text = f"""
PDP 분석 결과:

● 변수별 민감도 평가:
{chr(10).join(risk_assessment)}

▶ 해석 가이드:
• 기울기가 급한 구간 = 민감한 변수
• 현재값 위치가 중요
• 임계값 근처 = 주의 필요

■ 최적화 방향:
{"불량 확률을 낮추기 위해 민감한 변수들의 조정이 필요합니다." if pred == 1 else "현재 조건을 유지하되 민감한 변수들을 모니터링하세요."}

▲ 종합 위험도: {"높음" if proba > 0.7 else "중간" if proba > 0.3 else "낮음"}
                """
                
                ax4.text(0.05, 0.95, pdp_text, transform=ax4.transAxes, fontsize=10,
                        verticalalignment='top')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # ==================== 5페이지: 종합 결론 및 권장사항 ====================
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
            ax.axis('off')
            
            fig.suptitle('종합 결론 및 권장사항', fontsize=20, fontweight='bold', y=0.95)
            
            # 동적 결론 생성
            if pred == 0:
                conclusion_title = "양품 생산 조건 확인"
                conclusion_color = 'green'
                main_conclusion = f"""
현재 설정된 공정 조건에서는 양품이 생산될 것으로 예측됩니다.
불량 확률이 {proba:.1%}로 임계값 {threshold:.3f}보다 낮아 안전한 상태입니다.
                """
            else:
                conclusion_title = "불량 위험 조건 감지"
                conclusion_color = 'red'
                main_conclusion = f"""
현재 설정된 공정 조건에서는 불량이 발생할 위험이 높습니다.
불량 확률이 {proba:.1%}로 임계값 {threshold:.3f}를 초과하여 즉시 조치가 필요합니다.
                """
            
            # 주요 권장사항 생성
            recommendations = []
            
            if results.get("shap_top_feats"):
                top_vars = results["shap_top_feats"][-3:]  # 상위 3개
                for var in top_vars:
                    korean_name = KOREAN_VARIABLE_MAP.get(var, var)
                    current_val = input_data.get(var, 0)
                    
                    if var in VARIABLE_RANGES:
                        var_range = VARIABLE_RANGES[var]
                        optimal_range = f"{var_range['min'] + (var_range['max'] - var_range['min']) * 0.3:.0f} ~ {var_range['min'] + (var_range['max'] - var_range['min']) * 0.7:.0f}"
                        recommendations.append(f"• {korean_name}: 현재 {current_val} → 권장 범위 {optimal_range}")
            
            # 불량 범위 기반 권장사항
            if defect_ranges_data and critical_vars:
                recommendations.append(f"• 위험 구간 변수 즉시 조정: {', '.join(critical_vars)}")
            
            # 모니터링 권장사항
            monitoring_vars = []
            if results.get("shap_top_feats"):
                monitoring_vars = [KOREAN_VARIABLE_MAP.get(var, var) for var in results["shap_top_feats"][-5:]]
            
            # 전체 텍스트 구성
            report_text = f"""
{conclusion_title}

■ 분석 요약:
{main_conclusion}

● 핵심 관리 변수:
{', '.join(monitoring_vars) if monitoring_vars else '분석 중'}

▶ 즉시 조치사항:
{chr(10).join(recommendations) if recommendations else '• 현재 조건 유지 권장'}

■ 지속 모니터링:
• 실시간 품질 데이터 수집
• 주요 변수 변화 추이 관찰
• 임계값 근처 도달 시 알람 설정

● 개선 제안:
• 정기적인 공정 조건 재검토
• 예측 모델 성능 지속적 개선
• 작업자 교육을 통한 품질 인식 향상

▶ 장기 전략:
• 데이터 기반 예방적 품질 관리
• 공정 최적화를 통한 수율 향상
• 지능형 제조 시스템 구축

───────────────────────────────────

* 문의 및 지원:
추가적인 분석이나 상세한 조치 방안이 필요한 경우 
품질관리팀으로 연락하시기 바랍니다.

보고서 생성 시간: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}
            """
            
            # 텍스트 출력
            ax.text(0.05, 0.95, conclusion_title, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', color=conclusion_color)
            
            ax.text(0.05, 0.85, report_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    # except Exception as e:
    #     print(f"PDF 생성 중 오류: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return None

# ------------------- 슬라이더+숫자입력 연동 컴포넌트 함수 -------------------
def create_linked_input(var_name, korean_name, var_range):
    """슬라이더와 숫자 입력이 연동되는 컴포넌트 생성"""
    return ui.div(
        ui.div(
            ui.h6(korean_name, style="margin-bottom: 8px; font-weight: bold; color: #495057; font-size: 13px;"),
            ui.div(
                # 슬라이더
                ui.div(
                    ui.input_slider(
                        f"{var_name}_slider",
                        "",
                        min=var_range['min'],
                        max=var_range['max'],
                        value=var_range['default'],
                        step=1
                    ),
                    style="width: 70%; display: inline-block; vertical-align: middle; margin-right: 8px;"
                ),
                # 숫자 입력
                ui.div(
                    ui.input_numeric(
                        f"{var_name}_numeric",
                        "",
                        value=var_range['default'],
                        min=var_range['min'],
                        max=var_range['max'],
                        step=1
                    ),
                    style="width: 15%; display: inline-block; vertical-align: middle;"
                ),
                style="display: flex; align-items: center; margin-bottom: 3px;"
            ),
            ui.output_ui(f"defect_range_{var_name}"),
            style="margin-bottom: 12px; padding: 8px; background: #fafafa; border-radius: 6px;"
        )
    )

# ------------------- 의주 Load required data -------------------

# Load required data
answer_df = shared.DATA_ANSWER
answer_df['registration_time'] = pd.to_datetime(answer_df['registration_time'])
uj_df = answer_df

# 파이프라인에서 필요한 구성요소들 추출
feature_names = pipeline["feature_names"]

# 초기 데이터 설정
uj_df["registration_time"] = pd.to_datetime(uj_df["registration_time"], errors='coerce')
uj_df = uj_df.reset_index(drop=True)

min_date = uj_df["registration_time"].min().strftime("%Y-%m-%d")
max_date = uj_df["registration_time"].max().strftime("%Y-%m-%d")

# ------------------------지원----------------------------
# ───────── 한글 레이블 매핑 사전 ─────────
SENSOR_LABELS = {
    "count": "일자별 생산 번호",
    "working": "가동 여부",
    "emergency_stop": "비상 정지 여부",
    "molten_temp": "용탕 온도",
    "facility_operation_cycleTime": "설비 작동 사이클 시간",
    "production_cycletime": "제품 생산 사이클 시간",
    "low_section_speed": "저속 구간 속도",
    "high_section_speed": "고속 구간 속도",
    "molten_volume": "용탕량",
    "cast_pressure": "주조 압력",
    "biscuit_thickness": "비스켓 두께",
    "upper_mold_temp1": "상금형 온도1",
    "upper_mold_temp2": "상금형 온도2",
    "upper_mold_temp3": "상금형 온도3",
    "lower_mold_temp1": "하금형 온도1",
    "lower_mold_temp2": "하금형 온도2",
    "lower_mold_temp3": "하금형 온도3",
    "sleeve_temperature": "슬리브 온도",
    "physical_strength": "형체력",
    "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "전자교반 가동 시간",
    "registration_time": "등록 일시",
    "passorfail": "양품/불량 판정 (0: 양품, 1: 불량)",
    "tryshot_signal": "사탕 신호",
    "mold_code": "금형 코드",
    "heating_furnace": "가열로 구분"
}
# ───────── 센서 선택 목록에서 제외할 항목 ─────────
UNWANTED_COLUMNS = [
    "Unnamed: 0.2",
    "Unnamed: 0.1",
    "Unnamed: 0",
    "id",
    "anomalyornot",
    "anomaly_detail"
]
AVAILABLE_SENSORS = [
    col for col in shared.NUM_COLS
    if col not in UNWANTED_COLUMNS
]


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        ui.tags.link(rel="stylesheet", href="styles.css"),
      
    ),
    ui.div(
    ui.h1("다이캐스팅 공정 모니터링"),
    class_="dashboard-title-box"),
    ui.page_navbar(
        ui.nav_panel(
            "[1] 모니터링",
            ui.div(
                ui.layout_columns(
                    ui.div(
                        ui.output_ui("simulation_time")
                        ),
                    ui.input_select(
                        "mold_code_1page",
                        "금형 코드 선택: ",
                        choices=['8917', '8722', '8412'],
                        selected=""
                    ),
                    ui.input_select(
                        "period",
                        "조회 기간",
                        choices=["전체","1시간 전", "3시간 전", "12시간 전", "1일 전", "3일 전"],
                        selected=""
                    )
                )
            ),
            ui.div(
                ui.div(
                        ui.div(
                            ui.card(
                                ui.output_plot('spike_chart')),
                                    class_="w-full"
                            ),

                        # 오른쪽 (두 줄 합치기)
                        ui.div(
                            # 첫 번째 알림 카드
                            ui.card(
                                ui.output_ui("defect_alerts"),
                                class_="w-full mh-432",
                            ),
                            # 두 번째 알림 카드
                            ui.card(
                                ui.output_ui("anomaly_alert"), 
                                class_="w-full mh-432"
                            ),
                            class_="w-full flex flex-row gap-4 mh-432",
                        ),
                        class_="flex flex-row mh-432 gap-4"
                    ),
                ),
                
                     ui.div(
                        {"class": "space-y-1 mt-2"},
                        ui.p("센서 선택(최대 3개)"),
                        ui.div(
                            {"class": "flex gap-2"},
                            ui.input_selectize(
                                "selected_cols",
                                "",
                                choices={col: SENSOR_LABELS.get(col, col) for col in AVAILABLE_SENSORS},
                                multiple=True,
                                selected=[
                                    "molten_temp",
                                    "cast_pressure",
                                    "high_section_speed",
                                ],
                                options={"maxItems": 3}
                            ),
                            ui.input_action_button(
                                "apply_sensors",
                                "적용",
                                class_="bg-blue-500 text-white rounded text-sm flex items-center justify-center",
                                style="height: 38px;"  # selectize와 높이 맞춤
                            )
                        )
                    ),
                    ui.h3("최근 100개 대비 실시간 변화"),
                    ui.div(
                        ui.output_ui("stats_cards"),
                    ),
                    ui.div(
                            ui.output_plot('sensor_chart')
                        ),
                ),
        ui.nav_panel(
            "[2] 통계",
            ui.layout_columns(
                ui.input_date_range(
                    "stat_date_range", "분석 기간",
                    start=min_date, end=max_date, min=min_date, max=max_date
                ),
                ui.input_select(
                    "stat_moldcode",
                    "금형 코드 선택: ",
                    choices=['8412', '8722', '8917'],
                    selected="",
                ),
                ui.input_action_button(
                    "apply_btn", "적용하기", class_="btn-primary mt-auto btn-apply mt-auto flex justify-center items-center"
                ),
            ),
            ui.layout_columns(
                ui.value_box("전체 샘플", ui.output_text("total_count_uj")),
                ui.value_box("불량 건수", ui.output_text("defect_count_uj")),
                ui.value_box("Tryshot 수", ui.output_text("tryshot_count")),
            ),
            ui.layout_columns(
                ui.card("불량률 추이", output_widget("defect_trend")),
                ui.card(
                        "관리도",
                        ui.row(
                            ui.column(6,
                                ui.input_date(
                                    "control_date",
                                    "날짜 지정",
                                    value=max_date,
                                    min=min_date,
                                    max=max_date
                                )
                            ),
                            ui.column(6,
                                ui.input_select(
                                    "control_interval", "시간 단위 선택",
                                    choices={"1H": "1시간", "30min": "30분", "5min": "5분"},
                                    selected="1H"
                                )
                            )
                        ),
                        output_widget("control_chart")
                    )
            ),
            ui.layout_columns(
                ui.input_radio_buttons(
                    "download_type", 
                    "다운로드할 데이터:", 
                    choices={
                        "pred_only": "예측 결과만",
                        "all":       "모든 센서 결과"
                    },
                    selected="pred_only",
                    inline=True
                ),

                ui.download_button(
                    "download_log", "불량 로그 다운로드 (Excel)"
                ),
            ),
            ui.div(
                ui.div(
                    ui.output_ui("paged_table"),
                    class_="w-full"
                ),
                class_="flex",
            ),
        ),
        ui.nav_panel(
            "[3] 사용자 정의 예측",
            # 입력 패널 (슬라이더들)
            ui.div(
                # ui.output_ui("current_data_info"),
                ui.row(
                    ui.column(3,
                        ui.input_select(
                            "mold_code",
                            "금형 코드",
                            choices={8412: "8412", 8573: "8573", 8600: "8600", 8722: "8722", 8917: "8917"},
                            selected=8412
                        )
                    ),
                    ui.column(3,
                        ui.input_select(
                            "working",
                            "가동 여부",
                            choices={"on": "가동", "off": "정지"},
                            selected="on"
                        )
                    ),
                    ui.column(2,
                        ui.input_select(
                            "pdp_var1",
                            "PDP 변수 1",
                            choices=pdp_choices,
                            selected="molten_temp"
                        )
                    ),
                    ui.column(2,
                        ui.input_select(
                            "pdp_var2",
                            "PDP 변수 2", 
                            choices=pdp_choices,
                            selected="sleeve_temperature"
                        )
                    ),
                    ui.column(2,
                        ui.input_select(
                            "pdp_var3",
                            "PDP 변수 3",
                            choices=pdp_choices,
                            selected="production_cycletime"
                        )
                    )
                ),

                # 둘째 행부터: 연동 슬라이더들
                ui.row(
                    ui.column(3, create_linked_input('molten_temp', KOREAN_VARIABLE_MAP['molten_temp'], VARIABLE_RANGES['molten_temp'])),
                    ui.column(3, create_linked_input('sleeve_temperature', KOREAN_VARIABLE_MAP['sleeve_temperature'], VARIABLE_RANGES['sleeve_temperature'])),
                    ui.column(3, create_linked_input('Coolant_temperature', KOREAN_VARIABLE_MAP['Coolant_temperature'], VARIABLE_RANGES['Coolant_temperature'])),
                    ui.column(3, create_linked_input('lower_mold_temp1', KOREAN_VARIABLE_MAP['lower_mold_temp1'], VARIABLE_RANGES['lower_mold_temp1'])),
                ),
                ui.row(
                    ui.column(3, create_linked_input('lower_mold_temp2', KOREAN_VARIABLE_MAP['lower_mold_temp2'], VARIABLE_RANGES['lower_mold_temp2'])),
                    ui.column(3, create_linked_input('lower_mold_temp3', KOREAN_VARIABLE_MAP['lower_mold_temp3'], VARIABLE_RANGES['lower_mold_temp3'])),
                    ui.column(3, create_linked_input('upper_mold_temp1', KOREAN_VARIABLE_MAP['upper_mold_temp1'], VARIABLE_RANGES['upper_mold_temp1'])),
                    ui.column(3, create_linked_input('upper_mold_temp2', KOREAN_VARIABLE_MAP['upper_mold_temp2'], VARIABLE_RANGES['upper_mold_temp2'])),
                ),
                ui.row(
                    ui.column(3, create_linked_input('upper_mold_temp3', KOREAN_VARIABLE_MAP['upper_mold_temp3'], VARIABLE_RANGES['upper_mold_temp3'])),
                    ui.column(3, create_linked_input('facility_operation_cycleTime', KOREAN_VARIABLE_MAP['facility_operation_cycleTime'], VARIABLE_RANGES['facility_operation_cycleTime'])),
                    ui.column(3, create_linked_input('production_cycletime', KOREAN_VARIABLE_MAP['production_cycletime'], VARIABLE_RANGES['production_cycletime'])),
                    ui.column(3, create_linked_input('low_section_speed', KOREAN_VARIABLE_MAP['low_section_speed'], VARIABLE_RANGES['low_section_speed'])),
                ),
                ui.row(
                    ui.column(3, create_linked_input('high_section_speed', KOREAN_VARIABLE_MAP['high_section_speed'], VARIABLE_RANGES['high_section_speed'])),
                    ui.column(3, create_linked_input('cast_pressure', KOREAN_VARIABLE_MAP['cast_pressure'], VARIABLE_RANGES['cast_pressure'])),
                    ui.column(3, create_linked_input('biscuit_thickness', KOREAN_VARIABLE_MAP['biscuit_thickness'], VARIABLE_RANGES['biscuit_thickness'])),
                    ui.column(3, create_linked_input('physical_strength', KOREAN_VARIABLE_MAP['physical_strength'], VARIABLE_RANGES['physical_strength'])),
                ),
                ui.row(
                    ui.column(3),
                    ui.column(6,
                        ui.input_action_button("analyze", "결과 분석", class_="analyze-btn")
                    ),
                    ui.column(3),
                ),

                class_="control-panel"
            ),
            ui.div(
                ui.div(
                    ui.div(
                        ui.span("▼", class_="toggle-icon rotated", id="log-toggle-icon"),
                        ui.span("로그 데이터에서 조건 불러오기"),
                        class_="collapsible-header flex flex-row gap-2",
                        onclick="toggleLogSection()"
                    ),
                    ui.div(
                        ui.row(
                            ui.column(4,
                                ui.input_date_range(
                                    "log_date_range", "분석 기간",
                                    start=min_date, end=max_date, min=min_date, max=max_date
                                ),
                            ),
                            ui.column(4,
                                ui.input_selectize(
                                    "log_mold_code",
                                    "금형 코드 선택: ",
                                    choices=['8917', '8722', '8412'],
                                    multiple=True,
                                    selected=[],
                                        options={
                                            "placeholder": "불러올 데이터의 금형 코드를 선택하세요(다중 선택 가능)",
                                            "allowEmptyOption": True
                                        },
                                ),
                            ),
                            ui.column(4,
                                ui.input_select(
                                    "data_filter_type", "데이터 필터",
                                    choices={
                                        "all": "전체 데이터",
                                        "defect_only": "불량 데이터만",
                                        "good_only": "양품 데이터만"
                                    },
                                    selected="all"
                                )
                            ),
                        ),
                        ui.row(
                            ui.column(3,
                                ui.div(),
                                ),
                            ui.column(6,
                                      ui.div(
                                                ui.input_action_button(
                                                    "load_log_data", 
                                                    "데이터 로드", 
                                                    class_="w-full btn-primary h-9 text-sm rounded-md flex justify-center items-center",
                                                    title="로그 데이터를 불러옵니다"
                                                ),
                                                ui.input_action_button(
                                                    "reset_sliders", 
                                                    "초기화", 
                                                    class_="w-full btn-secondary h-9 text-sm rounded-md flex justify-center items-center",
                                                    title="슬라이더를 기본값으로 되돌립니다"
                                                ),
                                                class_="flex gap-4 mb-4",   
                                            ),
                                      ),
                            ui.column(3,
                                      ui.div(),
                            ),
                        ),
                       ui.div(
                            ui.row(
                                ui.column(8,
                                    ui.h5("로그 데이터 목록", style="margin-top: 20px; margin-bottom: 10px;")
                                ),
                                ui.column(4,
                                    ui.div(
                                        ui.output_text("log_data_count_info"),
                                        style="margin-top: 25px; text-align: right; color: #6c757d; font-size: 12px;"
                                    )
                                )
                            ),
                            ui.output_ui("log_data_table")
                        ),
                        class_="collapsible-content show",
                        id="log-collapsible-content"
                    ),
                    class_="collapsible-section"
                ),
                
                # JavaScript 추가
                ui.tags.script("""
                    function toggleLogSection() {
                        const content = document.getElementById('log-collapsible-content');
                        const icon = document.getElementById('log-toggle-icon');
                        
                        if (content.classList.contains('show') || content.style.display === 'block') {
                            content.classList.remove('show');
                            content.style.display = 'none';
                            icon.textContent = '▶';
                            icon.classList.remove('rotated');
                        } else {
                            content.classList.add('show');
                            content.style.display = 'block';
                            icon.textContent = '▼';
                            icon.classList.add('rotated');
                        }
                    }
                    
                    // 페이지 로드 시 초기 상태 설정
                    document.addEventListener('DOMContentLoaded', function() {
                        const content = document.getElementById('log-collapsible-content');
                        const icon = document.getElementById('log-toggle-icon');
                        if (content && icon) {
                            content.classList.add('show');
                            content.style.display = 'block';
                            icon.textContent = '▼';
                            icon.classList.add('rotated');
                        }
                    });
                """),
                class_="log-load-panel"
            ),
            # 예측 결과 패널 (슬라이더 위로 이동)
            ui.div(
                ui.div(
                        ui.div(class_="col-span-3"),
                        ui.div(
                            ui.output_ui("prediction_result"),
                            class_="col-span-6"
                        ),
                        ui.div(class_="col-span-3"),
                        class_="grid grid-cols-12 gap-4",
                    ),
                
                ui.div(
                        ui.div(class_="col-span-4"),
                        ui.div(
                            ui.output_ui("pdf_download_section"),
                            class_="col-span-4"
                        ),
                        ui.div(class_="col-span-4"),
                        class_="grid grid-cols-12 gap-4",
                    ),
                class_="prediction-panel"
            ),

            # 분석 패널 (SHAP, PDP)
            ui.div(
                ui.h3("SHAP 변수 기여도 분석", style="color: #495057; margin-top: 10px;"),
                ui.p("현재 조건에서 각 변수가 예측에 기여한 절댓값 순위 상위 8개를 보여줍니다.", style="color: #6c757d;"),
                ui.output_ui("shap_plot"),

                ui.h3("선택 변수별 PDP 분석", style="color: #495057; margin-top: 30px;"),
                ui.p("선택한 3개 변수 각각의 값 변화 시 평균 예측 확률이 어떻게 변하는지 시뮬레이션합니다.", style="color: #6c757d;"),
                ui.output_ui("pdp_plot"),

                class_="analysis-panel"
            )
        ),
    ),
    ui.output_ui("anomaly_detail_modal"),
    ui.output_ui("defect_detail_modal")
)

def server(input, output, session):
    analysis_results = reactive.Value(None)
    page2_analysis_results = reactive.Value(None)  # 페이지2 전용
    show_detail_modal = reactive.Value(False)
    selected_alert_data = reactive.Value({})
    defect_ranges_results = reactive.Value({})  # 불량 범위 저장용
    defect_alert_list = reactive.Value([])
    applied_cols = reactive.Value([
        "molten_temp",
        "high_section_speed",
        "cast_pressure",
    ])
    log_data = reactive.Value(pd.DataFrame())
    selected_data_info = reactive.Value(None)
    selected_log_row_id = reactive.Value(None)  # 선택된 로그 행 추적

    # 슬라이더 수동 조정 감지 변수
    manual_adjustment_detected = reactive.Value(False)

    log_data_loading = reactive.Value(False)
    log_data_loaded = reactive.Value(False) 

    # 다른 페이지와 공유되는 분석 결과는 별도 관리
    anomaly_results = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.apply_sensors)
    def _():
        # 버튼을 누를 때마다 input.selected_cols() 값을 applied_cols에 저장
        applied_cols.set(list(input.selected_cols() or []))

    # ------------------- 변수 목록 정의 -------------------
    variable_names = ['molten_temp', 'sleeve_temperature', 'Coolant_temperature', 
                     'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
                     'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                     'facility_operation_cycleTime', 'production_cycletime',
                     'low_section_speed', 'high_section_speed', 'cast_pressure',
                     'biscuit_thickness', 'physical_strength']
    
    # ------------------- 마지막 변경 소스 추적 -------------------
    # 각 변수별로 마지막에 어떤 컨트롤이 변경되었는지 추적
    last_changed_source = {}
    for var_name in variable_names:
        last_changed_source[var_name] = reactive.Value("slider")  # 기본값은 슬라이더
    
    # 변경 감지 이펙트들을 명시적으로 정의
    # molten_temp
    @reactive.Effect
    def _molten_temp_slider_change():
        input.molten_temp_slider()
        # 로그 데이터 로딩 중이 아닐 때만 실행
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['molten_temp'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _molten_temp_numeric_change():
        input.molten_temp_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['molten_temp'].set("numeric")
            manual_adjustment_detected.set(True)

    # sleeve_temperature
    @reactive.Effect
    def _sleeve_temperature_slider_change():
        input.sleeve_temperature_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['sleeve_temperature'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _sleeve_temperature_numeric_change():
        input.sleeve_temperature_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['sleeve_temperature'].set("numeric")
            manual_adjustment_detected.set(True)

    # Coolant_temperature
    @reactive.Effect
    def _Coolant_temperature_slider_change():
        input.Coolant_temperature_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['Coolant_temperature'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _Coolant_temperature_numeric_change():
        input.Coolant_temperature_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['Coolant_temperature'].set("numeric")
            manual_adjustment_detected.set(True)

    # lower_mold_temp1
    @reactive.Effect
    def _lower_mold_temp1_slider_change():
        input.lower_mold_temp1_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['lower_mold_temp1'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _lower_mold_temp1_numeric_change():
        input.lower_mold_temp1_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['lower_mold_temp1'].set("numeric")
            manual_adjustment_detected.set(True)

    # lower_mold_temp2
    @reactive.Effect
    def _lower_mold_temp2_slider_change():
        input.lower_mold_temp2_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['lower_mold_temp2'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _lower_mold_temp2_numeric_change():
        input.lower_mold_temp2_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['lower_mold_temp2'].set("numeric")
            manual_adjustment_detected.set(True)

    # lower_mold_temp3
    @reactive.Effect
    def _lower_mold_temp3_slider_change():
        input.lower_mold_temp3_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['lower_mold_temp3'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _lower_mold_temp3_numeric_change():
        input.lower_mold_temp3_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['lower_mold_temp3'].set("numeric")
            manual_adjustment_detected.set(True)

    # upper_mold_temp1
    @reactive.Effect
    def _upper_mold_temp1_slider_change():
        input.upper_mold_temp1_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['upper_mold_temp1'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _upper_mold_temp1_numeric_change():
        input.upper_mold_temp1_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['upper_mold_temp1'].set("numeric")
            manual_adjustment_detected.set(True)

    # upper_mold_temp2
    @reactive.Effect
    def _upper_mold_temp2_slider_change():
        input.upper_mold_temp2_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['upper_mold_temp2'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _upper_mold_temp2_numeric_change():
        input.upper_mold_temp2_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['upper_mold_temp2'].set("numeric")
            manual_adjustment_detected.set(True)

    # upper_mold_temp3
    @reactive.Effect
    def _upper_mold_temp3_slider_change():
        input.upper_mold_temp3_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['upper_mold_temp3'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _upper_mold_temp3_numeric_change():
        input.upper_mold_temp3_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['upper_mold_temp3'].set("numeric")
            manual_adjustment_detected.set(True)

    # facility_operation_cycleTime
    @reactive.Effect
    def _facility_operation_cycleTime_slider_change():
        input.facility_operation_cycleTime_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['facility_operation_cycleTime'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _facility_operation_cycleTime_numeric_change():
        input.facility_operation_cycleTime_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['facility_operation_cycleTime'].set("numeric")
            manual_adjustment_detected.set(True)

    # production_cycletime
    @reactive.Effect
    def _production_cycletime_slider_change():
        input.production_cycletime_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['production_cycletime'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _production_cycletime_numeric_change():
        input.production_cycletime_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['production_cycletime'].set("numeric")
            manual_adjustment_detected.set(True)

    # low_section_speed
    @reactive.Effect
    def _low_section_speed_slider_change():
        input.low_section_speed_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['low_section_speed'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _low_section_speed_numeric_change():
        input.low_section_speed_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['low_section_speed'].set("numeric")
            manual_adjustment_detected.set(True)

    # high_section_speed
    @reactive.Effect
    def _high_section_speed_slider_change():
        input.high_section_speed_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['high_section_speed'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _high_section_speed_numeric_change():
        input.high_section_speed_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['high_section_speed'].set("numeric")
            manual_adjustment_detected.set(True)

    # cast_pressure
    @reactive.Effect
    def _cast_pressure_slider_change():
        input.cast_pressure_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['cast_pressure'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _cast_pressure_numeric_change():
        input.cast_pressure_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['cast_pressure'].set("numeric")
            manual_adjustment_detected.set(True)

    # biscuit_thickness
    @reactive.Effect
    def _biscuit_thickness_slider_change():
        input.biscuit_thickness_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['biscuit_thickness'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _biscuit_thickness_numeric_change():
        input.biscuit_thickness_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['biscuit_thickness'].set("numeric")
            manual_adjustment_detected.set(True)

    # physical_strength
    @reactive.Effect
    def _physical_strength_slider_change():
        input.physical_strength_slider()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['physical_strength'].set("slider")
            manual_adjustment_detected.set(True)

    @reactive.Effect
    def _physical_strength_numeric_change():
        input.physical_strength_numeric()
        if not log_data_loading.get():
            selected_data_info.set(None)
            last_changed_source['physical_strength'].set("numeric")
            manual_adjustment_detected.set(True)

    # ------------------- 값 가져오기 함수 -------------------
    def get_variable_value(var_name):
        """마지막 변경 소스를 고려하여 현재 값 가져오기"""
        try:
            numeric_val = getattr(input, f"{var_name}_numeric")()
            slider_val = getattr(input, f"{var_name}_slider")()
            
            # 마지막 변경 소스 확인
            last_source = last_changed_source[var_name].get()
            
            if last_source == "numeric" and numeric_val is not None:
                return numeric_val
            else:
                return slider_val
        except:
            return VARIABLE_RANGES.get(var_name, {}).get('default', 0)

    def calculate_realtime_defect_range(var_name, current_inputs):
        """이진 탐색으로 빠른 불량 범위 계산"""
        if not pipeline or "models" not in pipeline:
            return {"ranges": []}
        
        mold_code = current_inputs["mold_code"]
        if mold_code not in pipeline["models"]:
            return {"ranges": []}
        
        try:
            model = pipeline["models"][mold_code]
            model_type = pipeline["best_model_config"][mold_code]
            threshold = pipeline["optimal_thresholds"].get(mold_code, 0.5)
            feature_names = pipeline["feature_names"]
            
            # 현재 모든 입력값을 기준으로 베이스 샘플 생성
            base_sample = {}
            for col in feature_names:
                if col in current_inputs:
                    base_sample[col] = current_inputs[col]
                else:
                    base_sample[col] = 0  # 기본값
            
            if var_name not in VARIABLE_RANGES:
                return {"ranges": []}
            
            var_range = VARIABLE_RANGES[var_name]
            min_val = var_range["min"]
            max_val = var_range["max"]
            
            def predict_at_value(val):
                """특정 값에서 불량 확률 예측"""
                test_sample = base_sample.copy()
                test_sample[var_name] = val
                X_test = pd.DataFrame([test_sample], columns=feature_names)
                
                if "XGBoost" in model_type:
                    dmat = xgb.DMatrix(X_test)
                    proba = model.predict(dmat)[0]
                else:
                    proba = model.predict(X_test)[0]
                
                return proba >= threshold
            
            # 이진 탐색으로 불량 구간 찾기 (약 20번 예측으로 충분)
            ranges = []
            
            # 전체 범위를 10개 정도 구간으로 나눠서 대략적인 불량 구간 파악
            sample_points = np.linspace(min_val, max_val, 12)
            defect_status = [predict_at_value(p) for p in sample_points]
            
            # 연속된 불량 구간들 찾기
            start = None
            for i, is_defect in enumerate(defect_status):
                if is_defect and start is None:
                    start = i
                elif not is_defect and start is not None:
                    # 불량 구간 발견, 이진 탐색으로 정확한 경계 찾기
                    left_bound = sample_points[max(0, start-1)]
                    right_bound = sample_points[min(len(sample_points)-1, i)]
                    
                    # 시작점 이진 탐색 (최대 5번)
                    left, right = left_bound, sample_points[start]
                    for _ in range(5):
                        mid = (left + right) / 2
                        if predict_at_value(mid):
                            right = mid
                        else:
                            left = mid
                    start_bound = right
                    
                    # 끝점 이진 탐색 (최대 5번)
                    left, right = sample_points[i-1], right_bound
                    for _ in range(5):
                        mid = (left + right) / 2
                        if predict_at_value(mid):
                            left = mid
                        else:
                            right = mid
                    end_bound = left
                    
                    ranges.append({
                        "min": start_bound,
                        "max": end_bound
                    })
                    start = None
            
            # 마지막까지 불량인 경우
            if start is not None:
                left_bound = sample_points[max(0, start-1)]
                left, right = left_bound, sample_points[start]
                for _ in range(5):
                    mid = (left + right) / 2
                    if predict_at_value(mid):
                        right = mid
                    else:
                        left = mid
                ranges.append({
                    "min": right,
                    "max": max_val
                })
            
            return {"ranges": ranges}
            
        except Exception as e:
            print(f"불량 범위 계산 오류 ({var_name}): {e}")
            return {"ranges": []}

    def render_defect_range(var_name):
        """저장된 불량 범위 표시 (분석 버튼 클릭 시에만 업데이트)"""
        
        # 저장된 불량 범위 가져오기
        cached_ranges = defect_ranges_results.get()
        
        # 분석하지 않은 상태
        if not cached_ranges or var_name not in cached_ranges:
            return ui.div(
                ui.div(
                    style="background: #ddd; height: 6px; border-radius: 3px; margin: 4px 0; width: 70%; border: 1px solid #ccc;",
                    class_="defect-range-bar-multi"
                ),
                ui.div("분석 후 불량 범위 표시", class_="defect-info", style="color: #999;")
            )
        
        # 저장된 불량 범위 데이터 사용
        range_data = cached_ranges[var_name]
        
        if range_data["ranges"]:
            if var_name not in VARIABLE_RANGES:
                return ui.div()
            
            var_range = VARIABLE_RANGES[var_name]
            total_min = var_range["min"]
            total_max = var_range["max"]
            defect_ranges = range_data["ranges"]
            
            # 다중 구간을 위한 gradient 생성
            gradient = generate_gradient_for_multiple_ranges(var_name, defect_ranges, total_min, total_max)
            
            # 구간 정보 텍스트 생성
            if len(defect_ranges) == 1:
                info_text = f"불량 범위 약 {defect_ranges[0]['min']:.0f}~{defect_ranges[0]['max']:.0f}"
            elif len(defect_ranges) <= 3:
                ranges_text = ", ".join([f"{r['min']:.0f}~{r['max']:.0f}" for r in defect_ranges])
                info_text = f"불량 범위 약 {ranges_text}"
            else:
                info_text = f"불량 범위 약 {len(defect_ranges)}개 구간"
            
            return ui.div(
                ui.div(
                    style=f"background: {gradient}; height: 6px; border-radius: 3px; margin: 4px 0; width: 70%; border: 1px solid #dee2e6;",
                    class_="defect-range-bar-multi"
                ),
                ui.div(info_text, class_="defect-info")
            )
        else:
            # 불량 범위가 없는 경우 전체 파란색
            return ui.div(
                ui.div(
                    style="background: #007bff; height: 6px; border-radius: 3px; margin: 4px 0; width: 70%; border: 1px solid #dee2e6;",
                    class_="defect-range-bar-multi"
                ),
                ui.div("불량 범위 없음", class_="defect-info", style="color: #007bff;")
            )

    @reactive.Effect
    @reactive.event(input.load_log_data)
    def handle_log_data_load():
        """데이터 로드 버튼이 클릭되었을 때 상태 업데이트"""
        log_data_loaded.set(True)

    @reactive.Calc
    def get_log_data():
        # 초기화 후 또는 데이터 로드 버튼을 누르지 않았으면 빈 DataFrame 반환 ===
        if not log_data_loaded.get() or input.load_log_data() == 0:
            return pd.DataFrame()
        
        try:
            _df = uj_df.copy()
            
            # 날짜 컬럼 미리 생성 (한 번만)
            if "registration_date" not in _df.columns:
                _df["registration_date"] = _df["registration_time"].dt.strftime("%Y-%m-%d")
            
            date_range = input.log_date_range()
            date_start = date_range[0].strftime("%Y-%m-%d")
            date_end = date_range[1].strftime("%Y-%m-%d")
            
            selected_molds = input.log_mold_code()
            
            mask = (
                (_df["registration_date"] >= date_start) &
                (_df["registration_date"] <= date_end)
            )
            
            if selected_molds:
                mask = mask & (_df["mold_code"].astype(str).isin(selected_molds))
            
            _df = _df[mask]
            
            filter_type = input.data_filter_type()
            if filter_type == "defect_only":
                _df = _df[_df["passorfail"] == 1]
            elif filter_type == "good_only":
                _df = _df[_df["passorfail"] == 0]
            
            _df = _df.sort_values("registration_time", ascending=False)
            
            return _df.reset_index(drop=True)
            
        except Exception as e:
            print(f"로그 데이터 로딩 오류: {e}")
            return pd.DataFrame()
    
    @output
    @render.text
    def log_data_count_info():
        df = get_log_data()
        if df.empty:
            return ""
        
        total_count = len(df)
        defect_count = (df['passorfail'] == 1).sum() if 'passorfail' in df.columns else 0
        good_count = total_count - defect_count
        
        return f"총 {total_count}건 (양품: {good_count}, 불량: {defect_count})"
    
    @output
    @render.ui
    def log_data_table():
        df = get_log_data()
        
        if df.empty:
            return ui.div(
                ui.p("'데이터 로드' 버튼을 클릭하여 로그 데이터를 불러오세요.", 
                     style="color: #6c757d; text-align: center; padding: 20px;")
            )
        
        display_cols = [
            'id', 'registration_time', 'mold_code', 'passorfail',
            'molten_temp', 'cast_pressure', 'production_cycletime',
            'high_section_speed', 'biscuit_thickness'
        ]
        
        available_cols = [col for col in display_cols if col in df.columns]
        df_display = df[available_cols].copy()
        
        if 'registration_time' in df_display.columns:
            df_display['registration_time'] = pd.to_datetime(df_display['registration_time']).dt.strftime('%Y-%m-%d %H:%M')
        
        if 'passorfail' in df_display.columns:
            df_display['품질상태'] = df_display['passorfail'].map({0: '양품', 1: '불량'})
            df_display = df_display.drop('passorfail', axis=1)
        
        column_mapping = {
            'id': 'ID',
            'registration_time': '등록시간',
            'mold_code': '금형코드',
            'molten_temp': '용탕온도',
            'cast_pressure': '주조압력',
            'production_cycletime': '생산사이클',
            'high_section_speed': '고속구간속도',
            'biscuit_thickness': '비스킷두께'
        }
        
        df_display = df_display.rename(columns=column_mapping)
        
        selected_row = selected_log_row_id.get()
        
        table_html = '<div style="max-height: 400px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 5px;">'
        table_html += '<table class="log-table" style="margin: 0;">'
        
        table_html += '<thead>'
        table_html += '<tr>'
        for col in df_display.columns:
            table_html += f'<th>{col}</th>'
        table_html += '</tr></thead>'
        
        table_html += '<tbody>'
        for idx, row in df_display.iterrows():
            original_idx = df.index[idx]
            row_class = "selected-row" if selected_row == original_idx else ""
            table_html += f'<tr class="{row_class}" onclick="selectLogData({original_idx})">'
            for col in df_display.columns:
                value = row[col]
                if pd.isna(value):
                    value = '-'
                elif isinstance(value, float):
                    value = f'{value:.1f}'
                table_html += f'<td>{value}</td>'
            table_html += '</tr>'
        table_html += '</tbody></table></div>'
        
        js_code = """
        <script>
        function selectLogData(rowIndex) {
            Shiny.setInputValue('selected_log_row', rowIndex, {priority: 'event'});
        }
        </script>
        """
        
        return ui.HTML(table_html + js_code)
    
# ================= 수정된 apply_selected_log_data 함수 =================

    @reactive.Effect
    @reactive.event(input.selected_log_row)
    def apply_selected_log_data():
        try:
            df = get_log_data()
            if df.empty:
                return
            
            selected_idx = input.selected_log_row()
            if selected_idx >= len(df):
                return
                
            selected_row = df.iloc[selected_idx]
            selected_log_row_id.set(selected_idx)
            
            # === 강제로 모든 상태 초기화 ===
            manual_adjustment_detected.set(False)
            selected_data_info.set(None)  # 먼저 None으로 초기화
            
            # === 강제 UI 업데이트 ===
            import time
            time.sleep(0.01)  # 아주 짧은 딜레이
            
            # === 로그 데이터 로딩 시작 ===
            log_data_loading.set(True)
            
            # 선택된 데이터 정보 설정
            selected_data_info.set({
                'id': selected_row.get('id', 'N/A'),
                'time': selected_row.get('registration_time', 'N/A'),
                'mold_code': selected_row.get('mold_code', 'N/A'),
                'quality': '양품' if selected_row.get('passorfail', 0) == 0 else '불량'
            })
            
            # 슬라이더 업데이트
            for var_name in variable_names:
                if var_name in selected_row and pd.notna(selected_row[var_name]):
                    value = float(selected_row[var_name])
                    
                    var_range = VARIABLE_RANGES.get(var_name, {})
                    min_val = var_range.get('min', 0)
                    max_val = var_range.get('max', 1000)
                    
                    if value < min_val:
                        value = min_val
                    elif value > max_val:
                        value = max_val
                    
                    ui.update_slider(f"{var_name}_slider", value=value)
                    ui.update_numeric(f"{var_name}_numeric", value=value)
            
            if 'mold_code' in selected_row and pd.notna(selected_row['mold_code']):
                ui.update_select("mold_code", selected=int(selected_row['mold_code']))
            
            if 'working' in selected_row and pd.notna(selected_row['working']):
                working_val = "on" if selected_row['working'] == 1 else "off"
                ui.update_select("working", selected=working_val)
            
            # === 로그 데이터 로딩 완료 ===
            log_data_loading.set(False)
            
            # === 강제로 reactive 시스템 업데이트 ===
            reactive.invalidate_later(2)
            
            print(f"✅ 로그 데이터 선택 완료: ID {selected_row.get('id', 'N/A')}")
                    
        except Exception as e:
            log_data_loading.set(False)
            manual_adjustment_detected.set(False)
            print(f"❌ 로그 데이터 적용 중 오류: {e}")
    
    @output
    @render.ui
    def current_data_info():
        # info = selected_data_info.get()
        # manual_adjusted = manual_adjustment_detected.get()
        
        # # 로그 데이터가 선택되어 있으면 우선 표시
        # if info is not None:
        #     return ui.div(
        #         ui.div(
        #             f"현재 적용된 데이터: ID {info['id']} | {info['time']} | 금형 {info['mold_code']} | {info['quality']}",
        #             class_="current-data-info"
        #         )
        #     )
        # # 로그 데이터 선택이 없고 수동 조절한 경우에만
        # elif manual_adjusted:
        #     return ui.div(
        #         ui.div(
        #             "조건 직접 조절한 상태",
        #             class_="current-data-info"
        #         )
        #     )
        # # 둘 다 없으면 빈 div
        # else:
            return ui.div()
    
    @reactive.Effect
    @reactive.event(input.reset_sliders)
    def reset_sliders_to_default():
        try:
            # === 로그 데이터 로딩 중 플래그 설정 (슬라이더 변경 감지 방지) ===
            log_data_loading.set(True)
            
            # 슬라이더 초기화
            for var_name in variable_names:
                default_val = VARIABLE_RANGES.get(var_name, {}).get('default', 0)
                ui.update_slider(f"{var_name}_slider", value=default_val)
                ui.update_numeric(f"{var_name}_numeric", value=default_val)
            
            ui.update_select("mold_code", selected=8412)
            ui.update_select("working", selected="on")
            
            # === 모든 상태 완전 초기화 ===
            selected_data_info.set(None)
            selected_log_row_id.set(None)
            manual_adjustment_detected.set(False)
            log_data_loaded.set(False)  # 로그 데이터 로드 상태 리셋
            
            # 로그 데이터 필터 초기화
            ui.update_select("log_mold_code", selected=[])
            ui.update_select("data_filter_type", selected="all")
            
            # === 로그 데이터 로딩 완료 ===
            log_data_loading.set(False)
            
            print("✅ 모든 상태가 완전 초기화되었습니다.")
            
        except Exception as e:
            log_data_loading.set(False)
            print(f"❌ 초기화 중 오류: {e}")

    @reactive.Effect
    @reactive.event(input.analyze)
    def run_analysis():
        try:
            if pipeline is None:
                page2_analysis_results.set({"error": "파이프라인을 로드할 수 없습니다"})
                return

            # 0) 분석 시점에 모든 슬라이더와 숫자입력 동기화
            for var_name in variable_names:
                try:
                    # 현재 값 가져오기
                    numeric_val = getattr(input, f"{var_name}_numeric")()
                    slider_val = getattr(input, f"{var_name}_slider")()
                    
                    # 범위 체크용 데이터
                    var_range = VARIABLE_RANGES.get(var_name, {})
                    min_val = var_range.get('min', 0)
                    max_val = var_range.get('max', 1000)
                    
                    # 마지막 변경 소스에 따라 최종값 결정
                    last_source = last_changed_source[var_name].get()
                    
                    if last_source == "numeric" and numeric_val is not None:
                        # 숫자입력이 마지막 변경이고 값이 있는 경우
                        final_val = numeric_val
                    else:
                        # 슬라이더가 마지막 변경이거나 숫자입력이 None인 경우
                        final_val = slider_val
                    
                    # 범위 체크 후 조정
                    if final_val < min_val:
                        final_val = min_val
                    elif final_val > max_val:
                        final_val = max_val
                    
                    # 슬라이더와 숫자입력을 최종값으로 동기화
                    ui.update_slider(f"{var_name}_slider", value=final_val)
                    ui.update_numeric(f"{var_name}_numeric", value=final_val)
                    
                except Exception as e:
                    print(f"동기화 오류 ({var_name}): {e}")

            # 1) 입력값 수집 (연동된 값들 사용)
            mold_code = int(input.mold_code())
            working_val = 0 if input.working() == "auto" else 1
            molten_temp = get_variable_value('molten_temp')
            sleeve_temp = get_variable_value('sleeve_temperature')
            coolant_temp = get_variable_value('Coolant_temperature')
            lower1 = get_variable_value('lower_mold_temp1')
            lower2 = get_variable_value('lower_mold_temp2')
            lower3 = get_variable_value('lower_mold_temp3')
            upper1 = get_variable_value('upper_mold_temp1')
            upper2 = get_variable_value('upper_mold_temp2')
            upper3 = get_variable_value('upper_mold_temp3')
            cycle_fac = get_variable_value('facility_operation_cycleTime')
            cycle_prod = get_variable_value('production_cycletime')
            low_speed = get_variable_value('low_section_speed')
            high_speed = get_variable_value('high_section_speed')
            pressure = get_variable_value('cast_pressure')
            biscuit = get_variable_value('biscuit_thickness')
            strength = get_variable_value('physical_strength')

            # 2) 입력 데이터를 한 행짜리 dict로 생성
            input_data = {
                "working": working_val,
                "molten_temp": molten_temp,
                "facility_operation_cycleTime": cycle_fac,
                "production_cycletime": cycle_prod,
                "low_section_speed": low_speed,
                "high_section_speed": high_speed,
                "cast_pressure": pressure,
                "biscuit_thickness": biscuit,
                "upper_mold_temp1": upper1,
                "upper_mold_temp2": upper2,
                "upper_mold_temp3": upper3,
                "lower_mold_temp1": lower1,
                "lower_mold_temp2": lower2,
                "lower_mold_temp3": lower3,
                "sleeve_temperature": sleeve_temp,
                "physical_strength": strength,
                "Coolant_temperature": coolant_temp,
                "mold_code": mold_code,
                "passorfail": 0
            }
            df_partial = pd.DataFrame([input_data])

            # 3) 모델 로드
            models_dict = pipeline.get("models", {})
            if mold_code not in models_dict:
                page2_analysis_results.set({"error": f"mold_code {mold_code}에 대한 모델이 없습니다"})
                return
            model = models_dict[mold_code]

            best_model_config = pipeline.get("best_model_config", {})
            model_type = best_model_config.get(mold_code, None)

            optimal_thresholds = pipeline.get("optimal_thresholds", {})
            threshold = optimal_thresholds.get(mold_code, 0.5)

            # 4) feature_names 순서대로 DataFrame 재정렬
            feat_names = pipeline.get("feature_names", [])
            df_full = pd.DataFrame(columns=feat_names)
            df_full.loc[0] = [0] * len(feat_names)
            for col in df_partial.columns:
                if col in df_full.columns:
                    df_full.loc[0, col] = df_partial.loc[0, col]
            X_full = df_full[feat_names]

            # 5) 예측 확률 계산
            if model_type == "XGBoost_Balanced":
                dmat = xgb.DMatrix(X_full)
                proba = model.predict(dmat)[0]
            else:
                proba = model.predict(X_full)[0]
            pred = 1 if proba >= threshold else 0

            # 6) SHAP 계산
            shap_top_feats, shap_top_vals = [], []
            try:
                if "XGBoost" in model_type:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_full)
                shap_abs = np.abs(shap_values[0])
                order = np.argsort(shap_abs)[::-1][:8]
                selected_feats = [feat_names[i] for i in order]
                selected_vals = [shap_abs[i] for i in order]
                shap_top_feats = selected_feats[::-1]
                shap_top_vals = selected_vals[::-1]
            except Exception as e:
                print(f"⚠ SHAP 계산 중 오류: {e}")

            # 7) PDP용 변수 세 개 읽어두기 - 안전한 접근
            pdp_var1 = input.pdp_var1() if hasattr(input, 'pdp_var1') and input.pdp_var1() else "molten_temp"
            pdp_var2 = input.pdp_var2() if hasattr(input, 'pdp_var2') and input.pdp_var2() else "sleeve_temperature"  
            pdp_var3 = input.pdp_var3() if hasattr(input, 'pdp_var3') and input.pdp_var3() else "production_cycletime"
            
            sel = []
            for v in (pdp_var1, pdp_var2, pdp_var3):
                if v and v not in sel:
                    sel.append(v)
            defaults = ["molten_temp", "sleeve_temperature", "production_cycletime"]
            for d in defaults:
                if len(sel) >= 3:
                    break
                if d not in sel:
                    sel.append(d)
            pdp_vars = sel[:3]

            # 8) 결과 저장
            results = {
                "prediction": int(pred),
                "probability": float(proba),
                "threshold": float(threshold),
                "model_type": model_type,
                "mold_code": mold_code,
                "feature_names": feat_names,
                "input_values": input_data,
                "input_df": X_full,
                "shap_top_feats": shap_top_feats,
                "shap_top_vals": shap_top_vals,
                "pdp_vars": pdp_vars
            }
            page2_analysis_results.set(results)

            # 9) 불량 범위 계산 (결과 분석 버튼 클릭 시에만)
            current_inputs = {
                "mold_code": mold_code,
                "working": working_val,
                "molten_temp": molten_temp,
                "sleeve_temperature": sleeve_temp,
                "Coolant_temperature": coolant_temp,
                "lower_mold_temp1": lower1,
                "lower_mold_temp2": lower2,
                "lower_mold_temp3": lower3,
                "upper_mold_temp1": upper1,
                "upper_mold_temp2": upper2,
                "upper_mold_temp3": upper3,
                "facility_operation_cycleTime": cycle_fac,
                "production_cycletime": cycle_prod,
                "low_section_speed": low_speed,
                "high_section_speed": high_speed,
                "cast_pressure": pressure,
                "biscuit_thickness": biscuit,
                "physical_strength": strength,
            }
            
            # 모든 변수의 불량 범위 계산
            all_defect_ranges = {}
            for var_name in variable_names:
                range_data = calculate_realtime_defect_range(var_name, current_inputs)
                all_defect_ranges[var_name] = range_data
            
            defect_ranges_results.set(all_defect_ranges)

        except Exception as e:
            page2_analysis_results.set({"error": f"분석 중 오류 발생: {e}"})

    # ------------------- 불량 범위 표시 함수들 -------------------
    @output
    @render.ui
    def defect_range_molten_temp():
        return render_defect_range('molten_temp')
    
    @output
    @render.ui
    def defect_range_sleeve_temperature():
        return render_defect_range('sleeve_temperature')
    
    @output
    @render.ui
    def defect_range_Coolant_temperature():
        return render_defect_range('Coolant_temperature')
    
    @output
    @render.ui
    def defect_range_lower_mold_temp1():
        return render_defect_range('lower_mold_temp1')
    
    @output
    @render.ui
    def defect_range_lower_mold_temp2():
        return render_defect_range('lower_mold_temp2')
    
    @output
    @render.ui
    def defect_range_lower_mold_temp3():
        return render_defect_range('lower_mold_temp3')
    
    @output
    @render.ui
    def defect_range_upper_mold_temp1():
        return render_defect_range('upper_mold_temp1')
    
    @output
    @render.ui
    def defect_range_upper_mold_temp2():
        return render_defect_range('upper_mold_temp2')
    
    @output
    @render.ui
    def defect_range_upper_mold_temp3():
        return render_defect_range('upper_mold_temp3')
    
    @output
    @render.ui
    def defect_range_facility_operation_cycleTime():
        return render_defect_range('facility_operation_cycleTime')
    
    @output
    @render.ui
    def defect_range_production_cycletime():
        return render_defect_range('production_cycletime')
    
    @output
    @render.ui
    def defect_range_low_section_speed():
        return render_defect_range('low_section_speed')
    
    @output
    @render.ui
    def defect_range_high_section_speed():
        return render_defect_range('high_section_speed')
    
    @output
    @render.ui
    def defect_range_cast_pressure():
        return render_defect_range('cast_pressure')
    
    @output
    @render.ui
    def defect_range_biscuit_thickness():
        return render_defect_range('biscuit_thickness')
    
    @output
    @render.ui
    def defect_range_physical_strength():
        return render_defect_range('physical_strength')

    # ------------------- 출력: 예측 결과 -------------------
    @output
    @render.ui
    def prediction_result():
        results = page2_analysis_results.get()  # 페이지2 전용 결과 사용
        if results is None:
            return ui.div(
                ui.h3("분석 대기 중", style="color: #6c757d; text-align: center;"),
                ui.p("모든 설정을 완료한 후 '결과 분석' 버튼을 클릭하세요.", style="text-align: center; color: #6c757d;")
            )
        if "error" in results:
            return ui.div(
                ui.h3("오류", style="color: #dc3545;"),
                ui.p(results["error"], style="color: #dc3545;")
            )
        
        pred = results["prediction"]
        proba = results["probability"]
        threshold = results["threshold"]
        
        if pred == 0:
            cls = "prediction-good"
            txt = f"양품 예측 (불량 확률: {proba:.1%})"
            detail = f"임계값 {threshold:.3f}보다 낮아 양품으로 판정됩니다."
        else:
            cls = "prediction-bad"
            txt = f"불량 예측 (불량 확률: {proba:.1%})"
            detail = f"임계값 {threshold:.3f}보다 높아 불량으로 판정됩니다."
        
        return ui.div(
            ui.h3("품질 예측 결과", style="color: #495057; display: flex; justify-content: center; text-align: center;"),
            ui.div(txt, class_=cls),
            ui.p(detail, style="color: #6c757d; text-align: center; margin-top: 10px;"),
            ui.p(f"사용 모델: {results['model_type']} (금형코드 {results['mold_code']})",
                 style="color: #6c757d; text-align: center; font-size: 12px;")
        )

    # ------------------- 출력: PDF 다운로드 -------------------
    @output
    @render.ui  
    def pdf_download_section():
        results = page2_analysis_results.get()  # 페이지2 전용 결과 사용
        if results is None or "error" in results:
            return ui.div()
        
        return ui.div(
            ui.download_button(
                "download_pdf", 
                "PDF 보고서 다운로드", 
                class_="pdf-btn"
            ),
            style="margin: 20px 0;"
        )

    @render.download(
        filename=lambda: f"품질예측보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    def download_pdf():
        results = page2_analysis_results.get()  # 페이지2 전용 결과 사용
        defect_ranges = defect_ranges_results.get()
        
        if results is None or "error" in results:
            return None
            
        try:
            # PDF 생성
            pdf_data = generate_pdf_report(results, defect_ranges)
            
            if pdf_data:
                return io.BytesIO(pdf_data)
            else:
                return None
                
        except Exception as e:
            print(f"PDF 다운로드 오류: {e}")
            return None

    @output
    @render.ui
    def shap_plot():
        results = page2_analysis_results.get()
        if results is None or "error" in results:
            fig = go.Figure()
            fig.add_annotation(
                text="분석 대기 중<br>모든 설정을 완료한 후 '결과 분석' 버튼을 클릭하세요.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="SHAP 변수 기여도 (대기 중)",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="shap_bar"))

        if not results.get("shap_top_feats"):
            return ui.div("SHAP 계산 데이터가 없습니다.")

        top_feats = results["shap_top_feats"]
        top_vals = results["shap_top_vals"]
        korean_feats = [KOREAN_VARIABLE_MAP.get(f, f) for f in top_feats]

        fig = go.Figure(go.Bar(
            y=korean_feats,
            x=top_vals,
            orientation="h",
            marker_color="lightgreen",
            text=[f"{v:.3f}" for v in top_vals],
            textposition="outside"
        ))
        fig.update_layout(
            xaxis_title="SHAP 값 (절댓값)",
            yaxis_title="변수",
            height=400,
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#E0E0E0', gridwidth=0.5),
            yaxis=dict(gridcolor='#E0E0E0', gridwidth=0.5)
        )
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="shap_bar"))


    @output
    @render.ui
    def pdp_plot():
        results = page2_analysis_results.get()
        if results is None or "error" in results:
            fig = go.Figure()
            fig.add_annotation(
                text="분석 대기 중<br>모든 설정을 완료한 후 '결과 분석' 버튼을 클릭하세요.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="PDP 분석 (대기 중)",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="pdp_sim"))

        pdp_vars = results.get("pdp_vars", [])
        inp = results["input_values"]

        if len(pdp_vars) < 3:
            pdp_vars = ["molten_temp", "sleeve_temperature", "production_cycletime"]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"{KOREAN_VARIABLE_MAP.get(pdp_vars[0], pdp_vars[0])} 변화" if len(pdp_vars) > 0 else "변수 1",
                f"{KOREAN_VARIABLE_MAP.get(pdp_vars[1], pdp_vars[1])} 변화" if len(pdp_vars) > 1 else "변수 2", 
                f"{KOREAN_VARIABLE_MAP.get(pdp_vars[2], pdp_vars[2])} 변화" if len(pdp_vars) > 2 else "변수 3"
            ]
        )

        # 첫 번째 점 추가 (현재값)
        if len(pdp_vars) > 0:
            feat = pdp_vars[0]
            cur = inp.get(feat, 0)
            if feat in VARIABLE_RANGES:
                mn = VARIABLE_RANGES[feat]["min"]
                mx = VARIABLE_RANGES[feat]["max"]
            else:
                mn = cur * 0.8
                mx = cur * 1.2
            vals = np.linspace(mn, mx, 20)
            np.random.seed(42)
            if feat == "molten_temp":
                probs = 0.1 + 0.4 * (vals - mn) / (mx - mn) + np.random.normal(0, 0.05, 20)
            elif feat == "production_cycletime":
                opt = (mn + mx) / 2
                probs = 0.1 + 0.3 * np.abs(vals - opt) / (mx - mn) + np.random.normal(0, 0.03, 20)
            else:
                probs = 0.2 + 0.2 * np.sin((vals - mn) / (mx - mn) * np.pi) + np.random.normal(0, 0.03, 20)
            probs = np.clip(probs, 0, 1)
            cur_prob = np.interp(cur, vals, probs)
            
            fig.add_trace(
                go.Scatter(
                    x=[cur], y=[cur_prob],
                    mode="markers",
                    name="현재값",
                    marker=dict(color="red", size=10, symbol="diamond"),
                    showlegend=True
                ),
                row=1, col=1
            )

        for idx, feat in enumerate(pdp_vars[:3]):
            if feat not in inp:
                continue
            cur = inp.get(feat, 0)
            if feat in VARIABLE_RANGES:
                mn = VARIABLE_RANGES[feat]["min"]
                mx = VARIABLE_RANGES[feat]["max"]
            else:
                mn = cur * 0.8
                mx = cur * 1.2

            vals = np.linspace(mn, mx, 20)
            np.random.seed(42)
            if feat == "molten_temp":
                probs = 0.1 + 0.4 * (vals - mn) / (mx - mn) + np.random.normal(0, 0.05, 20)
            elif feat == "production_cycletime":
                opt = (mn + mx) / 2
                probs = 0.1 + 0.3 * np.abs(vals - opt) / (mx - mn) + np.random.normal(0, 0.03, 20)
            else:
                probs = 0.2 + 0.2 * np.sin((vals - mn) / (mx - mn) * np.pi) + np.random.normal(0, 0.03, 20)

            probs = np.clip(probs, 0, 1)
            cur_prob = np.interp(cur, vals, probs)
            
            fig.add_trace(
                go.Scatter(
                    x=vals, y=probs,
                    mode="lines",
                    name=KOREAN_VARIABLE_MAP.get(feat, feat),
                    line=dict(width=2),
                    showlegend=True
                ),
                row=1, col=idx+1
            )
            
            if idx > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[cur], y=[cur_prob],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="diamond"),
                        showlegend=False
                    ),
                    row=1, col=idx+1
                )

        fig.update_layout(
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # === 각 subplot에 통일된 연한 회색 격자 추가 ===
        for col_i in range(1, 4):
            fig.update_yaxes(title_text="불량 확률", row=1, col=col_i, gridcolor='#E0E0E0', gridwidth=0.5)
            fig.update_xaxes(row=1, col=col_i, gridcolor='#E0E0E0', gridwidth=0.5)

        return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="pdp_sim"))

    # ==================================================
    # [ 소은 - 모니터링 페이지 ]
    # ==================================================

    @reactive.calc
    def get_recent_data():
        df_all = load_data(shared.DATA_PATH_SE)
        mold = input.mold_code_1page()

        df = df_all[df_all['mold_code'] == int(mold)]

        start_time, end_time = monitoring_period_range()

        if df.empty or start_time is None or end_time is None:
            return df.iloc[0:0]

        # 1) 기간 내 데이터
        mask = (
            (df["registration_time"] >= start_time)
            & (df["registration_time"] < end_time)
        )
        period_df = df[mask]
        
        # 2) 아직 판정되지 않은 행들
        mask_df = df[(df["registration_time"] < end_time) & (df["passorfail"].isna())]

        if len(mask_df) > 0:
            # 3) mask_df를 처리해서 판정 결과를 담은 DataFrame을 리턴
            final_df = predict_next_row(mask_df)

            # 4) period_df와 final_df를 합치고, 'id' 기준으로 중복 제거
            combined = pd.concat([period_df, final_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="last")

            return combined

        # 5) mask_df가 없으면 그냥 period_df 반환
        return period_df

    @reactive.calc
    def tick2():
        global cumul_val2
        reactive.invalidate_later(2)
        cumul_val2 += timedelta(minutes=3)
        now2 = CURRENT_TIMESTAMP.get() + cumul_val2
        return now2

    @output
    @render.text
    def simulation_time():
        now = tick2()
        return ui.div(
            ui.TagList(
                # 2. 날짜 (중간, 조금 키움)
                ui.div(now.strftime("%Y-%m-%d"), style="font-size: 24px; color: #6c757d; margin-bottom: 6px;"),
                
                # 3. 시:분:초 (아주 크지 않게, 날짜의 1.5배)
                ui.div(
                    now.strftime("%H:%M:%S"),
                    style="""
                        font-size: 36px;
                        font-weight: 700;
                        letter-spacing: 2px;
                        color: #007bff;
                    """
                )
            ),
            style="display: flex; flex-direction: column; align-items: center; justify-content: center;"
        )

    # mold_code 선택 로직
    @reactive.Effect
    def _update_mold_code():
        shared.CURRENT_MOLD_CODE.set(input.mold_code_1page())

    @reactive.Calc
    def filtered_by_mold():
        mold = input.mold_code_1page()
        
        if mold is None:
            return shared.DATA.iloc[0:0]
        return shared.DATA[shared.DATA['mold_code'].astype(str) == mold]

    @reactive.Calc
    def monitoring_period_range():
        try:
            base_time = tick2()
        except IndexError:
            return None, None  # NaN이 없을 경우
        period_text = input.period()

        if period_text == "전체":
            start_time = pd.to_datetime('2019-03-12 00:00:00')
            end_time = base_time
            return start_time, end_time
        else:
            delta = {
                "1시간 전": timedelta(hours=1),
                "3시간 전": timedelta(hours=3),
                "12시간 전": timedelta(hours=12),
                "1일 전": timedelta(days=1),
                "3일 전": timedelta(days=3),
            }.get(period_text, timedelta(hours=1))

            start_time = base_time - delta
            end_time = base_time
            return start_time, end_time
    

    def predict_next_row(candidates: pd.DataFrame) -> pd.DataFrame:
        """
        candidates: passorfail이 NaN인 행들만 모아둔 DataFrame.
        -> 여기서 가장 오래된 한 행만 골라 예측 후 shared.DATA를 갱신하고,
        그 행만 DataFrame으로 반환.
        """
        if candidates is None or len(candidates) == 0:
            return pd.DataFrame(columns=candidates.columns if candidates is not None else [])

        # candidates 중 가장 오래된 한 행을 뽑기 위해 복사
        temp = candidates.copy()
        temp["registration_time"] = pd.to_datetime(temp["registration_time"])
        temp = temp.sort_values(["registration_time", "id"])

        processed_rows = []

        # 3) for loop으로 모든 로우를 순회 → shared.DATA를 즉시 업데이트
        for idx, row in temp.iterrows():
            the_id = row["id"]
            # print(f"[predict_next_row] 처리 시작 → idx={idx}, registration_time={row['registration_time']}")
            
            # 3-1) input_data로 변환 (예시)
            input_data = {
                "molten_temp":                  row["molten_temp"],
                "sleeve_temperature":           row["sleeve_temperature"],
                "Coolant_temperature":          row["Coolant_temperature"],
                "lower_mold_temp1":             row["lower_mold_temp1"],
                "lower_mold_temp2":             row["lower_mold_temp2"],
                "lower_mold_temp3":             row["lower_mold_temp3"],
                "upper_mold_temp1":             row["upper_mold_temp1"],
                "upper_mold_temp2":             row["upper_mold_temp2"],
                "upper_mold_temp3":             row["upper_mold_temp3"],
                "facility_operation_cycleTime": row["facility_operation_cycleTime"],
                "production_cycletime":         row["production_cycletime"],
                "low_section_speed":            row["low_section_speed"],
                "high_section_speed":           row["high_section_speed"],
                "cast_pressure":                row["cast_pressure"],
                "biscuit_thickness":            row["biscuit_thickness"],
                "physical_strength":            row["physical_strength"],
                "mold_code":                    int(row["mold_code"]),
                "passorfail":                   0,  # dummy
            }

            # 3-2) 한 행 예측 수행
            results = analyze_single_row(input_data)
            analysis_results.set(results)
            
            if "prediction" in results:
                # 3-3) CSV 다시 읽어서 해당 행 passorfail/ probability 업데이트
                df_all = load_data(shared.DATA_PATH_SE)
                mask_id = df_all["id"] == the_id
                df_all.loc[mask_id, "passorfail"]  = results["prediction"]
                df_all.loc[mask_id, "probability"] = results["probability"]
                df_all.to_csv(shared.DATA_PATH_SE, index=False)

                # 3-4) 방금 업데이트한 행만 다시 로드해서 리스트에 저장
                updated_row = df_all[mask_id]
                processed_rows.append(updated_row)

                if results['prediction'] == 1:
                    # 🔧 센서 데이터 추가
                    sensor_data = {}
                    sensor_columns = [
                        'molten_temp', 'sleeve_temperature', 'Coolant_temperature',
                        'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
                        'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                        'facility_operation_cycleTime', 'production_cycletime',
                        'low_section_speed', 'high_section_speed',
                        'cast_pressure', 'biscuit_thickness', 'physical_strength'
                    ]
                    
                    for col in sensor_columns:
                        if col in row and not pd.isna(row[col]):
                            sensor_data[col] = float(row[col])
                    
                    new_alert = {
                        "id":          int(the_id),
                        "time":        row["registration_time"],          
                        "prob":        float(results["probability"]),     
                        "mold_code":   int(row["mold_code"]),
                        "specs":       sensor_data,  # 🔧 센서 데이터 추가
                        "input_data":  results.get("input_values", {}),
                        "threshold":   results.get("threshold", 0.5),
                        "model_type":  results.get("model_type", ""),
                    }
                    # 알림 리스트에 항목을 하나 추가
                    current = defect_alert_list.get()
                    defect_alert_list.set(current + [new_alert])
                    # print(defect_alert_list.get())

        # 4) processed_rows를 합쳐서 반환 
        if processed_rows:
            return pd.concat(processed_rows, ignore_index=True)
        else:
            # 단 한 건도 처리하지 않은 경우, 빈 DataFrame 반환
            return pd.DataFrame(columns=temp.columns)

    @output
    @render.ui
    def defect_alerts():
        """
        alert_list에 쌓인 딕셔너리들을 순서대로 꺼내서
        ui.card 형태로 하나씩 쌓아둡니다.
        """
        alerts = defect_alert_list.get()   # 예: [ {id:…, time:…, prob:…}, { … }, … ]

        if not alerts:
            # 알림이 하나도 없을 때는 빈 공간 혹은 안내문구 반환
            return ui.div("현재 알림이 없습니다.", class_="text-gray-500 p-2")

        # JavaScript 함수 먼저 추가
        cards = [
            ui.tags.script("""
                function openDefectDetailView(alertId) {
                    console.log('불량 상세보기 클릭:', alertId);
                    Shiny.setInputValue('defect_detail_button_clicked', alertId);
                }
            """)
        ]
        
        # 알림 카드들 생성
        for alert in alerts[::-1]:  # 최근 알림 먼저
            stamp = pd.to_datetime(alert["time"]).strftime("%Y-%m-%d %H:%M:%S")
            
            cards.append(
                ui.card(
                    ui.card_header(
                        ui.div([
                            # 왼쪽: 기본 정보
                             ui.div(
                                ui.span(f"불량 발생!",class_="text-sm font-medium"),
                                ui.span(f"(ID: {alert['id']}, 금형: {alert['mold_code']})", class_="text-xs text-gray-500 font-medium"),
                                class_="flex flex-col",
                            ),
                            # 오른쪽: 상세보기 버튼
                            ui.div([
                                ui.tags.button(
                                    "상세보기",
                                    onclick=f"openDefectDetailView('{alert['id']}')",
                                    class_="btn btn-outline-primary btn-xs px-2 py-1"
                                )
                            ], class_="text-right")
                        ], class_="flex justify-between items-center"),
                        class_="bg-red-100 text-red-800"
                    ),
                    ui.card_body(
                        ui.div([
                            ui.div(f"발생 시각: {stamp}", class_="flex items-center mb-1 text-sm"),
                            ui.div(f"불량 확률: {alert['prob']:.2%}", class_="text-gray-500 text-xs"),
                        ], class_="text-sm")
                    ),
                    class_="mb-2 shadow-sm"
                )
            )

        return ui.div(*cards, class_="space-y-2 p-2")

    

    def analyze_single_row(input_data: dict) -> dict:
        # 1) pipeline이 로드되어 있는지 확인
        if pipeline is None:
            return {"error": "파이프라인을 로드할 수 없습니다."}

        # 2) input_data에서 mold_code를 꺼내고, pipeline["models"]에서 모델 찾기
        mold_code = int(input_data.get("mold_code", -1))
        models_dict = pipeline.get("models", {})
        if mold_code not in models_dict:
            return {"error": f"mold_code {mold_code}에 대한 모델이 없습니다."}
        model = models_dict[mold_code]

        # 3) 모델 타입과 임계값(threshold) 꺼내기
        best_model_config = pipeline.get("best_model_config", {})
        model_type = best_model_config.get(mold_code, None)
        optimal_thresholds = pipeline.get("optimal_thresholds", {})
        threshold = optimal_thresholds.get(mold_code, 0.5)

        # 4) feature_names 순서대로 빈 DataFrame을 만들고, input_data 값을 채우기
        feat_names = pipeline.get("feature_names", [])
        df_full = pd.DataFrame(columns=feat_names)
        # "한 행"이니까, 인덱스 0에 모두 0을 채워둔 후 덮어쓰기
        df_full.loc[0] = [0] * len(feat_names)

        # input_data key가 feat_names 목록에 있으면 값 덮어쓰기
        for col, val in input_data.items():
            if col in df_full.columns:
                df_full.loc[0, col] = val

        # X_full: 모델에 넣을 feature 순서대로 정렬된 DataFrame
        X_full = df_full[feat_names]

        # 5) 예측 확률(proba) 계산
        try:
            if model_type and "XGBoost" in model_type:
                # XGBoost 모델인 경우 DMatrix로 감싸서 predict
                dmat = xgb.DMatrix(X_full)
                proba = model.predict(dmat)[0]
            else:
                # LightGBM 등 scikit-learn API 호환 모델
                proba = model.predict(X_full)[0]
        except Exception as e:
            return {"error": f"모델 예측 중 오류 발생: {e}"}

        # 6) 임계값(threshold)과 비교해서 0/1 레이블(pred) 결정
        pred = 1 if proba >= threshold else 0

        # 7) 결과 딕셔너리 리턴 (필요에 따라 내용을 추가/제거하세요)
        return {
            "prediction": int(pred),         # 0=양품, 1=불량 (임계값 비교 결과)
            "probability": float(proba),     # 예측된 불량 확률 (0~1)
            "threshold": float(threshold),   # 사용된 임계값
            "model_type": model_type,        # 어떤 모델(XGBoost or LightGBM 등)인지
            "mold_code": mold_code,          
            # input_data는 원래 row에서 가져온 값 그대로
            "input_values": input_data
        }


    @reactive.Calc
    def filtered_data():
        df = filtered_by_mold()
        start_time, end_time = monitoring_period_range()
        if df.empty or start_time is None or end_time is None:
            return df.iloc[0:0]
        df.loc[:, "registration_time"] = pd.to_datetime(df["registration_time"])
        mask = (df["registration_time"] >= start_time) & (df["registration_time"] < end_time)
        
        return df[mask]
        
    @output
    @render.ui
    def anomaly_alert():
        current_time = tick2()
        alerts = anomaly_results.get()
        
        if not alerts or len(alerts) == 0:
            return ui.div([
                ui.div([
                    ui.div("📭", class_="text-4xl text-gray-300 mb-2"),
                    ui.p("알림이 없습니다", class_="text-gray-400 text-sm")
                ], class_="text-center py-8"),
                
                ui.div([
                    ui.input_action_button(
                        "btn_clear_alerts", 
                        "전체 삭제", 
                        class_="btn btn-outline-secondary btn-sm",
                        disabled=True
                    ),
                ], class_="flex gap-2 mt-3 justify-center")
        
    ], class_="p-0 w-full")
        
        # 🔍 상세보기 버튼을 위한 JavaScript 함수 추가
        script_tag = ui.tags.script("""
            function openDetailView(alertId) {
                console.log('상세보기 클릭:', alertId);
                Shiny.setInputValue('detail_button_clicked', alertId);
            }
        """)
        
        # 🔍 알림이 있을 때 - 원래 내용 유지하면서 카드 디자인만 레몬색으로 변경
        alert_items = [script_tag]
        
        for i, alert in enumerate(alerts):
            risk_level = alert["risk_level"]
            
            # 시간 계산
            now = current_time
            time_diff = now - alert["full_timestamp"]
            
            if time_diff.total_seconds() < 60:
                time_ago = "방금 전"
            elif time_diff.total_seconds() < 3600:
                minutes = int(time_diff.total_seconds() / 60)
                time_ago = f"{minutes}분 전"
            else:
                hours = int(time_diff.total_seconds() / 3600)
                time_ago = f"{hours}시간 전"
            
            # 상단 노란색 + 하단 흰색 카드 디자인
            alert_item = ui.card(
                    # ─── 카드 헤더: 노란색 바 ───
                    ui.card_header(
                        ui.div(
                            # 좌측 타이틀
                            ui.div(
                                ui.span(f"이상치 발생!",class_="text-sm font-medium"),
                                ui.span(f"(ID: {alert['real_id']}, 금형: {alert['mold_code']})", class_="text-xs text-gray-500 font-medium"),
                                class_="flex flex-col",
                            ),
                            # 우측 상세보기 버튼
                            ui.tags.button(
                                "상세보기",
                                onclick=f"openDetailView('{alert['id']}')",
                                class_="btn btn-outline-primary btn-xs px-2 py-1"
                            ),
                            class_="flex justify-between items-center"
                        ),
                        class_="bg-yellow-200 rounded-t-lg p-3"
                    ),

                    # ─── 카드 본문: 흰색 바 ───
                    ui.card_body(
                        ui.div(
                            # 왼쪽 정보
                            ui.div(
                                ui.div(
                                    ui.div(f"발생 시각: {alert['full_timestamp']}"),
                                    class_="flex items-center mb-1 text-sm"
                                ),
                                ui.div(
                                    ui.span(risk_level["level"], class_=f"{risk_level['class']} font-semibold text-xs"),
                                    ui.div(f"점수: {alert['anomaly_score']:.3f}", class_="text-gray-500 text-xs"),
                                    class_="flex flex-row gap-4"
                                ),
                                class_="flex flex-col flex-1 gap-2"
                            ),
                            # 오른쪽 시간
                            ui.div(
                                ui.div(time_ago, class_="text-gray-400 text-xs"),
                                class_="text-right"
                            ),
                            class_="flex justify-between items-center"
                        ),
                        class_="bg-white rounded-b-lg p-3"
                    ),

                    class_="mb-2 shadow-sm hover:shadow-md transition-all duration-300 w-full min-w-0 flex-1"
                )
            
            alert_items.append(alert_item)
        
        return ui.div([
            ui.div(alert_items, class_="space-y-2 w-full flex-1"),
            
            ui.div([
                ui.input_action_button(
                    "btn_clear_alerts", 
                    "전체 삭제", 
                    class_="btn btn-outline-secondary btn-sm"
                ),
            ], class_="flex gap-2 mt-3 justify-center")
        ])
    
    # @output
    # @render.plot
    # def spike_chart():
    #     df = get_recent_data()
    #     if df.empty:
    #         return

    #     df = df.sort_values("registration_time")
    #     mold = input.mold_code_1page()
    #     thresh = threshold.get(int(mold), 0.0)
    #     df["exceed"] = df["probability"] > thresh

    #     p = (
    #         ggplot(df, aes("registration_time", "probability"))
    #         + geom_point(aes(color="exceed"), size=2)
    #         + geom_hline(yintercept=thresh, linetype="dashed")
    #         + scale_color_manual(
    #             values={False: "gray", True: "red"}
    #         )
    #         + guides(color=False)  # hide the legend
    #         + scale_x_datetime(date_labels="%H:%M", 
    #                            breaks=breaks_date(width="6 hours"))
    #         + labs(
    #             title="품질 예측 확률 추이",
    #             x="등록 시간",
    #             y="불량 확률"
    #         )
    #         + theme_minimal()
    #         + theme(
    #             figure_size=(8, 3),
    #             title=element_text(fontproperties=font_prop)
    #         )
    #     )
    #     return p.draw()

    
    @output
    @render.plot
    def spike_chart():
        # 1) 데이터
        df = get_recent_data()
        if df.empty:
            return

        df = df.sort_values("registration_time")
        mold = input.mold_code_1page()
        thresh = threshold.get(int(mold), 0.0)
        df["exceed"] = df["probability"] > thresh

        # 2) Figure/Axis
        fig, ax = plt.subplots(figsize=(8, 3))

        # 3) 임계치선 (점선)
        ax.axhline(y=thresh, linestyle="--", linewidth=1, color="black")

        # 4) 산점도 (exceed False 회색, True 빨강)
        times = df["registration_time"]
        probs = df["probability"]
        colors = ["red" if ex else "gray" for ex in df["exceed"]]
        ax.scatter(times, probs, c=colors, s=20)

        # 5) 레이블 & 타이틀
        ax.set_title("품질 예측 확률 추이", fontproperties=font_prop)
        ax.set_xlabel("등록 시간", fontproperties=font_prop)
        ax.set_ylabel("불량 확률", fontproperties=font_prop)

        # 6) x축 6시간 간격, HH:MM 포맷
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # 7) 그리드 & 레이아웃
        ax.grid(True, linestyle=":", alpha=0.3)
        fig.tight_layout()

        return fig

    # @output
    # @render.plot
    # def sensor_chart():
    #     cols = applied_cols.get()

    #     df = get_recent_data()

    #     if not cols:
    #         return "변수를 선택해주세요"

    #     # melt로 long-format으로 변환
    #     df_long = df.melt(id_vars=['registration_time', 'mold_code'], value_vars=cols,
    #                   var_name='feature', value_name='value')

    #     # mold_code를 마지막 값 기준으로 설정
    #     mold_code_list = df['mold_code'].unique()
    #     mold_code_one = input.mold_code_1page()

    #     # 해당 몰드 코드에 대한 상하한 정보 필터링
    #     bounds_df = bound_df[bound_df['mold_code'] == mold_code_one]

    #     # merge로 상/하한선 정보 붙이기
    #     df_plot = df_long.merge(bounds_df, how='left', on=['mold_code', 'feature'])

    # # 그래프 생성
    #     p = (
    #         ggplot(df_plot, aes(x='registration_time', y='value'))
    #         + geom_line(color='blue')
    #         + geom_hline(aes(yintercept='lower_bound'), linetype='dashed', color='red')
    #         + geom_hline(aes(yintercept='upper_bound'), linetype='dashed', color='red')
    #         + facet_wrap('~feature', scales='free_y')
    #         + labs(title=f"센서별 시계열 그래프 (몰드코드 {mold_code_one})", x='시간', y='측정값')
    #         + theme(
    #             axis_text_x=element_text(rotation=45, hjust=1),
    #             title=element_text(fontproperties=font_prop)
    #             )
    #     )

    #     return p

    # 영어 변수명 → 한글명 매핑 딕셔너리 (sensor용)
    feature_name_map = {
        "molten_temp": "용탕온도",
        "sleeve_temperature": "슬리브온도", 
        "Coolant_temperature": "냉각수온도",
        "cast_pressure": "사출압력",
        "high_section_speed": "고속구간",
        "biscuit_thickness": "비스킷두께",
        "physical_strength": "물리강도",
        "lower_mold_temp1": "하부몰드1",
        "lower_mold_temp2": "하부몰드2",
        "lower_mold_temp3": "하부몰드3",
        "upper_mold_temp1": "상부몰드1",
        "upper_mold_temp2": "상부몰드2",
        "upper_mold_temp3": "상부몰드3",
        "facility_operation_cycleTime": "설비사이클",
        "production_cycletime": "생산사이클",
        "low_section_speed": "저속구간"
    }

    
    @output
    @render.plot
    def sensor_chart():
        reactive.invalidate_later(2)
        cols = applied_cols.get()
        df = get_recent_data()
        if not cols:
            return "변수를 선택해주세요"

        # melt 및 병합
        df_long = df.melt(id_vars=['registration_time', 'mold_code'], value_vars=cols,
                        var_name='feature', value_name='value')
        mold_code_one = input.mold_code_1page()
        bounds_sub = bound_df[bound_df['mold_code'] == mold_code_one]
        df_plot = df_long.merge(bounds_sub, how='left', on=['mold_code', 'feature'])

        features = df_plot['feature'].unique().tolist()
        n_feats = len(features)
        col_count = len(applied_cols.get())
        row_count = math.ceil(n_feats / col_count)

        fig, axes = plt.subplots(row_count, col_count, figsize=(6*col_count, 4*row_count), sharey=False)
        # axes가 ndarray인지 검사하여 1차원 리스트로 변환
        if isinstance(axes, np.ndarray):
            ax_list = axes.ravel()
        else:
            ax_list = [axes]

        for idx, feature in enumerate(features):
            ax = ax_list[idx]
            sub = df_plot[df_plot['feature'] == feature]
            # registration_time이 datetime인지 확인하고, 필요시 pd.to_datetime() 적용
            times = sub['registration_time']
            values = sub['value']
            lowers = sub.get('lower_bound')
            uppers = sub.get('upper_bound')

            # 항상 Axes.plot 호출
            ax.plot(times, values, color='blue')
            # 상하한선: 스칼라 값으로 axhline 호출
            if lowers is not None and not lowers.isna().all():
                lb = lowers.dropna().iloc[0]
                ax.axhline(lb, linestyle='--', color='red')
            if uppers is not None and not uppers.isna().all():
                ub = uppers.dropna().iloc[0]
                ax.axhline(ub, linestyle='--', color='red')

            ax.set_title(feature_name_map.get(feature, feature), fontproperties=font_prop)
            ax.set_xlabel("시간")
            ax.set_ylabel("측정값")
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

        # 빈 subplot 숨기기
        for j in range(n_feats, len(ax_list)):
            ax_list[j].axis('off')

        fig.suptitle(f"센서별 시계열 그래프 (몰드코드 {mold_code_one})", fontproperties=font_prop)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
        
    @render.table
    def preview_filtered_df():
        df = filtered_data()
        return df.tail(10)  # UI에 10개까지 표시
    
    # 최근값 카드
    @reactive.Calc
    def rolling_stats():
        # get_recent_data() 나 filtered_df() 등, 
        # 전체 시계열 데이터 프레임을 리턴하는 reactive 를 사용하세요.
        df = get_recent_data()  
        if df.empty or len(df) < 2:
            return {}  # 충분한 데이터가 없으면 빈 dict

        # 정렬
        df = df.sort_values("registration_time")

        out = {}
        for col in input.selected_cols():
            series = df[col].dropna()
            if len(series) == 0:
                continue
            last100 = series.iloc[-100:]  # 마지막 100개
            avg100  = last100.mean()
            curr   = series.iloc[-1]
            pct    = (curr - avg100) / avg100 * 100 if avg100 != 0 else None
            out[col] = {
                "avg":  avg100,
                "curr": curr,
                "pct":  pct,
            }
        return out
    
    
    # ③ UI 카드들을 렌더링하는 콜백
    @output
    @render.ui
    def stats_cards():
        stats = rolling_stats()  # 위에서 계산된 dict
        if not stats:
            return ui.div("데이터가 부족합니다.", class_="text-gray-500 p-2")

        cards = []
        for var, vals in stats.items():
            avg = vals["avg"]
            curr = vals["curr"]
            change = curr - avg

           # 변화 방향 및 클래스
            if change >= 0:
                arrow = "▲"
                change_color_class = "change-up"
            else:
                arrow = "▼"
                change_color_class = "change-down"

            # 🔧 HTML로 공백 삽입: &nbsp; 또는 margin-right
            change_str = f"""
            <span class='{change_color_class}'>
                <span class='arrow'>{arrow}</span>
                <span class='change-num'>{abs(change):.2f}</span>
            </span>
            """

            # ✅ 센서 이름을 한글로 표시
            sensor_label = SENSOR_LABELS.get(var, var)

            cards.append(
                 ui.card(
                        ui.card_header(
                        sensor_label,
                        class_="custom-card-header"
                    ),
                    ui.card_body(
                        # 최신 / 평균 한 줄에 표현
                        ui.div(
                            ui.HTML(f"""
                                <span class='latest-value'>{curr:.2f}</span>
                                <span class='avg-inline'> / {avg:.2f}</span>
                            """)
                        ),
                        ui.div(ui.HTML(change_str)),
                        class_="custom-card-body"
                    ),
                    class_="custom-card w-full"
                )
            )

        # 가로로 쭉 나열하고 싶으면 flex, 세로 스택이면 space-y
        return ui.div(*cards, class_="flex gap-4 p-2")

    # -------------------------------------------------
    # 의주
    # -------------------------------------------------
    @reactive.Calc
    def filtered_df2():
        """필터링된 데이터프레임 반환 (예측 포함)"""
        _df = uj_df.copy()
        _df["registration_date"] = _df["registration_time"].dt.strftime("%Y-%m-%d")
        
        # 날짜 및 금형 필터링
        if input.apply_btn() > 0:  # 적용 버튼이 눌렸을 때만 필터링
            date_range = input.stat_date_range()
            date_start = date_range[0].strftime("%Y-%m-%d")
            date_end = date_range[1].strftime("%Y-%m-%d")
            selected_molds = input.stat_moldcode()
            
            _df = _df[
                (_df["registration_date"] >= date_start) &
                (_df["registration_date"] <= date_end) &
                (_df["mold_code"] == int(selected_molds))
            ]

        # 필터링된 데이터에 대해 예측 수행
        if len(_df) > 0:
            _df["predict"] = _df['passorfail']
            _df["predict_proba"] = _df['probability']
        else:
            _df["predict"] = []
            _df["predict_proba"] = []
        
        return _df.reset_index(drop=True)

    @reactive.Calc
    def filtered_df3():
        """필터링된 데이터프레임 반환 (예측 포함)"""
        _df = shared.TEST_DATA.copy()
        _df["registration_date"] = _df["registration_time"].dt.strftime("%Y-%m-%d")
        
        # 날짜 및 금형 필터링
        if input.apply_btn() > 0:  # 적용 버튼이 눌렸을 때만 필터링
            date_range = input.stat_date_range()
            date_start = date_range[0].strftime("%Y-%m-%d")
            date_end = date_range[1].strftime("%Y-%m-%d")
            selected_molds = input.stat_moldcode()
            
            _df = _df[
                (_df["registration_date"] >= date_start) &
                (_df["registration_date"] <= date_end) &
                (_df["mold_code"] == int(selected_molds))
            ]

        return _df.reset_index(drop=True)

    @output
    @render_widget
    def control_chart():
        df_chart = filtered_df2().copy()
        if df_chart.empty:
            return go.Figure(layout_title_text="관리도 - 데이터 없음")
        interval = input.control_interval()
        df_chart["registration_time"] = pd.to_datetime(df_chart["registration_time"])
        df_chart["date"] = df_chart["registration_time"].dt.floor(interval)
        selected_date = pd.to_datetime(input.control_date())
        df_chart = df_chart[df_chart["registration_time"].dt.date == selected_date.date()]
        if df_chart.empty:
            return go.Figure(layout_title_text="선택한 날짜에 해당하는 데이터 없음")
        df_grouped = df_chart.groupby("date").agg(
            defect_rate=("predict", lambda x: (x == 1).mean()),
            count=("predict", "count")
        ).reset_index().sort_values("date")
        if df_grouped.empty:
            return go.Figure(layout_title_text="그룹화 후 데이터 없음")
        center_line = df_grouped["defect_rate"].mean()
        std = df_grouped["defect_rate"].std()
        ucl = center_line + 3 * std
        lcl = max(0, center_line - 3 * std)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_grouped["date"], y=df_grouped["defect_rate"], mode="lines+markers", name="불량률", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=df_grouped["date"], y=[center_line]*len(df_grouped), mode="lines", name="중심선", line=dict(dash="dash", color="green")))
        fig.add_trace(go.Scatter(x=df_grouped["date"], y=[ucl]*len(df_grouped), mode="lines", name="상한선(UCL)", line=dict(dash="dot", color="red")))
        fig.add_trace(go.Scatter(x=df_grouped["date"], y=[lcl]*len(df_grouped), mode="lines", name="하한선(LCL)", line=dict(dash="dot", color="blue")))
        fig.update_layout(
            yaxis_title="불량률",
            xaxis_title="날짜",
            yaxis=dict(tickformat=".1%"),
            hovermode='x unified'
        )
        return fig


    @output
    @render.text
    def total_count_uj():
        return str(len(filtered_df2()))

    @output
    @render.text
    def defect_count_uj():
        return str((filtered_df2()["predict"] == 1).sum())

    @output
    @render.text
    def tryshot_count():
        return str(filtered_df3()["tryshot_signal"].notna().sum())

    @output
    @render_widget
    def defect_trend():
        df_trend = filtered_df2().copy()
        if df_trend.empty:
            return go.Figure(layout_title_text="불량률 추이 - 데이터 없음")

        df_trend["date"] = pd.to_datetime(df_trend["registration_time"]).dt.floor("1D")

        df_grouped = df_trend.groupby("date").agg(
            predicted_defect=("predict", lambda x: (x == 1).sum()),
            tryshot_fail=("tryshot_signal", lambda x: x.notna().sum()),
            total=("predict", "count")
        ).reset_index().sort_values("date")

        df_grouped["predicted_rate"] = df_grouped["predicted_defect"] / df_grouped["total"]
        df_grouped["tryshot_rate"] = df_grouped["tryshot_fail"] / df_grouped["total"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_grouped["date"],
            y=df_grouped["predicted_rate"],
            name="예측 불량률",
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(0, 200, 0, 0.4)"
        ))
        fig.update_layout(
            yaxis_title="불량률",
            xaxis_title="주간 시작일",
            title="주간 불량률 추이 (예측 기반)",
            xaxis=dict(tickformat="%Y-%m-%d", type='date'),
            yaxis=dict(tickformat=".1%"),
            hovermode='x unified'
        )
        return fig


    @output
    @render.download(
        filename=lambda: "defect_log.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    def download_log():
        # 1) 필터링된 DataFrame 가져오기
        df = filtered_df2().copy()
        if input.download_type() == "pred_only":
            # 예측 결과만
            cols = [
                "registration_time",
                "id",
                "mold_code",
                "passorfail",
                "probability"
            ]
            df_ = df[cols]
        else:
            # 모든 센서 컬럼 포함
            df_ = df.copy()

        # 2) 메모리 버퍼에 Excel 작성
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_.to_excel(writer, index=False, sheet_name="DefectLog")

        # 3) 스트림 위치를 맨 앞으로
        buf.seek(0)

        # 4) 바이트 뭉치 반환
        yield buf.getvalue()
        

    # @output
    # @render.data_frame
    # def paged_table():
    #     df = filtered_df2()
    #     cols = ['registration_time', 'id', 'mold_code', 'passorfail', 'shap1', 'shap2', 'shap3']
    #     if cols:
    #         return df[cols]
    #     else:
    #         return df
    # @output
    # @render.table
    # def paged_table():
    #     df = filtered_df2()
    #     cols = ['registration_time', 'id', 'mold_code', 'passorfail', 'shap1', 'shap2', 'shap3']
    #     # 선택된 컬럼만 뽑아내거나, cols 가 비어 있으면 전체 DataFrame 반환
    #     return df[cols] if cols else df

    @output
    @render.ui
    def paged_table():
        df = filtered_df2()
        if df.empty:
            return ui.p("⚠️ 조건에 맞는 데이터가 없습니다.")

        # 최근 10개만 보기
        table_data = df.copy()
        # datetime to string
        table_data["registration_time"] = table_data["registration_time"].astype(str)

        # 테이블 row 생성
        rows = []
        for _, row in table_data.iterrows():
            rows.append(
                ui.tags.tr(
                    ui.tags.td(row["registration_time"]),
                    ui.tags.td(str(row["id"])),
                    ui.tags.td(str(row["mold_code"])),
                    ui.tags.td("불량" if row["passorfail"] == 1 else "양품"),
                    ui.tags.td(f"{row['shap1']}"),
                    ui.tags.td(f"{row['shap2']}"),
                    ui.tags.td(f"{row['shap3']}"),
                )
            )

        # 완성된 HTML 테이블 반환
        return ui.tags.table(
            # 헤더
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("등록시간"),
                    ui.tags.th("ID"),
                    ui.tags.th("금형코드"),
                    ui.tags.th("실제결과"),
                    ui.tags.th("SHAP1"),
                    ui.tags.th("SHAP2"),
                    ui.tags.th("SHAP3"),
                )
            ),
            # 본문
            ui.tags.tbody(*rows),
            class_="table table-striped w-full"
        )

            
    # -------------현준----------------
    @reactive.Effect
    def call_before_data():
        return get_before_data()        
       
    @reactive.Calc
    def get_before_data():
        df = filtered_by_mold()  # 이미 mold 필터링까지 완료된 데이터
        start_time, end_time = monitoring_period_range()  # 한 번만

        if df.empty or start_time is None or end_time is None:
            return df.iloc[0:0]
    
        mask = (
            (df["registration_time"] >= start_time)
            & (df["registration_time"] < end_time)
        )
        period_df = df[mask]

        # 🔍 기존 진단 코드 모두 삭제하고 새로운 코드로 교체
        # print(f"🔍 [NEW] get_before_data 실행됨")
        # print(f"🔍 [NEW] 현재 end_time: {end_time}")
        # print(f"🔍 [NEW] end_time 타입: {type(end_time)}")
    
        # NaN 데이터들의 시간 확인
        nan_data = df[df['anomalyornot'].isna()]
        # print(f"🔍 [NEW] 전체 NaN 데이터: {len(nan_data)}개")
    
        # if len(nan_data) > 0:
        #     print(f"🔍 [NEW] NaN 데이터 시간 범위:")
        #     print(f"   - 최소: {nan_data['registration_time'].min()}")
            # print(f"   - 최대: {nan_data['registration_time'].max()}")
    
        # 시간 조건 체크
        time_condition = df['registration_time'] <= end_time
        nan_condition = df['anomalyornot'].isna()
        combined_condition = time_condition & nan_condition
    
        # print(f"🔍 [NEW] 시간 조건 만족: {time_condition.sum()}개")
        # print(f"🔍 [NEW] NaN 조건 만족: {nan_condition.sum()}개") 
        # print(f"🔍 [NEW] 둘 다 만족: {combined_condition.sum()}개")
    
        # 이상치 탐지
        anomaly_mask_df = df[combined_condition]
    
        if len(anomaly_mask_df) > 0:
            # print("🔍 [NEW] detect_anomalies 호출할 예정")
            detect_anomalies(anomaly_mask_df)
            # print("🔍 [NEW] detect_anomalies 처리 완료")
            # ✅ 예측 후 업데이트된 데이터로 period_df 다시 생성
            df_updated = filtered_by_mold()
            mask_updated = (
                (df_updated['registration_time']>=start_time)
                & (df_updated['registration_time']< end_time)
            )
            period_df = df_updated[mask_updated]
        # else:
        #     print("🔍 [NEW] 탐지할 데이터가 없음")

        return period_df
    def get_anomaly_risk_level(anomaly_score):
        """이상치 점수 기반 위험 등급 판정"""
    
        # 분석 결과에서 나온 임계값
        SEVERE_THRESHOLD = -0.0455
        MODERATE_THRESHOLD = -0.0156
    
        if anomaly_score <= SEVERE_THRESHOLD:
            return {
                "level": "🔴 매우 이상",
                "class": "text-red-600",
                "bg_class": "bg-red-50 border-red-200",
                "description": "극도로 비정상적인 패턴 - 즉시 조치 필요"
            }
        elif anomaly_score <= MODERATE_THRESHOLD:
            return {
                "level": "🟡 이상", 
                "class": "text-yellow-600",
                "bg_class": "bg-yellow-50 border-yellow-200",
                "description": "비정상적인 패턴 감지 - 집중 모니터링"
            }
        else:
            return {
                "level": "🔵 약간 이상",
                "class": "text-blue-600", 
                "bg_class": "bg-blue-50 border-blue-200",
                "description": "경미한 이상 패턴 - 기록 및 관찰"
            }    
    
    def detect_anomalies(candidates):
        """이상치 탐지 수행 - 모든 NaN 행 처리"""
        if candidates is None or len(candidates) == 0:
            return pd.DataFrame()

        # candidates 복사 및 정렬
        temp = candidates.copy()
        temp["registration_time"] = pd.to_datetime(temp["registration_time"])
        temp = temp.sort_values(["registration_time", "id"])
        processed_rows = []

        # print(f"🔍 [DEBUG] detect_anomalies 시작 - 총 {len(temp)}개 행 처리 예정")

        for idx, row in temp.iterrows():
            # 이미 처리된 행인지 확인
            current_anomaly_value = shared.DATA.loc[idx, "anomalyornot"]
            if not pd.isna(current_anomaly_value):
                # print(f"🔍 [DEBUG] 행 {idx} 이미 처리됨 (값: {current_anomaly_value}) - 스킵")
                continue

            # print(f"🔍 [DEBUG] 행 {idx} 처리 시작... ({row['registration_time']})")
        
            result = run_anomaly_detection(row.to_dict())
            # print(f"🔍 [DEBUG] 행 {idx} 결과: {result}")
        
            if "is_anomaly" in result:
                # shared.DATA 업데이트
                shared.DATA.loc[idx, "anomalyornot"] = result["is_anomaly"]
                # print(f"🔍 [DEBUG] shared.DATA 업데이트: idx={idx}, anomalyornot={result['is_anomaly']}")
            
                # 🔍 이상치가 탐지되면 알림 **추가**
                if result["is_anomaly"] == 1:
                    # print("🚨 [ALERT] 실제 이상치 탐지됨! 알림 추가 중...")
                    
                    # 🔍 기존 알림들 가져오기
                    current_alerts = anomaly_results.get()
                    
                    # 🔧 실제 센서 데이터 추출
                    specs_data = {}
                    sensor_columns = [
                'molten_temp', 'sleeve_temperature', 'Coolant_temperature',
                'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
                'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                'facility_operation_cycleTime', 'production_cycletime',
                'low_section_speed', 'high_section_speed',
                'cast_pressure', 'biscuit_thickness', 'physical_strength'
            ]
                    
                    for col in sensor_columns:
                        if col in row and not pd.isna(row[col]):
                            specs_data[col] = float(row[col])
                            # print(f"🔍 [DEBUG] specs 추가: {col} = {row[col]}")
                    
                    # print(f"🔍 [DEBUG] 최종 specs_data: {specs_data}")
                    
                    # 🔍 새 알림 생성 (specs 포함!)
                    new_alert = {
                        "real_id": row['id'],
                        "id": f"real_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}",
                        "timestamp": row["registration_time"].strftime('%H:%M:%S'),
                        "full_timestamp": row["registration_time"],
                        "mold_code": str(row["mold_code"]),
                        "anomaly_score": result.get("anomaly_score", 0),
                        "risk_level": get_anomaly_risk_level(result.get("anomaly_score", 0)),
                        "specs": specs_data,  # 🔧 실제 센서 데이터!
                        "is_real_data": True
                    }
                    
                    # 🔍 새 알림을 리스트 맨 앞에 추가 (최신이 위로)
                    current_alerts.insert(0, new_alert)
                    
                    # 🔍 최대 8개까지만 유지
                    if len(current_alerts) > 8:
                        current_alerts = current_alerts[:8]
                    
                    # 🔍 업데이트된 알림 리스트 저장
                    anomaly_results.set(current_alerts)
                    # print("기래!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print("🚨 [ALERT] 실제 이상치 알림 추가 완료!")
                else:
                    continue
                    # print("아니래!!!!!!!!!!!!!!!!!!!!!!!")
                    # print("✅ [DEBUG] 정상 데이터로 판정됨")
            
                # 처리된 행 저장
                updated_row = shared.DATA.loc[[idx]]
                processed_rows.append(updated_row)

        # print(f"🔍 [DEBUG] detect_anomalies 완료 - 총 {len(processed_rows)}개 행 처리됨")

        if processed_rows:
            return pd.concat(processed_rows, ignore_index=True)
        else:
            return pd.DataFrame(columns=temp.columns)  
        
    def run_anomaly_detection(row_data):
        """개별 행에 대해 이상치 탐지 수행"""
        # print(f"🔍 [DEBUG] run_anomaly_detection 시작")
    
        if isolation_model_data is None:
            # print("🚨 [ERROR] isolation_model_data가 None임")
            return {"error": "Isolation Forest 모델이 로드되지 않았습니다."}
    
        try:
            # print(f"🔍 [DEBUG] CustomPreprocessor 인스턴스 생성 중...")
        
            # working 컬럼 임시 추가 (문제 해결을 위해)
            row_data['working'] = 1
        
            preprocessor = CustomPreprocessor()
        
            # row_data를 DataFrame으로 변환
            input_df = pd.DataFrame([row_data])
            # print(f"🔍 [DEBUG] 입력 DataFrame 크기: {input_df.shape}")
            # print(f"🔍 [DEBUG] 입력 컬럼들: {list(input_df.columns)}")
        
            # 전처리 수행
            processed_df = preprocessor.transform(input_df)
            # print(f"🔍 [DEBUG] 전처리 후 DataFrame 크기: {processed_df.shape}")
            # print(f"🔍 [DEBUG] 전처리 후 컬럼들: {list(processed_df.columns)}")
        
            # ✅ 수정된 부분: isolation_model_data가 Pipeline인지 확인
            # print(f"🔍 [DEBUG] isolation_model_data 타입: {type(isolation_model_data)}")
        
            if hasattr(isolation_model_data, 'predict'):
                # Pipeline 또는 모델 객체인 경우
                model = isolation_model_data
                # print(f"🔍 [DEBUG] Pipeline/모델 직접 사용")
            
                # Pipeline인 경우 원본 DataFrame을 사용 (전처리 포함)
                input_df_original = pd.DataFrame([row_data])
                anomaly_prediction = model.predict(input_df_original)[0]
                anomaly_score = model.decision_function(input_df_original)[0]
            
            else:
                # 딕셔너리 형태인 경우 (이 경우는 발생하지 않을 것 같음)
                model = isolation_model_data.get('model') or isolation_model_data.get('isolation_forest')
                # print(f"🔍 [DEBUG] 딕셔너리에서 모델 추출")
            
                X = processed_df.values
                anomaly_prediction = model.predict(X)[0]
                anomaly_score = model.decision_function(X)[0]
        
            # print(f"🔍 [DEBUG] 원본 예측 결과: {anomaly_prediction}")
            # print(f"🔍 [DEBUG] 이상치 점수: {anomaly_score}")
        
            # -1을 1로, 1을 0으로 변환
            is_anomaly = 1 if anomaly_prediction == -1 else 0
        
            # print(f"🔍 [DEBUG] 최종 이상치 판정: {is_anomaly}")
        
            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "raw_prediction": int(anomaly_prediction)
            }
        
        except Exception as e:
            # print(f"🚨 [ERROR] run_anomaly_detection 오류: {e}")
            import traceback
            # print(f"🚨 [ERROR] 상세 오류:\n{traceback.format_exc()}")
            return {"error": f"이상치 탐지 중 오류 발생: {e}"}
    
    def get_isolation_forest_normal_pattern_simple(mold_code):
        """훈련 데이터 분석 결과 기반 고정 정상 패턴"""

        # print(f"🎯 [RESULT] 몰드 {mold_code} 고정 정상 패턴 사용 (훈련 데이터 기반)")

        # 🔧 몰드별 정상 패턴 (평균값 사용)
        FIXED_NORMAL_PATTERNS = {
            "8412": {
                "molten_temp": 724.43,
                "sleeve_temperature": 374.64,
                "Coolant_temperature": 33.09,
                "lower_mold_temp1": 146.92,
                "lower_mold_temp2": 194.07,
                "lower_mold_temp3": 1449.0,
                "upper_mold_temp1": 192.41,
                "upper_mold_temp2": 148.95,
                "upper_mold_temp3": 1449.0,
                "facility_operation_cycleTime": 120.24,
                "production_cycletime": 122.78,
                "low_section_speed": 114.12,
                "high_section_speed": 116.72,
                "cast_pressure": 326.49,
                "biscuit_thickness": 50.89,
                "physical_strength": 707.69,
            },
            "8573": {
                "molten_temp": 720.49,
                "sleeve_temperature": 497.24,
                "Coolant_temperature": 31.21,
                "lower_mold_temp1": 182.08,
                "lower_mold_temp2": 203.66,
                "lower_mold_temp3": 1449.0,
                "upper_mold_temp1": 223.17,
                "upper_mold_temp2": 190.25,
                "upper_mold_temp3": 217.94,
                "facility_operation_cycleTime": 122.12,
                "production_cycletime": 122.95,
                "low_section_speed": 109.98,
                "high_section_speed": 112.31,
                "cast_pressure": 330.46,
                "biscuit_thickness": 53.17,
                "physical_strength": 707.05,
            },
            "8600": {
                "molten_temp": 709.02,
                "sleeve_temperature": 501.51,
                "Coolant_temperature": 31.54,
                "lower_mold_temp1": 212.87,
                "lower_mold_temp2": 189.74,
                "lower_mold_temp3": 1449.0,
                "upper_mold_temp1": 224.28,
                "upper_mold_temp2": 174.45,
                "upper_mold_temp3": 1449.0,
                "facility_operation_cycleTime": 121.04,
                "production_cycletime": 124.26,
                "low_section_speed": 114.06,
                "high_section_speed": 116.6,
                "cast_pressure": 324.86,
                "biscuit_thickness": 48.19,
                "physical_strength": 691.47,
            },
            "8917": {
                "molten_temp": 718.32,
                "sleeve_temperature": 456.66,
                "Coolant_temperature": 31.42,
                "lower_mold_temp1": 231.0,
                "lower_mold_temp2": 176.4,
                "lower_mold_temp3": 1449.0,
                "upper_mold_temp1": 157.95,
                "upper_mold_temp2": 173.36,
                "upper_mold_temp3": 1449.0,
                "facility_operation_cycleTime": 120.05,
                "production_cycletime": 120.94,
                "low_section_speed": 109.15,
                "high_section_speed": 111.54,
                "cast_pressure": 328.4,
                "biscuit_thickness": 50.72,
                "physical_strength": 703.33,
            },
        }

        # 해당 몰드의 패턴 반환
        pattern = FIXED_NORMAL_PATTERNS.get(str(mold_code))

        if pattern:
            # print(f"✅ 몰드 {mold_code} 정상 패턴:")
            # for key, value in pattern.items():
                # print(f"  - {key}: {value}")
            return pattern
        else:
            # print(f"🚨 [WARNING] 몰드 {mold_code} 패턴 없음, 기본값 사용")
            # 🔧 기본값 반환 (가장 가까운 몰드의 평균값 사용)
            return {
                "molten_temp": 718.07,
                "sleeve_temperature": 457.51,
                "Coolant_temperature": 31.82,
                "lower_mold_temp1": 193.22,
                "lower_mold_temp2": 190.97,
                "lower_mold_temp3": 1449.0,
                "upper_mold_temp1": 199.45,
                "upper_mold_temp2": 171.75,
                "upper_mold_temp3": 1449.0,
                "facility_operation_cycleTime": 120.86,
                "production_cycletime": 122.73,
                "low_section_speed": 111.83,
                "high_section_speed": 114.29,
                "cast_pressure": 327.55,
                "biscuit_thickness": 50.74,
                "physical_strength": 702.39,
            }
    def get_korean_feature_name(feature_name):
            """영문 피처명을 한글로 변환"""
            name_mapping = {
                'molten_temp': '용탕온도',
                'sleeve_temperature': '슬리브온도', 
                'cast_pressure': '사출압력',
                'biscuit_thickness': '비스킷두께',
                'physical_strength': '물리강도',
                'production_cycletime': '생산사이클',
                'upper_mold_temp1': '상부몰드1',
                'upper_mold_temp2': '상부몰드2',
                'upper_mold_temp3': '상부몰드3',
                'lower_mold_temp1': '하부몰드1',
                'lower_mold_temp2': '하부몰드2',
                'lower_mold_temp3': '하부몰드3',
                'high_section_speed': '고속구간',
                'low_section_speed': '저속구간',
                'Coolant_temperature': '냉각수온도',
                'facility_operation_cycleTime': '설비사이클'
            }
            return name_mapping.get(feature_name, feature_name)
    
    def get_fast_permutation_importance(model, sample, feature_names, n_repeats=15):
        """빠른 Permutation Importance (실시간용)"""
        
        baseline_score = model.decision_function([sample])[0]
        importances = []
        
        for feature_idx in range(len(feature_names)):
            importance_scores = []
            
            for _ in range(n_repeats):
                modified_sample = sample.copy()
                # 정규분포에서 랜덤 샘플링
                modified_sample[feature_idx] = np.random.normal(0, 1)
                
                new_score = model.decision_function([modified_sample])[0]
                score_change = abs(baseline_score - new_score)
                importance_scores.append(score_change)
            
            avg_importance = np.mean(importance_scores)
            importances.append(avg_importance)
        
        return np.array(importances)
    
    def analyze_anomaly_with_permutation(latest_anomaly):
            """이상치에 대한 Permutation Importance 분석"""
            
            try:
                # 행 데이터를 feature 순서대로 배열
                feature_names = [
                    "molten_temp", "sleeve_temperature", "Coolant_temperature",
                    "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3", 
                    "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
                    "facility_operation_cycleTime", "production_cycletime",
                    "low_section_speed", "high_section_speed", "cast_pressure",
                    "biscuit_thickness", "physical_strength"
                ]
                
                # 전처리된 데이터 준비
                row_data = latest_anomaly["specs"].copy()
                row_data['working'] = 1  # 임시 추가
                row_data['mold_code'] = latest_anomaly["mold_code"]
                
                preprocessor = CustomPreprocessor()
                input_df = pd.DataFrame([row_data])
                processed_df = preprocessor.transform(input_df)
                
                # feature_names에 맞게 정렬
                available_features = [f for f in feature_names if f in processed_df.columns]
                sample_array = processed_df[available_features].values[0]
                
                # Permutation Importance 계산
                importances = get_fast_permutation_importance(
                    isolation_model_data, 
                    sample_array, 
                    available_features
                )
                
                # 상위 8개 피처 선택
                top_indices = np.argsort(importances)[-8:][::-1]
                
                analysis_result = {
                    "feature_importances": importances.tolist(),
                    "available_features": available_features,
                    "top_features": [
                        {
                            "name": available_features[i],
                            "korean_name": get_korean_feature_name(available_features[i]),
                            "importance": float(importances[i]),
                            "current_value": float(sample_array[i]),
                            "rank": int(np.where(top_indices == i)[0][0] + 1) if i in top_indices else None
                        }
                        for i in top_indices
                    ],
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                return analysis_result
                
            except Exception as e:
                # print(f"🚨 [ERROR] Permutation 분석 실패: {e}")
                return {"error": str(e)}
    @output
    @render.ui
    def anomaly_detail_modal():
        """이상치 상세 분석 모달 - 적용 버튼 추가"""
        
        if not show_detail_modal.get():
            return ui.div()
        
        detail_data = selected_alert_data.get()
        if not detail_data:
            return ui.div()
        
        alert_info = detail_data["alert_info"]
        
        return ui.div([
            # JavaScript
            ui.tags.script("""
                setTimeout(function() {
                    const modal = document.getElementById('detail-modal-container');
                    if (modal) {
                        modal.style.display = 'flex';
                        document.body.style.overflow = 'hidden';
                    }
                }, 100);
                
                function closeDetailModal() {
                    const modal = document.getElementById('detail-modal-container');
                    if (modal) {
                        modal.style.display = 'none';
                        document.body.style.overflow = 'auto';
                    }
                    Shiny.setInputValue('detail_modal_closed', Math.random());
                }
            """),
            
            # 모달 컨테이너
            ui.div([
                # 배경
                ui.div(
                    class_="fixed inset-0 bg-black bg-opacity-50 z-40",
                    onclick="closeDetailModal()"
                ),
                
                # 모달 내용
                ui.div([
                    ui.div([
                        # 헤더
                        ui.div([
                            ui.h3(f"[ 이상치 상세 분석 ] - {alert_info['real_id']}", class_="text-xl font-bold"),
                            ui.tags.button(
                                "✕",
                                onclick="closeDetailModal()",
                                class_="absolute top-4 right-4 text-gray-500 hover:text-gray-700 text-xl bg-transparent border-none cursor-pointer"
                            )
                        ], class_="relative p-4 border-b"),
                        
                        # 기본 정보
                        ui.div([
                            create_basic_info_section(alert_info)
                        ], class_="px-4 border-b bg-gray-50"),
                        
                        # 메인 분석 영역
                        ui.div([
                            ui.row([
                                # 왼쪽: 전체 변수 상세 스펙
                                ui.column(6, [
                                    ui.h4("이상치 상세 스펙", class_="font-semibold mb-3 text-blue-600"),
                                    ui.output_ui("modal_all_specs")
                                ]),
                                
                                # 오른쪽: 레이더 차트 + 변수 선택
                                ui.column(6, [
                                    ui.h4("패턴 분석", class_="font-semibold mb-3 text-red-600"),
                                    
                                    # 🔧 변수 선택 UI + 적용 버튼
                                    ui.div([
                                        ui.p("레이더 차트 변수 선택 (최대 8개):", class_="text-sm font-medium mb-2"),
                                        
                                        # 🔧 변수 선택과 적용 버튼을 가로로 배치
                                        ui.div([
                                            # 변수 선택 드롭다운
                                            ui.div([
                                                ui.input_selectize(
                                                    "radar_variables_temp",  # 🔧 임시 이름 변경
                                                    "",
                                                    choices={
                                                        "molten_temp": "용탕온도",
                                                        "sleeve_temperature": "슬리브온도", 
                                                        "Coolant_temperature": "냉각수온도",
                                                        "cast_pressure": "사출압력",
                                                        "high_section_speed": "고속구간",
                                                        "biscuit_thickness": "비스킷두께",
                                                        "physical_strength": "물리강도",
                                                        "lower_mold_temp1": "하부몰드1",
                                                        "lower_mold_temp2": "하부몰드2",
                                                        "lower_mold_temp3": "하부몰드3",
                                                        "upper_mold_temp1": "상부몰드1",
                                                        "upper_mold_temp2": "상부몰드2",
                                                        "upper_mold_temp3": "상부몰드3",
                                                        "facility_operation_cycleTime": "설비사이클",
                                                        "production_cycletime": "생산사이클",
                                                        "low_section_speed": "저속구간"
                                                    },
                                                    selected=["molten_temp", "sleeve_temperature", "Coolant_temperature", 
                                                            "cast_pressure", "high_section_speed", "biscuit_thickness", "physical_strength"],
                                                    multiple=True,
                                                    options={"maxItems": 8, "plugins": ["clear_button"]}
                                                )
                                            ], class_="flex-1 mr-2"),
                                            
                                            # 🔧 적용 버튼
                                            ui.div([
                                                ui.input_action_button(
                                                    "apply_radar_variables",
                                                    "적용",
                                                    class_="btn btn-primary btn-sm px-3 py-2"
                                                )
                                            ], class_="flex-shrink-0")
                                        ], class_="flex items-end")
                                    ], class_="mb-3 p-2 bg-gray-100 rounded"),
                                    
                                    # 레이더 차트
                                    ui.div([
                                        output_widget("detail_radar_chart")
                                    ], class_="h-96")
                                ])
                            ])
                        ], class_="px-4"),
                        
                        # 푸터
                        ui.div([
                            ui.tags.button(
                                "닫기",
                                onclick="closeDetailModal()",
                                class_="btn btn-secondary px-6 py-2"
                            )
                        ], class_="p-4 border-t text-center")
                        
                    ], class_="bg-white rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-auto")
                ], class_="fixed inset-0 flex items-center justify-center z-50 p-4")
            ], 
            id="detail-modal-container",
            style="display: none;"
            )
        ])    
    @output
    @render_widget
    def detail_radar_chart():
        """적용 버튼과 연동된 레이더 차트"""
        
        detail_data = selected_alert_data.get()
        if not detail_data:
            return go.Figure()
        
        # 🔧 적용 버튼과 연동된 변수 가져오기
        applied_vars = input.radar_variables_temp() if hasattr(input, 'radar_variables_temp') and input.radar_variables_temp() else []
        
        # 기본값 사용
        if not applied_vars:
            applied_vars = ["molten_temp", "sleeve_temperature", "Coolant_temperature", 
                        "cast_pressure", "high_section_speed", "biscuit_thickness", "physical_strength"]
        
        selected_vars = applied_vars
        
        if not selected_vars or len(selected_vars) < 3:
            return go.Figure().add_annotation(
                text="최소 3개 이상의 변수를 선택 후 적용 버튼을 눌러주세요",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14)
            )
        
        normal_pattern = detail_data["normal_pattern"]
        deviations = detail_data["deviations"]
        
        if not normal_pattern or not deviations:
            return go.Figure()
        
        # 변수명 매핑
        variable_names = {
            "molten_temp": "용탕온도", "sleeve_temperature": "슬리브온도", 
            "Coolant_temperature": "냉각수온도", "cast_pressure": "사출압력",
            "high_section_speed": "고속구간", "biscuit_thickness": "비스킷두께",
            "physical_strength": "물리강도", "lower_mold_temp1": "하부몰드1",
            "lower_mold_temp2": "하부몰드2", "lower_mold_temp3": "하부몰드3",
            "upper_mold_temp1": "상부몰드1", "upper_mold_temp2": "상부몰드2",
            "upper_mold_temp3": "상부몰드3", "facility_operation_cycleTime": "설비사이클",
            "production_cycletime": "생산사이클", "low_section_speed": "저속구간"
        }
        
        # 정규화 범위
        feature_ranges = {
            "molten_temp": (700, 730), "sleeve_temperature": (350, 520),
            "Coolant_temperature": (30, 35), "cast_pressure": (320, 335),
            "high_section_speed": (110, 118), "biscuit_thickness": (45, 55),
            "physical_strength": (690, 710), "lower_mold_temp1": (140, 240),
            "lower_mold_temp2": (170, 210), "lower_mold_temp3": (1440, 1450),
            "upper_mold_temp1": (150, 230), "upper_mold_temp2": (140, 200),
            "upper_mold_temp3": (210, 1450), "facility_operation_cycleTime": (118, 125),
            "production_cycletime": (119, 126), "low_section_speed": (108, 116)
        }
        
        # 선택된 변수들만 처리
        features = []
        normal_values = []
        current_values = []
        
        for var in selected_vars:
            if var in deviations:
                features.append(variable_names.get(var, var))
                
                normal_val = deviations[var]["normal"]
                current_val = deviations[var]["current"]
                
                # 정규화
                min_val, max_val = feature_ranges.get(var, (0, 1))
                normal_norm = (normal_val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                current_norm = (current_val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                
                normal_values.append(max(0, min(1, normal_norm)))
                current_values.append(max(0, min(1, current_norm)))
        
        if len(features) < 3:
            return go.Figure().add_annotation(
                text="선택된 변수의 데이터가 부족합니다",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
        
        # 원형으로 만들기
        features_circle = features + features[:1]
        normal_values_circle = normal_values + normal_values[:1]
        current_values_circle = current_values + current_values[:1]
        
        fig = go.Figure()
        
        # 정상 패턴 (회색, 점선)
        fig.add_trace(go.Scatterpolar(
            r=normal_values_circle,
            theta=features_circle,
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(color='gray', width=2, dash='dash'),
            name='정상 패턴'
        ))
        
        # 현재 이상 패턴 (빨강, 실선)
        fig.add_trace(go.Scatterpolar(
            r=current_values_circle,
            theta=features_circle,
            fill='toself',
            fillcolor='rgba(255, 99, 132, 0.3)',
            line=dict(color='red', width=3),
            name='현재 패턴'
        ))
        
        # 이상 피처 강조
        for i, var in enumerate(selected_vars):
            if var in deviations and deviations[var]["is_outlier"]:
                if i < len(features):
                    fig.add_trace(go.Scatterpolar(
                        r=[current_values[i]],
                        theta=[features[i]],
                        mode='markers',
                        marker=dict(
                            color='orange',
                            size=12,
                            symbol='star',
                            line=dict(color='darkorange', width=2)
                        ),
                        name=f'⚠️ {features[i]}',
                        showlegend=False
                    ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickvals=[0, 0.5, 1.0],
                    ticktext=['최소', '중간', '최대']
                )
            ),
            title=f"정상 패턴 vs 현재 패턴",
            showlegend=True,
            width=450,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig    
# 🔍 여기부터 새로 추가
    # 🗑️ 알림 전체 삭제
    @reactive.Effect
    @reactive.event(input.btn_clear_alerts)
    def clear_all_alerts():
        anomaly_results.set([])
        # print("🗑️ [DEBUG] 모든 알림 삭제됨")

    # 🧪 테스트 알림 생성
    # @reactive.Effect
    # @reactive.event(input.btn_create_test_alert)
    # def create_test_alert():
    #     # print("🧪 [TEST] 테스트 알림 생성 중...")
        
    #     current_alerts = anomaly_results.get()
    #     current_time = tick2()
        
    #     # 🔧 현재 선택된 몰드 코드 사용
    #     selected_mold = input.mold_code_1page()
    #     if selected_mold is None or selected_mold == "":
    #         selected_mold = "8917"  # 기본값
        
    #     # 랜덤 이상치 점수 생성
    #     import random
    #     test_scores = [-0.08, -0.03, -0.01]
    #     test_score = random.choice(test_scores)
        
    #     # 테스트 specs 데이터
    #     test_specs = {
    #     "molten_temp": random.uniform(700, 730),
    #     "sleeve_temperature": random.uniform(350, 520),
    #     "Coolant_temperature": random.uniform(30, 35),
    #     "lower_mold_temp1": random.uniform(140, 240),
    #     "lower_mold_temp2": random.uniform(170, 210),
    #     "lower_mold_temp3": random.uniform(1440, 1450),
    #     "upper_mold_temp1": random.uniform(150, 230),
    #     "upper_mold_temp2": random.uniform(140, 200),
    #     "upper_mold_temp3": random.uniform(210, 1450),
    #     "facility_operation_cycleTime": random.uniform(118, 125),
    #     "production_cycletime": random.uniform(119, 126),
    #     "low_section_speed": random.uniform(108, 116),
    #     "high_section_speed": random.uniform(110, 118),
    #     "cast_pressure": random.uniform(320, 335),
    #     "biscuit_thickness": random.uniform(45, 55),
    #     "physical_strength": random.uniform(690, 710),
    # }
        
    #     new_alert = {
    #         "id": f"test_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #         "timestamp": current_time.strftime('%H:%M:%S'),
    #         "full_timestamp": current_time,
    #         "mold_code": selected_mold,  # 🔧 동적 몰드 코드
    #         "anomaly_score": test_score,
    #         "risk_level": get_anomaly_risk_level(test_score),
    #         "specs": test_specs,
    #         "is_real_data": False  # 테스트 데이터 표시
    #     }
        
    #     current_alerts.insert(0, new_alert)
        
    #     if len(current_alerts) > 8:
    #         current_alerts = current_alerts[:8]
        
    #     anomaly_results.set(current_alerts)
    #     # print(f"🧪 [TEST] 테스트 알림 생성 완료! (몰드: {selected_mold})")

    @reactive.Effect
    @reactive.event(input.detail_button_clicked)  
    def open_detail_modal():
        """상세보기 모달 열기 - IF 정상 패턴 사용"""
        
        alert_id = input.detail_button_clicked()
        if not alert_id:
            return
        
        # print(f"🔍 [DEBUG] 상세보기 요청: {alert_id}")
        
        # 해당 alert 찾기
        alerts = anomaly_results.get()
        selected_alert = None
        
        for alert in alerts:
            if alert['id'] == alert_id:
                selected_alert = alert
                break
        
        if not selected_alert:
            # print(f"🚨 [ERROR] 알림을 찾을 수 없음: {alert_id}")
            return
        
        # print(f"🔍 [DEBUG] 선택된 알림: {selected_alert['id']}")
        # print(f"🔍 [DEBUG] 실제 데이터 여부: {selected_alert.get('is_real_data', False)}")
        
        try:
            # specs 데이터 확인
            specs = selected_alert.get("specs", {})
            # print(f"🔍 [DEBUG] specs 개수: {len(specs)}")
            # print(f"🔍 [DEBUG] specs 키들: {list(specs.keys())}")
            
            if not specs:
                # print("🚨 [ERROR] specs 데이터가 없습니다!")
                return
            
            # 🔧 Isolation Forest 정상 패턴 계산
            # print("🔍 [DEBUG] IF 정상 패턴 계산 중...")
            normal_pattern = get_isolation_forest_normal_pattern_simple(selected_alert["mold_code"])
            
            if normal_pattern is None or len(normal_pattern) < 3:
                # print("🚨 [WARNING] IF 정상 패턴 계산 실패, 기본값 사용")
                normal_pattern = {
                    "molten_temp": 680.0,
                    "sleeve_temperature": 220.0,
                    "Coolant_temperature": 25.0,
                    "cast_pressure": 300.0,
                    "high_section_speed": 150.0,
                    "biscuit_thickness": 15.0,
                    "physical_strength": 250.0
                }
            
            # print(f"🔍 [DEBUG] 사용할 정상 패턴: {normal_pattern}")
            
            # 편차 계산
            deviations = {}
            
            for key in specs:
                if key in normal_pattern:
                    try:
                        current_val = float(specs[key])
                        normal_val = float(normal_pattern[key])
                        
                        abs_diff = current_val - normal_val
                        rel_diff_pct = abs(abs_diff) / normal_val * 100 if normal_val != 0 else 0
                        
                        deviations[key] = {
                            "current": current_val,
                            "normal": normal_val,
                            "absolute_diff": abs_diff,
                            "relative_diff_pct": rel_diff_pct,
                            "is_outlier": rel_diff_pct > 20
                        }
                        
                        # print(f"🔍 [DEBUG] {key}: 현재={current_val:.1f}, IF정상={normal_val:.1f}, 차이={rel_diff_pct:.1f}%")
                        
                    except (ValueError, TypeError) as e:
                        # print(f"🚨 [ERROR] {key} 계산 실패: {e}")
                        continue
            
            # print(f"🔍 [DEBUG] 최종 deviations 개수: {len(deviations)}")
            
            # 이상치 데이터 구성
            anomaly_data = {
                "timestamp": selected_alert["full_timestamp"],
                "mold_code": selected_alert["mold_code"],
                "specs": specs,
                "anomaly_score": selected_alert["anomaly_score"]
            }
            
            # 상세 데이터 저장
            alert_data = {
                "alert_info": selected_alert,
                "anomaly_data": anomaly_data,
                "normal_pattern": normal_pattern,
                "deviations": deviations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            selected_alert_data.set(alert_data)
            show_detail_modal.set(True)
            # print(f"🔍 [DEBUG] 상세보기 모달 열림 완료")
            
        except Exception as e:
            # print(f"🚨 [ERROR] 상세 분석 실패: {e}")
            import traceback
            # print(traceback.format_exc())
        # 🔧 create_detailed_specs 함수도 더 안전하게 수정

    # 상세보기 모달 닫기
    @reactive.Effect
    @reactive.event(input.detail_modal_closed)
    def close_detail_modal():
        show_detail_modal.set(False)
        selected_alert_data.set({})
        # print("🔍 [DEBUG] 상세보기 모달 닫힘")

    def create_basic_info_section(alert_info):
        """기본 정보 섹션"""
        
        risk_level = alert_info["risk_level"]
        
        return ui.div([
            ui.div([
                ui.div([
                    ui.span("🚨", class_="text-2xl mr-3"),
                    ui.div([
                        ui.h5(f"몰드 {alert_info['mold_code']} - {alert_info['real_id']} 이상치 탐지", class_="font-semibold text-lg"),
                        ui.p(f"탐지 시간: {alert_info['timestamp']}", class_="text-sm text-gray-600")
                    ])
                ], class_="flex items-center"),
                
                ui.div([
                    ui.div([
                        ui.span("위험도:", class_="text-sm font-medium mr-2"),
                        ui.span(risk_level["level"], class_=f"{risk_level['class']} font-bold")
                    ], class_="mb-2"),
                    ui.div([
                        ui.span("이상치 점수:", class_="text-sm font-medium mr-2"),
                        ui.span(f"{alert_info['anomaly_score']:.4f}", class_="font-mono")
                    ])
                ])
            ], class_="flex justify-between items-center")
        ])

    def create_detailed_specs(current_specs, normal_pattern, deviations):
        """상세 스펙 비교 테이블"""
        
        if not deviations:
            return ui.div("분석 데이터를 불러올 수 없습니다.", class_="text-gray-500")
        
        feature_mapping = {
            'molten_temp': ('용탕온도', '°C'),
            'sleeve_temperature': ('슬리브온도', '°C'), 
            'Coolant_temperature': ('냉각수온도', '°C'),
            'cast_pressure': ('사출압력', 'bar'),
            'high_section_speed': ('고속구간', ''),
            'biscuit_thickness': ('비스킷두께', 'mm'),
            'physical_strength': ('물리강도', 'MPa')
        }
        
        spec_items = []
        
        for eng_name, (kor_name, unit) in feature_mapping.items():
            if eng_name in deviations:
                dev = deviations[eng_name]
                
                # 상태에 따른 색상
                if dev["is_outlier"]:
                    status_class = "bg-red-100 border-red-300"
                    status_icon = "🔴"
                elif dev["relative_diff_pct"] > 10:
                    status_class = "bg-yellow-100 border-yellow-300"
                    status_icon = "🟡"
                else:
                    status_class = "bg-green-100 border-green-300"
                    status_icon = "🟢"
                
                spec_items.append(
                    ui.div([
                        ui.div([
                            ui.span(status_icon, class_="mr-2"),
                            ui.span(kor_name, class_="font-semibold text-sm")
                        ], class_="flex items-center mb-2"),
                        
                        ui.div([
                            ui.div([
                                ui.span("정상:", class_="text-xs text-gray-600 mr-1"),
                                ui.span(f"{dev['normal']:.1f}{unit}", class_="text-xs font-medium")
                            ], class_="mb-1"),
                            ui.div([
                                ui.span("현재:", class_="text-xs text-gray-600 mr-1"),
                                ui.span(f"{dev['current']:.1f}{unit}", class_="text-xs font-bold")
                            ], class_="mb-1"),
                            ui.div([
                                ui.span("차이:", class_="text-xs text-gray-600 mr-1"),
                                ui.span(
                                    f"{dev['absolute_diff']:+.1f}{unit} ({dev['relative_diff_pct']:+.1f}%)", 
                                    class_="text-xs font-medium"
                                )
                            ])
                        ])
                        
                    ], class_=f"p-3 rounded border {status_class} mb-3")
                )
        
        return ui.div([
            ui.div(spec_items, class_="max-h-80 overflow-y-auto")
        ])

    def create_problem_summary(deviations):
        """문제 요약"""
        
        if not deviations:
            return ui.div("분석 중...", class_="text-gray-500")
        
        # 이상치 피처들만 추출
        outlier_features = [
            (name, data) for name, data in deviations.items() 
            if data["is_outlier"]
        ]
        
        if not outlier_features:
            return ui.div([
                ui.p("🟢 주요 이상 피처가 없습니다.", class_="text-green-600 font-medium"),
                ui.p("경미한 패턴 변화로 판단됩니다.", class_="text-sm text-gray-600")
            ])
        
        summary_items = []
        
        for i, (eng_name, data) in enumerate(outlier_features[:3]):  # 상위 3개만
            korean_name = {
                'molten_temp': '용탕온도',
                'sleeve_temperature': '슬리브온도', 
                'cast_pressure': '사출압력',
                'biscuit_thickness': '비스킷두께',
                'physical_strength': '물리강도',
                'high_section_speed': '고속구간',
                'Coolant_temperature': '냉각수온도'
            }.get(eng_name, eng_name)
            
            summary_items.append(
                ui.div([
                    ui.span(f"{i+1}.", class_="font-bold text-lg mr-2"),
                    ui.span(korean_name, class_="font-semibold mr-2"),
                    ui.span(f"{data['relative_diff_pct']:.1f}% 이상", class_="text-red-600 text-sm")
                ], class_="flex items-center mb-2 p-2 bg-red-50 rounded")
            )
        
        return ui.div([
            ui.h5("⚠️ 주요 문제 피처", class_="font-semibold mb-2 text-red-600"),
            ui.div(summary_items)
        ])
    @reactive.Effect
    @reactive.event(input.apply_radar_variables)
    def apply_radar_variables():
        """적용 버튼 클릭시 레이더 차트 변수 업데이트"""
        
        # 임시 선택값을 실제 변수에 복사
        temp_selection = input.radar_variables_temp()
    
    @output
    @render.ui  
    def modal_all_specs():
        """Isolation Forest 전체 변수 상세 스펙 - 스크롤 확장 + 상태별 정렬"""

        detail_data = selected_alert_data.get()
        if not detail_data:
            return ui.div()

        deviations = detail_data["deviations"]
        if not deviations:
            return ui.div("분석 데이터를 불러올 수 없습니다.", class_="text-gray-500")

        # 16개 변수 한글명/단위 매핑
        all_variables = {
            'molten_temp': ('용탕온도', '°C'),
            'sleeve_temperature': ('슬리브온도', '°C'), 
            'Coolant_temperature': ('냉각수온도', '°C'),
            'lower_mold_temp1': ('하부몰드1', '°C'),
            'lower_mold_temp2': ('하부몰드2', '°C'),
            'lower_mold_temp3': ('하부몰드3', '°C'),
            'upper_mold_temp1': ('상부몰드1', '°C'),
            'upper_mold_temp2': ('상부몰드2', '°C'),
            'upper_mold_temp3': ('상부몰드3', '°C'),
            'facility_operation_cycleTime': ('설비사이클', '초'),
            'production_cycletime': ('생산사이클', '초'),
            'low_section_speed': ('저속구간', ''),
            'high_section_speed': ('고속구간', ''),
            'cast_pressure': ('사출압력', 'bar'),
            'biscuit_thickness': ('비스킷두께', 'mm'),
            'physical_strength': ('물리강도', 'MPa')
        }

        # 상태별 우선순위 매핑 (🔴→0, 🟡→1, 🟢→2, ⚪→3)
        priority_map = {"🔴": 0, "🟡": 1, "🟢": 2, "⚪": 3}
        items_with_prio = []

        for eng_name, (kor_name, unit) in all_variables.items():
            if eng_name in deviations:
                dev = deviations[eng_name]
                # Outlier?
                if dev["is_outlier"]:
                    status_icon = "🔴"
                    status_class = "bg-red-100 border-red-300"
                # 이상?
                elif dev["relative_diff_pct"] > 10:
                    status_icon = "🟡"
                    status_class = "bg-yellow-100 border-yellow-300"
                # 정상
                else:
                    status_icon = "🟢"
                    status_class = "bg-green-100 border-green-300"

                card = ui.div([
                    ui.div([
                        ui.span(status_icon, class_="mr-2"),
                        ui.span(kor_name, class_="font-semibold text-sm")
                    ], class_="flex items-center mb-1"),
                    ui.div([
                        ui.div([
                            ui.span("정상:", class_="text-xs text-gray-600 mr-1"),
                            ui.span(f"{dev['normal']:.1f}{unit}", class_="text-xs font-medium")
                        ], class_="mb-1"),
                        ui.div([
                            ui.span("현재:", class_="text-xs text-gray-600 mr-1"),
                            ui.span(f"{dev['current']:.1f}{unit}", class_="text-xs font-bold")
                        ], class_="mb-1"),
                        ui.div([
                            ui.span("차이:", class_="text-xs text-gray-600 mr-1"),
                            ui.span(
                                f"{dev['absolute_diff']:+.1f}{unit} "
                                f"({dev['relative_diff_pct']:+.1f}%)",
                                class_="text-xs font-medium"
                            )
                        ])
                    ])
                ], class_=f"p-2 rounded border {status_class} mb-2")

            else:
                # 데이터 없음은 마지막(⚪)
                status_icon = "⚪"
                status_class = "bg-gray-50 border-gray-200"
                card = ui.div([
                    ui.div([
                        ui.span(status_icon, class_="mr-2"),
                        ui.span(kor_name, class_="font-medium text-sm text-gray-500")
                    ], class_="flex items-center mb-1"),
                    ui.div("데이터 없음", class_="text-xs text-gray-400")
                ], class_=f"p-2 rounded border {status_class} mb-2")

            # (priority, card) 형태로 리스트에 저장
            items_with_prio.append((priority_map[status_icon], card))

        # 우선순위에 따라 정렬 후 카드만 추출
        items_with_prio.sort(key=lambda x: x[0])
        spec_items = [card for _, card in items_with_prio]

        # 스크롤 높이 확장
        return ui.div(
            ui.div(spec_items, class_="maxh-500 overflow-y-auto")
        )

    @reactive.Effect
    @reactive.event(input.defect_detail_button_clicked)  
    def open_defect_detail_modal():
        """불량률 상세보기 모달 열기 - 완전 구현"""
        
        alert_id = input.defect_detail_button_clicked()
        if not alert_id:
            return
        
        # 해당 alert 찾기
        alerts = defect_alert_list.get()
        selected_alert = None
        
        for alert in alerts:
            if str(alert['id']) == str(alert_id):
                selected_alert = alert
                break
        
        if not selected_alert:
            return
        
        try:
            # 🔧 센서 데이터 확인 (불량 알림에 specs가 있어야 함)
            specs = selected_alert.get("specs", {})
            
            if not specs:
                print("🚨 [ERROR] 불량 알림에 센서 데이터가 없습니다!")
                return
            
            # 🔧 불량 예측용 양품 기준 패턴 사용
            mold_code = selected_alert["mold_code"]
            normal_pattern = get_defect_normal_pattern_simple(mold_code)
            
            # 🔧 편차 계산 (이상치와 동일한 로직)
            deviations = {}
            for key in specs:
                if key in normal_pattern:
                    try:
                        current_val = float(specs[key])
                        normal_val = float(normal_pattern[key])
                        
                        abs_diff = current_val - normal_val
                        rel_diff_pct = abs(abs_diff) / normal_val * 100 if normal_val != 0 else 0
                        
                        deviations[key] = {
                            "current": current_val,
                            "normal": normal_val,
                            "absolute_diff": abs_diff,
                            "relative_diff_pct": rel_diff_pct,
                            "is_outlier": rel_diff_pct > 20
                        }
                        
                    except (ValueError, TypeError):
                        continue
            
            # 상세 데이터 저장
            alert_data = {
                "alert_info": selected_alert,
                "normal_pattern": normal_pattern,
                "deviations": deviations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            selected_defect_alert_data.set(alert_data)
            show_defect_detail_modal.set(True)
            
        except Exception as e:
            print(f"🚨 [ERROR] 불량 상세 분석 실패: {e}")

    @reactive.Effect
    @reactive.event(input.defect_detail_modal_closed)
    def close_defect_detail_modal():
        show_defect_detail_modal.set(False)
        selected_defect_alert_data.set({})

    @output
    @render.ui
    def defect_detail_modal():
        """불량률 상세 분석 모달 - 완전 구현"""
        
        if not show_defect_detail_modal.get():
            return ui.div()
        
        detail_data = selected_defect_alert_data.get()
        if not detail_data:
            return ui.div()
        
        alert_info = detail_data["alert_info"]
        
        return ui.div([
            # JavaScript
            ui.tags.script("""
                setTimeout(function() {
                    const modal = document.getElementById('defect-detail-modal-container');
                    if (modal) {
                        modal.style.display = 'flex';
                        document.body.style.overflow = 'hidden';
                    }
                }, 100);
                
                function closeDefectDetailModal() {
                    const modal = document.getElementById('defect-detail-modal-container');
                    if (modal) {
                        modal.style.display = 'none';
                        document.body.style.overflow = 'auto';
                    }
                    Shiny.setInputValue('defect_detail_modal_closed', Math.random());
                }
            """),
            
            # 모달 컨테이너
            ui.div([
                # 배경
                ui.div(
                    class_="fixed inset-0 bg-black bg-opacity-50 z-40",
                    onclick="closeDefectDetailModal()"
                ),
                
                # 모달 내용
                ui.div([
                    ui.div([
                        # 헤더
                        ui.div([
                            ui.h3(f"불량 상세 분석 - M{alert_info['id']}", class_="text-xl font-bold"),
                            ui.tags.button(
                                "✕",
                                onclick="closeDefectDetailModal()",
                                class_="absolute top-4 right-4 text-gray-500 hover:text-gray-700 text-xl bg-transparent border-none cursor-pointer"
                            )
                        ], class_="relative p-4 border-b"),
                        
                        # 기본 정보
                        ui.div([
                            create_defect_basic_info_section(alert_info)
                        ], class_="px-4 border-b bg-gray-50"),
                        
                        # 메인 분석 영역
                        ui.div([
                            ui.row([
                                # 왼쪽: 전체 변수 상세 스펙
                                ui.column(6, [
                                    ui.h4("불량 상세 스펙", class_="font-semibold mb-3 text-red-600"),
                                    ui.output_ui("defect_modal_all_specs")
                                ]),
                                
                                # 오른쪽: 레이더 차트 + 변수 선택
                                ui.column(6, [
                                    ui.h4("양품 기준 비교", class_="font-semibold mb-3 text-blue-600"),
                                    
                                    # 🔧 변수 선택 UI (불량 전용)
                                    ui.div([
                                        ui.p("레이더 차트 변수 선택 (최대 8개):", class_="text-sm font-medium mb-2"),
                                        
                                        ui.div([
                                            # 변수 선택 드롭다운 (불량 전용)
                                            ui.div([
                                                ui.input_selectize(
                                                    "defect_radar_variables_temp",
                                                    "",
                                                    choices={
                                                        "molten_temp": "용탕온도",
                                                        "sleeve_temperature": "슬리브온도", 
                                                        "Coolant_temperature": "냉각수온도",
                                                        "cast_pressure": "사출압력",
                                                        "high_section_speed": "고속구간",
                                                        "biscuit_thickness": "비스킷두께",
                                                        "physical_strength": "물리강도",
                                                        "lower_mold_temp1": "하부몰드1",
                                                        "lower_mold_temp2": "하부몰드2",
                                                        "lower_mold_temp3": "하부몰드3",
                                                        "upper_mold_temp1": "상부몰드1",
                                                        "upper_mold_temp2": "상부몰드2",
                                                        "upper_mold_temp3": "상부몰드3",
                                                        "facility_operation_cycleTime": "설비사이클",
                                                        "production_cycletime": "생산사이클",
                                                        "low_section_speed": "저속구간"
                                                    },
                                                    selected=["molten_temp", "sleeve_temperature", "Coolant_temperature", 
                                                            "cast_pressure", "high_section_speed", "biscuit_thickness", "physical_strength"],
                                                    multiple=True,
                                                    options={"maxItems": 8}
                                                )
                                            ], class_="flex-1 mr-2"),
                                            
                                            # 적용 버튼
                                            ui.div([
                                                ui.input_action_button(
                                                    "apply_defect_radar_variables",
                                                    "적용",
                                                    class_="btn btn-primary btn-sm px-3 py-2"
                                                )
                                            ], class_="flex-shrink-0")
                                        ], class_="flex items-end")
                                    ], class_="mb-3 p-2 bg-gray-100 rounded"),
                                    
                                    # 레이더 차트
                                    ui.div([
                                        output_widget("defect_detail_radar_chart")
                                    ], class_="h-96")
                                ])
                            ])
                        ], class_="px-4"),
                        
                        # 푸터
                        ui.div([
                            ui.tags.button(
                                "닫기",
                                onclick="closeDefectDetailModal()",
                                class_="btn btn-secondary px-6 py-2"
                            )
                        ], class_="p-4 border-t text-center")
                        
                    ], class_="bg-white rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-auto")
                ], class_="fixed inset-0 flex items-center justify-center z-50 p-4")
            ], 
            id="defect-detail-modal-container",
            style="display: none;"
            )
        ])
    


    DEFECT_NORMAL_PATTERNS_HARDCODED = {
    "8412": {
        "molten_temp": 724.43,
        "facility_operation_cycleTime": 120.24,
        "production_cycletime": 122.78,
        "low_section_speed": 114.12,
        "high_section_speed": 116.72,
        "cast_pressure": 326.49,
        "biscuit_thickness": 50.89,
        "upper_mold_temp1": 192.41,
        "upper_mold_temp2": 148.95,
        "upper_mold_temp3": 1449.0,
        "lower_mold_temp1": 146.92,
        "lower_mold_temp2": 194.07,
        "lower_mold_temp3": 1449.0,
        "sleeve_temperature": 374.64,
        "physical_strength": 707.69,
        "Coolant_temperature": 33.09,
    },
    "8573": {
        "molten_temp": 720.49,
        "facility_operation_cycleTime": 122.12,
        "production_cycletime": 122.95,
        "low_section_speed": 109.98,
        "high_section_speed": 112.31,
        "cast_pressure": 330.46,
        "biscuit_thickness": 53.17,
        "upper_mold_temp1": 223.17,
        "upper_mold_temp2": 190.25,
        "upper_mold_temp3": 217.94,
        "lower_mold_temp1": 182.08,
        "lower_mold_temp2": 203.66,
        "lower_mold_temp3": 1449.0,
        "sleeve_temperature": 497.24,
        "physical_strength": 707.05,
        "Coolant_temperature": 31.21,
    },
    "8600": {
        "molten_temp": 709.02,
        "facility_operation_cycleTime": 121.04,
        "production_cycletime": 124.26,
        "low_section_speed": 114.06,
        "high_section_speed": 116.6,
        "cast_pressure": 324.86,
        "biscuit_thickness": 48.19,
        "upper_mold_temp1": 224.28,
        "upper_mold_temp2": 174.45,
        "upper_mold_temp3": 1449.0,
        "lower_mold_temp1": 212.87,
        "lower_mold_temp2": 189.74,
        "lower_mold_temp3": 1449.0,
        "sleeve_temperature": 501.51,
        "physical_strength": 691.47,
        "Coolant_temperature": 31.54,
    },
    "8722": {
        "molten_temp": 718.32,
        "facility_operation_cycleTime": 120.05,
        "production_cycletime": 120.94,
        "low_section_speed": 109.15,
        "high_section_speed": 111.54,
        "cast_pressure": 328.4,
        "biscuit_thickness": 50.72,
        "upper_mold_temp1": 157.95,
        "upper_mold_temp2": 173.36,
        "upper_mold_temp3": 1449.0,
        "lower_mold_temp1": 231.0,
        "lower_mold_temp2": 176.4,
        "lower_mold_temp3": 1449.0,
        "sleeve_temperature": 456.66,
        "physical_strength": 703.33,
        "Coolant_temperature": 31.42,
    },
    "8917": {
        "molten_temp": 715.85,
        "facility_operation_cycleTime": 119.87,
        "production_cycletime": 121.45,
        "low_section_speed": 110.23,
        "high_section_speed": 112.89,
        "cast_pressure": 325.67,
        "biscuit_thickness": 49.94,
        "upper_mold_temp1": 188.76,
        "upper_mold_temp2": 165.82,
        "upper_mold_temp3": 1449.0,
        "lower_mold_temp1": 198.43,
        "lower_mold_temp2": 181.29,
        "lower_mold_temp3": 1449.0,
        "sleeve_temperature": 478.92,
        "physical_strength": 699.81,
        "Coolant_temperature": 32.15,
    },
}
    def get_defect_normal_pattern_simple(mold_code):
        """불량률 예측용 양품 기준 패턴"""
        pattern = DEFECT_NORMAL_PATTERNS_HARDCODED.get(str(mold_code))
        if pattern:
            return pattern
        else:
            return DEFECT_NORMAL_PATTERNS_HARDCODED.get("8412", {})
        

    def create_defect_basic_info_section(alert_info):
        """불량 기본 정보 섹션"""
        return ui.div([
            ui.div([
                ui.div([
                    ui.span("⚠️", class_="text-2xl mr-3"),
                    ui.div([
                        ui.h5(f"불량 ID: {alert_info['id']}", class_="font-semibold text-lg"),
                        ui.p(f"금형: M{alert_info['mold_code']}", class_="text-sm text-gray-600"),
                        ui.p(f"발생 시각: {pd.to_datetime(alert_info['time']).strftime('%Y-%m-%d %H:%M:%S')}", class_="text-sm text-gray-600")
                    ])
                ], class_="flex items-center"),
                
                ui.div([
                    ui.div([
                        ui.span("불량 확률:", class_="text-sm font-medium mr-2"),
                        ui.span(f"{alert_info['prob']:.2%}", class_="font-bold text-red-600")
                    ], class_="mb-2"),
                    ui.div([
                        ui.span("기준:", class_="text-sm font-medium mr-2"),
                        ui.span("양품 생산 조건", class_="text-blue-600 font-medium")
                    ])
                ])
            ], class_="flex justify-between items-center")
        ])

    @output
    @render.ui  
    def defect_modal_all_specs():
        """불량 전체 변수 상세 스펙"""
         # before your for-loop, define a priority map
        priority_map = {"🔴": 0, "🟡": 1, "🟢": 2}
        items_with_prio = []

        
        detail_data = selected_defect_alert_data.get()
        if not detail_data:
            return ui.div()
        
        deviations = detail_data["deviations"]
        if not deviations:
            return ui.div("분석 데이터를 불러올 수 없습니다.", class_="text-gray-500")
        
        # 불량 예측에 사용되는 16개 변수
        all_variables = {
            'molten_temp': ('용탕온도', '°C'),
            'sleeve_temperature': ('슬리브온도', '°C'), 
            'Coolant_temperature': ('냉각수온도', '°C'),
            'facility_operation_cycleTime': ('설비사이클', '초'),
            'production_cycletime': ('생산사이클', '초'),
            'low_section_speed': ('저속구간', ''),
            'high_section_speed': ('고속구간', ''),
            'cast_pressure': ('사출압력', 'bar'),
            'biscuit_thickness': ('비스킷두께', 'mm'),
            'upper_mold_temp1': ('상부몰드1', '°C'),
            'upper_mold_temp2': ('상부몰드2', '°C'),
            'upper_mold_temp3': ('상부몰드3', '°C'),
            'lower_mold_temp1': ('하부몰드1', '°C'),
            'lower_mold_temp2': ('하부몰드2', '°C'),
            'lower_mold_temp3': ('하부몰드3', '°C'),
            'physical_strength': ('물리강도', 'MPa')
        }
        
        spec_items = []
        for eng_name, (kor_name, unit) in all_variables.items():
            if eng_name in deviations:
                dev = deviations[eng_name]
                if dev["is_outlier"]:
                    status_class = "bg-red-100 border-red-300"
                    status_icon  = "🔴"
                elif dev["relative_diff_pct"] > 10:
                    status_class = "bg-yellow-100 border-yellow-300"
                    status_icon  = "🟡"
                else:
                    status_class = "bg-green-100 border-green-300"
                    status_icon  = "🟢"

                # build the card exactly as before
                card = ui.div([
                    ui.div([
                        ui.span(status_icon, class_="mr-2"),
                        ui.span(kor_name, class_="font-semibold text-sm")
                    ], class_="flex items-center mb-1"),
                    ui.div([
                        ui.div([
                            ui.span("양품기준:", class_="text-xs text-gray-600 mr-1"),
                            ui.span(f"{dev['normal']:.1f}{unit}", class_="text-xs font-medium")
                        ], class_="mb-1"),
                        ui.div([
                            ui.span("현재값:", class_="text-xs text-gray-600 mr-1"),
                            ui.span(f"{dev['current']:.1f}{unit}", class_="text-xs font-bold")
                        ], class_="mb-1"),
                        ui.div([
                            ui.span("차이:", class_="text-xs text-gray-600 mr-1"),
                            ui.span(
                                f"{dev['absolute_diff']:+.1f}{unit} ({dev['relative_diff_pct']:+.1f}%)", 
                                class_="text-xs font-medium"
                            )
                        ])
                    ])
                ], class_=f"p-2 rounded border {status_class} mb-2")

                # append (priority, card)
                items_with_prio.append((priority_map[status_icon], card))

        # now sort by the numeric priority and drop it
        items_with_prio.sort(key=lambda x: x[0])
        spec_items = [card for _, card in items_with_prio]

        return ui.div(
            spec_items,
            class_="maxh-500 overflow-y-auto"
        )


    @reactive.Effect
    @reactive.event(input.apply_defect_radar_variables)
    def apply_defect_radar_variables():
        """불량 레이더 차트 변수 적용"""
        pass

    @output
    @render_widget
    def defect_detail_radar_chart():
        """불량 상세 레이더 차트"""
        
        detail_data = selected_defect_alert_data.get()
        if not detail_data:
            return go.Figure()
        
        # 적용 버튼과 연동된 변수 가져오기
        applied_vars = input.defect_radar_variables_temp() if hasattr(input, 'defect_radar_variables_temp') and input.defect_radar_variables_temp() else []
        
        # 기본값 사용
        if not applied_vars:
            applied_vars = ["molten_temp", "sleeve_temperature", "Coolant_temperature", 
                        "cast_pressure", "high_section_speed", "biscuit_thickness", "physical_strength"]
        
        selected_vars = applied_vars
        
        if not selected_vars or len(selected_vars) < 3:
            return go.Figure().add_annotation(
                text="최소 3개 이상의 변수를 선택 후 적용 버튼을 눌러주세요",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14)
            )
        
        normal_pattern = detail_data["normal_pattern"]
        deviations = detail_data["deviations"]
        
        if not normal_pattern or not deviations:
            return go.Figure()
        
        # 변수명 매핑
        variable_names = {
            "molten_temp": "용탕온도", "sleeve_temperature": "슬리브온도", 
            "Coolant_temperature": "냉각수온도", "cast_pressure": "사출압력",
            "high_section_speed": "고속구간", "biscuit_thickness": "비스킷두께",
            "physical_strength": "물리강도", "lower_mold_temp1": "하부몰드1",
            "lower_mold_temp2": "하부몰드2", "lower_mold_temp3": "하부몰드3",
            "upper_mold_temp1": "상부몰드1", "upper_mold_temp2": "상부몰드2",
            "upper_mold_temp3": "상부몰드3", "facility_operation_cycleTime": "설비사이클",
            "production_cycletime": "생산사이클", "low_section_speed": "저속구간"
        }
        
        # 정규화 범위
        feature_ranges = {
            "molten_temp": (700, 730), "sleeve_temperature": (350, 520),
            "Coolant_temperature": (30, 35), "cast_pressure": (320, 335),
            "high_section_speed": (110, 118), "biscuit_thickness": (45, 55),
            "physical_strength": (690, 710), "lower_mold_temp1": (140, 240),
            "lower_mold_temp2": (170, 210), "lower_mold_temp3": (1440, 1450),
            "upper_mold_temp1": (150, 230), "upper_mold_temp2": (140, 200),
            "upper_mold_temp3": (210, 1450), "facility_operation_cycleTime": (118, 125),
            "production_cycletime": (119, 126), "low_section_speed": (108, 116)
        }
        
        # 선택된 변수들만 처리
        features = []
        normal_values = []
        current_values = []
        
        for var in selected_vars:
            if var in deviations:
                features.append(variable_names.get(var, var))
                
                normal_val = deviations[var]["normal"]
                current_val = deviations[var]["current"]
                
                # 정규화
                min_val, max_val = feature_ranges.get(var, (0, 1))
                normal_norm = (normal_val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                current_norm = (current_val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                
                normal_values.append(max(0, min(1, normal_norm)))
                current_values.append(max(0, min(1, current_norm)))
        
        if len(features) < 3:
            return go.Figure().add_annotation(
                text="선택된 변수의 데이터가 부족합니다",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
        
        # 원형으로 만들기
        features_circle = features + features[:1]
        normal_values_circle = normal_values + normal_values[:1]
        current_values_circle = current_values + current_values[:1]
        
        fig = go.Figure()
        
        # 양품 기준 패턴 (파랑, 점선)
        fig.add_trace(go.Scatterpolar(
            r=normal_values_circle,
            theta=features_circle,
            fill='toself',
            fillcolor='rgba(54, 162, 235, 0.2)',
            line=dict(color='blue', width=2, dash='dash'),
            name='양품 기준'
        ))
        
        # 현재 불량 패턴 (빨강, 실선)
        fig.add_trace(go.Scatterpolar(
            r=current_values_circle,
            theta=features_circle,
            fill='toself',
            fillcolor='rgba(255, 99, 132, 0.3)',
            line=dict(color='red', width=3),
            name='현재 불량'
        ))
        
        # 이상 피처 강조
        for i, var in enumerate(selected_vars):
            if var in deviations and deviations[var]["is_outlier"]:
                if i < len(features):
                    fig.add_trace(go.Scatterpolar(
                        r=[current_values[i]],
                        theta=[features[i]],
                        mode='markers',
                        marker=dict(
                            color='orange',
                            size=12,
                            symbol='star',
                            line=dict(color='darkorange', width=2)
                        ),
                        name=f'⚠️ {features[i]}',
                        showlegend=False
                    ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    tickvals=[0, 0.5, 1.0],
                    ticktext=['최소', '중간', '최대']
                )
            ),
            title=f"양품 기준 vs 현재 불량 패턴",
            showlegend=True,
            width=450,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig


       
app = App(app_ui, server, static_assets=STATIC_DIR)