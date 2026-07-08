# -*- coding: utf-8 -*-
"""
최종 결합 모델 학습 스크립트: 두 사전 단계의 강점을 결합하여 최종 부정맥 탐지 모델을 구축합니다.
1. CLIP 대조학습으로 PPG 신호와 임상 정보(Clinical Info) 간의 관계를 학습한 Pre-trained ResNet Encoder.
2. 부정맥 분류(Classification) 및 이상 구간 분할(Segmentation)을 수행하는 Hybrid U-Net.

[최종 결합 전략]
- Encoder: CLIP으로 사전 학습된 ResNet1DEncoder를 가져와 U-Net의 인코더(Backbone)로 사용 (Transfer Learning).
- Decoder: ResNet의 특징맵(Feature Maps) 차원에 맞춰 최적화된 U-Net Decoder 사용.
- Input: PPG 신호(Waveform) + 임상 데이터(CSV) + HRV 특징.
"""


# 1. 라이브러리 임포트 (이후 셀에서 사용할 핵심 라이브러리 사전 로드)
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
import os

# 1. 기존 drive 폴더가 있다면 삭제 (초기화)
if os.path.exists('/content/drive'):
    !rm -rf /content/drive

# 2. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 3. 기본 작업 경로 설정
# 앞으로 사용할 train.csv, test.csv 및 파형 데이터들이 있는 폴더로 작업 경로를 고정합니다.
PROJECT_PATH = '/content/drive/MyDrive/종합설계/PPG_AD_Dataset'

if os.path.exists(PROJECT_PATH):
    os.chdir(PROJECT_PATH)
    print(f"✅ 작업 디렉토리 변경 완료: {os.getcwd()}")
    print("   이제 'train.csv', 'valid.csv' 등을 바로 불러올 수 있습니다.")
else:
    print(f"⚠️ 경로를 찾을 수 없습니다: {PROJECT_PATH}")
    print("   마운트 상태나 폴더 경로를 확인해주세요.")

import os
import re
import glob
import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

# Scikit-learn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# --- 1. 기본 경로 설정 ---

# 드라이브 경로 (이전 셀에서 PROJECT_PATH로 이동했으므로 상대 경로 활용 가능하나, 명시적으로 전체 경로 사용)
BASE_PROJECT_PATH = "/content/drive/MyDrive/종합설계"
DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "PPG_AD_Dataset")

# CLIP 사전학습 단계에서 저장한 가중치 경로
CLIP_WEIGHTS_PATH = os.path.join(DATASET_PATH, "checkpoints", "clip_biobert_hyh_xai.pth")

# 최종 모델 저장 경로
MODEL_SAVE_PATH = os.path.join(BASE_PROJECT_PATH, "unet_clip_biobert_hyh_xai.pth")

print(f"프로젝트 경로: {BASE_PROJECT_PATH}")
print(f"데이터셋 경로: {DATASET_PATH}")
print(f"CLIP 가중치 경로: {CLIP_WEIGHTS_PATH}")
print(f"최종 모델 저장 경로: {MODEL_SAVE_PATH}")

# --- 2. 시드 고정 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 시드 고정 완료 (seed={seed})")

set_seed(42)

# --- 3. 모델 구성 요소 정의 ---

# [3-1] ResNet Block (1125영호CLIP.ipynb와 동일)
class ResidualBlock(nn.Module):
    """ResNet의 기본 블록: Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# [3-2] CLIP Encoder Backbone (1125영호CLIP.ipynb 기반 + 수정 없음)
# 중요: 가중치 로드를 위해 레이어 이름(self.initial, self.layer1 등)은 CLIP 코드와 똑같아야 함.
class ResNet1D_Backbone(nn.Module):
    """CLIP의 ResNet1DEncoder와 동일한 구조를 가지며, U-Net을 위해 중간 특징맵을 반환"""
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()

        # 1. 초기 진입 (Stride 2 -> L/2, MaxPool -> L/4)
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Residual Blocks (각 단계마다 채널 2배, 길이 1/2)
        self.layer1 = ResidualBlock(base_filters, base_filters*2, stride=2) # 32 -> 64, L/8
        self.layer2 = ResidualBlock(base_filters*2, base_filters*4, stride=2) # 64 -> 128, L/16
        self.layer3 = ResidualBlock(base_filters*4, base_filters*8, stride=2) # 128 -> 256, L/32

    def forward(self, x):
        # U-Net의 Skip Connection을 위해 중간 특징맵들을 리스트로 반환
        features = []

        x = self.initial(x)
        features.append(x) # [0]: Base features (L/4, 32ch)

        x = self.layer1(x)
        features.append(x) # [1]: Layer1 features (L/8, 64ch)

        x = self.layer2(x)
        features.append(x) # [2]: Layer2 features (L/16, 128ch)

        x = self.layer3(x)
        features.append(x) # [3]: Layer3 features (L/32, 256ch) - Bottleneck

        return features

print("✅ ResNet1D Backbone 정의 완료 (Skip Connection 지원).")


# [3-3] U-Net Decoder Components (1124Unet.ipynb와 동일)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, gn_groups=8):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: Up-sampled feature (Decoder 쪽)
        # x2: Skip Connection feature (Encoder 쪽)
        x1 = self.up(x1)

        # 크기 보정 (Padding)
        diff = x2.size(2) - x1.size(2)
        if diff != 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SE1D(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, C, r=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, C // r), nn.ReLU(inplace=True),
            nn.Linear(C // r, C), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w

print("✅ U-Net Decoder 구성 요소 정의 완료.")

# --- 4. 데이터 통계, HRV 함수 및 임상 데이터 설정 ---

# [4-1] data_stats.json 로드
# 1124unet.py에서 계산한 통계치(평균, 표준편차, 타겟 길이, HRV 통계) 로드
CONFIG_PATH = os.path.join(DATASET_PATH, "data_stats.json")

N_HRV_FEATURES = 4 # HR, SDNN, RMSSD, pNN50
eps_hrv = 1e-6 # HRV 정규화 시 분모 0 방지

try:
    with open(CONFIG_PATH, "r") as f:
        stats_cfg = json.load(f)

    # 1) 파형 정규화 통계
    # 주의: 이 통계치는 글로벌 정규화에 사용되며, CLIP이 샘플별 정규화를 사용했다면 불일치가 발생할 수 있음
    TRAIN_MEAN = float(stats_cfg.get("train_mean", 0.0))
    TRAIN_STD = float(stats_cfg.get("train_std", 1.0))
    if TRAIN_STD == 0: TRAIN_STD = 1.0

    # 2) HRV 정규화 통계
    def _as_vec(x, default_val):
        default_vec = [default_val] * N_HRV_FEATURES
        arr = np.array(x if x is not None else default_vec, dtype=np.float32).reshape(-1)
        if arr.size != N_HRV_FEATURES:
            # print(f"⚠️ HRV 통계치 크기 불일치! 기본값 사용. (기대: {N_HRV_FEATURES}, 실제: {arr.size})")
            arr = np.array(default_vec, dtype=np.float32)
        return arr

    HRV_MEAN  = _as_vec(stats_cfg.get("hrv_mean"),  0.0)
    HRV_STD  = _as_vec(stats_cfg.get("hrv_std"), 1.0)
    HRV_MEDIAN = _as_vec(stats_cfg.get("hrv_median"), 0.0) # 결측 대치용

    # 3) 타겟 길이
    TARGET_LENGTH = int(stats_cfg.get("target_length", 286)) # 1124unet과 동일해야 함

    print("✅ data_stats.json 로드 완료:")
    print(f"  - TRAIN_MEAN: {TRAIN_MEAN:.4f}, TRAIN_STD: {TRAIN_STD:.4f}")
    print(f"  - TARGET_LENGTH: {TARGET_LENGTH}")
    # print(f"  - HRV_MEAN (4D): {HRV_MEAN}")

except FileNotFoundError:
    print(f"❌ '{CONFIG_PATH}'를 찾을 수 없습니다. 1124unet.py를 먼저 실행하세요.")
    # Fallback
    TRAIN_MEAN, TRAIN_STD, TARGET_LENGTH = 0.0, 1.0, 286
    HRV_MEAN  = np.zeros(N_HRV_FEATURES, dtype=np.float32)
    HRV_STD = np.ones (N_HRV_FEATURES, dtype=np.float32)
    HRV_MEDIAN = np.zeros(N_HRV_FEATURES, dtype=np.float32)


# [4-2] 클래스 맵 및 디바이스 설정
NUM_CLASSES = 4 # normal, af, b, t
CLASS_MAP = {'normal': 0, 'af': 1, 'b': 2, 't': 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 클래스 맵: {CLASS_MAP}")
print(f"✅ 사용 디바이스: {device}")


# [4-3] HRV 특징 추출 함수 (1124unet.py와 동일)
def _compute_hrv_raw(signal: np.ndarray, fs: float = 100.0) -> np.ndarray | None:
    """HRV 4특성(정규화 전)을 계산. [HR, SDNN, RMSSD, pNN50]"""
    if signal is None or len(signal) < 10 or not np.any(np.isfinite(signal)):
        return None
    try:
        min_height = np.nanquantile(signal, 0.6)
    except Exception:
        return None

    peaks, _ = find_peaks(signal, height=min_height, distance=int(fs * 0.3))
    if len(peaks) < 3: return None

    rri_ms = np.diff(peaks) * (1000.0 / fs)
    if rri_ms.size < 2 or not np.all(np.isfinite(rri_ms)): return None

    mean_rri = np.mean(rri_ms)
    hr_bpm  = 60000.0 / mean_rri if mean_rri > 0 else np.nan
    sdnn  = np.std(rri_ms)
    diff_rri = np.diff(rri_ms)
    if diff_rri.size == 0: return None
    rmssd = np.sqrt(np.mean(diff_rri ** 2))
    pnn50 = (np.sum(np.abs(diff_rri) > 50.0) / diff_rri.size) * 100.0

    feats = np.array([hr_bpm, sdnn, rmssd, pnn50], dtype=np.float32)
    if not np.any(np.isfinite(feats)): return None
    return feats

def get_hrv_features(signal: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """HRV 4특성 계산 -> 결측 대치 -> 표준화"""
    raw = _compute_hrv_raw(signal, fs=fs)
    if raw is None:
        raw = HRV_MEDIAN.copy()
    else:
        bad_mask = ~np.isfinite(raw)
        if np.any(bad_mask):
            raw[bad_mask] = HRV_MEDIAN[bad_mask]

    standardized = (raw - HRV_MEAN) / (HRV_STD + eps_hrv)
    return standardized.astype(np.float32)

print("✅ HRV 특징 추출 함수 정의 완료.")


# [4-4] 임상 데이터(Clinical Info) 설정: CLIP 학습 시 사용한 컬럼 정의와 동일하게 유지 (17개 특징)

# 1. 연속형 컬럼 (9개)
CONTINUOUS_COLS = [
    "bmi", "opdur", "preop_na", "preop_bun", "preop_cr", "age",
    "preop_k", "intraop_eph", "intraop_phe"
]

# 2. 범주형 컬럼 (2개 -> OHE 시 4개)
CATEGORICAL_COLS = {"sex": ["M", "F"], "emop": ["N", "Y"]}

# 3. 이진 문자열 컬럼 (2개 -> OHE 시 4개)
BINARY_STR_COLS = {"preop_dm": ["N", "Y"], "preop_htn": ["N", "Y"]}

# One-Hot Encoding (OHE) 시 예상되는 총 특징 수 계산 (모든 카테고리 포함)
N_CONT = len(CONTINUOUS_COLS)  # 9
N_CAT_OHE = sum(len(v) for v in CATEGORICAL_COLS.values()) # 2 + 2 = 4
N_BIN_OHE = sum(len(v) for v in BINARY_STR_COLS.values()) # 2 + 2 = 4

N_CLINICAL_FEATURES = N_CONT + N_CAT_OHE + N_BIN_OHE # 9 + 4 + 4 = 17

# 전역 스케일러 (나중에 fit_clinical_scaler로 업데이트됨)
CONT_MEAN = {c: 0.0 for c in CONTINUOUS_COLS}
CONT_STD = {c: 1.0 for c in CONTINUOUS_COLS}


print(f"✅ 임상 데이터 총 특징 수 설정 완료: {N_CLINICAL_FEATURES}")
print("✅ 임상 데이터 설정 및 전처리 함수 정의 완료.")

# --- 5. 임상 데이터 전처리 및 로드 (CLIP 방식 준수) ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

# 임상 특징 리스트 (CLIP 모델 학습 시 사용된 13개 특징을 그대로 사용)
# [Continuous(9)] + [OHE, drop_first=True (4)] = 13
CLINICAL_FEATURES_LIST = [
    'age', 'bmi', 'opdur', 'preop_na', 'preop_bun', 'preop_cr', 'preop_k',
    'intraop_eph', 'intraop_phe',
    'sex_M', 'emop_Y', 'preop_dm_Y', 'preop_htn_Y'
]
N_CLINICAL_FEATURES = len(CLINICAL_FEATURES_LIST)

def preprocess_clinical_data(csv_path):
    """
    임상 정보 CSV 파일을 로드하고, CLIP 모델 학습 시 사용된 방식으로 전처리 및 정규화합니다.
    (OHE, NaN 처리, StandardScaler 후 Min-Max 유사 스케일링 적용)

    반환: {caseid: {feature1: value, feature2: value, ...}, ...} 형태의 딕셔너리
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ 임상 데이터 파일 없음: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    # 1. caseid를 인덱스로 설정
    df = df.set_index('caseid')

    # 2. One-Hot Encoding (OHE) - 범주형 변수 처리 (Drop First = True)
    # sex, emop, preop_dm, preop_htn
    categorical_cols = ['sex', 'emop', 'preop_dm', 'preop_htn']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. CLIP 모델에서 사용된 13개 특징만 선택 및 보정
    df_selected = df.copy()

    # 누락된 OHE 컬럼 (예: 해당 데이터셋에 'emop_Y'가 전혀 없는 경우)을 0으로 채움
    for col in CLINICAL_FEATURES_LIST:
        if col not in df_selected.columns:
            if col.endswith('_M') or col.endswith('_Y'):
                df_selected[col] = 0
            else:
                # 원본 연속형 컬럼이 누락된 경우 (예상치 못한 오류)
                raise ValueError(f"Missing original clinical feature column: {col}")

    # 최종 13개 특징으로 DF 정렬/선택
    df_selected = df_selected[CLINICAL_FEATURES_LIST]

    # 4. 결측치(NaN) 처리 - 평균값으로 대체
    df_selected = df_selected.fillna(df_selected.mean())

    # 5. 스케일링 (CLIP 코드의 Min-Max 유사 스케일링 로직 재현)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_selected)

    min_val = np.min(scaled_features, axis=0)
    max_val = np.max(scaled_features, axis=0)

    # 분모가 0인 경우 (단일 값만 있는 특징)를 1로 설정하여 0-1 범위 유지
    diff = max_val - min_val
    diff[diff == 0] = 1

    normalized_features = (scaled_features - min_val) / diff

    df_normalized = pd.DataFrame(normalized_features, columns=CLINICAL_FEATURES_LIST, index=df_selected.index)

    # 6. 빠른 조회를 위한 딕셔너리 형태로 변환
    clinical_map = df_normalized.to_dict('index')

    print(f"✅ Clinical data loaded and preprocessed. Total cases: {len(clinical_map)}, Features per case: {N_CLINICAL_FEATURES}")
    return clinical_map

print(f"N_CLINICAL_FEATURES (CLIP): {N_CLINICAL_FEATURES}")

# --- 6. 결합 모델 (ResNet-UNet Hybrid) 정의 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# N_CLINICAL_FEATURES 상수는 13으로 정의되었습니다.
# N_HRV_FEATURES 상수는 4로 정의되었습니다.

class UNet1D_ResNet_Combined(nn.Module):
    # n_hrv_features 인자를 추가합니다.
    def __init__(self, n_channels, n_classes, n_clinical_features=N_CLINICAL_FEATURES, n_hrv_features=N_HRV_FEATURES):
        super().__init__()
        self.n_classes = n_classes
        self.n_clinical_features = n_clinical_features
        self.n_hrv_features = n_hrv_features
        self.encoder_feature_dim = 256
        self.clip_feature_dim = 128  # CLIP의 최종 특징 차원

        # 1. Encoder (CLIP Backbone)
        # ResNet1D_Backbone의 최종 출력 (Bottleneck)은 256 채널입니다.
        self.encoder = ResNet1D_Backbone(in_channels=n_channels, base_filters=32)

        # CLIP 특징 추출 레이어 (256 -> 128), CLIP Encoder의 self.fc에 해당하며 CLIP 가중치를 로드해야 함
        self.clip_projection = nn.Linear(self.encoder_feature_dim, self.clip_feature_dim)

        # 2. Decoder (U-Net 구조)
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)

        # Final Upsample to Original Length
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )

        # 3. Heads
        self.seg_head = nn.Conv1d(16, 1, kernel_size=1)

        # 분류 헤드 입력 차원: 128 (CLIP Feature) + 13 (Clinical) + 4 (HRV) = 145
        clf_in_dim = self.clip_feature_dim + self.n_clinical_features + self.n_hrv_features

        self.clf_head = nn.Sequential(
            nn.Linear(clf_in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, n_classes)
        )

    def forward(self, signal_input, clinical_features, hrv_features):
        # 1. Encoder (ResNet)
        features = self.encoder(signal_input)
        x3 = features[3] # Bottleneck (256ch)

        # 2. Classification Path
        gap = F.adaptive_avg_pool1d(x3, 1).squeeze(-1) # [B, 256]


        # 256차원 GAP 특징을 128차원 CLIP 특징 공간으로 투영
        clip_feature = self.clip_projection(gap) # [B, 128]

        # 128차원 CLIP 특징, 임상 특징, HRV 특징 결합
        combined = torch.cat([clip_feature, clinical_features, hrv_features], dim=1) # [B, 145]
        clf_logits = self.clf_head(combined)

        # Segmentation Path
        d1 = self.up1(x3, features[2])
        d2 = self.up2(d1, features[1])
        d3 = self.up3(d2, features[0])
        d_final = self.final_up(d3)

        seg_logits = self.seg_head(d_final)

        # [크기 보정]
        if seg_logits.size(2) != signal_input.size(2):
            seg_logits = F.interpolate(
                seg_logits,
                size=signal_input.size(2),
                mode='linear',
                align_corners=False
            )

        y_seg = seg_logits.squeeze(1)

        return clf_logits, y_seg

    def load_clip_weights(self, weights_path, device):
        """CLIP 체크포인트 로드 (wave_enc 키 처리)"""
        if not os.path.exists(weights_path):
            print(f"⚠️ 가중치 파일 없음: {weights_path}")
            return

        print(f"🔄 CLIP 가중치 로드 시도: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)

        # 키 자동 탐색
        source_state = checkpoint
        if 'wave_enc' in checkpoint:
            source_state = checkpoint['wave_enc']
            print("   -> 'wave_enc' 키 감지됨.")
        elif 'wave_encoder_state_dict' in checkpoint:
            source_state = checkpoint['wave_encoder_state_dict']

        model_state = self.state_dict()
        loaded_state = {}
        match_count = 0

        # Encoder (ResNet Backbone)와 CLIP Projection Layer 가중치 로드
        for k, v in source_state.items():
            k = k.replace('module.', '')

            # 1) ResNet Backbone 매핑
            # wave_enc.initial.* -> encoder.initial.*
            if k.startswith('initial.') or k.startswith('layer'):
                target_key = f"encoder.{k}"
                if target_key in model_state and model_state[target_key].shape == v.shape:
                    loaded_state[target_key] = v
                    match_count += 1

            # 2) Projection Layer 매핑 (fc 레이어)
            # BioBERT CLIP 코드에서 ResNet의 마지막은 'self.fc' (256->128) 였습니다.
            # 이를 Combined Model의 'clip_projection'에 매핑합니다.
            elif k.startswith('fc.'):
                target_key = f"clip_projection.{k.split('fc.')[1]}"
                if target_key in model_state and model_state[target_key].shape == v.shape:
                    loaded_state[target_key] = v
                    match_count += 1


        if match_count > 0:
            # strict=False를 사용하여 Decoder, Seg Head, Clf Head는 무시하고 로드합니다.
            self.load_state_dict(loaded_state, strict=False)
            print(f"✅ Pre-trained 가중치 로드 완료! (Encoder/Projection {match_count} 레이어 매칭)")
        else:
            print("⚠️ 매칭되는 가중치가 없습니다. 학습이 잘 안 될 수 있습니다.")

print("✅ 최종 결합 모델 (UNet1D_ResNet_Combined) 정의 수정 완료 (CLIP 특징 128차원 활용).")

# --- 7. 데이터셋 클래스 정의 (CLIP의 샘플별 정규화 반영) ---

import torch.nn.functional as F # F.interpolate 사용을 위해 추가/확인
from scipy import stats # stats.mode 사용을 위해 추가/확인
import random # random.randint 사용을 위해 추가/확인
import os
import torch
import numpy as np
import pandas as pd # pd.read_csv 사용을 위해 추가/확인
from torch.utils.data import Dataset # Dataset 임포트 확인

class PPGDataset_Combined(Dataset):
    """
    [최종 결합 데이터셋]
    1. 파형(Waveform) 로드 및 리샘플링 (Target Length)
    2. 파형 정규화: CLIP과의 일관성을 위해 샘플별 정규화(Sample-wise Normalization) 적용.
    3. 임상 데이터(Clinical Info) 및 HRV 특징 매칭.
    """
    def __init__(self, data_dirs, clinical_csv_path, target_length, fs=100.0, is_test=False):
        """
        data_dirs: ['.', './train'] 처럼 탐색할 폴더 리스트
        clinical_csv_path: 'train.csv' 등 임상 정보 파일 경로
        """
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.target_length = int(target_length)
        self.fs = float(fs)
        self.is_test = is_test

        # 1. 임상 데이터 로드
        self.clinical_map = preprocess_clinical_data(clinical_csv_path)

        # 2. 파일 매칭 및 리스트 생성
        self.file_list = []
        self._scan_and_match_files()

        # 3. 라벨 미리 로드 (Sampler용)
        self.labels = self._get_all_labels()

        print(f"✅ 데이터셋 초기화 완료: {len(self.file_list)} 샘플")
        if self.clinical_map:
            print(f"   (임상 데이터 매칭: {len(self.clinical_map)}명 정보 로드됨)")

    # _scan_and_match_files 메소드는 원본과 동일하게 유지
    def _scan_and_match_files(self):
        """폴더를 스캔하여 파형 파일과 임상 정보를 매칭"""
        # 중복 방지를 위한 세트
        seen_files = set()

        for d in self.data_dirs:
            if not os.path.isdir(d):
                continue

            # 해당 폴더의 모든 파일 스캔
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.npy', '.npz')):
                        # 중복 체크
                        full_path = os.path.join(root, f)
                        if full_path in seen_files:
                            continue
                        seen_files.add(full_path)

                        # CaseID 추출 (파일명: 1234_win0.npz -> 1234)
                        try:
                            caseid = int(f.split('_')[0])
                        except:
                            caseid = -1 # 매칭 불가

                        self.file_list.append({
                            'path': full_path,
                            'filename': f,
                            'caseid': caseid
                        })

    # _get_all_labels 메소드는 원본과 동일하게 유지
    def _get_all_labels(self):
        labels = []
        label_map_vec = {0: 'normal', 1: 'normal', 2: 'af', 3: 'b', 4: 't'}

        for item in self.file_list:
            fname = item['filename']
            fpath = item['path']
            label_name = 'normal'

            if fname.endswith('.npz'):
                try:
                    with np.load(fpath, allow_pickle=True) as data:
                        cls_raw = data.get('class')
                        if cls_raw is not None:
                            if isinstance(cls_raw, np.ndarray) and cls_raw.size > 1:
                                nz = cls_raw[cls_raw != 0]
                                if nz.size > 0:
                                    mode_val = stats.mode(nz, keepdims=True)[0][0]
                                    label_name = label_map_vec.get(int(mode_val), 'normal')
                            else:
                                label_name = label_map_vec.get(int(np.array(cls_raw).item()), 'normal')
                except:
                    pass

            labels.append(CLASS_MAP.get(label_name, 0))
        return labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        fpath = item['path']
        fname = item['filename']
        caseid = item['caseid']

        # 1. 파형 & 마스크 로드
        label_name = 'normal'
        try:
            if fname.endswith('.npy'):
                sig = np.load(fpath).astype(np.float32)
                mask = np.zeros_like(sig, dtype=np.float32)
            else: # .npz
                with np.load(fpath, allow_pickle=True) as data:
                    sig = data['signal'].astype(np.float32)
                    mask = data['mask'].astype(np.float32)

                    # 라벨 추출 (학습용)
                    cls_raw = data.get('class')
                    if cls_raw is not None:
                        # (위와 동일한 라벨 추출 로직)
                        if isinstance(cls_raw, np.ndarray) and cls_raw.size > 1:
                            nz = cls_raw[cls_raw != 0]
                            if nz.size > 0:
                                mode_val = stats.mode(nz, keepdims=True)[0][0]
                                label_map = {0: 'normal', 1: 'normal', 2: 'af', 3: 'b', 4: 't'}
                                label_name = label_map.get(int(mode_val), 'normal')
                        else:
                            label_map = {0: 'normal', 1: 'normal', 2: 'af', 3: 'b', 4: 't'}
                            label_name = label_map.get(int(np.array(cls_raw).item()), 'normal')

        except Exception as e:
            # 에러 시 랜덤 샘플 대체
            return self.__getitem__(random.randint(0, len(self)-1))

        # HRV 특징 계산 및 표준화 (get_hrv_features는 글로벌 HRV 통계 사용)
        hrv_feat = get_hrv_features(sig, fs=self.fs)
        hrv_t = torch.from_numpy(hrv_feat).float() # [4]


        # 2. 임상 특징 로드
        try:
            clinical_features = self.clinical_map[caseid]
        except KeyError:
            # print(f"Warning: caseid {caseid} not found in clinical_map. Returning zero vector for clinical features.")
            clinical_features_list = [0.0] * N_CLINICAL_FEATURES
        else:
            # 특징 값을 리스트로 변환
            clinical_features_list = [clinical_features[f] for f in CLINICAL_FEATURES_LIST]

        clinical_features_t = torch.tensor(clinical_features_list, dtype=torch.float32) # [13]


        # 3. 전처리 (Resampling & Norm)
        sig_t = torch.from_numpy(sig).unsqueeze(0).unsqueeze(0) # (1,1,L)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        sig_res = F.interpolate(sig_t, size=self.target_length, mode='linear', align_corners=False).squeeze(0) # (1, L)
        mask_res = F.interpolate(mask_t, size=self.target_length, mode='nearest').squeeze(0)

        # PPG 파형 정규화: 샘플별 정규화 (CLIP 학습 방식과 동일)
        x = sig_res.float() # (1, L)

        current_mean = x.mean()
        current_std = x.std()

        # 샘플별 정규화: (X - mean) / std
        if current_std > 1e-8:
            x = (x - current_mean) / current_std
        else:
            # 신호가 플랫한 경우: 평균만 빼서 0으로 중심 맞춤
            x = x - current_mean

        y_mask = mask_res.float()
        y_label = torch.tensor(CLASS_MAP.get(label_name, 0), dtype=torch.long)

        # 순서: (파형, 마스크, 라벨, 임상특징, HRV특징)
        return x, y_mask, y_label, clinical_features_t, hrv_t

print("✅ 최종 데이터셋 클래스 (PPGDataset_Combined) 정의 수정 완료 (CLIP 샘플별 정규화 반영).")

# --- 8. 학습/평가 함수, 손실 함수, EarlyStopping 정의 ---\n
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# [8-1] EarlyStopping 클래스
class EarlyStopping:
    """Validation Loss가 개선되지 않으면 학습을 조기 종료"""
    def __init__(self, patience=50, verbose=False, delta=1e-3,
                 path='new_best_model.pth', mode='min', monitor_name='val_total'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = float(delta)
        self.path = path
        self.mode = mode
        self.monitor_name = monitor_name

    def _is_improved(self, score):
        if self.best_score is None: return True
        if self.mode == 'max':
            return (score - self.best_score) > self.delta
        else:
            return (self.best_score - score) > self.delta

    def __call__(self, score, model, ref_metric_for_log=None, ref_name='ref'):
        if hasattr(score, "item"): score = float(score.item())

        if self.best_score is None or self._is_improved(score):
            self.best_score = score
            if self.verbose:
                log_msg = f"✅ {self.monitor_name} improved to {score:.6f}."
                if ref_metric_for_log is not None:
                    log_msg += f" ({ref_name}: {float(ref_metric_for_log):.6f})."
                print(log_msg + " Saving model...")
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping counter: {self.counter} / {self.patience} "
                      f"(best {self.monitor_name}: {self.best_score:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True

# [8-2] 손실 함수 (Clf: Focal, Seg: BCE+Dice)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.view(-1, 1)).view(-1)
        pt = logpt.exp()
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum() + self.smooth
        dice = (2 * inter + self.smooth) / denom
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, weight_bce=0.5, weight_dice=0.5):
        super().__init__(); self.bce = nn.BCEWithLogitsLoss(); self.dice = DiceLoss(smooth); self.wb, self.wd = weight_bce, weight_dice
    def forward(self, logits, targets):
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)

# [8-3] 학습 및 평가 함수
def _dice_coefficient_bin(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_bin = pred_bin.float(); target_bin = target_bin.float()
    intersect = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum() + eps
    dice = (2.0 * intersect + eps) / denom
    return dice

def train_one_epoch(model, dataloader, criterion_clf, criterion_seg, optimizer, device, alpha=1.0, beta=1.0, max_norm=1.0):
    model.train()
    total_loss, total_clf_loss, total_seg_loss = 0.0, 0.0, 0.0
    all_labels, all_preds = [], []

    for signals, masks, labels, clinical_features, hrv_features in tqdm(dataloader, desc="Training"):
        signals = signals.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        clinical_features = clinical_features.to(device)
        hrv_features = hrv_features.to(device)

        optimizer.zero_grad()

        clf_logits, seg_logits = model(signals, clinical_features, hrv_features)

        if masks.dim() == 3 and masks.size(1) == 1:
            target_mask = masks.squeeze(1)
        else:
            target_mask = masks

        clf_loss = criterion_clf(clf_logits, labels)
        seg_loss = criterion_seg(seg_logits, target_mask)
        loss = clf_loss * alpha + seg_loss * beta

        loss.backward()

        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss += float(loss.item())
        total_clf_loss += float(clf_loss.item())
        total_seg_loss += float(seg_loss.item())

        preds = torch.argmax(clf_logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    denom = max(1, len(dataloader))
    return total_loss/denom, total_clf_loss/denom, total_seg_loss/denom, \
           f1_score(all_labels, all_preds, average='macro', zero_division=0), \
           accuracy_score(all_labels, all_preds)

def evaluate(model, dataloader, criterion_clf, criterion_seg, device, alpha=1.0, beta=1.0, thr: float = 0.5):
    model.eval()
    total_loss, total_clf_loss, total_seg_loss = 0.0, 0.0, 0.0
    all_labels, all_preds, all_confidences = [], [], []
    total_pixel_acc = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for signals, masks, labels, clinical_features, hrv_features in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            clinical_features = clinical_features.to(device)
            hrv_features = hrv_features.to(device)

            clf_logits, seg_logits = model(signals, clinical_features, hrv_features)

            if masks.dim() == 3 and masks.size(1) == 1:
                target_mask = masks.squeeze(1)
            else:
                target_mask = masks

            clf_loss = criterion_clf(clf_logits, labels)
            seg_loss = criterion_seg(seg_logits, target_mask)
            loss = clf_loss * alpha + seg_loss * beta

            total_loss += float(loss.item())
            total_clf_loss += float(clf_loss.item())
            total_seg_loss += float(seg_loss.item())

            # 1. Classification Predictions
            probs = F.softmax(clf_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(torch.max(probs, dim=1)[0].cpu().numpy())

            # 2. Segmentation Predictions (Thresholding)
            seg_probs = torch.sigmoid(seg_logits)
            seg_bin = (seg_probs > thr).to(target_mask.dtype)

            # --- Post-processing: 분류 결과가 'Normal'(Index 0)이면 해당 샘플의 Segmentation Mask를 강제로 0 초기화 ---
            # preds는 배치 내의 예측 클래스 인덱스 [Batch_size]
            # 0번 클래스가 'normal'이라고 가정 (CLASS_MAP 확인: {'normal': 0, ...})
            normal_indices = (preds == 0) # Boolean Mask: Normal인 위치만 True

            # Normal로 예측된 샘플들의 마스크를 0으로 설정
            if normal_indices.any():
                seg_bin[normal_indices] = 0.0
            # ------------------------------------------

            # Pixel Accuracy 및 Dice 계산 (후처리된 seg_bin 사용)
            total_pixel_acc += float((seg_bin == target_mask).float().mean().item())
            total_dice += float(_dice_coefficient_bin(seg_bin, target_mask).item())

    denom = max(1, len(dataloader))
    return (
        total_loss/denom, total_clf_loss/denom, total_seg_loss/denom,
        f1_score(all_labels, all_preds, average='macro', zero_division=0),
        accuracy_score(all_labels, all_preds),
        all_preds, all_labels, all_confidences,
        total_pixel_acc/denom, total_dice/denom
    )

print("✅ 학습/평가 함수 정의 수정 완료 (Normal 예측 시 Segmentation 결과 무시 로직 추가).")

# --- 9. 데이터셋 및 데이터 로더 생성 ---
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler # Sampler를 위해 명시적 임포트

print("--- 데이터셋 및 데이터 로더 준비 (Normal CSV 병합 포함) ---")

# --- 1. 경로 정의 ---
# 파형 데이터 경로
TRAIN_WAVE_DIRS = [os.path.join(DATASET_PATH, "train"), os.path.join(DATASET_PATH, "train_normal")]
VALID_WAVE_DIRS = [os.path.join(DATASET_PATH, "valid"), os.path.join(DATASET_PATH, "val_normal")]

# CSV 파일 경로 정의
train_csv_main = os.path.join(DATASET_PATH, "train.csv")
train_csv_normal = os.path.join(DATASET_PATH, "train_normal.csv")

valid_csv_main = os.path.join(DATASET_PATH, "valid.csv")
valid_csv_normal = os.path.join(DATASET_PATH, "valid_normal.csv")

# --- CSV 병합 함수 ---
def prepare_merged_csv(main_path, normal_path, output_name="combined.csv"):
    """
    메인 CSV와 Normal CSV를 병합하여 저장합니다.
    Normal CSV가 없으면 메인 CSV 경로를 그대로 반환합니다.
    """
    # 저장할 경로 (dataset 폴더 내)
    output_path = os.path.join(DATASET_PATH, output_name)

    if os.path.exists(normal_path):
        print(f"🔄 병합 진행 중: {os.path.basename(main_path)} + {os.path.basename(normal_path)}")
        try:
            df_main = pd.read_csv(main_path)
            df_normal = pd.read_csv(normal_path)

            # 컬럼이 다를 경우를 대비해 공통 컬럼만 사용하거나, 단순히 concat (여기선 concat)
            df_combined = pd.concat([df_main, df_normal], ignore_index=True)

            # 병합된 파일 저장
            df_combined.to_csv(output_path, index=False)
            print(f"✅ 병합 완료 및 저장: {output_path} (총 {len(df_combined)} 행)")
            return output_path
        except Exception as e:
            print(f"❌ CSV 병합 중 오류 발생: {e}")
            print(f"⚠️ 기존 {main_path}만 사용합니다.")
            return main_path
    else:
        print(f"ℹ️ {os.path.basename(normal_path)} 파일이 없습니다. {os.path.basename(main_path)}만 사용합니다.")
        return main_path

# --- 병합 실행 ---
FINAL_TRAIN_CSV = prepare_merged_csv(train_csv_main, train_csv_normal, "train_combined.csv")
FINAL_VALID_CSV = prepare_merged_csv(valid_csv_main, valid_csv_normal, "valid_combined.csv")


# --- 2. 데이터셋 인스턴스 생성 (PPGDataset_Combined, 샘플별 정규화 사용) ---

print("\n[Train Dataset 생성]")
train_dataset = PPGDataset_Combined(
    data_dirs=TRAIN_WAVE_DIRS,
    clinical_csv_path=FINAL_TRAIN_CSV, # 병합된 CSV 경로 사용
    target_length=TARGET_LENGTH,
    fs=100.0,
    is_test=False
)

print("\n[Valid Dataset 생성]")
valid_dataset = PPGDataset_Combined(
    data_dirs=VALID_WAVE_DIRS,
    clinical_csv_path=FINAL_VALID_CSV, # 병합된 CSV 경로 사용
    target_length=TARGET_LENGTH,
    fs=100.0,
    is_test=False
)

# --- 3. 클래스 가중치 및 샘플러 설정 ---
# train_dataset.labels는 데이터셋 생성 시 __init__에서 로드됨
class_counts = np.bincount(train_dataset.labels, minlength=len(CLASS_MAP))
print(f"\nTrain 클래스 분포 (병합 후): {class_counts.tolist()}")
print(f" -> {INV_CLASS_MAP}")

# 0으로 나누는 것을 방지하기 위해 최소 1로 클리핑
safe_counts = np.clip(class_counts, 1, None)
class_weights_for_sampler = 1.0 / torch.tensor(safe_counts, dtype=torch.float32)

# 각 샘플에 대한 가중치 할당
sample_weights = class_weights_for_sampler[train_dataset.labels]

# WeightedRandomSampler 정의
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print("✅ WeightedRandomSampler (클래스 불균형 처리용) 설정 완료.")

# --- 4. 데이터 로더 생성 ---
BATCH_SIZE = 32 # L4 GPU 적합

# 재현성 보장을 위해 num_workers=0 유지 (필요시 2로 변경)
num_workers = 0

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler, # shuffle=False 대신 sampler 사용
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=bool(num_workers)
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, #검증 시에는 순서대로
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=bool(num_workers)
)

print(f"✅ Train/Valid 데이터 로더 생성 완료 (Batch Size: {BATCH_SIZE})")

# --- 10. 모델, 손실, 옵티마이저, 스케줄러, EarlyStopping 설정 ---
import torch.optim as optim
import math
import numpy as np # np.bincount, np.clip 사용을 위해 추가

print("--- 트레이너 설정 ---")

# 1. 하이퍼파라미터 설정 (규제 강화)
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 5e-5 # 디코더/헤드에 적용될 기본 LR
ENCODER_LR_RATIO = 0.1 # 인코더에 적용될 LR 비율 (5e-6)
ALPHA = 2.0  # Clf 손실 가중치
BETA = 0.8 # Seg 손실 가중치
MAX_NORM = 1.0# Gradient Clipping
WEIGHT_DECAY = 5e-4  # L2 규제
UNFREEZE_EPOCH = 50 # 인코더 동결을 해제할 에포크


print(f"✅ 하이퍼파라미터: LR={LEARNING_RATE}, Enc_Ratio={ENCODER_LR_RATIO}, Alpha={ALPHA}, Beta={BETA}, L2={WEIGHT_DECAY}")

# 2. 모델 초기화 및 가중치 로드 (UNet1D_ResNet_Combined는 128차원 CLIP 특징을 사용)
model = UNet1D_ResNet_Combined(
    n_channels=1,
    n_classes=len(CLASS_MAP),
    n_clinical_features=N_CLINICAL_FEATURES,
    n_hrv_features=N_HRV_FEATURES
).to(device)

model.load_clip_weights(CLIP_WEIGHTS_PATH, device)

# 3. 인코더 동결 설정
FREEZE_ENCODER = True
if FREEZE_ENCODER:
    # CLIP ResNet Encoder 및 Projection 가중치 동결
    for name, param in model.named_parameters():
        if name.startswith('encoder.') or name.startswith('clip_projection.'):
            param.requires_grad = False
    print(f"✅ CLIP ResNet Encoder 및 Projection 가중치 동결 완료 (학습 중 {UNFREEZE_EPOCH} Epoch 후 해제).")


# 4. 손실 함수 설정 (클래스 가중치 수동 설정)
class_counts = np.bincount(train_dataset.labels, minlength=len(CLASS_MAP))
# [normal, af, b, t] 순서에 맞게 설정 (총합이 1이 아니어도 됨)
custom_weights = [0.2, 0.6, 0.6, 0.2] # 이 가중치는 사용자가 설정한 값입니다.
special_weights = torch.tensor(custom_weights, dtype=torch.float32).to(device)

criterion_clf = FocalLoss(gamma=2.2, alpha=special_weights)
criterion_seg = BCEDiceLoss()

print(f"   -> 클래스 분포: {INV_CLASS_MAP} = {class_counts.tolist()}")
print(f"✅ Focal Loss 가중치 설정 완료: {custom_weights} (재현율 개선 목표)")

# 5. 옵티마이저 및 스케줄러 설정 (계층별 학습률)

# Fine-tuning을 위한 계층별 LR 설정: 인코더 그룹은 낮은 LR 적용
encoder_group_params = []
other_group_params = []

for name, p in model.named_parameters():
    if p.requires_grad:
        if name.startswith('encoder.') or name.startswith('clip_projection.'):
            encoder_group_params.append(p)
        else:
            other_group_params.append(p)

# 인코더 그룹에 대한 낮은 LR 설정
optimizer = optim.AdamW([
    {'params': encoder_group_params, 'lr': LEARNING_RATE * ENCODER_LR_RATIO, 'weight_decay': WEIGHT_DECAY, 'name': 'encoder'},
    {'params': other_group_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'name': 'other'}
], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Warmup + Cosine Annealing 스케줄러
total_epochs = EPOCHS
warmup_epochs = max(1, int(0.05 * total_epochs))

def lr_lambda(epoch_idx):
    if epoch_idx < warmup_epochs:
        return float(epoch_idx + 1) / float(warmup_epochs)
    progress = (epoch_idx - warmup_epochs + 1) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"✅ 옵티마이저 및 스케줄러 설정 완료 (Layer-wise LR 적용, Warmup: {warmup_epochs} epochs)")

# 6. EarlyStopping 설정
early_stopping = EarlyStopping(
    patience=50,
    verbose=True,
    path=MODEL_SAVE_PATH,
    mode='min',
    monitor_name='val_total',
    delta=1e-3
)

print(f"✅ EarlyStopping 설정 완료. (모니터링: val_total)")
print("\n🎉 모든 준비가 완료되었습니다! 이제 '11. 학습 루프 실행' 셀을 실행하세요.")

# --- 11. 학습 루프 실행 ---

# 설정된 EPOCHS 및 FREEZE_ENCODER 변수 사용
print(f"--- 🚀 학습을 시작합니다 (총 {EPOCHS} 에포크) ---")
print(f"모델 저장 경로: {MODEL_SAVE_PATH}")
# 주의: FREEZE_ENCODER 변수는 Encoder와 clip_projection 파라미터를 함께 제어합니다.
print(f"초기 인코더/프로젝션 동결 상태: {FREEZE_ENCODER} ({UNFREEZE_EPOCH} 에포크 후 해제 예정)")

start_time = datetime.now()

history = {
    'train_total': [], 'val_total': [],
    'train_clf': [], 'val_clf': [],
    'train_seg': [],  'val_seg': [],
    'train_f1': [], 'val_f1': [],
    'train_acc': [], 'val_acc': [],
    'val_seg_pixel_acc': [], 'val_seg_dice': []
}

# 인코더를 해제할 에포크 (Fine-tuning 시작 시점)
UNFREEZE_EPOCH = 50

for epoch in range(EPOCHS):
    epoch_start_time = datetime.now()
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

    # --- [Fine-tuning] 인코더/프로젝션 동결 해제 로직 ---
    if FREEZE_ENCODER and (epoch + 1) == UNFREEZE_EPOCH:
        print(f"\n🔥 Epoch {UNFREEZE_EPOCH}: CLIP ResNet Encoder/Projection 동결 해제! (Fine-tuning 시작)")

        # model.encoder와 model.clip_projection의 모든 파라미터 학습 활성화
        for name, param in model.named_parameters():
            if name.startswith('encoder.') or name.startswith('clip_projection.'):
                param.requires_grad = True

        # 옵티마이저를 갱신해야 하지만, AdamW의 경우 requires_grad=True로 바뀌면
        # 다음 step부터 자동으로 기울기 계산에 포함됩니다.
        # (별도로 optimizer를 재정의하지 않아도 되지만, Layer-wise LR이 적용되었으므로 LR은 이미 설정된 대로 유지됩니다.)

        FREEZE_ENCODER = False
        print("   -> Encoder 및 Projection 파라미터가 이제 업데이트됩니다.")

    # --- 1. Train ---
    tr_total, tr_clf, tr_seg, tr_f1, tr_acc = train_one_epoch(
        model, train_dataloader, criterion_clf, criterion_seg,
        optimizer, device, alpha=ALPHA, beta=BETA, max_norm=MAX_NORM
    )

    # --- 2. Scheduler Step ---
    scheduler.step()

    # --- 3. Evaluate ---
    (va_total, va_clf, va_seg, va_f1, va_acc,
     _, _, _, va_seg_pixel_acc, va_seg_dice) = evaluate(
        model, valid_dataloader, criterion_clf, criterion_seg,
        device, alpha=ALPHA, beta=BETA
    )

    epoch_end_time = datetime.now()
    epoch_duration = epoch_end_time - epoch_start_time

    # --- 4. Log ---
    # 인코더 그룹의 현재 LR을 로그에 표시
    current_lr = optimizer.param_groups[0]['lr']

    print(f"  [Train] | total: {tr_total:.4f} | clf: {tr_clf:.4f} | seg: {tr_seg:.4f} | F1: {tr_f1:.4f} | Acc: {tr_acc:.4f}")
    print(f"  [Valid] | total: {va_total:.4f} | clf: {va_clf:.4f} | seg: {va_seg:.4f} | "
          f"F1: {va_f1:.4f} | Acc: {va_acc:.4f}")
    print(f"  [Seg(V)]| PixelAcc: {va_seg_pixel_acc:.4f} | Dice: {va_seg_dice:.4f}")
    print(f"  [Info] | Enc LR: {current_lr:.2e} | Time: {epoch_duration}")

    # --- 5. History Update ---
    history['train_total'].append(tr_total); history['val_total'].append(va_total)
    history['train_clf'].append(tr_clf);  history['val_clf'].append(va_clf)
    history['train_seg'].append(tr_seg);  history['val_seg'].append(va_seg)
    history['train_f1'].append(tr_f1);  history['val_f1'].append(va_f1)
    history['train_acc'].append(tr_acc);  history['val_acc'].append(va_acc)
    history['val_seg_pixel_acc'].append(va_seg_pixel_acc)
    history['val_seg_dice'].append(va_seg_dice)

    # --- 6. EarlyStopping ---
    early_stopping(va_total, model, ref_metric_for_log=va_seg_pixel_acc, ref_name='seg_pixel_acc')
    if early_stopping.early_stop:
        print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}.")
        break

end_time = datetime.now()
print(f"\n--- 🏁 총 학습 시간: {end_time - start_time} ---")

# --- 7. 최적 가중치 로드 ---
try:
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"\n✅ Best checkpoint loaded from '{MODEL_SAVE_PATH}' (Best Score: {early_stopping.best_score:.6f})")
    else:
        print("\n⚠️ 저장된 모델 파일이 없습니다.")
except Exception as e:
    print(f"\n⚠️ 최적 가중치 로드 실패: {e}. (현재 마지막 에포크 모델 상태 유지)")

# --- 11. 학습 결과 시각화 ---
import matplotlib.pyplot as plt
import numpy as np

# history 딕셔너리는 학습 루프에서 채워짐
if 'history' not in locals() or len(history.get('train_total', [])) == 0:
    print("❌ 'history' 딕셔너리가 비어 있습니다. 학습 루프를 먼저 실행하세요.")
else:
    # 실행된 에폭 수 (조기 종료된 에폭)
    epochs_ran = len(history.get('train_total', []))
    x = range(1, epochs_ran + 1)
    print(f"총 {epochs_ran} 에포크의 학습 기록을 시각화합니다.")

    # 베스트 에폭 (검증 총손실 기준)
    best_epoch_idx = int(np.argmin(history['val_total']))
    best_epoch = best_epoch_idx + 1
    best_val_loss = history['val_total'][best_epoch_idx]
    print(f"Best Epoch (min val_total): {best_epoch} (Loss: {best_val_loss:.4f})")

    # 3x2 레이아웃
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Combined Model Training & Validation History', fontsize=16, y=1.02)

    # 1) Total Loss
    ax = axes[0, 0]
    ax.plot(x, history['train_total'], label='Train Total', marker='.', markersize=5)
    ax.plot(x, history['val_total'],  label='Valid Total', marker='.', markersize=5)
    ax.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax.set_title(f'Total Loss (Best Val: {best_val_loss:.4f})'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

    # 2) Classification Loss
    ax = axes[0, 1]
    ax.plot(x, history['train_clf'], label='Train Clf', marker='.', markersize=5)
    ax.plot(x, history['val_clf'],  label='Valid Clf', marker='.', markersize=5)
    ax.set_title('Classification Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

    # 3) Segmentation Loss
    ax = axes[1, 0]
    ax.plot(x, history['train_seg'], label='Train Seg', marker='.', markersize=5)
    ax.plot(x, history['val_seg'], label='Valid Seg', marker='.', markersize=5)
    ax.set_title('Segmentation Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

    # 4) Classification Accuracy & F1-Score (Valid 기준)
    ax = axes[1, 1]
    ax.plot(x, history['val_acc'],  label='Valid Acc', linestyle='-', color='C0')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax.twinx() # Y축 공유
    ax2.plot(x, history['val_f1'],  label='Valid F1', linestyle='--', color='C1')
    ax2.set_ylabel('F1-Score (Macro)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    ax.set_title('Validation Classification Metrics')
    ax.axvline(best_epoch, color='r', linestyle='--')
    # 범례 합치기
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    # 5) Segmentation Pixel Accuracy (Valid)
    ax = axes[2, 0]
    ax.plot(x, history['val_seg_pixel_acc'], label='Valid Seg PixelAcc', marker='.', markersize=5, color='C2')
    ax.set_title('Validation Segmentation Pixel Accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Pixel Accuracy')
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

    # 6) Segmentation Dice (Valid)
    ax = axes[2, 1]
    ax.plot(x, history['val_seg_dice'], label='Valid Seg Dice', marker='.', markersize=5, color='C3')
    ax.set_title('Validation Segmentation Dice (Reference)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice Coefficient')
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

    plt.tight_layout()
    plt.show()

# --- 12. 최종 모델 평가 (Test Set) ---

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

print("--- 🏁 최종 모델 평가 (Test Set) 시작 ---")

# --- 1. Test 데이터셋/로더 생성 ---
# Test 셋 파형 경로
TEST_WAVE_DIR = os.path.join(DATASET_PATH, "test")

# Test 셋 임상 정보 경로 (test.csv)
TEST_CSV_PATH = os.path.join(DATASET_PATH, "test.csv")

print("\n[Test Dataset 생성]")
test_dataset = PPGDataset_Combined(
    data_dirs=[TEST_WAVE_DIR], # 리스트 형태로 전달
    clinical_csv_path=TEST_CSV_PATH,
    target_length=TARGET_LENGTH,
    fs=100.0,
    is_test=False # 평가를 위해 라벨이 필요하므로 False (라벨 로드 시도)
)

# DataLoader 설정
# (Colab 안정성을 위해 num_workers=0 권장)
num_workers = 0
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # 평가는 순서대로
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=bool(num_workers)
)
print(f"✅ Test 데이터 로더 생성 완료 (총 {len(test_dataset)}개 샘플)")


# --- 2. 최종 모델 로드 ---
print(f"\n[최종 모델 로드]")
final_model = UNet1D_ResNet_Combined(
    n_channels=1,
    n_classes=len(CLASS_MAP),
    n_clinical_features=N_CLINICAL_FEATURES,
    n_hrv_features=N_HRV_FEATURES
).to(device)

if os.path.exists(MODEL_SAVE_PATH):
    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"✅ '{MODEL_SAVE_PATH}'에서 최적 모델을 성공적으로 로드했습니다.")
else:
    print(f"❌ 오류: '{MODEL_SAVE_PATH}'에서 모델을 찾을 수 없습니다. (현재 모델 상태로 평가 진행)")


# --- 3. Test 데이터셋으로 최종 성능 평가 ---
print("\n[Test Set 평가 실행]")
(test_loss, test_clf_loss, test_seg_loss, test_f1, test_acc,
 all_preds, all_labels, all_confidences,
 test_seg_pixel_acc, test_seg_dice) = evaluate(
    final_model,
    test_dataloader,
    criterion_clf,
    criterion_seg,
    device
)

# --- 4. 최종 결과 출력 ---
print("\n--- 📊 최종 성능 평가 결과 (Test Set / 결합 모델) ---")
print(f"  - Test Loss (Total)         : {test_loss:.4f}")
print(f"  - Classification")
print(f"      · Accuracy              : {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"      · F1 (Macro)            : {test_f1:.4f}")
print(f"      · Classification Loss   : {test_clf_loss:.4f}")
print(f"  - Segmentation (1D)")
print(f"      · Pixel Accuracy        : {test_seg_pixel_acc:.4f}")
print(f"      · Dice (reference)      : {test_seg_dice:.4f}")
print(f"      · Segmentation Loss     : {test_seg_loss:.4f}")

# --- 5. Confusion Matrix & Classification Report ---
labels_order = list(range(len(CLASS_MAP)))
target_names = [INV_CLASS_MAP[i] for i in labels_order]

print("\n[Classification Report (Test Set)]\n")
print(classification_report(
    all_labels,
    all_preds,
    labels=labels_order,
    target_names=target_names,
    digits=4,
    zero_division=0
))

print("\n[Confusion Matrix (Test Set)]")
cm = confusion_matrix(all_labels, all_preds, labels=labels_order)
import matplotlib.pyplot as plt # plt 임포트 확인
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
             xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.show()

# 13. 최종 결과 분석 및 LLM 입력 생성
# (기존 시각화 로직을 LLM 입력 생성 로직으로 대체)

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# --- (수정) LLM 입력을 저장할 디렉토리 ---
LLM_BASE_DIR = '/content/drive/MyDrive/종합설계/PPG_AD_Dataset/input_for_llm'

LLM_IMAGE_DIR = os.path.join(LLM_BASE_DIR, 'images') # 이미지용 하위 폴더
LLM_HINT_DIR = os.path.join(LLM_BASE_DIR, 'hints')   # 힌트용 하위 폴더

os.makedirs(LLM_IMAGE_DIR, exist_ok=True)
os.makedirs(LLM_HINT_DIR, exist_ok=True)
print(f"🖼️ LLM용 순수 파형 이미지 저장 경로: '{LLM_IMAGE_DIR}'")
print(f"📝 LLM용 모델 힌트 텍스트 저장 경로: '{LLM_HINT_DIR}'")

# --- (신규) 마스크 배열을 텍스트로 변환하는 헬퍼 함수 ---
def find_mask_indices(mask_array, offset=0):
    """
    U-Net의 boolean/binary 마스크 배열(1D)을 받아서
    "index A to B" 형태의 문자열로 변환합니다.
    """
    if not np.any(mask_array):
        return "No anomaly mask detected."

    # 마스크가 1인 인덱스를 찾음
    indices = np.where(mask_array > 0)[0]
    if len(indices) == 0:
        return "No anomaly mask detected."

    # 마스크 세그먼트의 시작과 끝 찾기
    segments = []
    start_index = indices[0]
    for i in range(1, len(indices)):
        # 인덱스가 연속되지 않으면, 이전 세그먼트 종료
        if indices[i] != indices[i-1] + 1:
            segments.append((start_index, indices[i-1]))
            start_index = indices[i]
    # 마지막 세그먼트 추가
    segments.append((start_index, indices[-1]))

    # 텍스트로 변환 (오프셋 적용)
    segment_texts = [f"index {s+offset} to {e+offset}" for s, e in segments]

    if len(segment_texts) == 0:
        return "No anomaly mask detected."

    return "Anomaly mask detected from " + ", ".join(segment_texts) + "."


# --- 1) Test 전 구간 예측 (분류/세그) ---
final_model.eval()
all_labels, all_preds = [], []
all_pred_confidences = [] # 예측 신뢰도 저장
all_pixel_acc = []        # (참고용)
all_dice = []             # (참고용)

with torch.no_grad():
    for signals, masks, labels, clinical_features, hrv_features in tqdm(test_dataloader, desc="Test set 재-예측 중 (LLM용)"):
        signals = signals.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        clinical_features = clinical_features.to(device)
        hrv_features = hrv_features.to(device)

        clf_logits, seg_logits = final_model(signals, clinical_features, hrv_features)

        # 1. 분류 결과 저장
        clf_probs = torch.softmax(clf_logits, dim=1)
        batch_confidences, batch_preds = torch.max(clf_probs, dim=1)

        all_preds.extend(batch_preds.cpu().numpy())
        all_pred_confidences.extend(batch_confidences.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 2. 세그멘테이션 성능 계산
        seg_probs = torch.sigmoid(seg_logits)
        seg_bin = (seg_probs > 0.5).to(masks.dtype)

        if masks.ndim == 3:
            masks = masks.squeeze(1)
        if seg_bin.ndim == 3:
            seg_bin = seg_bin.squeeze(1)

        batch_pixel_acc = (seg_bin == masks).float().mean(dim=1)
        all_pixel_acc.extend(batch_pixel_acc.cpu().numpy().tolist())

        intersect = (seg_bin * masks).sum(dim=1)
        denom = seg_bin.sum(dim=1) + masks.sum(dim=1) + 1e-7
        batch_dice = (2.0 * intersect + 1e-7) / denom
        all_dice.extend(batch_dice.cpu().numpy().tolist())

all_labels_np = np.array(all_labels)
all_preds_np  = np.array(all_preds)
all_pred_confidences_np = np.array(all_pred_confidences)
all_pixel_acc_np = np.array(all_pixel_acc)
all_dice_np   = np.array(all_dice)


# --- 2) LLM 입력 생성 함수 ---

def save_llm_input(idx):
    """
    단일 인덱스(idx)에 대해 [순수 파형 이미지]와 [모델 힌트 텍스트]를
    LLM 입력 형식으로 저장합니다.
    """
    # 1. 데이터 로드 및 파일명 파싱
    x, y_mask, y_label, clinical_t, hrv_t = test_dataset[idx]

    file_info = test_dataset.file_list[idx] # list of dicts
    file_name = file_info['filename']       # '1234_win0.npz'

    try:
        parts = file_name.replace('.npz', '').replace('.npy', '').split('_')
        patient_id = parts[0]
        window_id = parts[1].replace('win', '') if len(parts) > 1 else "N-A"
    except Exception:
        patient_id, window_id = "N-A", "N-A"

    # 2. LLM [Hint] 텍스트 정보 추출
    pred_class_idx = int(all_preds_np[idx])
    pred_class_name = INV_CLASS_MAP.get(pred_class_idx, str(pred_class_idx))
    pred_confidence_i = all_pred_confidences_np[idx]

    # 2-3. [Mask Info] (모델 재실행하여 마스크 추출)
    with torch.no_grad():
        _, pred_seg_logits = final_model(
            x.unsqueeze(0).to(device),
            clinical_t.unsqueeze(0).to(device),
            hrv_t.unsqueeze(0).to(device)
        )
    pred_probs = torch.sigmoid(pred_seg_logits)
    pred_mask_np = (pred_probs > 0.5).int().squeeze().cpu().numpy()

    # 분류가 Normal(0)이면 마스크를 0으로 초기화 (후처리)
    if pred_class_idx == 0:
        pred_mask_np[:] = 0

    mask_location_text = find_mask_indices(pred_mask_np)

    # 3. 파일명 및 경로
    base_filename = f"{patient_id}_{window_id}"
    image_save_path = os.path.join(LLM_IMAGE_DIR, f"{base_filename}.png")
    hint_save_path = os.path.join(LLM_HINT_DIR, f"{base_filename}.txt")

    # 원본 파형
    processed_signal_np = x.squeeze().numpy()

    # 4. 이미지 저장
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=120)
    indices = np.arange(len(processed_signal_np))
    ax.plot(indices, processed_signal_np, color='blue', label=f'Signal')
    ax.set_title(f"PID: {patient_id}, Win: {window_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    fig.savefig(image_save_path)
    plt.close(fig)

    # 5. 텍스트 저장
    hint_text = (
        f"[Prediction Class]: {pred_class_name}\n"
        f"[Logit Value]: {pred_confidence_i:.4f}\n"
        f"[Mask Info]: {mask_location_text}\n"
    )
    with open(hint_save_path, 'w', encoding='utf-8') as f:
        f.write(hint_text)


# --- 3) 실행 ---
total_samples = len(test_dataset)
if total_samples == 0:
    print("⚠️ 테스트 데이터셋이 비어있습니다.")
else:
    print(f"\n🚀 {total_samples}개의 테스트 샘플 처리 시작...")
    for i in tqdm(range(total_samples), desc="LLM 입력 생성 중"):
        try:
            save_llm_input(int(i))
        except Exception as e:
            print(f"🔥 [오류] 인덱스 {i}: {e}")

print("\n--- 모든 LLM 입력 생성이 완료되었습니다. ---")

!pip install -q -U google-generativeai

"""### 설명가능한 인공지능을 만들기 위해 clip에서 저장한 데이터베이스를 가져와서 소견서에 적용"""

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import ipywidgets as widgets
from IPython.display import display, clear_output
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. 환경 설정 & 모델/API 초기화
# -----------------------------------------------------------------------------
BASE_PROJECT_PATH = "/content/drive/MyDrive/종합설계"
DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "PPG_AD_Dataset")
LLM_IMAGE_DIR = os.path.join(DATASET_PATH, 'input_for_llm', 'images')
os.makedirs(LLM_IMAGE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Gemini API 설정

from dotenv import load_dotenv
load_dotenv()
MY_API_KEY = os.getenv("GEMINI_API_KEY", "")
try:
    genai.configure(api_key=MY_API_KEY)
except Exception as e:
    print(f"⚠️ Gemini API 설정 오류: {e}")

# 모델 설정
llm_model = genai.GenerativeModel('gemini-flash-latest')
safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}

# -----------------------------------------------------------------------------
# 2. 헬퍼 함수 및 데이터 매핑
# -----------------------------------------------------------------------------

# 소문자 라벨을 대문자(가이드라인 키)로 변환하는 맵
LABEL_DISPLAY_MAP = {
    'normal': 'Normal',
    'af': 'AF',
    'b': 'B',
    't': 'T'
}

# (1) 가이드라인 DB
MEDICAL_GUIDELINES = {
    "AF": """
    [Reference: 2020 ESC Guidelines for Atrial Fibrillation (Adapted for PPG)]
    - **진단 기준**: PPG 상 맥박 간격의 불규칙성이 30초 이상 지속 시 의심 (확진은 ECG 필요).
    - **특징**: 'Irregularly Irregular' R-R Interval, P파 소실(ECG).
    - **권고 (Management)**:
      1. 65세 이상 환자에서 기회적 선별검사(Opportunistic Screening) 권장.
      2. CHA2DS2-VASc 점수 >= 2 (남성)인 경우 경구 항응고제(NOAC) 투여 고려.
      3. 심박수 조절(Rate Control)을 위해 베타차단제(Beta-blockers) 투여 고려.
    """,
    "T": """
    [Reference: ACC/AHA Guidelines for Tachycardia]
    - **정의**: 안정 시 심박수 > 100 bpm.
    - **평가**: 12-lead ECG를 통해 동성 빈맥(Sinus Tachycardia)과 부정맥 감별 필수.
    - **처치**:
      1. 2차적 원인(발열, 빈혈, 불안, 탈수 등) 교정.
      2. 증상이 있는 상심실성 빈맥(SVT)의 경우 미주신경 자극법(Vagal maneuvers) 시도.
    """,
    "B": """
    [Reference: ACC/AHA Guidelines for Bradycardia]
    - **정의**: 안정 시 심박수 < 60 bpm.
    - **처치**:
      1. 무증상인 경우 경과 관찰 (Observation).
      2. 증상(실신, 현기증 등) 동반 시 원인 약물 중단 또는 아트로핀/페이싱 고려.
    """,
    "Normal": """
    [Reference: General Health Checkup Guidelines]
    - **정상 소견**: 규칙적인 동리듬, 심박수 60-100 bpm.
    - **권고**: 특이 소견 없음. 정기적인 건강 관리 유지.
    """
}

# (2) 모델 정의
class ClinicalMLP(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class Projection(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, proj_dim)
    def forward(self, x): return F.normalize(self.fc(x), dim=-1)

# (3) 데이터 변환 (13 -> 17)
def transform_13_to_17(vec_13):
    if vec_13.dim() == 1: vec_13 = vec_13.unsqueeze(0)
    cont = vec_13[:, :9]
    sex_m = vec_13[:, 9:10]; sex_f = 1.0 - sex_m
    emop_y = vec_13[:, 10:11]; emop_n = 1.0 - emop_y
    dm_y = vec_13[:, 11:12]; dm_n = 1.0 - dm_y
    htn_y = vec_13[:, 12:13]; htn_n = 1.0 - htn_y
    return torch.cat([cont, sex_m, sex_f, emop_n, emop_y, dm_n, dm_y, htn_n, htn_y], dim=1)

# (4) KB 로드
KB_PATH = os.path.join(DATASET_PATH, "checkpoints", "knowledge_base.pt")
knowledge_base = None
if os.path.exists(KB_PATH):
    knowledge_base = torch.load(KB_PATH)
    knowledge_base["clinical_vectors"] = knowledge_base["clinical_vectors"].to(device)
    print("✅ Knowledge Base 로드 완료.")

# (5) 모델 가중치 로드
rag_clin_enc = ClinicalMLP(in_dim=17).to(device)
rag_proj_c = Projection(in_dim=128).to(device)
CLIP_CKPT_PATH = os.path.join(DATASET_PATH, "checkpoints", "clip_biobert_hyh_xai.pth")
if os.path.exists(CLIP_CKPT_PATH):
    ckpt = torch.load(CLIP_CKPT_PATH, map_location=device)
    rag_clin_enc.load_state_dict(ckpt['clin_enc'])
    rag_proj_c.load_state_dict(ckpt['proj_c'])
    rag_clin_enc.eval(); rag_proj_c.eval()
    print("✅ 검색용 인코더 로드 완료.")

# (6) KB 통계 추출
def get_kb_patient_stats(target_caseid):
    if knowledge_base is None: return None
    kb_ids = knowledge_base["caseids"]
    if isinstance(kb_ids, torch.Tensor): kb_ids = kb_ids.tolist()

    indices = [i for i, x in enumerate(kb_ids) if x == target_caseid]
    if not indices: return None

    # 카운트 초기화 (대문자 키 사용)
    counts = {"Normal": 0, "AF": 0, "B": 0, "T": 0}
    kb_labels = knowledge_base["labels"]

    for idx in indices:
        lbl = kb_labels[idx]
        # 정수 라벨이면 문자열로 변환
        if isinstance(lbl, int): lbl = INV_CLASS_MAP[lbl] # 'b', 'af' 등
        # 소문자 -> 대문자 매핑 ('b' -> 'B')
        display_lbl = LABEL_DISPLAY_MAP.get(lbl, lbl)
        counts[display_lbl] = counts.get(display_lbl, 0) + 1

    total = len(indices)
    ratios = {k: v/total*100 for k, v in counts.items()}

    # Normal 제외한 최다 빈도 찾기
    arrs = {k: v for k, v in counts.items() if k != "Normal"}
    if sum(arrs.values()) > 0:
        dom = max(arrs, key=arrs.get)
    else:
        dom = "Normal"

    return {"counts": counts, "ratios": ratios, "dominant": dom, "total": total}

# (7) 유사 환자 검색
def find_similar_patients_with_stats(query_vec_13, k=3):
    if knowledge_base is None: return []
    with torch.no_grad():
        q_17 = transform_13_to_17(query_vec_13.to(device))
        q_emb = rag_proj_c(rag_clin_enc(q_17))
        db_embs = knowledge_base["clinical_vectors"]
        sim = torch.mm(F.normalize(q_emb, p=2, dim=1), F.normalize(db_embs, p=2, dim=1).T).squeeze(0)
        top_vals, top_idxs = torch.topk(sim, k=k*10)

    results = []
    seen = set()
    for i, idx in enumerate(top_idxs):
        idx = idx.item()
        caseid = knowledge_base["caseids"][idx]
        if isinstance(caseid, torch.Tensor): caseid = caseid.item()
        if caseid in seen: continue
        seen.add(caseid)

        stats = get_kb_patient_stats(caseid)
        if stats:
            results.append({"caseid": caseid, "similarity": top_vals[i].item(), "stats": stats})
        if len(results) >= k: break
    return results

# -----------------------------------------------------------------------------
# 3. Ultimate Patient Viewer Class
# -----------------------------------------------------------------------------

class UltimatePatientViewer:
    def __init__(self, dataset, model, device):
        self.dataset = dataset
        self.model = model
        self.device = device

        self.current_pid = ""
        self.target_indices = []
        self.current_idx_pointer = 0
        self.current_cache = None
        self.current_img_buffer = None

        # UI
        self.input_pid = widgets.Text(value='114', description='ID:', layout=widgets.Layout(width='180px'))
        self.btn_search = widgets.Button(description="검색", icon="search", button_style='info')
        self.btn_prev = widgets.Button(description="◀", disabled=True)
        self.lbl_seg = widgets.Label(value="Seg: - / -")
        self.btn_next = widgets.Button(description="▶", disabled=True)
        self.out_plot = widgets.Output()

        self.out_summary = widgets.Output(layout={'border': '2px solid #A9D0F5', 'padding': '10px', 'margin': '0 0 10px 0'})
        self.btn_rag = widgets.Button(description="RAG 리포트", button_style='primary', layout=widgets.Layout(width='48%'))
        self.btn_vision = widgets.Button(description="순수 파형 이미지 분석", button_style='warning', layout=widgets.Layout(width='48%'))
        self.out_report = widgets.Output(layout={'border': '1px solid #ddd', 'padding': '10px'})

        self.btn_search.on_click(self.on_search)
        self.btn_prev.on_click(self.on_prev)
        self.btn_next.on_click(self.on_next)
        self.btn_rag.on_click(self.on_rag_report)
        self.btn_vision.on_click(self.on_vision_check)

        self.ui = widgets.VBox([
            widgets.HBox([self.input_pid, self.btn_search]),
            self.out_summary,
            widgets.HTML("<h4>📉 세그먼트별 상세 분석</h4>"),
            widgets.HBox([self.btn_prev, self.lbl_seg, self.btn_next], layout=widgets.Layout(justify_content='center')),
            self.out_plot,
            widgets.HBox([self.btn_rag, self.btn_vision], layout=widgets.Layout(justify_content='space-between')),
            self.out_report
        ])

    def show(self): display(self.ui)

    def on_search(self, b):
        pid = self.input_pid.value.strip()
        if not pid: return
        self.current_pid = pid

        self.target_indices = [i for i, f in enumerate(self.dataset.file_list) if f['filename'].startswith(f"{pid}_")]

        # 정렬 (숫자 추출)
        def get_win_num(idx):
            fname = self.dataset.file_list[idx]['filename']
            if '_win' in fname:
                try: return int(fname.split('_win')[1].split('.')[0].split('_')[0])
                except: return 0
            return 0
        self.target_indices.sort(key=get_win_num)

        if not self.target_indices:
            with self.out_summary: clear_output(); print(f"❌ ID {pid} 데이터 없음.")
            return

        self._show_global_stats()
        self.current_idx_pointer = 0
        self.update_view()

    def _show_global_stats(self):
        with self.out_summary:
            clear_output(wait=True); print("⏳ 환자 전체 통계 분석 중...")

            # 1. 카운터 초기화 (대문자 키 사용)
            counts = {"Normal": 0, "AF": 0, "B": 0, "T": 0}

            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.dataset, self.target_indices),
                batch_size=64, shuffle=False
            )
            self.model.eval()
            with torch.no_grad():
                for sig, _, _, clin, hrv in loader:
                    clf, _ = self.model(sig.to(device), clin.to(device), hrv.to(device))
                    preds = torch.argmax(clf, dim=1).cpu().numpy()
                    for p in preds:
                        # 소문자 라벨 ('b') -> 대문자 라벨 ('B') 변환
                        raw_lbl = INV_CLASS_MAP[p]
                        display_lbl = LABEL_DISPLAY_MAP.get(raw_lbl, raw_lbl)
                        counts[display_lbl] = counts.get(display_lbl, 0) + 1

            total = sum(counts.values())
            arrs = {k: v for k, v in counts.items() if k != "Normal"}
            dom = max(arrs, key=arrs.get) if sum(arrs.values()) > 0 else "Normal"
            guide = MEDICAL_GUIDELINES.get(dom, "정보 없음")

            clear_output()
            display(widgets.HTML(f"""
            <div style='line-height:1.4;'>
                <b>🏥 환자 {self.current_pid} 요약 (Main: <span style='color:red'>{dom}</span>)</b>
                <ul><li>Total: {total} | AF: {counts['AF']}, B: {counts['B']}, T: {counts['T']}, N: {counts['Normal']}</li></ul>
                <div style='background:#f4f4f4;color:black; padding:5px; margin-top:5px; font-size:0.9em;'><b>📘 가이드라인:</b> {guide}</div>
            </div>
            """))

    def update_view(self):
        total = len(self.target_indices)
        idx = self.target_indices[self.current_idx_pointer]
        self.btn_prev.disabled = (self.current_idx_pointer == 0)
        self.btn_next.disabled = (self.current_idx_pointer == total - 1)
        self.lbl_seg.value = f"{self.current_idx_pointer + 1} / {total}"

        with self.out_plot:
            clear_output(wait=True)
            self._plot(idx)

    def on_prev(self, b):
        if self.current_idx_pointer > 0:
            self.current_idx_pointer -= 1
            self.update_view()

    def on_next(self, b):
        if self.current_idx_pointer < len(self.target_indices) - 1:
            self.current_idx_pointer += 1
            self.update_view()

    def _plot(self, idx):
        x, y_mask, y_label, clin, hrv = self.dataset[idx]
        fname = self.dataset.file_list[idx]['filename']

        self.model.eval()
        with torch.no_grad():
            sig = x.unsqueeze(0).to(device)
            clf, seg = self.model(sig, clin.unsqueeze(0).to(device), hrv.unsqueeze(0).to(device))
            probs = torch.softmax(clf, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pred_raw = INV_CLASS_MAP[pred.item()]
            # 대문자 변환
            pred_str = LABEL_DISPLAY_MAP.get(pred_raw, pred_raw)

            seg_map = (torch.sigmoid(seg) > 0.5).float().cpu().numpy().squeeze()
            if pred_raw == 'normal': seg_map[:] = 0

        # 이미지 찾기
        img_path = None
        try:
            clean = fname.replace('_seg', '').replace('.npz', '').replace('win', '')
            parts = clean.split('_')
            if len(parts) >= 2:
                img_name = f"{parts[0]}_{parts[1]}.png"
                img_path = os.path.join(LLM_IMAGE_DIR, img_name)
        except: pass

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax1.plot(x.numpy().squeeze(), 'k', lw=1)
        ax1.set_title(f"AI: {pred_str} ({conf.item()*100:.1f}%)", color='red' if pred_raw!='normal' else 'blue', fontweight='bold')
        ax2.plot(seg_map, 'r', lw=2, label='AI Mask')
        ax2.set_ylim(-0.1, 1.1); ax2.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.current_img_buffer = buf
        plt.show()

        self.current_cache = {
            "patient_id": self.current_pid,
            "clin_vec": clin,
            "pred_class": pred_str, # 대문자
            "pred_conf": conf.item() * 100,
            "mask_detected": np.any(seg_map > 0),
            "img_path": img_path
        }

    def on_rag_report(self, b):
        if not self.current_cache: return
        c = self.current_cache
        with self.out_report:
            clear_output(); print("⏳ RAG 분석 중...")
            sims = find_similar_patients_with_stats(c['clin_vec'].unsqueeze(0), k=3)
            evidence = ""
            if sims:
                for i, s in enumerate(sims):
                    dom = s['stats']['dominant']
                    ratio = s['stats']['ratios'][dom]
                    evidence += f"- [ID:{s['caseid']}] ({s['similarity']:.2f}): {dom} ({ratio:.1f}%)\n"
            else: evidence = "유사 사례 없음."

            prompt = f"""
            당신은 광용적맥파(PPG) 데이터를 분석하는 심장내과 전문의입니다. (ECG 아님)

            환자 ID: {c['patient_id']}
            CLIP + UNet 모델 예측: {c['pred_class']} ({c['pred_conf']:.1f}%)
            이상 파형: {'있음' if c['mask_detected'] else '없음'}
            유사 환자 통계:
            {evidence}

            ### 작성 요청 (한국어)
            1. **근거 분석**: 유사 환자들의 **진단 비율(%)**을 인용하여 예측의 타당성을 설명하세요.
            2. 위 정보를 바탕으로 전문적인 소견서를 작성하세요.
            """
            try: print(llm_model.generate_content(prompt).text)
            except Exception as e: print(f"Error: {e}")
# --- 4. 기능: Vision Consistency Check ---
    def on_vision_check(self, b):
        # 메모리 버퍼에 있는 이미지(현재 플롯)를 최우선으로 사용
        img_source = self.current_img_buffer

        # 만약 디스크에 저장된 원본 이미지를 쓰고 싶다면 아래 주석 해제
        # if self.current_cache and self.current_cache['img_path'] and os.path.exists(self.current_cache['img_path']):
        #    img_source = self.current_cache['img_path']

        if not img_source:
            with self.out_report: print("⚠️ 분석할 이미지가 없습니다.")
            return

        with self.out_report:
            clear_output()
            print("👁️ Gemini Vision이 파형 이미지를 분석 중입니다...")
            try:
                img = PIL.Image.open(img_source)
                prompt = """
                이 이미지는 환자의 PPG 신호입니다. (ECG 아님)
                당신은 광용적맥파(PPG) 데이터를 분석하는 심장내과 전문의입니다. (ECG 아님)
                이미지만을 보고 다음을 분석하세요:
                1. **리듬 규칙성**: 규칙적/불규칙적 여부
                2. **노이즈**: 신호가 깨끗한지 여부
                3. **부정맥 소견**: 심방세동(Irregular), 빈맥, 서맥 등 의심 여부

                **결론**: "이미지 시각적 분석 결과, [진단명] 패턴과 유사합니다." (한 문단 요약)
                """
                print(llm_model.generate_content([prompt, img]).text)

                # 썸네일 표시
                thumb = img.copy()
                thumb.thumbnail((400, 150))
                display(thumb)

            except Exception as e: print(f"Vision Error: {e}")

if __name__ == "__main__":
    try:
        viewer = UltimatePatientViewer(test_dataset, final_model, device)
        viewer.show()
    except NameError: print("⚠️ 모델/데이터셋을 먼저 로드하세요.")