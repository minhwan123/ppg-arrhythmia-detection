import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd

# =============================================================================
# 1. 설정 및 통계치 (Constants & Stats)
# =============================================================================

# data_stats.json: PPG/HRV 통계, clinical_stats.json: 임상 특징 스케일링 통계
CONFIG_PATH = "data_stats.json"
CLINICAL_STATS_PATH = "clinical_stats.json" 


N_HRV_FEATURES = 4  # HR, SDNN, RMSSD, pNN50
eps_hrv = 1e-6      # HRV 정규화 시 분모 0 방지

# [임상 데이터 컬럼 정의]
CLINICAL_FEATURES_LIST = [
    'age', 'bmi', 'opdur', 'preop_na', 'preop_bun', 'preop_cr', 'preop_k',
    'intraop_eph', 'intraop_phe',
    'sex_M', 'emop_Y', 'preop_dm_Y', 'preop_htn_Y'
]
N_CLINICAL_FEATURES = len(CLINICAL_FEATURES_LIST) # 13개

# [연속형 컬럼] (StandardScaler 적용 대상)
CONTINUOUS_COLS = CLINICAL_FEATURES_LIST[:9]

# [이진형 컬럼] (OHE 적용 대상)
BINARY_STR_COLS = {
    "sex": ["M", "F"], "emop": ["N", "Y"], 
    "preop_dm": ["N", "Y"], "preop_htn": ["N", "Y"]
}

CLASS_MAP = {'normal': 0, 'af': 1, 'b': 2, 't': 3}
INV_CLASS_MAP = {0: 'normal', 1: 'af', 2: 'b', 3: 't'}
N_CLASSES = 4 

# -----------------------------------------------------------------------------
# 통계치 로드 로직
# -----------------------------------------------------------------------------
# [A. data_stats.json 로드] (PPG/HRV 통계)
stats_cfg = {}
try:
    with open(CONFIG_PATH, "r") as f:
        stats_cfg = json.load(f)
except FileNotFoundError:
    pass

def _as_vec(x, default_val):
    default_vec = [default_val] * N_HRV_FEATURES
    arr = np.array(x if x is not None else default_vec, dtype=np.float32).reshape(-1)
    if arr.size != N_HRV_FEATURES:
        arr = np.array(default_vec, dtype=np.float32)
    return arr

HRV_MEAN   = _as_vec(stats_cfg.get("hrv_mean"),   0.0)
HRV_STD    = _as_vec(stats_cfg.get("hrv_std"),    1.0)
HRV_MEDIAN = _as_vec(stats_cfg.get("hrv_median"), 0.0)

CONT_MEAN = {c: 0.0 for c in CONTINUOUS_COLS}
CONT_STD = {c: 1.0 for c in CONTINUOUS_COLS}
for col in CONTINUOUS_COLS:
    CONT_MEAN[col] = float(stats_cfg.get(f"cont_mean_{col}", 0.0))
    CONT_STD[col] = float(stats_cfg.get(f"cont_std_{col}", 1.0))
    if CONT_STD[col] == 0: CONT_STD[col] = 1.0

TARGET_LENGTH = int(stats_cfg.get("target_length", 286))

# [B. clinical_stats.json 로드] (임상 스케일링 통계)
clinical_stats_cfg = {}
try:
    with open(CLINICAL_STATS_PATH, "r") as f:
        clinical_stats_cfg = json.load(f)
except FileNotFoundError:
    pass

# [핵심] StandardScaler 통계치 (1차 Z-score에 사용)
STD_SCALER_MEAN = np.array(clinical_stats_cfg.get("std_scaler_mean", [0.0] * N_CLINICAL_FEATURES), dtype=np.float32)
STD_SCALER_STD = np.array(clinical_stats_cfg.get("std_scaler_std", [1.0] * N_CLINICAL_FEATURES), dtype=np.float32)
if np.any(STD_SCALER_STD == 0):
    STD_SCALER_STD[STD_SCALER_STD == 0] = 1.0

# [핵심] Min-Max 유사 스케일링 통계치 (2차 Min-Max에 사용)
STDS_MIN = np.array(clinical_stats_cfg.get("scaled_features_min", [0.0] * N_CLINICAL_FEATURES), dtype=np.float32)
STDS_MAX = np.array(clinical_stats_cfg.get("scaled_features_max", [1.0] * N_CLINICAL_FEATURES), dtype=np.float32)


# =============================================================================
# 2. HRV 및 임상 특징 전처리 함수
# =============================================================================

# [HRV 특징 추출 함수]
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
    hr_bpm   = 60000.0 / mean_rri if mean_rri > 0 else np.nan
    sdnn     = np.std(rri_ms)
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
            raw[bad_mask] = HRV_MEDIAN[bad_mask] # 전역 중앙값 대치

    # 전역 Z-score 표준화
    standardized = (raw - HRV_MEAN) / (HRV_STD + eps_hrv)
    return standardized.astype(np.float32)

# [임상 특징 벡터 변환 함수] 학습 시 사용한 2단계 스케일링(StandardScaler -> Min-Max)을 그대로 재현
def get_clinical_feature_vector(patient_data: dict, fill_value: float = 0.0) -> np.ndarray:
    """
    DB에서 불러온 원시 임상 정보를 학습 시 사용된 2단계 스케일링을 거친 13개 특징 벡터로 변환합니다.
    (1. StandardScaler 적용 -> 2. Min-Max 유사 스케일링 적용)
    """
    
    # 1. 원시 데이터를 CLIP의 13개 특징 순서에 맞게 OHE 변환 (StandardScaler 직전 상태 재현)
    df_raw = pd.DataFrame([patient_data])
    
    # 누락된 OHE 기준 컬럼은 'N'/'M' 등 기본값으로 채움
    for col_raw in list(BINARY_STR_COLS.keys()):
        if col_raw not in df_raw.columns:
            # 기본값 설정: sex('M'), emop('N'), preop_dm('N'), preop_htn('N') 등
            df_raw[col_raw] = BINARY_STR_COLS[col_raw][0]
            
    # OHE 적용 (drop_first=True)
    categorical_cols = list(BINARY_STR_COLS.keys())
    df_ohe = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=True)
    
    # 최종 13개 특징 리스트 (Raw Features) 생성
    raw_features_list = []
    
    # 4단계: 연속형 값의 결측치 처리 및 삽입
    for col in CONTINUOUS_COLS:
        val = df_ohe.get(col, np.nan).iloc[0] if col in df_ohe.columns else np.nan
        
        # 학습 시 사용한 평균(STD_SCALER_MEAN)으로 결측치 대체 (fillna 로직 모방)
        if np.isnan(val) or val is None:
            mean_index = CLINICAL_FEATURES_LIST.index(col)
            val = STD_SCALER_MEAN[mean_index] 
            
        raw_features_list.append(val)
        
    # 4단계: 이진형 특징 값 삽입 (누락된 OHE 컬럼은 0으로 채워짐)
    for col in CLINICAL_FEATURES_LIST[len(CONTINUOUS_COLS):]:
        val = df_ohe.get(col, 0.0).iloc[0] if col in df_ohe.columns else 0.0
        raw_features_list.append(val)

    feature_vector = np.array(raw_features_list, dtype=np.float32)
    
    # --------------------------------------------------------------------------
    # 2. [핵심 재현 1단계] StandardScaler (Z-score) 적용 (13개 특징 모두)
    # --------------------------------------------------------------------------
    # 학습 시 사용된 StandardScaler의 mean/std를 사용하여 Z-score 변환을 수동으로 재현
    # 분모 0 방지 로직은 이미 STD_SCALER_STD 로드 시 적용됨
    scaled_features = (feature_vector - STD_SCALER_MEAN) / STD_SCALER_STD
    
    # --------------------------------------------------------------------------
    # 3. [핵심 재현 2단계] Min-Max 유사 스케일링 최종 적용
    # --------------------------------------------------------------------------
    # 학습 시 계산된 scaled_features의 Min/Max 값을 사용
    diff = STDS_MAX - STDS_MIN
    
    # 분모 0인 경우 (단일 값만 있는 특징) 1로 설정하여 0-1 범위 유지 (학습 로직 모방)
    diff[diff == 0] = 1 
    
    normalized_features = (scaled_features - STDS_MIN) / diff
    
    return normalized_features.astype(np.float32)


def preprocess_for_inference(signal_raw, patient_clinical_data: dict):
    """
    [앱 전용] 신호와 임상 데이터를 받아 모델 입력 텐서와 시각화용 파형을 반환합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. HRV 추출 (전역 통계 사용)
    hrv = get_hrv_features(signal_raw)
    hrv_tensor = torch.from_numpy(hrv).float().unsqueeze(0).to(device) # (1, 4)

    # 2. 임상 특징 벡터 추출 (2단계 스케일링 적용)
    clinical_feat = get_clinical_feature_vector(patient_clinical_data)
    clinical_vector_tensor = torch.from_numpy(clinical_feat).float().unsqueeze(0).to(device) # (1, 13)

    # 3. 리샘플링 (Interpolation)
    sig_tensor = torch.from_numpy(signal_raw).float().view(1, 1, -1) # (1, 1, L_raw)
    sig_resampled = F.interpolate(sig_tensor, size=TARGET_LENGTH, mode='linear', align_corners=False)
    
    # 4. 정규화 (Normalization) - 샘플별 Z-score 표준화
    x = sig_resampled.squeeze(0).float() # (1, L_target)
    
    current_mean = x.mean()
    current_std = x.std()
    
    # 샘플별 Z-score 표준화: (X - mean) / std
    if current_std > 1e-8:
        x = (x - current_mean) / current_std
    else:
        # 신호가 플랫한 경우: 평균만 빼서 0으로 중심 맞춤
        x = x - current_mean
        
    # 모델 입력 형태 (1, 1, L_target)로 복구
    model_input = x.unsqueeze(0) 

    # plot_signal은 원본 스케일의 리샘플링된 신호
    return model_input, hrv_tensor, clinical_vector_tensor, sig_resampled.squeeze().cpu().numpy()


# =============================================================================
# 3. 모델 구성 요소 정의
# =============================================================================

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

# [3-2] CLIP Encoder Backbone
# 중요: 가중치 로드를 위해 레이어 이름(self.initial, self.layer1 등)은 CLIP 사전학습 코드와 똑같아야 함.
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


# [3-3] U-Net Decoder Components
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
    
    


# [최종 결합 모델]
class UNet1D_ResNet_Combined(nn.Module):
    """CLIP Pre-trained ResNet Encoder와 U-Net Decoder를 결합한 하이브리드 모델"""
    def __init__(self, n_channels, n_classes, n_clinical_features=N_CLINICAL_FEATURES, n_hrv_features=N_HRV_FEATURES):
        super().__init__()
        self.n_classes = n_classes
        self.n_clinical_features = n_clinical_features
        self.n_hrv_features = n_hrv_features

        self.encoder_feature_dim = 256 # ResNet Backbone의 최종 출력 필터 수
        self.clip_feature_dim = 128    # CLIP의 최종 특징 차원

        # 1. Encoder (CLIP Backbone)
        self.encoder = ResNet1D_Backbone(in_channels=n_channels, base_filters=32)

        # CLIP 특징 추출 레이어 (256 -> 128)
        self.clip_projection = nn.Linear(self.encoder_feature_dim, self.clip_feature_dim) 

        # 2. Decoder (U-Net 구조)
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )

        # 3. Heads
        self.seg_head = nn.Conv1d(16, 1, kernel_size=1)

        # 분류 헤드 입력 차원: 128 (CLIP Feature) + 13 (Clinical) + 4 (HRV) = 145
        clf_in_dim = self.clip_feature_dim + self.n_clinical_features + self.n_hrv_features # 128 + 13 + 4 = 145

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
        x_stem = features[0] 
        x1 = features[1]
        x2 = features[2]
        x3 = features[3] # Bottleneck (B, 256, L/32)

        # 2. Classification Path
        gap = F.adaptive_avg_pool1d(x3, 1).squeeze(-1) # [B, 256]

        # 256차원 GAP 특징을 128차원 CLIP 특징 공간으로 투영
        clip_feature = self.clip_projection(gap) # [B, 128]

        # 128차원 CLIP 특징, 임상 특징, HRV 특징 결합
        combined = torch.cat([clip_feature, clinical_features, hrv_features], dim=1) # [B, 145]
        clf_logits = self.clf_head(combined)

        # 3. Segmentation Path (Decoder)
        d1 = self.up1(x3, x2)
        d2 = self.up2(d1, x1)
        d3 = self.up3(d2, x_stem)
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


def find_mask_indices(mask_array, offset=0, threshold=0.5):
    """
    U-Net의 boolean/binary 마스크 배열(1D)을 받아서
    "index A to B" 형태의 문자열로 변환합니다.
    (여러 세그먼트 감지 가능)
    """
    binary_mask = (mask_array > threshold).astype(np.int32)

    if not np.any(binary_mask):
        return "No anomaly mask detected."

    # 마스크가 1인 인덱스를 찾음
    indices = np.where(binary_mask > 0)[0]
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