# -*- coding: utf-8 -*-
# 베이스라인 PPG 부정맥 분류·segmentation 학습 스크립트 (Colab 환경 기준)

# 라이브러리
import os
import re
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from scipy import stats

# 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 데이터 이해를 위해 우선 train + train_normal 경로 설정
DATA_ROOT = "/content/drive/MyDrive/종합설계/PPG_AD_Dataset"
TRAIN_PATH = f"{DATA_ROOT}/train"
TRAIN_NORMAL_PATH = f"{DATA_ROOT}/train_normal"

# 파일 목록 확인
try:
    train_files = os.listdir(TRAIN_PATH)
    train_n_files = os.listdir(TRAIN_NORMAL_PATH) if os.path.exists(TRAIN_NORMAL_PATH) else []

    # 정상(.npy): train + train_normal 합집합
    npy_files = [f for f in train_files if f.endswith('.npy')]
    npy_files_extra = [f for f in train_n_files if f.endswith('.npy')]
    all_npy_files = [(TRAIN_PATH, f) for f in npy_files] + [(TRAIN_NORMAL_PATH, f) for f in npy_files_extra]

    # 부정맥(.npz): train에서만
    npz_files = [f for f in train_files if f.endswith('.npz')]

    print(f"✅ 데이터 루트 확인 완료: {DATA_ROOT}")
    print(f"- train 폴더: {len(train_files)}개 파일")
    print(f"- train_normal 폴더: {len(train_n_files)}개 파일")
    print(f"정상(.npy) 총 개수(통합): {len(all_npy_files)}개")
    print(f"부정맥(.npz) 개수(train): {len(npz_files)}개")
    print("-" * 50)

except FileNotFoundError as e:
    print(f"❌ 경로 오류: {e}")
    assert False, "데이터 경로를 확인해주세요."

# .npy (정상) 샘플 로드 및 시각화 — 랜덤 선택
if all_npy_files:
    sample_dir, sample_npy_file = random.choice(all_npy_files)
    sample_npy_path = os.path.join(sample_dir, sample_npy_file)

    try:
        normal_signal = np.load(sample_npy_path)
        print(f"📂 정상(.npy) 샘플 파일 로드: {sample_npy_file} (from {os.path.basename(sample_dir)}/)")
        print(f"데이터 형태(shape): {normal_signal.shape}")

        plt.figure(figsize=(6, 4))
        plt.title(f"Sample Normal Signal (.npy) - {sample_npy_file}")
        plt.plot(normal_signal, label='PPG Signal')
        plt.xlabel("Sample Index"); plt.ylabel("Amplitude")
        plt.legend(); plt.grid(True); plt.show()
        print("-" * 50)
    except Exception as e:
        print(f"❌ .npy 파일 로드/시각화 중 오류: {e}")
else:
    print("⚠️ normal(.npy) 파일이 없어 시각화를 건너뜁니다.")
    print("-" * 50)

# .npz (부정맥) 샘플 로드, 구조 확인 및 시각화 — train에서만
if npz_files:
    sample_npz_file = random.choice(npz_files)
    sample_npz_path = os.path.join(TRAIN_PATH, sample_npz_file)

    try:
        arrhythmia_data = np.load(sample_npz_path)
        print(f"📂 부정맥(.npz) 샘플 파일 로드: {sample_npz_file}")
        print(f".npz 파일 내부 key 목록: {list(arrhythmia_data.keys())}")

        signal_key, mask_key = 'signal', 'mask'
        if signal_key in arrhythmia_data.keys() and mask_key in arrhythmia_data.keys():
            arrhythmia_signal = arrhythmia_data[signal_key]
            arrhythmia_mask = arrhythmia_data[mask_key]

            print(f"신호 데이터 shape: {arrhythmia_signal.shape} | 마스크 shape: {arrhythmia_mask.shape}")

            plt.figure(figsize=(6, 4))
            plt.title(f"Sample Arrhythmia Signal with Mask (.npz) - {sample_npz_file}")
            plt.plot(arrhythmia_signal, color='blue', label='PPG Signal', zorder=2)
            y_min, y_max = plt.ylim()
            plt.fill_between(range(len(arrhythmia_mask)), y_min, y_max,
                             where=arrhythmia_mask > 0, color='red', alpha=0.3,
                             label='Arrhythmia Mask', zorder=1)
            plt.ylim(y_min, y_max)
            plt.xlabel("Sample Index"); plt.ylabel("Amplitude")
            plt.legend(); plt.grid(True); plt.show()
        else:
            print("⚠️ 'signal' 또는 'mask' key를 찾지 못했습니다.", list(arrhythmia_data.keys()))
    except Exception as e:
        print(f"❌ .npz 파일 로드/시각화 중 오류: {e}")
else:
    print("⚠️ train 경로에 .npz 파일이 없어 시각화를 건너뜁니다.")

# -- 1. 설정 --

# matplotlib 설정 초기화
plt.rcdefaults()

# 데이터 루트
DATA_ROOT = "/content/drive/MyDrive/종합설계/PPG_AD_Dataset"
assert os.path.exists(DATA_ROOT), f"❌ Path not found: '{DATA_ROOT}'"

# 분포 점검용으로 포함할 폴더
set_types = ['train', 'valid', 'test', 'train_normal', 'val_normal']

# 모든 데이터 파일에서 신호 길이 수집
all_signal_lengths = []
print("Scanning files from datasets:", set_types)

for set_type in set_types:
    current_path = os.path.join(DATA_ROOT, set_type)
    if not os.path.isdir(current_path):
        print(f"⚠️ Directory not found, skipping: '{current_path}'")
        continue

    file_list = os.listdir(current_path)
    for filename in tqdm(file_list, desc=f"Processing '{set_type}'"):
        file_path = os.path.join(current_path, filename)
        try:
            if filename.endswith('.npy'):
                signal = np.load(file_path)
                all_signal_lengths.append(len(signal))
            elif filename.endswith('.npz'):
                with np.load(file_path) as data:
                    if 'signal' in data.keys():
                        signal = data['signal']
                        all_signal_lengths.append(len(signal))
        except Exception as e:
            print(f"⚠️ Error processing file '{set_type}/{filename}': {e}")

# 전체 통계 계산 및 출력
if all_signal_lengths:
    lengths_np = np.array(all_signal_lengths)
    total_count = len(lengths_np)
    mean_length = np.mean(lengths_np)
    std_length = np.std(lengths_np)
    median_length = np.median(lengths_np)
    min_length = np.min(lengths_np)
    max_length = np.max(lengths_np)

    print("\n✅ Overall data length analysis complete!")
    print("-" * 50)
    print("        < Overall Signal Length Statistics >")
    print("-" * 50)
    print(f"Total signals analyzed: {total_count}")
    print(f"Mean Length:            {mean_length:.2f}")
    print(f"Standard Deviation:     {std_length:.2f}")
    print(f"Median Length:          {median_length:.0f}")
    print(f"Min Length:             {min_length}")
    print(f"Max Length:             {max_length}")

    # 전체 길이 분포 시각화
    plt.figure(figsize=(8, 5))
    plt.hist(lengths_np, bins=50, alpha=0.75, color='royalblue', edgecolor='black')
    plt.title("Overall Distribution of Signal Lengths", fontsize=16)
    plt.xlabel("Signal Length (number of samples)", fontsize=12)
    plt.ylabel("Frequency (number of files)", fontsize=12)
    plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.2f}')

    stats_text = (f"Total Count: {total_count}\n"
                  f"Mean: {mean_length:.2f}\n"
                  f"Std Dev: {std_length:.2f}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ No signal data found to analyze.")

# --- TRAIN 통계: train + train_normal 모두 포함해 계산 ---

TRAIN_DIRS = [
    "/content/drive/MyDrive/종합설계/PPG_AD_Dataset/train",
    "/content/drive/MyDrive/종합설계/PPG_AD_Dataset/train_normal",
]

# 파형이 아닐 확률이 높은 키(완전 제외)
EXCLUDE_KEY_PATTERNS = re.compile(
    r"(mask|label|ann|peak|rri|idx|index|seg|class|cls|y|target|meta|info)", re.IGNORECASE
)

# 파형 필터 기준
MIN_LEN = 200        # 사실상 1D 길이 최소
UNIQUE_MIN = 10      # 고유값 너무 적으면 제외(이진/라벨류 배제)

def iter_numeric_arrays_from_file(path):
    """파일에서 (key(optional), raw_array) 제너레이터"""
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        yield (None, data)
    else:
        with np.load(path, allow_pickle=True) as z:
            for k in z.files:
                yield (k, z[k])

def to_1d_float(arr):
    """수치형을 1D float로 변환(불가하면 None)"""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype == object:
        parts = []
        for x in arr.flat:
            if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
                parts.append(x.astype(np.float64).ravel())
        return np.concatenate(parts) if parts else None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if arr.ndim == 2 and 1 in arr.shape:  # (1,L) or (L,1)
        return arr.astype(np.float64).reshape(-1)
    return None

def is_wave_like(arr):
    """파형 여부 판정: 길이/연속값 기준"""
    L = arr.size
    if L < MIN_LEN:
        return False
    u = np.unique(arr)
    if u.size <= UNIQUE_MIN:
        return False
    if u.size <= 3 and set(np.round(u, 6)).issubset({0.0, 1.0}):  # 이진 마스크 배제
        return False
    return True

# ---- 집계 (오직 '포함된 파형'만 사용) ----
paths = []
for d in TRAIN_DIRS:
    if os.path.isdir(d):
        paths.extend(
            os.path.join(r, f)
            for r, _, fs in os.walk(d)
            for f in fs if f.endswith((".npy", ".npz"))
        )
    else:
        print(f"⚠️ Directory not found, skipping: {d}")

sum_x = 0.0
sum_x2 = 0.0
count = 0

for path in tqdm(paths, desc="Computing TRAIN mean/std (signals only)"):
    try:
        for key, raw in iter_numeric_arrays_from_file(path):
            if key is not None and EXCLUDE_KEY_PATTERNS.search(str(key)):
                continue
            arr = to_1d_float(raw)
            if arr is None or not is_wave_like(arr):
                continue
            sum_x  += float(arr.sum())
            sum_x2 += float((arr * arr).sum())
            count  += int(arr.size)
    except Exception as e:
        print(f"⚠️ Error processing '{path}': {e}")

if count == 0:
    raise RuntimeError("파형으로 판단되는 train 데이터가 없습니다. 필터 기준을 확인하세요.")

TRAIN_MEAN = sum_x / count
TRAIN_STD  = float(np.sqrt(max(0.0, (sum_x2 / count) - TRAIN_MEAN**2)))

print(f"\n[TRAIN Stats on train + train_normal]")
print(f"Samples counted : {count}")
print(f"TRAIN_MEAN      : {TRAIN_MEAN:.6f}")
print(f"TRAIN_STD       : {TRAIN_STD:.6f}")

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 기본 경로 설정
BASE_PROJECT_PATH = "/content/drive/MyDrive/종합설계"
DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "PPG_AD_Dataset")
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH  = os.path.join(DATASET_PATH, "test")
MODEL_SAVE_PATH = os.path.join(BASE_PROJECT_PATH, "best_resampling_model.pth")

# HRV feature 개수( HR, SDNN, RMSSD, pNN50 )
N_HRV_FEATURES = 4

# --- 통계 설정 로드 ---
CONFIG_PATH = os.path.join(DATASET_PATH, "data_stats.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        stats_cfg = json.load(f)

    # 파형 정규화 통계 (train + train_normal 기준)
    TRAIN_MEAN = float(stats_cfg.get("train_mean", 0.0))
    TRAIN_STD  = float(stats_cfg.get("train_std", 1.0))

    # HRV 정규화 통계 (train_normal 기준) — 벡터(길이 4)로 로드
    def _as_vec(x, default):
        arr = np.array(x if x is not None else default, dtype=float).reshape(-1)
        if arr.size != N_HRV_FEATURES:
            # 크기 불일치 시 안전한 기본값으로 보정
            arr = np.array(default, dtype=float)
        return arr

    HRV_MEAN   = _as_vec(stats_cfg.get("hrv_mean"),   [0.0]*N_HRV_FEATURES)
    HRV_STD    = _as_vec(stats_cfg.get("hrv_std"),    [1.0]*N_HRV_FEATURES)
    HRV_MEDIAN = _as_vec(stats_cfg.get("hrv_median"), [0.0]*N_HRV_FEATURES)  # 결측 대치용

    TARGET_LENGTH = int(stats_cfg.get("target_length", 286))
    print(f"✅ Loaded normalization stats from data_stats.json")
else:
    print("⚠️ data_stats.json not found. Default values used.")
    TRAIN_MEAN, TRAIN_STD = 0.0, 1.0
    HRV_MEAN   = np.zeros(N_HRV_FEATURES, dtype=float)
    HRV_STD    = np.ones(N_HRV_FEATURES, dtype=float)
    HRV_MEDIAN = np.zeros(N_HRV_FEATURES, dtype=float)
    TARGET_LENGTH = 286  # fallback

print(f"TARGET_LENGTH : {TARGET_LENGTH}")

CLASS_MAP = {'normal': 0, 'af': 1, 'b': 2, 't': 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 가능한 디바이스: {device}")

# --- HRV 특징 추출 (정규화 X, 4개 특징: HR, SDNN, RMSSD, pNN50%) ---
def get_hrv_features_for_stats(signal, fs=100.0):
    """
    입력: 1D 신호 (PPG)
    출력: np.array([HR(bpm), SDNN(ms), RMSSD(ms), pNN50(%)], dtype=float32)
    """
    if signal is None or len(signal) < 10:
        return None
    try:
        # 피크 임계 높이는 신호 내 분위수 기반으로 자동 설정
        min_height = np.nanquantile(signal, 0.6)
    except Exception:
        return None

    # 최소 간격: 0.3초 (100Hz 기준 30샘플)
    # fs가 변해도 0.3초 물리적 시간은 유지되도록 수식화
    min_dist = int(fs * 0.3)
    peaks, _ = find_peaks(signal, height=min_height, distance=min_dist)

    if len(peaks) < 3:  # RR 간격이 최소 2개 필요 → 피크 최소 3개
        return None

    # ms 단위 RR 간격 (1000/fs 곱하기)
    rri_ms = np.diff(peaks) * (1000.0 / fs)
    if np.any(~np.isfinite(rri_ms)) or rri_ms.size < 2:
        return None

    # HR(bpm)
    mean_rri = np.mean(rri_ms)
    hr_bpm = 60000.0 / mean_rri if mean_rri > 0 else np.nan

    # SDNN(ms): RR의 표준편차
    sdnn = np.std(rri_ms)

    # RMSSD(ms): 인접 RR 차이의 제곱평균 제곱근
    diff_rri = np.diff(rri_ms)
    rmssd = np.sqrt(np.mean(diff_rri ** 2))

    # pNN50(%): |ΔRR| > 50ms 비율(백분율)
    nn50_cnt = np.sum(np.abs(diff_rri) > 50.0)
    pnn50 = (nn50_cnt / diff_rri.size) * 100.0 if diff_rri.size > 0 else np.nan

    feats = np.array([hr_bpm, sdnn, rmssd, pnn50], dtype=np.float32)
    # 안전 처리 (inf/NaN → 유한값 대체)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats

# --- HRV 통계 계산: 기본은 새 정상인 train_normal만 사용 ---
def calculate_hrv_statistics_from_dirs(dir_list, include_npz=False):
    """
    dir_list의 폴더를 순회하며 HRV 4특징을 계산하고 (mean, std, median)를 반환.
    기본적으로 .npy만 사용(= normal). 필요 시 include_npz=True로 .npz도 포함 가능.
    """
    all_feats = []
    for d in dir_list:
        if not os.path.isdir(d):
            print(f"⚠️ Directory not found, skip: {d}")
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy') or (include_npz and f.endswith('.npz'))]
        for fn in tqdm(files, desc=f"HRV from '{os.path.basename(d)}'"):
            fp = os.path.join(d, fn)
            try:
                if fn.endswith('.npy'):
                    sig = np.load(fp).astype(np.float32)
                else:  # npz 포함 옵션
                    with np.load(fp, allow_pickle=True) as z:
                        sig = z.get('signal', None)
                        if sig is not None:
                            sig = sig.astype(np.float32)

                # 샘플링 주파수는 100.0Hz 고정
                feats = get_hrv_features_for_stats(sig, fs=100.0)
                if feats is not None:
                    all_feats.append(feats)
            except Exception as e:
                print(f"⚠️ {fp} -> {e}")

    if len(all_feats) == 0:
        print("오류: HRV 특징을 추출하지 못했습니다.")
        return None, None, None

    M = np.vstack(all_feats)  # (N, 4)
    mean = M.mean(axis=0)
    std  = M.std(axis=0)
    med  = np.median(M, axis=0)
    return mean, std, med

# --- 실행: train_normal 기준으로 계산하고 data_stats.json에 저장/갱신 ---
HRV_SOURCE_DIRS = [
    os.path.join(DATASET_PATH, "train_normal"),  # 기본: 새 정상인 normal
    # 필요 시 아래 주석 해제하여 기존 train의 normal(.npy)도 포함 가능
    # os.path.join(DATASET_PATH, "train")
]

new_hrv_mean, new_hrv_std, new_hrv_median = calculate_hrv_statistics_from_dirs(HRV_SOURCE_DIRS, include_npz=False)

if new_hrv_mean is not None:
    print("\n--- 계산 완료: HRV 통계 (mean/std/median) ---")
    print("HRV_MEAN  :", new_hrv_mean)
    print("HRV_STD   :", new_hrv_std)
    print("HRV_MEDIAN:", new_hrv_median)

    # 기존 data_stats.json과 병합 저장
    stats_path = os.path.join(DATASET_PATH, "data_stats.json")
    stats_cfg = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                stats_cfg = json.load(f)
        except Exception:
            stats_cfg = {}

    stats_cfg.update({
        "hrv_mean": new_hrv_mean.tolist(),
        "hrv_std": new_hrv_std.tolist(),
        "hrv_median": new_hrv_median.tolist(),   # 결측 대치용으로 활용
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hrv_source": [os.path.basename(d) for d in HRV_SOURCE_DIRS],
        "hrv_order": ["HR_bpm", "SDNN_ms", "RMSSD_ms", "pNN50_percent"]
    })

    with open(stats_path, "w") as f:
        json.dump(stats_cfg, f, indent=2)
    print(f"✅ HRV 통계를 '{stats_path}'에 저장했습니다.")
else:
    print("❌ HRV 통계 저장을 건너뜁니다.")

# --- 2. 데이터 전처리 및 증강 클래스 정의 ---

# HRV 통계 벡터 형태 보정 (길이 4: HR, SDNN, RMSSD, pNN50%)
N_HRV_FEATURES = 4
HRV_MEAN   = np.asarray(HRV_MEAN,   dtype=np.float32).reshape(-1)
HRV_STD    = np.asarray(HRV_STD,    dtype=np.float32).reshape(-1)
HRV_MEDIAN = np.asarray(HRV_MEDIAN, dtype=np.float32).reshape(-1) if 'HRV_MEDIAN' in globals() else np.zeros(N_HRV_FEATURES, dtype=np.float32)

if HRV_MEAN.size != N_HRV_FEATURES or HRV_STD.size != N_HRV_FEATURES or HRV_MEDIAN.size != N_HRV_FEATURES:
    print("⚠️ HRV 통계(hv_mean/std/median) 벡터 크기가 (4,)가 아닙니다. 안전한 기본값으로 대체합니다.")
    HRV_MEAN   = np.zeros(N_HRV_FEATURES, dtype=np.float32)
    HRV_STD    = np.ones (N_HRV_FEATURES, dtype=np.float32)
    HRV_MEDIAN = np.zeros(N_HRV_FEATURES, dtype=np.float32)

eps_hrv = 1e-6  # 분모 0 방지

def _compute_hrv_raw(signal: np.ndarray, fs: float = 100.0) -> np.ndarray | None:
    """
    HRV 4특성(정규화 전)을 계산.
    반환 순서: [HR(bpm), SDNN(ms), RMSSD(ms), pNN50(%)]
    실패 시 None 반환.
    """
    if signal is None or len(signal) < 10 or not np.any(np.isfinite(signal)):
        return None

    try:
        min_height = np.nanquantile(signal, 0.6)
    except Exception:
        return None

    # 최소 간격: 0.3초(fs * 0.3)
    peaks, _ = find_peaks(signal, height=min_height, distance=int(fs * 0.3))
    if len(peaks) < 3:  # RR 간격 최소 2개 필요
        return None

    # ms 단위 변환 시 fs 사용 필수
    rri_ms = np.diff(peaks) * (1000.0 / fs)  # ms
    if rri_ms.size < 2 or not np.all(np.isfinite(rri_ms)):
        return None

    mean_rri = np.mean(rri_ms)
    hr_bpm   = 60000.0 / mean_rri if mean_rri > 0 else np.nan
    sdnn     = np.std(rri_ms)

    diff_rri = np.diff(rri_ms)
    if diff_rri.size == 0:
        return None
    rmssd = np.sqrt(np.mean(diff_rri ** 2))
    pnn50 = (np.sum(np.abs(diff_rri) > 50.0) / diff_rri.size) * 100.0

    feats = np.array([hr_bpm, sdnn, rmssd, pnn50], dtype=np.float32)

    if not np.any(np.isfinite(feats)):
        return None
    return feats

def get_hrv_features(signal: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    HRV 4특성 계산 → 결측/비유한값은 HRV_MEDIAN으로 대치 → (x - mean) / std 표준화
    반환: 표준화된 길이 4 벡터 (float32)
    """
    raw = _compute_hrv_raw(signal, fs=fs)

    if raw is None:
        raw = HRV_MEDIAN.copy()
    else:
        raw = raw.astype(np.float32)
        # 비유한값은 각 차원별 median으로 대치
        bad = ~np.isfinite(raw)
        if np.any(bad):
            raw[bad] = HRV_MEDIAN[bad]

    standardized = (raw - HRV_MEAN) / (HRV_STD + eps_hrv)
    return standardized.astype(np.float32)

class PPGDataset_Resampling(Dataset):
    def __init__(self, data_path, target_length, mean, std, fs=100.0):
        """
        data_path: str 또는 [str, ...]
        fs: 데이터의 샘플링 레이트 (Hz)
        """
        self.paths = [data_path] if isinstance(data_path, str) else list(data_path)
        self.target_length = int(target_length)
        self.mean = float(mean)
        self.std = float(std) if float(std) != 0 else 1.0
        self.fs = float(fs) # ★ 클래스 변수로 저장

        # 파일 수집(중복 방지)
        all_files = []
        for p in self.paths:
            if not os.path.isdir(p):
                print(f"⚠️ Directory not found, skip: {p}")
                continue
            for f in os.listdir(p):
                if f.endswith(('.npy', '.npz')):
                    all_files.append((p, f))

        self.file_list = list(dict.fromkeys(all_files))
        self.labels = self._get_all_labels()

    def _get_all_labels(self):
        print(f"{self.paths} 경로에서 모든 라벨을 로드하는 중...")
        labels = []
        label_map_vec = {0: 'normal', 1: 'normal', 2: 'af', 3: 'b', 4: 't'}
        for base_dir, file_name in tqdm(self.file_list, desc="라벨 로딩"):
            file_path = os.path.join(base_dir, file_name)
            if file_name.endswith('.npy'):
                label_name = 'normal'
            elif file_name.endswith('.npz'):
                with np.load(file_path, allow_pickle=True) as data:
                    class_label_raw = data.get('class')
                    if class_label_raw is None:
                        label_name = 'normal'
                    elif isinstance(class_label_raw, np.ndarray) and class_label_raw.ndim > 0 and len(class_label_raw) > 1:
                        non_zero = class_label_raw[class_label_raw != 0]
                        if len(non_zero) > 0:
                            label_num = stats.mode(non_zero, keepdims=True)[0][0]
                            label_name = label_map_vec.get(int(label_num), 'normal')
                        else:
                            label_name = 'normal'
                    else:
                        label_name = label_map_vec.get(int(np.array(class_label_raw).item()), 'normal')
            else:
                label_name = 'normal'
            labels.append(CLASS_MAP.get(label_name, CLASS_MAP['normal']))
        return labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_dir, file_name = self.file_list[idx]
        file_path = os.path.join(base_dir, file_name)

        label_map_vec = {0: 'normal', 1: 'normal', 2: 'af', 3: 'b', 4: 't'}

        if file_name.endswith('.npy'):
            signal = np.load(file_path).astype(np.float32)
            mask = np.zeros_like(signal, dtype=np.float32)
            label_name = 'normal'
        elif file_name.endswith('.npz'):
            with np.load(file_path, allow_pickle=True) as data:
                signal = data['signal'].astype(np.float32)
                mask = data['mask'].astype(np.float32)
                class_label_raw = data.get('class')
                if class_label_raw is None:
                    label_name = 'normal'
                elif isinstance(class_label_raw, np.ndarray) and class_label_raw.ndim > 0 and len(class_label_raw) > 1:
                    non_zero = class_label_raw[class_label_raw != 0]
                    if len(non_zero) > 0:
                        label_num = stats.mode(non_zero, keepdims=True)[0][0]
                        label_name = label_map_vec.get(int(label_num), 'normal')
                    else:
                        label_name = 'normal'
                else:
                    label_name = label_map_vec.get(int(np.array(class_label_raw).item()), 'normal')
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_name}")

        label = CLASS_MAP.get(label_name, CLASS_MAP['normal'])

        # HRV는 리샘플링 전 원본 신호 기준으로 계산
        hrv_features = get_hrv_features(signal, fs=self.fs)

        # Resample to target length
        signal_tensor = torch.from_numpy(signal).unsqueeze(0)  # (1, L)
        mask_tensor   = torch.from_numpy(mask).unsqueeze(0)    # (1, L)

        resampled_signal = F.interpolate(signal_tensor.unsqueeze(0), size=self.target_length,
                                         mode='linear', align_corners=False).squeeze(0)  # (1, T)
        resampled_mask   = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_length,
                                         mode='nearest').squeeze(0)                      # (1, T)

        # Normalize wave
        x = resampled_signal.numpy()
        x = (x - self.mean) / (self.std + 1e-8)

        final_signal_tensor = torch.from_numpy(x).float()
        final_mask_tensor   = resampled_mask.float()
        label_tensor        = torch.tensor(label, dtype=torch.long)
        hrv_features_tensor = torch.from_numpy(hrv_features).float()

        return final_signal_tensor, final_mask_tensor, label_tensor, hrv_features_tensor

import os, json, numpy as np
from tqdm import tqdm
from datetime import datetime

# ===== 경로 =====
BASE_PROJECT_PATH = "/content/drive/MyDrive/종합설계"
DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "PPG_AD_Dataset")
TRAIN_DIRS = [
    os.path.join(DATASET_PATH, "train"),
    os.path.join(DATASET_PATH, "train_normal"),
]

assert any(os.path.isdir(d) for d in TRAIN_DIRS), "train/train_normal 폴더를 확인하세요."

# ===== 통계 계산 =====
sum_x = 0.0
sum_x2 = 0.0
count = 0
min_len = None
num_files = 0

def load_signal(fp):
    if fp.endswith(".npy"):
        return np.load(fp)
    elif fp.endswith(".npz"):
        with np.load(fp, allow_pickle=True) as z:
            # 부정맥 npz는 'signal' 키 기준
            if "signal" in z:
                return z["signal"]
    return None

for d in TRAIN_DIRS:
    if not os.path.isdir(d):
        print(f"⚠️ skip: {d}")
        continue
    files = [f for f in os.listdir(d) if f.endswith((".npy", ".npz"))]
    for fn in tqdm(files, desc=f"Scanning {os.path.basename(d)}"):
        fp = os.path.join(d, fn)
        try:
            sig = load_signal(fp)
            if sig is None:
                continue
            sig = np.asarray(sig, dtype=np.float64).reshape(-1)
            # 길이 집계
            L = sig.size
            if min_len is None or L < min_len:
                min_len = L
            # 평균/표준편차 집계
            sum_x  += sig.sum()
            sum_x2 += (sig*sig).sum()
            count  += L
            num_files += 1
        except Exception as e:
            print(f"⚠️ {fp} -> {e}")

if count == 0 or min_len is None:
    raise RuntimeError("유효한 학습 신호를 찾지 못했습니다.")

train_mean = float(sum_x / count)
train_std  = float(np.sqrt(max(0.0, (sum_x2 / count) - train_mean**2)))
target_length = int(min_len)

print("\n✅ TRAIN 통계 계산 결과 (train + train_normal)")
print(f"- 파일 수: {num_files}")
print(f"- TRAIN_MEAN      : {train_mean:.6f}")
print(f"- TRAIN_STD       : {train_std:.6f}")
print(f"- TARGET_LENGTH   : {target_length}")

# ===== data_stats.json 병합 저장 =====
stats_path = os.path.join(DATASET_PATH, "data_stats.json")
cfg = {}
if os.path.exists(stats_path):
    try:
        with open(stats_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}

cfg.update({
    "train_mean": train_mean,
    "train_std":  train_std if train_std != 0 else 1.0,
    "target_length": target_length,
    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "train_stats_source": ["train", "train_normal"],
})

with open(stats_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"\n💾 저장 완료: {stats_path}")

# --- 3. 모델 정의 (안정화: GroupNorm/LN + Bottleneck-SE 추가) ---
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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(2) - x1.size(2)
        if diff != 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SE1D(nn.Module):
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

class UNet1D_Hybrid(nn.Module):
    """
    하이브리드 멀티태스크 U-Net (1D)
    - Segmentation: 디코더 출력(feature map) → seg_head
    - Classification: 인코더 보틀넥(x5) 전역풀링 + HRV 4특징 → clf_head
    """
    def __init__(self, n_channels, n_classes, n_external_features=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_external_features = n_external_features

        # 인코더
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # bottleneck

        # bottleneck 채널 주의(SE)
        self.se_bottleneck = SE1D(1024, r=16)

        # 디코더
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        # 세그먼테이션 헤드
        self.seg_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=1)
        )

        # 분류 헤드: 1024(bottleneck GAP) + 외부 4특징
        self.clf_head = nn.Sequential(
            nn.Linear(1024 + n_external_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(64, n_classes)
        )

    def forward(self, signal_input, external_features):
        # 인코더
        x1 = self.inc(signal_input)   # (B,64,L)
        x2 = self.down1(x1)           # (B,128,L/2)
        x3 = self.down2(x2)           # (B,256,L/4)
        x4 = self.down3(x3)           # (B,512,L/8)
        x5 = self.down4(x4)           # (B,1024,L/16)

        # SE 주의 적용
        x5 = self.se_bottleneck(x5)

        # 디코더 → segmentation
        xd = self.up1(x5, x4)
        xd = self.up2(xd, x3)
        xd = self.up3(xd, x2)
        xd = self.up4(xd, x1)
        seg_logits = self.seg_head(xd)  # (B,1,L)

        # bottleneck 기반 classification
        g = F.adaptive_avg_pool1d(x5, 1).squeeze(-1)           # (B,1024)
        combined = torch.cat([g, external_features], dim=1)    # (B,1024+ext)
        clf_logits = self.clf_head(combined)                   # (B,C)

        return clf_logits, seg_logits

# --- 4. 학습/평가 함수 정의 ---

def _dice_coefficient_bin(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    pred_bin, target_bin: (B, 1, T) in {0,1}
    """
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    intersect = (pred_bin * target_bin).sum(dim=(1,2))
    denom = pred_bin.sum(dim=(1,2)) + target_bin.sum(dim=(1,2)) + eps
    dice = (2.0 * intersect + eps) / denom
    return dice.mean()

def train_one_epoch(model, dataloader, criterion_clf, criterion_seg, optimizer, device, alpha=1.0, beta=1.0):
    model.train()
    total_loss, total_clf_loss, total_seg_loss = 0.0, 0.0, 0.0
    all_labels, all_preds = [], []

    for signals, masks, labels, hrv_features in tqdm(dataloader, desc="Training"):
        signals = signals.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        hrv_features = hrv_features.to(device)

        optimizer.zero_grad()
        clf_logits, seg_logits = model(signals, hrv_features)

        clf_loss = criterion_clf(clf_logits, labels)
        seg_loss = criterion_seg(seg_logits, masks)
        loss = clf_loss * alpha + seg_loss * beta

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_clf_loss += float(clf_loss.item())
        total_seg_loss += float(seg_loss.item())

        preds = torch.argmax(clf_logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    denom = max(1, len(dataloader))
    avg_loss = total_loss / denom
    avg_clf_loss = total_clf_loss / denom
    avg_seg_loss = total_seg_loss / denom
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, avg_clf_loss, avg_seg_loss, f1, acc


def evaluate(model, dataloader, criterion_clf, criterion_seg, device, alpha=1.0, beta=1.0, thr: float = 0.5):
    """
    세그먼트 평가는 포인트 단위 Accuracy로 산출.
    Dice는 참고용으로 함께 계산하여 반환 마지막 항목으로 제공.
    반환:
      avg_loss, avg_clf_loss, avg_seg_loss, f1, acc, all_preds, all_labels, all_confidences, avg_seg_pixel_acc, avg_seg_dice
    """
    model.eval()
    total_loss, total_clf_loss, total_seg_loss = 0.0, 0.0, 0.0
    all_labels, all_preds, all_confidences = [], [], []
    total_pixel_acc = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for signals, masks, labels, hrv_features in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            hrv_features = hrv_features.to(device)

            clf_logits, seg_logits = model(signals, hrv_features)

            # --- 손실 ---
            clf_loss = criterion_clf(clf_logits, labels)
            seg_loss = criterion_seg(seg_logits, masks)
            loss = clf_loss * alpha + seg_loss * beta

            total_loss += float(loss.item())
            total_clf_loss += float(clf_loss.item())
            total_seg_loss += float(seg_loss.item())

            # --- 분류 성능 ---
            probs = F.softmax(clf_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(torch.max(probs, dim=1)[0].cpu().numpy())

            # --- 세그먼트 성능: 포인트 단위 Accuracy ---
            seg_probs = torch.sigmoid(seg_logits)                  # (B,1,T)
            seg_bin = (seg_probs > thr).to(masks.dtype)            # (B,1,T) in {0,1}
            pixel_acc = (seg_bin == masks).float().mean()          # scalar
            total_pixel_acc += float(pixel_acc.item())

            # 참고용 Dice
            dice = _dice_coefficient_bin(seg_bin, masks)
            total_dice += float(dice.item())

    denom = max(1, len(dataloader))
    avg_loss = total_loss / denom
    avg_clf_loss = total_clf_loss / denom
    avg_seg_loss = total_seg_loss / denom
    avg_seg_pixel_acc = total_pixel_acc / denom
    avg_seg_dice = total_dice / denom

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    return (
        avg_loss, avg_clf_loss, avg_seg_loss,
        f1, acc,
        all_preds, all_labels, all_confidences,
        avg_seg_pixel_acc, avg_seg_dice
    )

# --- 5. 학습 및 평가 파라미터 설정 (val_total 기준 EarlyStopping + AdamW + Scheduler + grad clip) ---
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

set_seed(42)

# --- EarlyStopping ---
class EarlyStopping:
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
        if self.best_score is None:
            return True
        if self.mode == 'max':
            return (score - self.best_score) > self.delta
        else:
            return (self.best_score - score) > self.delta

    def __call__(self, score, model, ref_metric_for_log=None, ref_name='ref'):
        if hasattr(score, "item"):
            score = float(score.item())
        if self.best_score is None or self._is_improved(score):
            self.best_score = score
            if self.verbose:
                if ref_metric_for_log is None:
                    print(f"✅ {self.monitor_name} improved to {score:.6f}. Saving model...")
                else:
                    print(f"✅ {self.monitor_name} improved to {score:.6f} "
                          f"({ref_name}: {float(ref_metric_for_log):.6f}). Saving model...")
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping counter: {self.counter} / {self.patience} "
                      f"(best {self.monitor_name}: {self.best_score:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
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
        if self.reduction == 'sum':  return loss.sum()
        return loss

# --- Asymmetric Focal Loss ---
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, focal_loss_criterion, penalty_factor=2.5, true_label_idx=None, pred_label_idx=None):
        super().__init__()
        self.focal_loss_criterion = focal_loss_criterion
        self.penalty_factor = penalty_factor
        self.true_label_idx = true_label_idx
        self.pred_label_idx = pred_label_idx
    def forward(self, inputs, targets):
        self.focal_loss_criterion.reduction = 'none'
        base_loss = self.focal_loss_criterion(inputs, targets)
        preds = torch.argmax(inputs, dim=1)
        mask = (targets == self.true_label_idx) & (preds == self.pred_label_idx)
        penalty = torch.ones_like(base_loss)
        penalty[mask] = self.penalty_factor
        return (base_loss * penalty).mean()

# --- Segmentation Loss ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        inter = (probs * targets).sum()
        dice = (2 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.wb, self.wd = weight_bce, weight_dice
    def forward(self, logits, targets):
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float().contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    inter = (preds * targets).sum()
    dice = (2 * inter + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

# ----------------- 하이퍼파라미터 -----------------
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ALPHA = 1.8   # 분류 손실 가중치
BETA  = 0.9   # 세그 손실 가중치
MAX_NORM = 1.0  # (옵션) grad clip, 사용하려면 train_one_epoch 안에서 적용

# --- 6. 데이터셋 및 데이터로더 ---
TRAIN_NORMAL_PATH = os.path.join(DATASET_PATH, "train_normal")
VALID_NORMAL_PATH = os.path.join(DATASET_PATH, "val_normal")

# dataset (★ fs=100.0 전달)
train_dataset = PPGDataset_Resampling([TRAIN_PATH, TRAIN_NORMAL_PATH], TARGET_LENGTH, TRAIN_MEAN, TRAIN_STD, fs=100.0)
valid_dataset = PPGDataset_Resampling([VALID_PATH, VALID_NORMAL_PATH], TARGET_LENGTH, TRAIN_MEAN, TRAIN_STD, fs=100.0)

class_counts = np.bincount(train_dataset.labels, minlength=len(CLASS_MAP))
safe_counts = np.clip(class_counts, 1, None)
class_weights_for_sampler = 1.0 / torch.tensor(safe_counts, dtype=torch.float32)
sample_weights = class_weights_for_sampler[train_dataset.labels]
sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

num_workers = 2 if torch.cuda.is_available() else 0
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    persistent_workers=bool(num_workers)
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    persistent_workers=bool(num_workers)
)

print("Train class counts (used):", class_counts.tolist())

# --- 7. 모델, 손실, 옵티마이저, 스케줄러, ES ---
model = UNet1D_Hybrid(n_channels=1, n_classes=len(CLASS_MAP), n_external_features=4).to(device)

train_labels = np.array(train_dataset.labels)
classes_present = np.unique(train_labels)
raw_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=train_labels)
class_weights_full = np.ones(len(CLASS_MAP), dtype=np.float32)
class_weights_full[classes_present] = raw_weights
class_weights_full *= (15.0 / class_weights_full.max())

# 클래스 부스팅
boost = {'t': 1.3, 'af': 1.4, 'b': 1.4}
for k, m in boost.items():
    class_weights_full[CLASS_MAP[k]] *= m

special_weights = torch.tensor(class_weights_full, dtype=torch.float, device=device)

base_criterion_clf = FocalLoss(gamma=2.2, alpha=special_weights)
criterion_clf = AsymmetricFocalLoss(
    focal_loss_criterion=base_criterion_clf,
    penalty_factor=2.5,
    true_label_idx=CLASS_MAP['b'],
    pred_label_idx=CLASS_MAP['af']
)
criterion_seg = BCEDiceLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

total_epochs = EPOCHS
warmup_epochs = max(1, int(0.05 * total_epochs))
def lr_lambda(epoch_idx):
    if epoch_idx < warmup_epochs:
        return float(epoch_idx + 1) / float(warmup_epochs)
    progress = (epoch_idx - warmup_epochs + 1) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# val_total 기준 조기 종료
early_stopping = EarlyStopping(
    patience=50, verbose=True, path=MODEL_SAVE_PATH,
    mode='min', monitor_name='val_total', delta=1e-3
)

# --- 8. 학습 루프 ---
print("--- 학습을 시작합니다 ---")
start_time = datetime.now()

history = {
    'train_total': [], 'val_total': [],
    'train_clf': [],   'val_clf': [],
    'train_seg': [],   'val_seg': [],
    'train_f1': [],    'val_f1': [],
    'train_acc': [],   'val_acc': [],
    'val_seg_pixel_acc': [], 'val_seg_dice': []
}

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

    tr_total, tr_clf, tr_seg, tr_f1, tr_acc = train_one_epoch(
        model, train_dataloader, criterion_clf, criterion_seg,
        optimizer, device, alpha=ALPHA, beta=BETA
    )

    scheduler.step()

    (va_total, va_clf, va_seg, va_f1, va_acc,
     _, _, _, va_seg_pixel_acc, va_seg_dice) = evaluate(
        model, valid_dataloader, criterion_clf, criterion_seg,
        device, alpha=ALPHA, beta=BETA
    )

    print(f"  - Train | total: {tr_total:.4f} | clf: {tr_clf:.4f} | seg: {tr_seg:.4f} | F1: {tr_f1:.4f} | Acc: {tr_acc:.4f}")
    print(f"  - Valid | total: {va_total:.4f} | clf: {va_clf:.4f} | seg: {va_seg:.4f} | "
          f"F1: {va_f1:.4f} | Acc: {va_acc:.4f} | Seg PixelAcc: {va_seg_pixel_acc:.4f} | Dice: {va_seg_dice:.4f}")

    history['train_total'].append(tr_total); history['val_total'].append(va_total)
    history['train_clf'].append(tr_clf);     history['val_clf'].append(va_clf)
    history['train_seg'].append(tr_seg);     history['val_seg'].append(va_seg)
    history['train_f1'].append(tr_f1);       history['val_f1'].append(va_f1)
    history['train_acc'].append(tr_acc);     history['val_acc'].append(va_acc)
    history['val_seg_pixel_acc'].append(va_seg_pixel_acc)
    history['val_seg_dice'].append(va_seg_dice)

    # val_total 손실 기준 EarlyStopping
    early_stopping(va_total, model, ref_metric_for_log=va_seg_pixel_acc, ref_name='seg_pixel_acc')
    if early_stopping.early_stop:
        print("\n🛑 Early stopping triggered.")
        break

end_time = datetime.now()
print(f"\n총 학습 시간: {end_time - start_time}")

# ✅ 최적 가중치 로드
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
print(f"✓ Best checkpoint loaded from '{MODEL_SAVE_PATH}'")

# -- 9. 학습 결과 시각화 (total/clf/seg 손실 + 분류 Acc/F1 + 세그 PixelAcc & Dice) --

# history 딕셔너리가 미리 생성되어있다고 가정
if 'history' not in locals():
    print("❌ [셀 5]를 먼저 실행하여 'history' 딕셔너리를 생성해주세요.")
else:
    import numpy as np
    import matplotlib.pyplot as plt

    # 실행된 에폭 수
    epochs_ran = len(history.get('train_total', []))
    if epochs_ran == 0:
        print("❌ history가 비어 있습니다. 학습을 먼저 수행하세요.")
    else:
        x = range(epochs_ran)

        # 베스트 에폭 (검증 총손실 기준)
        best_epoch = int(np.argmin(history['val_total']))

        # 보조 지표 존재 여부
        has_seg_pixel_acc = 'val_seg_pixel_acc' in history and len(history['val_seg_pixel_acc']) == epochs_ran
        has_seg_dice      = 'val_seg_dice' in history and len(history['val_seg_dice']) == epochs_ran

        # 3x2 레이아웃
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # 1) Total Loss
        ax = axes[0, 0]
        ax.plot(x, history['train_total'], label='Train Total', marker='o')
        ax.plot(x, history['val_total'],   label='Valid Total', marker='o')
        ax.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
        ax.set_title('Total Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.grid(True); ax.legend()

        # 2) Classification Loss
        ax = axes[0, 1]
        ax.plot(x, history['train_clf'], label='Train Clf', marker='o')
        ax.plot(x, history['val_clf'],   label='Valid Clf', marker='o')
        ax.set_title('Classification Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.grid(True); ax.legend()

        # 3) Segmentation Loss
        ax = axes[1, 0]
        ax.plot(x, history['train_seg'], label='Train Seg', marker='o')
        ax.plot(x, history['val_seg'],   label='Valid Seg', marker='o')
        ax.set_title('Segmentation Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.grid(True); ax.legend()

        # 4) Classification Accuracy & F1-Score
        ax = axes[1, 1]
        ax.plot(x, history['train_acc'], label='Train Acc', linestyle='-', marker='o')
        ax.plot(x, history['val_acc'],   label='Valid Acc', linestyle='--', marker='x')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.grid(True)
        ax2 = ax.twinx()
        ax2.plot(x, history['train_f1'], label='Train F1', linestyle='-', marker='s')
        ax2.plot(x, history['val_f1'],   label='Valid F1', linestyle='--', marker='^')
        ax.set_title('Classification: Accuracy & F1')
        ax.axvline(best_epoch, color='r', linestyle='--')
        # 범례 합치기
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines+lines2, labels+labels2, loc='lower right')

        # 5) Segmentation Pixel Accuracy (point-wise)
        ax = axes[2, 0]
        if has_seg_pixel_acc:
            ax.plot(x, history['val_seg_pixel_acc'], label='Valid Seg PixelAcc', marker='o')
            ax.set_title('Segmentation Pixel Accuracy (Valid)')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Pixel Accuracy')
            ax.grid(True); ax.legend()
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No seg pixel accuracy logged', ha='center', va='center')

        # 6) Segmentation Dice (reference)
        ax = axes[2, 1]
        if has_seg_dice:
            ax.plot(x, history['val_seg_dice'], label='Valid Seg Dice', marker='o')
            ax.set_title('Segmentation Dice (Valid) [Reference]')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Dice')
            ax.grid(True); ax.legend()
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No seg dice logged', ha='center', va='center')

        plt.tight_layout()
        plt.show()

# --- 10. 최종 모델 평가 ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 테스트 데이터셋/로더 (test만 사용)
test_dataset = PPGDataset_Resampling(TEST_PATH, TARGET_LENGTH, TRAIN_MEAN, TRAIN_STD)
num_workers = 2 if torch.cuda.is_available() else 0
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=bool(num_workers)
)

# 최종 모델 로드
final_model = UNet1D_Hybrid(
    n_channels=1,
    n_classes=len(CLASS_MAP),
    n_external_features=4
).to(device)

# 저장된 모델 경로가 올바른지 확인 후 로드
if os.path.exists(MODEL_SAVE_PATH):
    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"'{MODEL_SAVE_PATH}'에서 모델을 성공적으로 로드했습니다.")
else:
    raise FileNotFoundError(f"오류: '{MODEL_SAVE_PATH}'에서 모델을 찾을 수 없습니다. 경로를 확인해주세요.")

# --- Test 데이터셋으로 최종 성능 평가 ---
(test_loss, test_clf_loss, test_seg_loss, test_f1, test_acc,
 all_preds, all_labels, all_confidences, test_seg_pixel_acc, test_seg_dice) = evaluate(
    final_model,
    test_dataloader,
    criterion_clf,   # 학습 셀에서 정의된 동일 criterion 사용
    criterion_seg,
    device
)

# --- 최종 결과 출력 (분류/세그 분리) ---
print("\n--- 최종 성능 평가 결과 (Test Set / HRV 결합) ---")
print(f"  - Test Loss                 : {test_loss:.4f}")
print(f"  - Classification")
print(f"      · Accuracy              : {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"      · F1 (Macro)            : {test_f1:.4f}")
print(f"      · Classification Loss   : {test_clf_loss:.4f}")
print(f"  - Segmentation (1D)")
print(f"      · Pixel Accuracy        : {test_seg_pixel_acc:.4f}")
print(f"      · Dice (reference)      : {test_seg_dice:.4f}")
print(f"      · Segmentation Loss     : {test_seg_loss:.4f}")

# --- Confusion Matrix & Classification Report ---
labels_order = list(range(len(CLASS_MAP)))
target_names = [INV_CLASS_MAP[i] for i in labels_order]

cm = confusion_matrix(all_labels, all_preds, labels=labels_order)
print("\n[Classification Report]\n")
print(classification_report(all_labels, all_preds, labels=labels_order, target_names=target_names, digits=4, zero_division=0))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.show()

# -- 11. 최종 결과 분석 및 시각화 --
#   * 잘 맞춘 5개:  부정맥(종류 무관) 샘플 중 (분류 정답 AND PixelAcc >= THRESH_PIXEL_ACC)
#   * 못 맞춘 5개:  부정맥(종류 무관) 샘플 중 (분류 오답 OR  PixelAcc < THRESH_PIXEL_ACC)
#   * 색상: 위(파랑 신호 + 초록 GT), 아래(주황 신호 + 빨강 Pred)
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

THRESH_PIXEL_ACC = 0.90  # 세그멘테이션 정답 판단 임계값(포인트 단위 정확도), 필요 시 0.85~0.95 사이로 조정

if 'final_model' not in locals():
    print("❌ 이전 셀을 실행하여 'final_model'을 먼저 생성하고 가중치를 로드해주세요.")
else:
    OUTPUT_IMAGE_DIR = os.path.join(BASE_PROJECT_PATH, '.image')
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    print(f"📊 시각화 결과는 '{OUTPUT_IMAGE_DIR}' 폴더에 저장됩니다.")

    # --- 1) Test 전 구간 예측 (분류/세그) ---
    final_model.eval()
    all_labels, all_preds = [], []
    all_pixel_acc = []  # 샘플별 Pixel Accuracy 저장
    all_dice = []       # 참고용 Dice 저장

    with torch.no_grad():
        for signals, masks, labels, hrv_features in tqdm(test_dataloader, desc="Test set 예측 중"):
            signals = signals.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            hrv_features = hrv_features.to(device)

            clf_logits, seg_logits = final_model(signals, hrv_features)
            preds = torch.argmax(clf_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # --- 세그멘테이션 성능 (포인트 단위 정확도 + Dice[참고]) ---
            seg_probs = torch.sigmoid(seg_logits)           # (B,1,T)
            seg_bin = (seg_probs > 0.5).to(masks.dtype)     # (B,1,T) in {0,1}
            # batch별 pixel acc
            batch_pixel_acc = (seg_bin == masks).float().mean(dim=(1,2))
            all_pixel_acc.extend(batch_pixel_acc.cpu().numpy().tolist())

            # Dice (참고)
            intersect = (seg_bin * masks).sum(dim=(1,2))
            denom = seg_bin.sum(dim=(1,2)) + masks.sum(dim=(1,2)) + 1e-7
            batch_dice = (2.0 * intersect + 1e-7) / denom
            all_dice.extend(batch_dice.cpu().numpy().tolist())

    all_labels_np = np.array(all_labels)
    all_preds_np  = np.array(all_preds)
    all_pixel_acc_np = np.array(all_pixel_acc)
    all_dice_np   = np.array(all_dice)

    # --- 2) Confusion Matrix ---
    print("\n\n--- Confusion Matrix ---")
    label_order = list(range(len(CLASS_MAP)))
    display_labels = [INV_CLASS_MAP[i] for i in label_order] if 'INV_CLASS_MAP' in locals() else list(CLASS_MAP.keys())
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=label_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
    ax_cm.set_title("Confusion Matrix for Test Set")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_IMAGE_DIR, 'confusion_matrix_test.png')
    fig_cm.savefig(cm_path, dpi=200)
    plt.show()
    plt.close(fig_cm)
    print(f"✅ 혼동 행렬 저장: {cm_path}")

    # --- 3) 부정맥 샘플만 대상으로 선별 --------------------------
    normal_idx = CLASS_MAP['normal']
    is_arrhythmia = (all_labels_np != normal_idx)

    # 잘 맞춘 케이스: 부정맥 ∧ (분류 정답) ∧ (PixelAcc ≥ 임계값)
    good_idx = np.where(is_arrhythmia &
                        (all_labels_np == all_preds_np) &
                        (all_pixel_acc_np >= THRESH_PIXEL_ACC))[0]

    # 못 맞춘 케이스: 부정맥 ∧ (분류 오답 ∨ PixelAcc < 임계값)
    bad_idx  = np.where(is_arrhythmia &
                        ((all_labels_np != all_preds_np) | (all_pixel_acc_np < THRESH_PIXEL_ACC)))[0]

    # 무작위로 최대 5개씩 선택
    rng = np.random.default_rng(seed=42)
    sel_good = rng.choice(good_idx, size=min(5, len(good_idx)), replace=False) if len(good_idx) > 0 else []
    sel_bad  = rng.choice(bad_idx,  size=min(5, len(bad_idx)),  replace=False) if len(bad_idx)  > 0 else []

    # 역매핑 준비
    INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

    def _get_tuple_file(info):
        """(base_dir, filename) 혹은 filename(str) 모두 지원"""
        if isinstance(info, (list, tuple)) and len(info) == 2:
            return info[0], info[1]
        else:
            return "", str(info)

    def visualize_and_save(idx, tag):
        """단일 인덱스 시각화 및 저장"""
        signal_tensor, mask_tensor, label_tensor, hrv_features_tensor = test_dataset[idx]
        base_dir, file_name = _get_tuple_file(test_dataset.file_list[idx])

        # 파일명 파싱 (옵션)
        try:
            parts = file_name.replace('.npz', '').replace('.npy', '').split('_')
            patient_id = parts[0]
            window_id = parts[1].replace('win', '') if len(parts) > 1 else "N-A"
        except Exception:
            patient_id, window_id = "N-A", "N-A"

        true_class_idx = int(all_labels_np[idx])
        pred_class_idx = int(all_preds_np[idx])

        true_label_name = INV_CLASS_MAP.get(true_class_idx, str(true_class_idx))
        pred_class_name = INV_CLASS_MAP.get(pred_class_idx, str(pred_class_idx))

        # 세그 예측(단일 샘플)
        with torch.no_grad():
            _, pred_seg_logits = final_model(
                signal_tensor.unsqueeze(0).to(device),
                hrv_features_tensor.unsqueeze(0).to(device)
            )
        pred_probs = torch.sigmoid(pred_seg_logits)
        pred_mask = (pred_probs > 0.5).int().squeeze().cpu().numpy()

        # normal로 분류되면 시각화 편의를 위해 마스크 표시 끔
        if pred_class_name == 'normal':
            pred_mask = np.zeros_like(pred_mask)

        processed_signal = signal_tensor.squeeze().numpy()
        true_mask = mask_tensor.squeeze().numpy()

        # 샘플별 Pixel Accuracy / Dice(참고) 계산
        pixel_acc_i = (pred_mask == true_mask).mean().item()
        intersect_i = (pred_mask * true_mask).sum()
        denom_i = pred_mask.sum() + true_mask.sum() + 1e-7
        dice_i = (2.0 * intersect_i + 1e-7) / denom_i

        # --- 시각화 ---
        fig, axes = plt.subplots(2, 1, figsize=(6, 4), dpi=150)

        # 상단: GT
        ax = axes[0]
        title_text = (f'Patient {patient_id}, Window {window_id} | '
                      f'Real Class: {true_label_name.upper()}')
        ax.set_title(title_text)
        ax.plot(processed_signal, color='blue', label=f'Input Signal (Resampled to {TARGET_LENGTH})')
        y_min, y_max = ax.get_ylim()
        ax.fill_between(range(TARGET_LENGTH), y_min, y_max, where=true_mask > 0,
                        color='green', alpha=0.3, label='True Mask (Resampled)')
        ax.legend(); ax.grid(True); ax.set_ylim(y_min, y_max)

        # 하단: Pred
        ax = axes[1]
        ax.set_title(f"Model Prediction | Pred Class: {pred_class_name.upper()} | "
                     f"PixelAcc: {pixel_acc_i:.3f} | Dice: {dice_i:.3f}")
        ax.plot(processed_signal, color='darkorange', label=f'Input Signal (Resampled to {TARGET_LENGTH})')
        y_min, y_max = ax.get_ylim()
        ax.fill_between(range(TARGET_LENGTH), y_min, y_max, where=pred_mask > 0,
                        color='red', alpha=0.3, label='Predicted Mask')
        ax.legend(); ax.grid(True); ax.set_ylim(y_min, y_max)

        plt.tight_layout()

        # 저장 → 표시 → close
        save_name = (f"{tag}_{idx:05d}_pid-{patient_id}_win-{window_id}"
                     f"_gt-{true_label_name}_pred-{pred_class_name}"
                     f"_pxacc-{pixel_acc_i:.3f}_dice-{dice_i:.3f}.png")
        save_path = os.path.join(OUTPUT_IMAGE_DIR, save_name)
        fig.savefig(save_path, dpi=200)
        plt.show()
        plt.close(fig)
        print(f"🖼️ 저장: {save_path}")

    # --- 4) 시각화 실행 ----------------------------
    if len(sel_good) == 0:
        print(f"ℹ️ 기준(부정맥 ∧ 분류 정답 ∧ PixelAcc ≥ {THRESH_PIXEL_ACC:.2f})을 만족하는 샘플이 없습니다.")
    else:
        print(f"\n✅ 잘 맞춘 부정맥 샘플 {len(sel_good)}개 시각화")
        for i in sel_good:
            visualize_and_save(int(i), tag="correct")

    if len(sel_bad) == 0:
        print("🎉 못 맞춘 부정맥 샘플이 없습니다.")
    else:
        print(f"\n❌ 못 맞춘 부정맥 샘플 {len(sel_bad)}개 시각화")
        for i in sel_bad:
            visualize_and_save(int(i), tag="incorrect")