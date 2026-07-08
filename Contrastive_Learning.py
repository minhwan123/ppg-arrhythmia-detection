# -*- coding: utf-8 -*-
# CLIP 스타일 PPG-임상정보-BioBERT 텍스트 대조학습 사전훈련 스크립트 (Colab 환경 기준)

!pip install transformers

import os
import pandas as pd
import numpy as np
import torch
from google.colab import drive



# 1. 구글 드라이브 마운트 및 경로 설정
drive.mount('/content/drive')
target_folder_path = '/content/drive/MyDrive/종합설계/PPG_AD_Dataset'

if os.path.exists(target_folder_path):
    os.chdir(target_folder_path)
    print(f"작업 디렉토리 변경 완료: {os.getcwd()}")
else:
    print("경로를 찾을 수 없습니다. 마운트 상태를 확인하세요.")

# 체크포인트 저장 경로 설정 및 폴더 생성
# 현재 작업 디렉토리가 'PPG_AD_Dataset'이므로 상대 경로 'checkpoints' 사용
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)
print(f"체크포인트 저장 폴더 생성 완료: {os.path.abspath(save_dir)}")

# 2. 피처(Feature) 정의 (최종 버전 기준)
# 연속형 변수 (정규화 대상)
CONTINUOUS_COLS = [
    "age", "bmi", "opdur", "preop_na", "preop_bun", "preop_cr",
    "preop_k", "intraop_eph", "intraop_phe"
]
# 범주형 변수 (One-hot Encoding 대상)
CATEGORICAL_COLS = {"sex": ["M", "F"], "emop": ["N", "Y"]}
BINARY_STR_COLS = {"preop_dm": ["N", "Y"], "preop_htn": ["N", "Y"]}

# 3. CSV 데이터 로드 및 전처리 함수 통합
def preprocess_clinical_data(csv_path, is_train=True):
    """CSV를 로드하고 결측치를 처리하여 반환하는 함수"""
    df = pd.read_csv(csv_path)

    # 불필요 컬럼 제거 (Train/Valid 공통 적용)
    drop_cols = ['preop_ph', 'preop_hco3', 'preop_pao2', 'preop_paco2']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 결측치 처리
    all_features = CONTINUOUS_COLS + list(CATEGORICAL_COLS.keys()) + list(BINARY_STR_COLS.keys())
    for col in all_features:
        if col in df.columns:
            if col in CONTINUOUS_COLS:
                # 연속형: 평균 대치
                df[col] = df[col].fillna(df[col].mean())
            else:
                # 범주형: 최빈값 대치
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
    return df

# 데이터 로드
clinical_data = preprocess_clinical_data('train.csv')
normal_clinical_data = preprocess_clinical_data('train_normal.csv')

print("임상 데이터 전처리 완료.")

# 파일 매칭을 위한 헬퍼 함수
def match_files_with_clinical(root_dirs, clinical_dfs, file_exts, exclude_keywords=[]):
    """
    폴더를 순회하며 파일과 임상 데이터를 매칭합니다.
    root_dirs: 탐색할 폴더 리스트 (예: ['.'])
    clinical_dfs: 임상 데이터프레임 리스트
    file_exts: 찾을 확장자 (예: ['.npy', '.npz'])
    """
    # 1. 임상 데이터 딕셔너리 생성 (caseid 기준)
    combined_clinical = {}
    for df in clinical_dfs:
        combined_clinical.update(df.set_index('caseid').to_dict('index'))

    matched_pairs = []

    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # 제외 키워드 확인 (예: valid, test 폴더 제외)
                path_parts = os.path.join(root, file).lower().split(os.sep)
                if any(k in path_parts for k in exclude_keywords):
                    continue

                if any(file.endswith(ext) for ext in file_exts):
                    file_path = os.path.join(root, file)
                    # 파일명에서 Patient ID 추출 (파일명 규칙: ID_어쩌구.ext)
                    try:
                        patient_id = int(os.path.basename(file).split('_')[0])
                        if patient_id in combined_clinical:
                            matched_pairs.append((combined_clinical[patient_id], file_path, patient_id))
                    except ValueError:
                        continue
    return matched_pairs

# 학습 데이터 매칭 실행 (Validation 제외)
# 현재 경로('.')에서 탐색하되 'val', 'test'가 들어간 경로는 제외
train_pairs = match_files_with_clinical(
    root_dirs=['.'],
    clinical_dfs=[clinical_data, normal_clinical_data],
    file_exts=['.npy', '.npz'],
    exclude_keywords=['val', 'test']
)

print(f"학습용 매칭 데이터 수: {len(train_pairs)}")

import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# --- [전역 변수] 스케일러 초기화 ---
CONT_MEAN = {c: 0.0 for c in CONTINUOUS_COLS}
CONT_STD  = {c: 1.0 for c in CONTINUOUS_COLS}

def fit_clinical_scaler(pairs):
    """매칭된 쌍에서 연속형 변수의 평균과 표준편차를 계산하여 전역 변수 업데이트"""
    # 임상 정보만 추출하여 DataFrame 생성
    data_list = [p[0] for p in pairs]
    df = pd.DataFrame(data_list)

    for c in CONTINUOUS_COLS:
        vals = pd.to_numeric(df[c], errors='coerce').dropna()
        if not vals.empty:
            CONT_MEAN[c] = vals.mean()
            CONT_STD[c] = vals.std(ddof=0) if vals.std(ddof=0) > 0 else 1.0

# 스케일러 학습 실행
fit_clinical_scaler(train_pairs)
print("Clinical Scaler Fitted.")

# --- [유틸리티] 데이터 로딩 및 상태 추론 ---
STATE_LIST = ["Normal", "AF", "B", "T"] # 상태 정의
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LIST)}

def load_npz_sample(path):
    """npz 파일에서 파형, 마스크, 상태 추출"""
    d = np.load(path, allow_pickle=True)
    wave = d["signal"].astype(np.float32)
    mask = d["mask"].astype(np.int32)
    # 상태 추론 로직 (meta -> class majority 순)
    state_str = "Unknown"
    if d["meta"].size > 0 and isinstance(d["meta"][0], dict):
        state_str = str(d["meta"][0].get('arr_type', 'Unknown'))
    if state_str == "Unknown": # 마스크 구간의 class majority로 추론
        masked_class = d["class"][mask.astype(bool)]
        if masked_class.size > 0:
            vals, counts = np.unique(masked_class, return_counts=True)
            mj = vals[np.argmax(counts)]
            # 매핑 테이블 (0,1: Normal, 2: AF, 3: B, 4: T)
            mapper = {0:"Normal", 1:"Normal", 2:"AF", 3:"B", 4:"T"}
            state_str = mapper.get(mj, str(mj))
    return wave, mask, state_str

def augment_ppg(wave):
    """데이터 증강: 노이즈 추가 및 스케일링"""
    if np.random.rand() < 0.5: wave += np.random.normal(0, 0.01, wave.shape)
    if np.random.rand() < 0.5: wave *= np.random.uniform(0.9, 1.1)
    return wave.astype(np.float32)

def clinical_to_vector(clin_dict):
    """임상 딕셔너리를 정규화된 벡터로 변환"""
    vec = []
    # 1. Continuous (Normalize)
    for c in CONTINUOUS_COLS:
        v = float(clin_dict.get(c, CONT_MEAN[c]))
        vec.append((v - CONT_MEAN[c]) / (CONT_STD[c] + 1e-6))
    # 2. Categorical / Binary (One-hot)
    for col_map in [CATEGORICAL_COLS, BINARY_STR_COLS]:
        for col, cats in col_map.items():
            val = str(clin_dict.get(col, '')).upper()
            oh = [1.0 if val == str(c).upper() else 0.0 for c in cats]
            vec.extend(oh)
    return np.array(vec, dtype=np.float32)

from transformers import AutoTokenizer

# --- 1. 진단명 -> 의학적 텍스트 매핑 ---
# 클래스별 문장 구조와 핵심 단어를 다르게 하여 텍스트 임베딩 상에서 서로 멀어지도록 구성

LABEL_TEXT_MAP = {
    # Normal: 정상 동율동
    "Normal": (
        "Normal Sinus Rhythm (NSR). Heart rate is within normal range (60-100 bpm). "
        "Regular and consistent beats with stable R-R intervals."
    ),

    # AF: 속도보다 '불규칙성(Chaos)'을 핵심 특징으로 강조
    "AF": (
        "Atrial Fibrillation. The hallmark is an 'irregularly irregular' rhythm. "
        "P waves are completely absent and replaced by chaotic fibrillatory waves. "
        "R-R intervals vary unpredictably."
    ),

    # B: 느린 심박수
    "B": (
        "Sinus Bradycardia. Heart rate is strictly less than 60 bpm. "
        "Rhythm is regular but significantly slow. Long R-R intervals."
    ),

    # T: '빠름'과 '규칙적임(Regular)'을 함께 강조해 AF와 구분
    "T": (
        "Sinus Tachycardia. Heart rate exceeds 100 bpm but the rhythm remains regular. "
        "Fast, steady beating with consistent but shortened R-R intervals. "
        "Distinct from chaotic rhythms."
    )
}

# BioBERT 토크나이저 로드
print("BioBERT Tokenizer 로드 중...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# --- Dataset 클래스 ---
class WaveClinTextDataset(Dataset):
    def __init__(self, pairs, target_len=286, is_train=False):
        self.pairs = pairs
        self.target_len = target_len
        self.is_train = is_train

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        clin, path, caseid = self.pairs[idx]

        # 1. 파형 로드
        try:
            if path.endswith(".npz"):
                wave, mask, state = load_npz_sample(path)
            else: # .npy (Normal)
                wave = np.load(path).astype(np.float32)
                mask = np.zeros_like(wave, dtype=np.int32)
                state = "Normal"
        except:
            return self.__getitem__((idx + 1) % len(self))

        if state not in STATE_TO_IDX: return self.__getitem__((idx + 1) % len(self))

        # 2. Resampling
        wave_tensor = torch.from_numpy(wave).unsqueeze(0).unsqueeze(0)
        original_len = wave.shape[-1]

        if original_len > self.target_len:
            seg_tensor = F.interpolate(wave_tensor, size=self.target_len, mode='linear', align_corners=True).squeeze()
            seg = seg_tensor.numpy()
        else:
            seg = wave
            if len(seg) < self.target_len:
                seg = np.pad(seg, (0, self.target_len - len(seg)), mode='constant')

        # 3. 정규화 및 증강
        seg = seg.astype(np.float32)
        if self.is_train: seg = augment_ppg(seg)
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)

        # 4. 텍스트 토큰화 (BioBERT 입력용)
        # 해당 라벨(state)에 맞는 의학적 문장을 가져옵니다.
        text_desc = LABEL_TEXT_MAP.get(state, "Unknown heart rhythm condition.")

        encoding = tokenizer(
            text_desc,
            padding='max_length',
            max_length=32,  # 문장이 짧으므로 32 토큰이면 충분
            truncation=True,
            return_tensors='pt'
        )

        return {
            "wave": torch.from_numpy(seg).unsqueeze(0), # (1, T)
            "clin": torch.from_numpy(clinical_to_vector(clin)),
            "caseid": torch.tensor(caseid, dtype=torch.long),
            "state": state,
            # BERT 입력값 추가
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0)
        }

# DataLoader 생성 (새로운 Dataset 클래스 적용)
TARGET_SEG_LEN = 286
train_ds = WaveClinTextDataset(train_pairs, target_len=TARGET_SEG_LEN, is_train=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
print(f"Train Loader 준비 완료. 배차 수: {len(train_loader)}")

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ResidualBlock(nn.Module):
    """ResNet의 기본 블록: Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 입력과 출력의 차원(채널 수)이 다르거나 stride가 1이 아닐 때 차원을 맞춰주는 1x1 Conv
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 여기가 핵심: Skip Connection
        out = self.relu(out)
        return out

class ResNet1DEncoder(nn.Module):
    """성능이 강화된 ResNet 기반 1D Encoder"""
    def __init__(self, in_channels=1, base_filters=32, out_dim=128):
        super().__init__()

        # 1. 초기 진입 레이어 (Receptive Field 확보)
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Residual Blocks 쌓기 (채널 수를 늘려가며 깊게)
        # 예: 32 -> 64 -> 128 -> 256 채널로 확장
        self.layer1 = ResidualBlock(base_filters, base_filters*2, stride=2)    # 32 -> 64
        self.layer2 = ResidualBlock(base_filters*2, base_filters*4, stride=2)  # 64 -> 128
        self.layer3 = ResidualBlock(base_filters*4, base_filters*8, stride=2)  # 128 -> 256

        # 3. Global Average Pooling & Final Projection
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # 최종 임베딩 차원으로 변환
        self.fc = nn.Linear(base_filters*8, out_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)

        return F.normalize(x, dim=-1) # 코사인 유사도 학습을 위해 정규화 유지


# --- BioBERT 텍스트 인코더 ---
class BioBERTTextEncoder(nn.Module):
    def __init__(self, out_dim=128, freeze_bert=True):
        super().__init__()
        print("BioBERT 모델 로드 중... (최초 1회 다운로드 시간이 소요됩니다)")
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # 학습 효율을 위해 BERT의 무거운 가중치는 고정(Freeze)하고 마지막 fc만 학습
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # BERT 출력(768) -> 공통 임베딩 공간(128)
        self.fc = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # BERT 통과
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] 토큰 (문장 전체의 의미를 담은 벡터) 추출
        cls_token = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_token)
        x = self.fc(x)
        return F.normalize(x, dim=-1)

class ClinicalMLP(nn.Module):
    """임상 데이터를 입력받아 특징 벡터로 변환하는 MLP"""
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class Projection(nn.Module):
    """인코더 출력을 공통 잠재 공간(Latent Space)으로 투영"""
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, proj_dim)
    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)


# --- 모델 초기화 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
D_clin_in = train_ds[0]['clin'].shape[0]

# 1. PPG Encoder
wave_enc = ResNet1DEncoder(in_channels=1, base_filters=32, out_dim=128).to(device)
# 2. Clinical Encoder
clin_enc = ClinicalMLP(in_dim=D_clin_in).to(device)
# 3. Text Encoder (BioBERT)
text_enc = BioBERTTextEncoder(out_dim=128, freeze_bert=True).to(device)

# Projections
proj_w = Projection(128).to(device)
proj_c = Projection(128).to(device)
# Text Encoder는 내부에 이미 Linear Layer가 있으므로 별도 proj_t 불필요

# Optimizer (text_enc 파라미터 포함)
optimizer = torch.optim.Adam(
    list(wave_enc.parameters()) + list(clin_enc.parameters()) +
    list(proj_w.parameters()) + list(proj_c.parameters()) + list(text_enc.parameters()),
    lr=1e-3
)

def masked_info_nce(sim_matrix, same_case_mask, tau=0.07):
    """
    CLIP Loss의 변형. 같은 환자의 데이터는 Negative Sample에서 제외합니다.
    sim_matrix: (Batch, Batch) 코사인 유사도 행렬
    same_case_mask: (Batch, Batch) 같은 환자 여부 (True/False)
    """
    # 대각선(자기 자신)은 Positive
    pos_sim = torch.diag(sim_matrix) / tau

    # Negative Mask: 자기 자신 제외(eye)하고, 같은 환자도 제외(same_case_mask)
    B = sim_matrix.size(0)
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device) & ~same_case_mask

    # Loss 계산 (LogSumExp Trick 대신 간단한 형태)
    nll_list = []
    for i in range(B):
        pos_exp = torch.exp(pos_sim[i])
        neg_exps = torch.exp(sim_matrix[i][neg_mask[i]] / tau)
        nll = -torch.log(pos_exp / (pos_exp + neg_exps.sum() + 1e-8))
        nll_list.append(nll)

    return torch.stack(nll_list).mean()

# =============================================================================
# [1단계] Validation 데이터 준비 (학습 루프 실행 전에 수행해야 함)
# =============================================================================

print("Validation 데이터 준비 중...")

# 1. Validation 데이터 로드
valid_clinical = preprocess_clinical_data('valid.csv', is_train=False)
valid_normal_clinical = preprocess_clinical_data('valid_normal.csv', is_train=False)

# 2. Validation 파일 매칭
# (Train과 달리 'val' 폴더 등을 제외하면 안 되므로 exclude_keywords를 비웁니다)
valid_pairs = match_files_with_clinical(
    root_dirs=['.'],
    clinical_dfs=[valid_clinical, valid_normal_clinical],
    file_exts=['.npy', '.npz'],
    exclude_keywords=[]
)

# 3. Validation DataLoader 생성
valid_ds = WaveClinTextDataset(valid_pairs, target_len=TARGET_SEG_LEN, is_train=False)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=2)

print(f"Validation Loader 준비 완료. 배치 수: {len(valid_loader)}")

# 4. Validation 평가 함수 정의
def validate(loader, wave_enc, clin_enc, text_enc, proj_w, proj_c):
    """검증 세트에 대한 Loss 계산"""
    wave_enc.eval(); clin_enc.eval(); text_enc.eval(); proj_w.eval(); proj_c.eval()

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            # 데이터 로드
            wave = batch["wave"].to(device)
            clin = batch["clin"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            caseids = batch["caseid"].to(device)

            # 라벨 인덱스 변환 (Mask 생성용)
            state_indices = torch.tensor([STATE_TO_IDX[s] for s in batch["state"]], device=device)

            # Forward
            z_w = proj_w(wave_enc(wave))
            z_c = proj_c(clin_enc(clin))
            z_t = text_enc(input_ids, attention_mask)

            # Loss 1: Wave <-> Clinical (환자 매칭)
            sim_wc = torch.matmul(z_w, z_c.T)
            same_case_mask = (caseids[:, None] == caseids[None, :])
            loss_clin = masked_info_nce(sim_wc, same_case_mask)

            # Loss 2: Wave <-> Text (진단명 매칭 - BioBERT)
            sim_wt = torch.matmul(z_w, z_t.T)
            same_state_mask = (state_indices[:, None] == state_indices[None, :])
            loss_text = masked_info_nce(sim_wt, same_state_mask, tau=0.07)

            # Weighted Sum
            loss = (0.2 * loss_clin) + (2.0 * loss_text)

            total_loss += loss.item()
            steps += 1

    return total_loss / steps

# =============================================================================
# [2단계] 학습 루프 (Training Loop)
# =============================================================================
import os

NUM_EPOCHS = 100
train_losses = []
val_losses = []
best_val_loss = float('inf') # 초기값 무한대

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"학습 시작 (BioBERT 기반)...")

# 함수 전달용 모델 리스트


for epoch in range(NUM_EPOCHS):
    # Train Mode
    wave_enc.train(); clin_enc.train(); proj_w.train(); proj_c.train(); text_enc.train()
    running_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        # 데이터 로드
        wave = batch["wave"].to(device)
        clin = batch["clin"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        caseids = batch["caseid"].to(device)

        # 라벨 인덱스 (Loss 계산 마스크용)
        state_indices = torch.tensor([STATE_TO_IDX[s] for s in batch["state"]], device=device)

        # Forward Pass
        z_w = proj_w(wave_enc(wave))                # PPG 임베딩
        z_c = proj_c(clin_enc(clin))                # 임상정보 임베딩
        z_t = text_enc(input_ids, attention_mask)   # BioBERT 텍스트 임베딩

        # 1. Wave <-> Clinical Loss
        sim_wc = torch.matmul(z_w, z_c.T)
        same_case_mask = (caseids[:, None] == caseids[None, :])
        loss_clin = masked_info_nce(sim_wc, same_case_mask)

        # 2. Wave <-> Text Loss (BioBERT)
        # 같은 'state'를 가진 샘플은 같은 '텍스트'를 가지므로 Positive 관계
        sim_wt = torch.matmul(z_w, z_t.T)
        same_state_mask = (state_indices[:, None] == state_indices[None, :])
        loss_text = masked_info_nce(sim_wt, same_state_mask, tau=0.07)

        # Total Loss
        loss = (0.2 * loss_clin) + (2.0 * loss_text)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # --- Epoch 종료 후 처리 ---
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation 수행 (매 에폭마다)
    avg_val_loss = validate(valid_loader, wave_enc, clin_enc, text_enc, proj_w, proj_c)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Best Model 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'wave_enc': wave_enc.state_dict(),
            'clin_enc': clin_enc.state_dict(),
            'text_enc': text_enc.state_dict(), # BioBERT 인코더 저장
            'proj_w': proj_w.state_dict(),
            'proj_c': proj_c.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': best_val_loss,
        }, os.path.join(save_dir, 'clip_biobert_hyh_xai.pth'))
        print(f"    --> Best Model Saved! (Loss: {best_val_loss:.4f})")

print("모든 학습 완료.")
# =============================================================================
# [3단계] 결과 시각화
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import torch

# 1. Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 2. t-SNE (Best Model 로드 후 시각화)
print("Best Model 로드 중...")
checkpoint = torch.load(os.path.join(save_dir, 'clip_biobert_hyh_xai.pth'))
wave_enc.load_state_dict(checkpoint['wave_enc'])
text_enc.load_state_dict(checkpoint['text_enc'])
proj_w.load_state_dict(checkpoint['proj_w'])
wave_enc.eval(); proj_w.eval(); text_enc.eval()

all_wave_embeds, all_labels = [], []

# (시간 절약을 위해 최대 1000개 정도만 샘플링해서 시각화)
sample_limit = 1000
current_samples = 0

with torch.no_grad():
        for batch in valid_loader:
            wave = batch['wave'].to(device)
            # PPG Embedding (Projection 통과)
            z_w = proj_w(wave_enc(wave))

            all_wave_embeds.append(z_w.cpu().numpy())
            all_labels.extend(batch['state'])

            current_samples += wave.size(0)
            if current_samples >= sample_limit:
                break

all_wave_embeds = np.concatenate(all_wave_embeds)[:sample_limit]
all_labels = np.array(all_labels)[:sample_limit]
# --- 3. [핵심] 4가지 진단 텍스트(Anchor) 임베딩 추출 ---
# 모델이 학습한 '기준점'이 어디인지 확인합니다.
unique_states = ["Normal", "AF", "B", "T"]
text_anchor_embeds = []

with torch.no_grad():
    for s in unique_states:
        desc = LABEL_TEXT_MAP[s]
        inputs = tokenizer(desc, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        # Text Embedding
        z_t = text_enc(input_ids, mask)
        text_anchor_embeds.append(z_t.cpu().numpy())

text_anchor_embeds = np.concatenate(text_anchor_embeds)

# --- 4. t-SNE 차원 축소 (파형 + 텍스트 함께 수행) ---
    # 파형과 텍스트가 같은 공간에 있는지 보려면 같이 t-SNE를 돌려야 함
combined_features = np.vstack([all_wave_embeds, text_anchor_embeds])

print(f"t-SNE 계산 중... (데이터 개수: {len(combined_features)})")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
z_tsne = tsne.fit_transform(combined_features)

# 다시 분리
z_wave_tsne = z_tsne[:len(all_wave_embeds)]
z_text_tsne = z_tsne[len(all_wave_embeds):]

# --- 5. 시각화 ---
plt.figure(figsize=(12, 10))

# 1) PPG 파형 점 찍기
sns.scatterplot(
    x=z_wave_tsne[:,0], y=z_wave_tsne[:,1],
    hue=all_labels,
    palette='viridis',
    alpha=0.6,
    s=60  # 점 크기
)

# 2) 텍스트 앵커(기준점) 찍기 (별표)
# 각 클래스의 텍스트가 해당 클래스의 파형들 중심에 위치하는지 확인
# 마커 스타일: X, 크기: 300, 검은 테두리
for i, label in enumerate(unique_states):
    plt.scatter(
        z_text_tsne[i, 0], z_text_tsne[i, 1],
        marker='*',
        s=500,
        color='red',
        edgecolors='black',
        linewidth=1.5,
        label=f"Text Anchor: {label}"
    )
    # 텍스트 라벨 표시
    plt.text(
        z_text_tsne[i, 0], z_text_tsne[i, 1]+0.5,
        label,
        fontsize=12,
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

plt.title("BioBERT-CLIP: Alignment of PPG Signals and Medical Texts", fontsize=15)
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# Knowledge Base 구축 (Retrieval-Augmented Generation 용)
# =============================================================================

def create_and_save_knowledge_base(dataloader, device, save_path='checkpoints/knowledge_base.pt', model=None):
    print(f"\n[Knowledge Base] 학습 데이터 데이터베이스 구축 시작...")

    # model이 None이 아닐 때만 eval() 호출 (안전 장치)
    if model is not None:
        model.eval()

    # 개별 인코더들을 평가 모드로 전환 (이것들이 실제 모델입니다)
    wave_enc.eval()
    clin_enc.eval()
    proj_w.eval()
    proj_c.eval()

    kb_data = {
        "clinical_vectors": [],  # 검색용: 임상 정보 임베딩
        "wave_vectors": [],      # 검색용: 파형 정보 임베딩
        "raw_clinical": [],      # LLM 참고용: 실제 임상 수치
        "labels": [],            # LLM 참고용: 실제 진단명
        "caseids": []            # 참조용: 환자 ID
    }

    with torch.no_grad():
        for batch in dataloader: # tqdm이 없으면 그냥 dataloader 사용
            # 데이터 로드
            wave = batch["wave"].to(device)
            clin = batch["clin"].to(device)
            caseids = batch["caseid"]
            states = batch["state"]

            # 임베딩 추출 (Projection 된 최종 벡터)
            z_w = proj_w(wave_enc(wave))
            z_c = proj_c(clin_enc(clin))

            # CPU로 이동하여 저장
            kb_data["clinical_vectors"].append(z_c.cpu())
            kb_data["wave_vectors"].append(z_w.cpu())
            kb_data["labels"].extend(states)
            kb_data["caseids"].extend(caseids.tolist())

            # Raw Clinical Data (여기서는 전처리된 tensor를 저장)
            kb_data["raw_clinical"].append(clin.cpu())

    # 리스트 -> 텐서 변환
    kb_data["clinical_vectors"] = torch.cat(kb_data["clinical_vectors"])
    kb_data["wave_vectors"] = torch.cat(kb_data["wave_vectors"])
    kb_data["raw_clinical"] = torch.cat(kb_data["raw_clinical"])

    # 저장
    torch.save(kb_data, save_path)
    print(f"✅ Knowledge Base 저장 완료: {save_path}")
    print(f"   - 총 데이터 수: {len(kb_data['labels'])}")
    print(f"   - 벡터 차원: {kb_data['clinical_vectors'].shape}")

# 실행 (model 인자는 생략하거나 None으로 둬도 동작함)
create_and_save_knowledge_base(
    dataloader=train_loader,
    device=device,
    save_path='checkpoints/knowledge_base.pt'
)