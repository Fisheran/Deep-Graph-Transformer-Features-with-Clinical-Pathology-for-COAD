import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= ⚡ 配置区域 =================
CLINICAL_JSON_PATH = "/home/student2025/shirx2025/MUSK-surv/TCGA_DATA/gdc-client/COAD/clinical.cart.COAD.json"
DATA_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"

# 输出路径
OUTPUT_DIR = "outputs_clinical_only_jitter" 
MODEL_WEIGHT_PATH = os.path.join(OUTPUT_DIR, "best_clinical_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
EPOCHS = 60      
LR = 1e-3
BATCH_SIZE = 32

try: plt.style.use('seaborn-v0_8-darkgrid')
except: plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===============================================

class ClinicalUtils:
    @staticmethod
    def norm(s): return str(s).strip().lower() if s else ""
    @staticmethod
    def to_f(val): 
        try: return float(val) 
        except: return np.nan
    @staticmethod
    def stage_to_num(s):
        s = ClinicalUtils.norm(s)
        if "stage iv" in s: return 4.0
        if "stage iii" in s: return 3.0
        if "stage ii" in s: return 2.0
        if "stage i" in s: return 1.0
        return 0.0 
    @staticmethod
    def t_to_num(s): 
        s = ClinicalUtils.norm(s)
        if "t4" in s: return 4.0
        if "t3" in s: return 3.0
        if "t2" in s: return 2.0
        if "t1" in s: return 1.0
        return 0.0
    @staticmethod
    def n_to_num(s): 
        s = ClinicalUtils.norm(s)
        if "n3" in s: return 3.0
        if "n2" in s: return 2.0
        if "n1" in s: return 1.0
        return 0.0
    @staticmethod
    def m_to_num(s): 
        s = ClinicalUtils.norm(s)
        if "m1" in s: return 1.0
        return 0.0

def load_clinical_features(json_path):
    print(f"🔄 读取临床数据: {json_path}")
    clinical_map = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            if 'data' in data and isinstance(data['data'], list): data = data['data']
            else: data = []
        
        for case in data:
            pid = case.get("case_submitter_id") or case.get("submitter_id") or ""
            pid = pid[:12]
            if not pid: continue
            
            diagnoses = case.get("diagnoses", [])
            candidates = diagnoses if diagnoses else [case]
            
            final_stage, final_t, final_n, final_m = 0.0, 0.0, 0.0, 0.0
            for item in candidates:
                s = ClinicalUtils.stage_to_num(item.get("ajcc_pathologic_stage"))
                if s > final_stage:
                    final_stage = s
                    final_t = ClinicalUtils.t_to_num(item.get("ajcc_pathologic_t"))
                    final_n = ClinicalUtils.n_to_num(item.get("ajcc_pathologic_n"))
                    final_m = ClinicalUtils.m_to_num(item.get("ajcc_pathologic_m"))
            
            demo = case.get("demographic", {}) or {}
            age = ClinicalUtils.to_f(demo.get("age_at_diagnosis"))
            age = (age/365.25/100) if not np.isnan(age) else 0.6
            gender = 1.0 if "male" in ClinicalUtils.norm(demo.get("gender")) else 0.0
            
            new_vec = np.array([age, gender, final_stage, final_t, final_n, final_m], dtype=np.float32)
            
            if pid in clinical_map:
                if final_stage > 0 and clinical_map[pid][2] == 0:
                    clinical_map[pid] = new_vec
            else:
                clinical_map[pid] = new_vec
                
    except Exception as e: print(f"Error: {e}")
    return clinical_map

class ClinicalOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 增加一点网络复杂度，看能不能学到非线性关系
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, 1)

    def forward(self, clin, return_embedding=False):
        feat = self.net(clin)
        if return_embedding:
            return feat
        return self.head(feat)

def cox_loss(risk, t, e):
    if e.sum() == 0: return torch.tensor(0.0, requires_grad=True).to(risk.device)
    idx = t.argsort(descending=True)
    risk, e = risk[idx], e[idx]
    return -((risk - torch.log(torch.cumsum(torch.exp(risk), 0))) * e).sum() / (e.sum()+1e-6)

# --- 改进的可视化函数：加入 Jitter 防止点重叠 ---
def generate_visualization(model, dataloader, raw_clin_np, device, output_path):
    print("\n🎨 生成全量数据可视化 (应用 Jitter 技术)...")
    model.eval()
    all_risks, all_events, all_times, all_embeddings = [], [], [], []
    
    with torch.no_grad():
        for clin, t, e in dataloader:
            clin = clin.to(device)
            risk = model(clin, return_embedding=False)
            emb = model(clin, return_embedding=True)
            
            all_risks.extend(risk.cpu().numpy().flatten())
            all_events.extend(e.numpy().flatten())
            all_times.extend(t.numpy().flatten())
            all_embeddings.extend(emb.cpu().numpy())
            
    all_risks = np.array(all_risks)
    all_events = np.array(all_events)
    all_times = np.array(all_times)
    all_embeddings = np.array(all_embeddings)
    all_stages = raw_clin_np[:, 2] 
    
    print(f"📊 绘图总样本数: {len(all_risks)}")
    
    # KM Curve
    median_risk = np.median(all_risks)
    high_risk = all_risks >= median_risk
    low_risk = all_risks < median_risk
    plt.figure(figsize=(10, 6))
    kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
    kmf_h.fit(all_times[high_risk], all_events[high_risk], label='High Risk')
    kmf_l.fit(all_times[low_risk], all_events[low_risk], label='Low Risk')
    kmf_h.plot_survival_function(color='#e74c3c', ci_show=True)
    kmf_l.plot_survival_function(color='#2ecc71', ci_show=True)
    res = logrank_test(all_times[high_risk], all_times[low_risk], all_events[high_risk], all_events[low_risk])
    plt.text(0.05, 0.05, f'p={res.p_value:.4e}', 
             fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
    plt.savefig(os.path.join(output_path, "km_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # t-SNE with Jitter
    if len(all_embeddings) > 30:
        print("   -> 计算 t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(all_embeddings)
        
        # 🌟 核心修改：添加 Jitter (随机噪声) 以分散重叠的点 🌟
        # 计算坐标范围
        x_span = emb_2d[:,0].max() - emb_2d[:,0].min()
        y_span = emb_2d[:,1].max() - emb_2d[:,1].min()
        # 添加 2% 的抖动
        noise = np.random.randn(*emb_2d.shape) * np.array([x_span, y_span]) * 0.02
        emb_2d_jittered = emb_2d + noise
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Risk
        sc1 = axes[0].scatter(emb_2d_jittered[:,0], emb_2d_jittered[:,1], c=all_risks, cmap='RdYlGn_r', 
                              s=40, alpha=0.6, edgecolors='k', lw=0.2) # alpha=0.6 让重叠部分更明显
        plt.colorbar(sc1, ax=axes[0], label='Risk Score')
        axes[0].set_title('(a) Colored by Risk Score', fontweight='bold')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        
        # 2. Event
        colors_evt = ['#3498db' if e==0 else '#e74c3c' for e in all_events]
        axes[1].scatter(emb_2d_jittered[:,0], emb_2d_jittered[:,1], c=colors_evt, 
                        s=40, alpha=0.6, edgecolors='k', lw=0.2)
        axes[1].set_title('(b) Colored by Event Status', fontweight='bold')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "tsne_combined_jitter.png"), dpi=300, bbox_inches='tight')
        plt.close()

def train_clinical_only():
    print("🚀 开始训练纯临床模型 (使用全量数据)...")
    
    clin_map = load_clinical_features(CLINICAL_JSON_PATH)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    
    cache_clin, cache_t, cache_e = [], [], []
    
    # 为了保证对比公平性，我们只取那些有 NPZ 文件的病人
    # (即：我们是在同一个数据集上对比 Clinical Only vs Fusion)
    print("📦 加载数据...")
    for fname in tqdm(files):
        pid = os.path.basename(fname)[:12]
        if pid not in clin_map: continue
        try:
            with np.load(fname) as d:
                if 'time' not in d or 'event' not in d: continue
                cache_clin.append(clin_map[pid])
                cache_t.append(float(d['time']))
                cache_e.append(float(d['event']))
        except: continue

    raw_clin_np = np.vstack(cache_clin)
    
    # 归一化 (Safe copy)
    X_clin_norm = torch.from_numpy(raw_clin_np.copy()).float()
    for i in range(X_clin_norm.shape[1]):
        std = X_clin_norm[:, i].std()
        if std > 1e-6: X_clin_norm[:, i] = (X_clin_norm[:, i] - X_clin_norm[:, i].mean()) / std
    
    Y_t = torch.tensor(cache_t).float()
    Y_e = torch.tensor(cache_e).float()
    
    print(f"✅ 有效样本数: {len(Y_t)}")
    
    # 5-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_c = 0
    best_model_state = None
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_clin_norm)):
        train_ds = TensorDataset(X_clin_norm[train_idx], Y_t[train_idx], Y_e[train_idx])
        test_ds = TensorDataset(X_clin_norm[test_idx], Y_t[test_idx], Y_e[test_idx])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        
        model = ClinicalOnlyNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        best_fold_c = 0
        best_fold_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            for clin, t, e in train_loader:
                clin, t, e = clin.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
                optimizer.zero_grad()
                loss = cox_loss(model(clin).squeeze(), t, e)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                clin, t, e = next(iter(test_loader))
                clin = clin.to(DEVICE)
                pred = model(clin).squeeze()
                try: c_index = concordance_index(t.numpy(), -pred.cpu().numpy(), e.numpy())
                except: c_index = 0.5
                if c_index > best_fold_c:
                    best_fold_c = c_index
                    best_fold_state = model.state_dict().copy()
        
        print(f"Fold {fold+1}: Best C-Index = {best_fold_c:.4f}")
        if best_fold_c > best_overall_c:
            best_overall_c = best_fold_c
            best_model_state = best_fold_state

    if best_model_state:
        torch.save(best_model_state, MODEL_WEIGHT_PATH)
        print(f"\n✅ 最佳模型 (C-Index: {best_overall_c:.4f}) 已保存")
        
        # 使用全量数据进行绘图
        best_model = ClinicalOnlyNet().to(DEVICE)
        best_model.load_state_dict(best_model_state)
        
        full_ds = TensorDataset(X_clin_norm, Y_t, Y_e)
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        generate_visualization(best_model, full_loader, raw_clin_np, DEVICE, OUTPUT_DIR)

if __name__ == "__main__":
    train_clinical_only()