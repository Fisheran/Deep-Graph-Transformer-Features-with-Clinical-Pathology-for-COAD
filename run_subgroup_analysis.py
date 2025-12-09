import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# ================= ⚡ 配置区域 =================
# 1. 您的缓存文件路径
FEATURE_CACHE_PATH = "/home/student2025/shirx2025/outputs_viz_hybrid_tsne_umap/feature_cache.npz"

# 2. 临床 JSON 路径
CLINICAL_JSON_PATH = "/home/student2025/shirx2025/MUSK-surv/TCGA_DATA/gdc-client/COAD/clinical.cart.COAD.json"

# 3. MSI 标签文件
MSI_LABEL_FILE = "/home/student2025/shirx2025/MUSK-surv/MSI_val/coad_msi_labels.csv"

# 4. 输出路径
SAVE_CSV_PATH = "/home/student2025/shirx2025/MUSK-surv/MSI_val/COAD_Fusion_Ready.csv"
OUTPUT_IMG = "msi_validation_result.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================

# --- 1. 临床工具 ---
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
        if "iv" in s: return 4.0
        if "iii" in s: return 3.0
        if "ii" in s: return 2.0
        if "i" in s: return 1.0
        return 0.0 
    @staticmethod
    def t_to_num(s): return 4.0 if "t4" in ClinicalUtils.norm(s) else (3.0 if "t3" in ClinicalUtils.norm(s) else (2.0 if "t2" in ClinicalUtils.norm(s) else (1.0 if "t1" in ClinicalUtils.norm(s) else 0.0)))
    @staticmethod
    def n_to_num(s): return 3.0 if "n3" in ClinicalUtils.norm(s) else (2.0 if "n2" in ClinicalUtils.norm(s) else (1.0 if "n1" in ClinicalUtils.norm(s) else 0.0))
    @staticmethod
    def m_to_num(s): return 1.0 if "m1" in ClinicalUtils.norm(s) else 0.0

def load_clinical_map(json_path):
    print(f"📄 解析临床 JSON: {json_path}")
    with open(json_path, 'r') as f: data = json.load(f)
    clin_map = {}
    for entry in data:
        try:
            sid = entry.get('submitter_id', '')
            if not sid: continue
            pid = sid[:12]
            
            dx = entry.get('diagnoses', [{}])[0]
            demo = entry.get('demographic', {})
            
            age = ClinicalUtils.to_f(demo.get('age_at_diagnosis'))
            age = (age/365.25/100) if not np.isnan(age) else 0.6
            gender = 1.0 if "male" in ClinicalUtils.norm(demo.get("gender")) else 0.0
            
            vec = [age, gender, 
                   ClinicalUtils.stage_to_num(dx.get("ajcc_pathologic_stage")),
                   ClinicalUtils.t_to_num(dx.get("ajcc_pathologic_t")),
                   ClinicalUtils.n_to_num(dx.get("ajcc_pathologic_n")),
                   ClinicalUtils.m_to_num(dx.get("ajcc_pathologic_m"))]
            clin_map[pid] = vec
        except: pass
    return clin_map

# --- 2. 融合模型 ---
class FusionNet(nn.Module):
    def __init__(self, img_dim=512):
        super().__init__()
        self.img_fc = nn.Sequential(nn.Linear(img_dim, 64), nn.ReLU(), nn.Dropout(0.2))
        self.clin_fc = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, img, clin):
        return self.head(torch.cat([self.img_fc(img), self.clin_fc(clin)], 1))

def cox_loss(risk, t, e):
    if e.sum()==0: return torch.tensor(0.0, requires_grad=True).to(risk.device)
    idx = t.argsort(descending=True)
    risk, e = risk[idx], e[idx]
    return -((risk - torch.log(torch.cumsum(torch.exp(risk), 0))) * e).sum() / (e.sum()+1e-6)

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def process_and_validate():
    # --- Step 1: 读取 NPZ ---
    print(f"🚀 读取缓存文件: {FEATURE_CACHE_PATH}")
    if not os.path.exists(FEATURE_CACHE_PATH): return print("❌ 找不到文件")
    
    cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    keys = list(cache.keys())
    print(f"   Keys: {keys}")
    
    # 【修复点】使用正确的键名
    try:
        # 使用报错信息里提示的正确 key
        ids = cache['case_ids']
        feats = cache['img_features']
        times = cache['times']
        events = cache['events']
        
        print(f"   数据形状: Feats {feats.shape}, IDs {ids.shape}")
    except KeyError as e:
        return print(f"❌ Key 错误: {e}. 请检查 npz 内容。")

    # 加载临床数据 (用于补充 CSV 可读信息)
    clin_map = load_clinical_map(CLINICAL_JSON_PATH)
    
    dataset = []
    print("🔄 正在生成融合 CSV...")
    
    for i in range(len(ids)):
        pid = str(ids[i])[:12]
        
        # 如果没有 JSON 临床信息，就跳过
        if pid not in clin_map: continue
        clin = clin_map[pid]
        
        row = {
            'Case_ID': pid,
            'Time': float(times[i]),
            'Event': int(events[i]),
            'Clin_Age': clin[0], 'Clin_Sex': clin[1], 'Clin_Stage': clin[2],
            'Clin_T': clin[3], 'Clin_N': clin[4], 'Clin_M': clin[5]
        }
        
        # 展平图像特征
        feat_vec = feats[i]
        if len(feat_vec.shape) > 1: feat_vec = feat_vec.flatten()
        
        for j, val in enumerate(feat_vec):
            row[f'Img_{j}'] = val
            
        dataset.append(row)
        
    df = pd.DataFrame(dataset)
    df.to_csv(SAVE_CSV_PATH, index=False)
    print(f"✅ CSV 生成完毕: {SAVE_CSV_PATH} (N={len(df)})")
    
    # --- Step 2: 训练融合模型生成 Risk Score ---
    print("\n⚡ 训练融合模型生成 Risk Score...")
    img_cols = [c for c in df.columns if c.startswith('Img_')]
    X_img = df[img_cols].values.astype(np.float32)
    X_clin = df[['Clin_Age', 'Clin_Sex', 'Clin_Stage', 'Clin_T', 'Clin_N', 'Clin_M']].values.astype(np.float32)
    Y_t = df['Time'].values.astype(np.float32)
    Y_e = df['Event'].values.astype(np.float32)
    
    predictions = np.zeros(len(df))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(X_img):
        t_img, t_clin = torch.tensor(X_img[train_idx]).to(DEVICE), torch.tensor(X_clin[train_idx]).to(DEVICE)
        t_t, t_e = torch.tensor(Y_t[train_idx]).to(DEVICE), torch.tensor(Y_e[train_idx]).to(DEVICE)
        v_img, v_clin = torch.tensor(X_img[test_idx]).to(DEVICE), torch.tensor(X_clin[test_idx]).to(DEVICE)
        
        # 这里的 img_dim 动态获取，防止维度不匹配
        model = FusionNet(img_dim=X_img.shape[1]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            loss = cox_loss(model(t_img, t_clin), t_t, t_e)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            predictions[test_idx] = model(v_img, v_clin).cpu().numpy().flatten()
            
    df['Risk_Score'] = predictions
    
    # --- Step 3: MSI 验证 ---
    print("\n🔬 执行 MSI 亚组验证...")
    if not os.path.exists(MSI_LABEL_FILE): return print("❌ 没找到 MSI 标签文件")
    msi_df = pd.read_csv(MSI_LABEL_FILE)
    
    # 合并
    df['Short_ID'] = df['Case_ID'].apply(lambda x: str(x)[:12])
    msi_df['Short_ID'] = msi_df['Case_ID'].apply(lambda x: str(x)[:12])
    merged = pd.merge(df, msi_df[['Short_ID', 'MSI_Status']], on='Short_ID', how='inner')
    
    print(f"   匹配到 {len(merged)} 个 MSI 病例")
    
    plt.figure(figsize=(10, 6))
    groups = ['MSS', 'MSI-H']
    
    for i, grp in enumerate(groups):
        sub = merged[merged['MSI_Status'].astype(str).str.contains(grp, case=False)]
        if len(sub) < 5: continue
        
        c_idx = concordance_index(sub['Time'], -sub['Risk_Score'], sub['Event'])
        med = sub['Risk_Score'].median()
        high = sub[sub['Risk_Score'] > med]
        low = sub[sub['Risk_Score'] <= med]
        
        try:
            p_val = logrank_test(high['Time'], low['Time'], high['Event'], low['Event']).p_value
        except: p_val = 1.0
        
        ax = plt.subplot(1, 1, 1)
        kmf = KaplanMeierFitter()
        kmf.fit(high['Time'], high['Event'], label="High Risk"); kmf.plot(ax=ax, color='red')
        kmf.fit(low['Time'], low['Event'], label="Low Risk"); kmf.plot(ax=ax, color='blue')
        
        plt.text(0.05, 0.05, f'p={p_val:.4e}', 
             fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
        print(f"📊 {grp}: C-Index={c_idx:.3f}, P={p_val:.4e}")
        
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"✅ 验证图已保存: {OUTPUT_IMG}")

if __name__ == "__main__":
    process_and_validate()