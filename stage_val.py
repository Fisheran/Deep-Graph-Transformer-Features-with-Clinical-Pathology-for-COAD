import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import os

# ================= ⚡ 配置 =================
# 请确保这里的文件名是您上一步生成的那个
CSV_PATH = "/home/student2025/shirx2025/MUSK-surv/MSI_val/COAD_Fusion_Ready.csv" 
# 如果找不到，尝试: "COAD_Fusion_Ready.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
LR = 1e-3
# ==========================================

# --- 1. 修复后的数据集类 (适配 Clin_ 前缀) ---
class FusionDataset(Dataset):
    def __init__(self, df):
        # 1. 图像特征
        img_cols = [c for c in df.columns if c.startswith('Img_')]
        self.img = df[img_cols].values.astype(np.float32)
        
        # 2. 临床特征 (修复列名匹配)
        # 优先查找 'Clin_Age', 如果没有则查找 'Age' (兼容旧版CSV)
        if 'Clin_Age' in df.columns:
            age_col = 'Clin_Age'
            sex_col = 'Clin_Sex'
            stage_col = 'Clin_Stage'
        else:
            age_col = 'Age'
            sex_col = 'Gender'
            stage_col = 'Stage'
            
        # Age
        age = df[age_col].values.astype(np.float32)
        # Z-Score 归一化 (即使之前归一化过，再做一次Z-Score也没问题)
        if age.std() > 0:
            self.age = (age - age.mean()) / (age.std() + 1e-6)
        else:
            self.age = age # 方差为0就不动了

        # Gender (Sex)
        self.gender = df[sex_col].values.astype(np.float32).reshape(-1, 1)
        
        # Stage (转为整数用于Embedding)
        self.stage = df[stage_col].values.astype(np.int64)
        
        # 3. 生存标签
        self.t = df['Time'].values.astype(np.float32)
        self.e = df['Event'].values.astype(np.float32)

    def __len__(self): return len(self.img)
    def __getitem__(self, idx):
        return (torch.tensor(self.img[idx]), 
                torch.tensor(self.age[idx:idx+1]), 
                torch.tensor(self.gender[idx]),
                torch.tensor(self.stage[idx]),
                torch.tensor(self.t[idx]),
                torch.tensor(self.e[idx]))

# --- 2. 模型定义 ---
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.2))
        self.stage_emb = nn.Embedding(5, 8) 
        # Age(1)+Gender(1)+Stage(8)=10
        self.clin_fc = nn.Sequential(nn.Linear(10, 16), nn.ReLU()) 
        # Fusion: 64+16=80
        self.head = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 1))
        
    def forward(self, img, age, sex, stg):
        h_img = self.img_fc(img)
        # 限制分期在 0-4 之间，防止越界
        s_emb = self.stage_emb(torch.clamp(stg, 0, 4))
        h_clin = self.clin_fc(torch.cat([age, sex, s_emb], 1))
        return self.head(torch.cat([h_img, h_clin], 1))

def cox_loss(risk, t, e):
    if e.sum()==0: return torch.tensor(0.0, requires_grad=True).to(risk.device)
    idx = t.argsort(descending=True)
    risk, e = risk[idx], e[idx]
    return -((risk - torch.log(torch.cumsum(torch.exp(risk), 0))) * e).sum() / (e.sum()+1e-6)

# --- 3. 获取无偏预测值 ---
def get_oof_predictions(df):
    print(f"🚀 正在运行 5-Fold 交叉验证 (Total samples: {len(df)})...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_risks = np.zeros(len(df))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        # 这里的 Dataset 会自动处理列名
        train_ds = FusionDataset(df.iloc[train_idx])
        test_ds = FusionDataset(df.iloc[test_idx])
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        
        model = FusionNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        
        for epoch in range(EPOCHS):
            model.train()
            for img, age, sex, stg, t, e in train_loader:
                img, age, sex, stg = img.to(DEVICE), age.to(DEVICE), sex.to(DEVICE), stg.to(DEVICE)
                t, e = t.to(DEVICE), e.to(DEVICE)
                optimizer.zero_grad()
                pred = model(img, age, sex, stg)
                loss = cox_loss(pred, t, e)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for img, age, sex, stg, _, _ in test_loader:
                img, age, sex, stg = img.to(DEVICE), age.to(DEVICE), sex.to(DEVICE), stg.to(DEVICE)
                pred = model(img, age, sex, stg)
                all_risks[test_idx] = pred.cpu().numpy().flatten()
    
    return all_risks

# --- 4. 绘图函数 ---
def plot_subgroup_km(df, target_stages, group_name):
    # 动态获取列名
    stage_col = 'Clin_Stage' if 'Clin_Stage' in df.columns else 'Stage'
    
    # 筛选
    sub_df = df[df[stage_col].isin(target_stages)].copy()
    
    if len(sub_df) < 10:
        print(f"⚠️ {group_name} 样本太少 (N={len(sub_df)})，无法画图")
        return

    # 中位数分组
    median_risk = sub_df['Risk'].median()
    high_mask = sub_df['Risk'] > median_risk
    low_mask = sub_df['Risk'] <= median_risk
    
    T_high = sub_df[high_mask]['Time']
    E_high = sub_df[high_mask]['Event']
    T_low = sub_df[low_mask]['Time']
    E_low = sub_df[low_mask]['Event']
    
    # 统计检验
    try:
        results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
        p_val = results.p_value
    except:
        p_val = 1.0 # 异常情况
    
    c_index = concordance_index(sub_df['Time'], -sub_df['Risk'], sub_df['Event'])

    # 绘图
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    kmf.fit(T_high, E_high, label="High Risk")
    kmf.plot_survival_function(color='#d62728', linewidth=2) # 红色
    
    kmf.fit(T_low, E_low, label="Low Risk")
    kmf.plot_survival_function(color='#1f77b4', linewidth=2) # 蓝色
    
    plt.text(0.05, 0.05, f'p={p_val:.4e}', 
             fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
    
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    fname = f"KM_{group_name.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(fname, dpi=300)
    print(f"✅ {group_name}: C={c_index:.3f}, P={p_val:.4e} -> {fname}")

# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        # 尝试备选文件名
        if os.path.exists("COAD_Fusion_Ready.csv"):
            CSV_PATH = "COAD_Fusion_Ready.csv"
        else:
            raise FileNotFoundError(f"❌ 找不到CSV文件: {CSV_PATH}")
    
    print(f"📂 读取数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 1. 计算风险
    df['Risk'] = get_oof_predictions(df)
    
    # 2. 保存带风险分数的表 (备用)
    df.to_csv("COAD_Predictions_Result.csv", index=False)
    
    print("\n" + "="*40)
    print("🔬 开始临床亚组分析")
    print("="*40)
    
    # 注意：Stage 1.0, 2.0 在 DataFrame 里可能是 float
    # 这里 target_stages 用 float 匹配更稳妥
    
    plot_subgroup_km(df, target_stages=[1.0, 2.0], group_name="Early Stage (I & II)")
    plot_subgroup_km(df, target_stages=[2.0, 3.0], group_name="Intermediate Stage (II & III)")
    plot_subgroup_km(df, target_stages=[4.0], group_name="Late Stage (IV)")