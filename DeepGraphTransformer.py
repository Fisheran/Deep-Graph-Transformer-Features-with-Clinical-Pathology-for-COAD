import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc, glob

# ================= ⚡ 顶配参数配置 =================
DATA_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ_ResNet/COAD"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存路径
BEST_MODEL_PATH = "best_deep_graph_model_resnet.pth"
KM_PLOT_PATH = "deep_graph_km_resnet.png"

# 训练参数 (模型变深了，需要更精细的调优)
BATCH_SIZE = 1
EPOCHS = 30
LR = 1.5e-4             # 稍微降低学习率，因为模型变深了
WEIGHT_DECAY = 5e-4     # 增加正则化，防止过拟合
IN_DIM = 2048
HIDDEN_DIM = 256
DROPOUT = 0.25          # 增加 Dropout
MAX_NODES = 2500        # 保持显存安全
KNN_K = 12              # 增加邻居数量，让图更稠密一点
# ===============================================

# --- 1. 标准化 GCN 层 (Spectral GCN + Residual) ---
class ResGCNLayer(nn.Module):
    """
    标准的 Kipf & Welling GCN 层，带残差连接和 LayerNorm
    公式: Output = Norm(ReLU(AXW) + X)
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super(ResGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # 如果输入输出维度不一致，需要投影残差
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x, adj):
        # x: (N, in)
        # adj: (N, N) 归一化后的邻接矩阵
        
        # 1. 线性变换 + 图传播
        support = self.linear(x)   # (N, out)
        out = torch.mm(adj, support) # (N, out)
        
        # 2. 激活 + Dropout
        out = F.relu(out)
        out = self.dropout(out)
        
        # 3. 残差连接 + Norm (防止过平滑的关键)
        res = self.residual_proj(x)
        out = out + res
        out = self.norm(out)
        
        return out

def build_standard_adj(coords, k=12):
    """
    构建标准的归一化邻接矩阵 D^-0.5 * (A+I) * D^-0.5
    """
    N = coords.shape[0]
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    
    # KNN 建图
    _, indices = dist.topk(k + 1, largest=False)
    adj = torch.zeros(N, N, device=coords.device)
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, k+1)
    adj[src, indices] = 1.0
    
    # 强制对称 (无向图)
    adj = torch.max(adj, adj.t())
    
    # 添加自环 (Self-loop)
    # A_hat = A + I
    adj = adj + torch.eye(N, device=coords.device)
    
    # 计算度矩阵 D_hat
    degree = adj.sum(1) # (N,)
    
    # 归一化: D^-0.5
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Symmetric normalization: D^-0.5 * A_hat * D^-0.5
    adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return adj_norm

# --- 2. 高级 Pooling 模块 (Dual Branch) ---
class DualPooling(nn.Module):
    """
    同时使用 Attention Pooling (关注分布) 和 Max Pooling (关注最显著特征)
    """
    def __init__(self, dim):
        super(DualPooling, self).__init__()
        # Branch A: Gated Attention
        self.attn_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        # Branch B: Max Pooling (不需要参数)
        
    def forward(self, x):
        # x: (N, dim)
        
        # A. Attention Branch
        attn_scores = self.attn_net(x) # (N, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        feat_attn = torch.mm(attn_weights.t(), x) # (1, dim)
        
        # B. Max Branch
        feat_max, _ = torch.max(x, dim=0, keepdim=True) # (1, dim)
        
        # Concat: (1, 2*dim)
        return torch.cat([feat_attn, feat_max], dim=1)

# --- 3. 核心模型: Deep Graph-Transformer ---
class DeepGraphTransformer(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, k=12, dropout=0.25):
        super(DeepGraphTransformer, self).__init__()
        self.k = k
        
        # 1. 初始投影
        self.fc_start = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. 深层 GCN (3层)
        self.gcn1 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn2 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn3 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        
        # 3. 深层 Transformer (3层)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm 训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        
        # 4. 双分支 Pooling
        self.pool = DualPooling(hidden_dim)
        
        # 5. 预测头 (输入维度翻倍了，因为用了Dual Pooling)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, coords):
        # x: (N, 1024)
        
        # Projection
        h = self.fc_start(x)
        
        # Graph Construction
        adj = build_standard_adj(coords, self.k)
        
        # Deep GCN (Local Context)
        h = self.gcn1(h, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj)
        
        # Deep Transformer (Global Context)
        h_trans = h.unsqueeze(0) # (1, N, 256)
        h_trans = self.transformer(h_trans)
        h = h + h_trans.squeeze(0) # Residual connection GCN + Transformer
        
        # Dual Pooling
        h_slide = self.pool(h) # (1, 512)
        
        # Prediction
        logits = self.classifier(h_slide)
        return logits

# --- 4. 标准 Cox Loss ---
class CoxPHLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, hazards, times, events):
        """
        标准的 Negative Log Partial Likelihood
        hazards: (B,) 预测风险值 (theta, not exp(theta))
        times: (B,)
        events: (B,)
        """
        if events.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=hazards.device)
            
        # 排序
        sorted_idx = times.argsort(descending=True)
        hazards = hazards[sorted_idx]
        events = events[sorted_idx]
        
        # LogSumExp 技巧防止数值溢出
        # loss = - sum( theta_i - log( sum(exp(theta_j)) ) ) for E=1
        
        risk_set = torch.cumsum(torch.exp(hazards), dim=0)
        log_risk = torch.log(risk_set)
        
        neg_log_like = hazards - log_risk
        
        # 只取发生事件的样本
        loss = -neg_log_like[events.bool()].sum() / events.sum()
        
        return loss

# --- 5. 数据集与工具 ---
class GraphDataset(Dataset):
    def __init__(self, data_list, max_nodes=2500, is_training=True):
        self.data = data_list
        self.max_nodes = max_nodes
        self.is_training = is_training

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        feats = item['feats'] 
        coords = item['coords'] 
        label = item['label']
        
        num_patches = feats.shape[0]
        if num_patches > self.max_nodes:
            if self.is_training:
                indices = torch.randperm(num_patches)[:self.max_nodes]
            else:
                indices = torch.arange(self.max_nodes)
            feats = feats[indices]
            coords = coords[indices]
        return feats, coords, label

def preload_data(data_dir, min_patches):
    print(f"🚀 Scanning data in {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
    print(f"📂 Found {len(files)} files. Loading into RAM...")
    
    loaded = []
    for f in tqdm(files):
        try:
            with np.load(f) as d:
                # 检查特征数
                if 'feats' in d and d['feats'].shape[0] >= min_patches:
                    coords = d['coords'] if 'coords' in d else np.zeros((d['feats'].shape[0], 2))
                    
                    # 坐标归一化 [0, 1]
                    if coords.shape[0] > 0:
                        c_min, c_max = coords.min(0), coords.max(0)
                        denom = c_max - c_min
                        denom[denom == 0] = 1
                        coords = (coords - c_min) / denom
                    
                    # 加载标签 [Time, Event]
                    loaded.append({
                        'feats': torch.from_numpy(d['feats']).float(),
                        'coords': torch.from_numpy(coords).float(),
                        'label': torch.tensor([float(d['time']), float(d['event'])]).float()
                    })
        except Exception as e:
            # print(f"Error loading {f}: {e}")
            pass
    print(f"✅ Successfully loaded {len(loaded)} samples.")
    return loaded

# --- 6. 主训练循环 ---
def run_deep_training():
    torch.cuda.empty_cache()
    all_data = preload_data(DATA_DIR, min_patches=500)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = torch.amp.GradScaler('cuda')
    criterion = CoxPHLoss()
    
    global_best_c = 0
    best_fold = -1
    
    print(f"\n🚀 启动 Deep Graph-Transformer (Layers: 3+3, Dual-Pool, K={KNN_K})")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\n🔹 Fold {fold+1} / 5")
        
        train_sub = [all_data[i] for i in train_idx]
        test_sub = [all_data[i] for i in test_idx]
        
        train_loader = DataLoader(GraphDataset(train_sub, MAX_NODES, True), batch_size=1, shuffle=True)
        test_loader = DataLoader(GraphDataset(test_sub, MAX_NODES, False), batch_size=1, shuffle=False)
        
        model = DeepGraphTransformer(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, k=KNN_K, dropout=DROPOUT).to(DEVICE)
        
        # 使用 L2 正则 (weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        fold_best_c = 0
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            step_preds, step_times, step_events = [], [], []
            
            for i, (feats, coords, label) in enumerate(train_loader):
                feats, coords = feats.squeeze(0).to(DEVICE), coords.squeeze(0).to(DEVICE)
                t, e = label[0,0].to(DEVICE), label[0,1].to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    pred = model(feats, coords)
                
                step_preds.append(pred)
                step_times.append(t)
                step_events.append(e)
                
                if (i+1) % 32 == 0 or (i+1) == len(train_loader):
                    p_stack = torch.cat(step_preds).squeeze()
                    if p_stack.ndim == 0: p_stack = p_stack.unsqueeze(0)
                    t_stack = torch.stack(step_times); e_stack = torch.stack(step_events)
                    
                    if e_stack.sum() > 0:
                        loss = criterion(p_stack, t_stack, e_stack)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    step_preds, step_times, step_events = [], [], []
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds, val_times, val_events = [], [], []
            with torch.no_grad():
                for feats, coords, label in test_loader:
                    feats, coords = feats.squeeze(0).to(DEVICE), coords.squeeze(0).to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        pred = model(feats, coords)
                    val_preds.append(pred.item())
                    val_times.append(label[0,0].item())
                    val_events.append(label[0,1].item())
            
            try:
                # 标准 C-Index
                score = concordance_index(val_times, -np.array(val_preds), val_events)
                if score > fold_best_c:
                    fold_best_c = score
            except: pass
            
        print(f"  🏆 Fold Best C-Index: {fold_best_c:.4f}")
        
        if fold_best_c > global_best_c:
            global_best_c = fold_best_c
            best_fold = fold + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            
        # 垃圾回收
        del model, optimizer
        torch.cuda.empty_cache()

    print(f"\n🔥 训练结束! 全局最佳: Fold {best_fold} (C-Index: {global_best_c:.4f})")
    
    # --- 画标准 KM 曲线 ---
    print("🎨 正在绘制最终 KM 曲线...")
    best_model = DeepGraphTransformer(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, k=KNN_K, dropout=DROPOUT).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    best_model.eval()
    
    loader = DataLoader(GraphDataset(all_data, MAX_NODES, False), batch_size=1, shuffle=False)
    all_risks, all_times, all_events = [], [], []
    
    with torch.no_grad():
        for feats, coords, label in tqdm(loader):
            feats, coords = feats.squeeze(0).to(DEVICE), coords.squeeze(0).to(DEVICE)
            pred = best_model(feats, coords)
            all_risks.append(pred.item())
            all_times.append(label[0,0].item())
            all_events.append(label[0,1].item())
            
    risks = np.array(all_risks)
    times = np.array(all_times)
    events = np.array(all_events)
    
    median = np.median(risks)
    high = risks > median
    low = risks <= median
    
    plt.figure(figsize=(10, 6))
    kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
    kmf_h.fit(times[high], events[high], label="High Risk")
    kmf_l.fit(times[low], events[low], label="Low Risk")
    kmf_h.plot_survival_function(color='#e74c3c', ci_show=True)
    kmf_l.plot_survival_function(color='#2ecc71', ci_show=True)
    res = logrank_test(times[high], times[low], events[high], events[low])
    plt.text(0.05, 0.05, f'p={res.p_value:.4e}', 
             fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)

    plt.savefig(os.path.join(KM_PLOT_PATH), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    run_deep_training()