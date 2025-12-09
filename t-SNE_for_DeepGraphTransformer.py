import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.lines import Line2D

# ================= ⚡ 配置区域 =================
# 请确保路径正确
DATA_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ_ResNet/COAD"
WEIGHT_PATH = "best_deep_graph_model_resnet.pth"
SAVE_IMG_PATH = "deep_graph_analysis_4plots_resnet.png" # 保存的文件名

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数 (必须与训练时一致)
IN_DIM = 2048
HIDDEN_DIM = 256
KNN_K = 12
DROPOUT = 0.25
MAX_NODES = 2500
MIN_PATCHES = 500  # 过滤小样本
# ===============================================

# --- 1. 模型结构 (保持完全一致) ---
class ResGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(ResGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.mm(adj, support)
        out = F.relu(out)
        out = self.dropout(out)
        res = self.residual_proj(x)
        out = out + res
        out = self.norm(out)
        return out

def build_standard_adj(coords, k=12):
    N = coords.shape[0]
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    _, indices = dist.topk(min(k + 1, N), largest=False)
    adj = torch.zeros(N, N, device=coords.device)
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, indices.shape[1])
    adj[src, indices] = 1.0
    adj = torch.max(adj, adj.t())
    adj = adj + torch.eye(N, device=coords.device)
    degree = adj.sum(1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return adj_norm

class DualPooling(nn.Module):
    def __init__(self, dim):
        super(DualPooling, self).__init__()
        self.attn_net = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
    def forward(self, x):
        attn_scores = self.attn_net(x)
        attn_weights = torch.softmax(attn_scores, dim=0)
        feat_attn = torch.mm(attn_weights.t(), x)
        feat_max, _ = torch.max(x, dim=0, keepdim=True)
        return torch.cat([feat_attn, feat_max], dim=1)

class DeepGraphTransformer(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, k=12, dropout=0.25):
        super(DeepGraphTransformer, self).__init__()
        self.k = k
        self.fc_start = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.gcn1 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn2 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn3 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pool = DualPooling(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, coords, return_embedding=False):
        h = self.fc_start(x)
        adj = build_standard_adj(coords, self.k)
        h = self.gcn1(h, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj)
        h_trans = h.unsqueeze(0)
        h_trans = self.transformer(h_trans)
        h = h + h_trans.squeeze(0)
        h_slide = self.pool(h)
        
        if return_embedding:
            return h_slide # 返回 Embedding 用于 t-SNE
            
        return self.classifier(h_slide) # 返回 Risk Score 用于分布图

# --- 2. 数据处理 ---
def preload_data(data_dir, min_patches):
    print("🚀 Loading data...")
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    loaded = []
    for f in tqdm(files):
        try:
            with np.load(f) as d:
                if 'feats' in d and d['feats'].shape[0] >= min_patches:
                    coords = d['coords'] if 'coords' in d else np.zeros((d['feats'].shape[0], 2))
                    # 坐标归一化
                    if coords.shape[0] > 0:
                        c_min, c_max = coords.min(0), coords.max(0)
                        denom = c_max - c_min
                        denom[denom == 0] = 1
                        coords = (coords - c_min) / denom
                    
                    # 读取 Stage 和 Event
                    stage = d['cov_ajcc_stage_num'].item() if 'cov_ajcc_stage_num' in d else 0
                    event = d['event'].item() if 'event' in d else 0
                    
                    loaded.append({
                        'feats': torch.from_numpy(d['feats']).float(),
                        'coords': torch.from_numpy(coords).float(),
                        'extra': {'stage': stage, 'event': event}
                    })
        except: pass
    print(f"Loaded {len(loaded)} samples.")
    return loaded

class GraphDataset(Dataset):
    def __init__(self, data_list, max_nodes):
        self.data = data_list
        self.max_nodes = max_nodes
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        feats, coords = item['feats'], item['coords']
        if feats.shape[0] > self.max_nodes:
            indices = torch.randperm(feats.shape[0])[:self.max_nodes]
            feats, coords = feats[indices], coords[indices]
        return feats, coords, item['extra']

# --- 3. 核心逻辑 ---
def run_full_analysis():
    # A. 加载模型
    model = DeepGraphTransformer(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, k=KNN_K, dropout=DROPOUT).to(DEVICE)
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from {WEIGHT_PATH}")
    else:
        print(f"❌ Weight file not found: {WEIGHT_PATH}")
        return
    model.eval()

    # B. 准备数据
    all_data = preload_data(DATA_DIR, MIN_PATCHES)
    if not all_data: return
    loader = DataLoader(GraphDataset(all_data, MAX_NODES), batch_size=1, shuffle=False)

    embeddings = []
    risk_scores = []
    stages = []
    events = []

    print("🚀 Running Inference...")
    with torch.no_grad():
        for feats, coords, extra in tqdm(loader):
            feats, coords = feats.squeeze(0).to(DEVICE), coords.squeeze(0).to(DEVICE)
            
            # 1. 获取特征向量 (Embedding)
            emb = model(feats, coords, return_embedding=True)
            embeddings.append(emb.cpu().numpy().flatten())
            
            # 2. 获取风险分数 (Risk Score)
            risk = model.classifier(emb)
            risk_scores.append(risk.item())
            
            stages.append(extra['stage'].item())
            events.append(extra['event'].item())

    # C. 计算 t-SNE
    print("🎨 Computing t-SNE...")
    X_emb = np.array(embeddings)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto').fit_transform(X_emb)
    
    risk_scores = np.array(risk_scores)
    stages = np.array(stages)
    events = np.array(events)
    print("      * Running t-SNE for Risk/Event...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings = np.array(embeddings)
    emb_tsne = tsne.fit_transform(embeddings)
        
    # Plot 1: Risk (t-SNE)
    sc1 = axes[0].scatter(emb_tsne[:,0], emb_tsne[:,1], c=risk_scores, cmap='RdYlGn_r', s=50, alpha=0.7, edgecolors='k', linewidth=0.3)
    plt.colorbar(sc1, ax=axes[0], label='Risk Score')
    axes[0].set_title('(a) t-SNE: Colored by Risk', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
        
    # Plot 2: Event (t-SNE)
    colors_evt = ['#3498db' if e==0 else '#e74c3c' for e in events]
    axes[1].scatter(emb_tsne[:,0], emb_tsne[:,1], c=colors_evt, s=50, alpha=0.7, edgecolors='k', linewidth=0.3)
    leg_evt = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', label='Censored', markersize=10),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Event', markersize=10)]
    axes[1].legend(handles=leg_evt, loc='best')
    axes[1].set_title('(b) t-SNE: Colored by Event', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
        
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_IMG_PATH), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ All analysis plots saved to: {SAVE_IMG_PATH}")

if __name__ == "__main__":
    run_full_analysis()