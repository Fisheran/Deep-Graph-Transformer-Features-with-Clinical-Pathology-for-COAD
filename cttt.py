import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec  # <--- 新增：用于控制子图比例
from matplotlib.collections import LineCollection
import openslide
from PIL import Image

# ================= ⚡ 配置区域 =================
CASE_ID = "TCGA-5M-AATE"
SVS_PATH = "/home/student2025/shirx2025/SVS_HUGE_DOWNLOAD/5ad0faff-3ea0-46da-8b6b-a7c70d61fbec/TCGA-5M-AATE-01Z-00-DX1.483FFD2F-61A1-477E-8F94-157383803FC7.svs"
NPZ_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"
WEIGHT_PATH = "best_deep_graph_model.pth"
SAVE_PATH = f"Overlay_Huge_{CASE_ID}_LayoutFixed.png" # 改个名，防止覆盖

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_DIM = 1024
HIDDEN_DIM = 256
KNN_K = 12
PATCH_SIZE = 256 
# ===============================================

# --- 1. 模型定义 (保持一致) ---
class ResGCNLayer(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.norm = nn.LayerNorm(out_f)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Linear(in_f, out_f) if in_f!=out_f else nn.Identity()
    def forward(self, x, adj):
        return self.norm(self.res(x) + self.dropout(F.relu(torch.mm(adj, self.linear(x)))))

def build_standard_adj(coords, k=12):
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    _, indices = dist.topk(k + 1, largest=False)
    adj = torch.zeros(coords.shape[0], coords.shape[0], device=coords.device)
    src = torch.arange(coords.shape[0], device=coords.device).unsqueeze(1).expand(coords.shape[0], k+1)
    adj[src, indices] = 1.0
    adj = torch.max(adj, adj.t()) + torch.eye(coords.shape[0], device=coords.device)
    d_inv = torch.diag(torch.pow(adj.sum(1), -0.5))
    return torch.mm(torch.mm(d_inv, adj), d_inv), indices

class DualPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_net = nn.Sequential(nn.Linear(dim, dim//2), nn.Tanh(), nn.Linear(dim//2, 1))
    def forward(self, x):
        attn_scores = self.attn_net(x)
        attn_weights = torch.softmax(attn_scores, dim=0)
        return attn_weights 

class DeepGraphTransformer(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, k=12, dropout=0.25):
        super().__init__()
        self.k = k
        self.fc_start = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.gcn1 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn2 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn3 = ResGCNLayer(hidden_dim, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, 512, dropout, batch_first=True, norm_first=True), 3)
        self.pool = DualPooling(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim // 2))

    def forward_viz(self, x, coords):
        h = self.fc_start(x)
        adj, knn_indices = build_standard_adj(coords, self.k)
        h = self.gcn1(h, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj)
        h = h + self.transformer(h.unsqueeze(0)).squeeze(0)
        attn_weights = self.pool(h)
        return attn_weights, knn_indices

# --- 2. 可视化主程序 (布局优化版) ---
def visualize_huge_overlay_layout_fixed():
    print(f"🚀 开始绘图 (布局优化版): {CASE_ID}")
    
    # A. 寻找 NPZ
    real_npz_path = None
    for f in os.listdir(NPZ_DIR):
        if CASE_ID in f and f.endswith('.npz'):
            real_npz_path = os.path.join(NPZ_DIR, f); break
    if not real_npz_path: return print("❌ 找不到 NPZ")

    # B. 加载模型
    model = DeepGraphTransformer(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, k=KNN_K).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH), strict=False)
    model.eval()

    # C. 计算 Attention
    with np.load(real_npz_path) as d:
        feats = d['feats']
        coords_raw = d['coords']
        N = feats.shape[0]
        
        coords_norm = coords_raw.copy()
        c_min, c_max = coords_norm.min(0), coords_norm.max(0)
        denom = c_max - c_min; denom[denom==0] = 1.0
        coords_norm = (coords_norm - c_min) / denom
        
        feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
        coords_t = torch.from_numpy(coords_norm).float().to(DEVICE)
        
        with torch.no_grad():
            attn_weights, knn_idx = model.forward_viz(feats_t.squeeze(0), coords_t)
            attn_weights = attn_weights.cpu().numpy().flatten()
            knn_idx = knn_idx.cpu().numpy()

    # D. 锁定高危区
    top_idx = np.argmax(attn_weights)
    center_x, center_y = coords_raw[top_idx]
    VIEW_PATCHES = 10
    window_px = int(VIEW_PATCHES * PATCH_SIZE)
    read_x = int(center_x - window_px // 2)
    read_y = int(center_y - window_px // 2)
    read_x, read_y = max(0, read_x), max(0, read_y)

    # E. 读取 SVS
    slide = openslide.OpenSlide(SVS_PATH)
    roi_image = slide.read_region((read_x, read_y), 0, (window_px, window_px)).convert("RGB")

    # F. 绘图 (GridSpec 布局控制)
    print("🎨 正在绘制...")
    
    # 调整画板大小：宽度加大到 22，高度保持 10
    fig = plt.figure(figsize=(22, 10), dpi=300) 
    
    # --- 关键修改：使用 GridSpec 控制比例 ---
    # width_ratios=[1.5, 1] -> 左图宽度是右图的 1.5 倍
    # wspace=0.15 -> 左右图之间留一点空隙
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)
    
    # --- 左图：全景 Heatmap ---
    ax1 = fig.add_subplot(gs[0]) # 使用第一个格子
    
    thumb_size = (2000, 2000)
    thumb = slide.get_thumbnail(thumb_size).convert("RGB")
    ax1.imshow(thumb)
    
    scale_x = thumb.size[0] / slide.dimensions[0]
    scale_y = thumb.size[1] / slide.dimensions[1]
    w_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
    
    sc = ax1.scatter(coords_raw[:,0]*scale_x, coords_raw[:,1]*scale_y, 
                     c=w_norm, cmap='jet', s=1, alpha=0.5)
    
    rect = patches.Rectangle((read_x*scale_x, read_y*scale_y), window_px*scale_x, window_px*scale_y, 
                             linewidth=3, edgecolor='red', facecolor='none') # 线宽加粗到3
    ax1.add_patch(rect)
    
    # Title 加上 pad=20 防止贴太近
    ax1.set_title(f"Whole Slide Heatmap (N={N} Patches)\nRed Box = Zoom Region", fontsize=18, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Colorbar 调整
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Attention Score", fontsize=12)

    # --- 右图：局部高清 + Graph ---
    ax2 = fig.add_subplot(gs[1]) # 使用第二个格子
    ax2.imshow(roi_image)
    
    x_min, x_max = read_x, read_x + window_px
    y_min, y_max = read_y, read_y + window_px
    
    local_points = []
    local_indices = []
    
    for i, (cx, cy) in enumerate(coords_raw):
        if x_min <= cx < x_max and y_min <= cy < y_max:
            lx = cx - read_x
            ly = cy - read_y
            local_points.append([lx, ly])
            local_indices.append(i)
            
    local_points = np.array(local_points)
    idx_map = {orig: loc for loc, orig in enumerate(local_indices)}
    
    lines = []
    line_colors = []
    for i, orig_idx in enumerate(local_indices):
        neighbors = knn_idx[orig_idx, 1:] 
        for n_idx in neighbors:
            if n_idx in idx_map:
                p1 = local_points[i]
                p2 = local_points[idx_map[n_idx]]
                lines.append([p1, p2])
                line_colors.append(attn_weights[orig_idx])

    lc = LineCollection(lines, cmap='Reds', array=np.array(line_colors), linewidths=2, alpha=0.6)
    ax2.add_collection(lc)
    
    for i, (lx, ly) in enumerate(local_points):
        orig_idx = local_indices[i]
        score = attn_weights[orig_idx]
        norm_score = (score - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
        edge = 'yellow' if score == attn_weights.max() else 'none'
        rect = patches.Rectangle(
            (lx - PATCH_SIZE//2, ly - PATCH_SIZE//2), 
            PATCH_SIZE, PATCH_SIZE,
            linewidth=2 if edge=='none' else 4, edgecolor=edge,
            facecolor=plt.cm.jet(norm_score),
            alpha=0.3
        )
        ax2.add_patch(rect)

    ax2.set_title("Local Topology & Attention\nRed Lines = Graph Connections", fontsize=18, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # --- 关键修改：调整顶部边距，防止标题溢出 ---
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"✅ 布局优化版绘图完成: {os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    visualize_huge_overlay_layout_fixed()