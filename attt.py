import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import openslide
from PIL import Image

# ================= ⚡ 已为您自动填好配置 =================
# 1. 病人 ID
CASE_ID = "TCGA-G4-6320"

# 2. SVS 原文件路径 (您刚下载的)
SVS_PATH = "/home/student2025/shirx2025/SVS_SINGLE_DOWNLOAD/cea26052-2806-45cb-a27d-e1e6bdaaf3e7/TCGA-G4-6320-01Z-00-DX1.09f11d38-4d47-44c9-b8d6-4d4910c6280e.svs"

# 3. NPZ 文件夹路径 (脚本会自动在这个文件夹里搜包含 CASE_ID 的文件)
NPZ_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"

# 4. 模型权重
WEIGHT_PATH = "best_deep_graph_model.pth"

# 5. 输出图片文件名
SAVE_PATH = f"Overlay_{CASE_ID}.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数 (必须与训练时一致)
IN_DIM = 1024
HIDDEN_DIM = 256
KNN_K = 12
PATCH_SIZE = 256 # MUSK 默认通常是 256
# ========================================================

# --- 1. 模型定义 (保持完全一致) ---
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
        # classifier 占位
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

# --- 2. 可视化主程序 ---
def visualize_wsi_overlay():
    print(f"🚀 开始处理病例: {CASE_ID}")
    
    # A. 寻找 NPZ 文件
    print(f"🔍 在 {NPZ_DIR} 中搜索特征文件...")
    real_npz_path = None
    for f in os.listdir(NPZ_DIR):
        if CASE_ID in f and f.endswith('.npz'):
            real_npz_path = os.path.join(NPZ_DIR, f)
            break
    
    if not real_npz_path:
        return print(f"❌ 错误: 找不到包含 {CASE_ID} 的 .npz 文件！")
    print(f"✅ 找到特征文件: {os.path.basename(real_npz_path)}")

    # B. 加载模型
    print("🧠 加载模型权重...")
    model = DeepGraphTransformer(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, k=KNN_K).to(DEVICE)
    if not os.path.exists(WEIGHT_PATH): return print("❌ 找不到模型权重文件")
    model.load_state_dict(torch.load(WEIGHT_PATH), strict=False)
    model.eval()

    # C. 计算 Attention
    print("🧮 计算 Attention 热力值...")
    with np.load(real_npz_path) as d:
        feats = d['feats']
        coords_raw = d['coords']
        
        # 归一化坐标用于模型
        coords_norm = coords_raw.copy()
        c_min, c_max = coords_norm.min(0), coords_norm.max(0)
        denom = c_max - c_min; denom[denom==0] = 1.0
        coords_norm = (coords_norm - c_min) / denom
        
        # 转 Tensor
        feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE) # (1, N, 1024)
        coords_t = torch.from_numpy(coords_norm).float().to(DEVICE)       # (N, 2)
        
        with torch.no_grad():
            # 需要 squeeze 掉 batch 维度传给 forward_viz 中相应的逻辑
            attn_weights, knn_idx = model.forward_viz(feats_t.squeeze(0), coords_t)
            attn_weights = attn_weights.cpu().numpy().flatten()
            knn_idx = knn_idx.cpu().numpy()

    # D. 锁定高危区域
    top_idx = np.argmax(attn_weights)
    center_x, center_y = coords_raw[top_idx]
    
    # 裁剪窗口: 8x8 个 Patch 的大小
    VIEW_PATCHES = 8 
    window_px = int(VIEW_PATCHES * PATCH_SIZE)
    read_x = int(center_x - window_px // 2)
    read_y = int(center_y - window_px // 2)
    read_x, read_y = max(0, read_x), max(0, read_y) # 边界保护

    print(f"📍 锁定最高关注点: ({center_x}, {center_y}) -> 读取窗口: {window_px}x{window_px} px")

    # E. 读取 SVS
    print("📷 正在从 SVS 读取高清大图 (OpenSlide)...")
    try:
        slide = openslide.OpenSlide(SVS_PATH)
        # 尝试读取，如果坐标超出范围可能会报错，加个保护
        try:
            roi_image = slide.read_region((read_x, read_y), 0, (window_px, window_px)).convert("RGB")
        except:
            print("⚠️ 读取 Level 0 失败 (可能超出边界)，尝试读取 Level 1...")
            roi_image = slide.read_region((read_x, read_y), 1, (window_px, window_px)).convert("RGB")
    except Exception as e:
        return print(f"❌ SVS 读取失败: {e}")

    # F. 绘图
    print("🎨 正在绘制叠加图...")
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ax.imshow(roi_image)
    
    # 筛选视野内的 Patch 并转换坐标
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
    
    # 1. 画边 (红色连线)
    lines = []
    line_colors = []
    for i, orig_idx in enumerate(local_indices):
        neighbors = knn_idx[orig_idx, 1:] # 邻居
        for n_idx in neighbors:
            if n_idx in idx_map:
                p1 = local_points[i]
                p2 = local_points[idx_map[n_idx]]
                lines.append([p1, p2])
                line_colors.append(attn_weights[orig_idx]) # 颜色随权重

    # 绘制半透明红线
    lc = LineCollection(lines, cmap='Reds', array=np.array(line_colors), linewidths=2, alpha=0.5)
    ax.add_collection(lc)
    
    # 2. 画热力方块
    for i, (lx, ly) in enumerate(local_points):
        orig_idx = local_indices[i]
        score = attn_weights[orig_idx]
        
        # 归一化用于配色
        norm_score = (score - attn_weights.min()) / (attn_weights.max() - attn_weights.min())
        
        # 最高分画个金框
        edge = 'yellow' if score == attn_weights.max() else 'none'
        lw = 2 if score == attn_weights.max() else 0
        
        rect = patches.Rectangle(
            (lx - PATCH_SIZE//2, ly - PATCH_SIZE//2), # 假设坐标是中心，往左上偏一点覆盖
            PATCH_SIZE, PATCH_SIZE,
            linewidth=lw, edgecolor=edge,
            facecolor=plt.cm.jet(norm_score),
            alpha=0.35
        )
        ax.add_patch(rect)

    ax.axis('off')
    plt.title(f"Graph Attention Overlay (Case: {CASE_ID})\nRed=High Risk, Lines=Graph Topology", fontsize=16)
    
    plt.savefig(SAVE_PATH, bbox_inches='tight', pad_inches=0)
    print(f"✅ 完成！图片已保存至: {os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    visualize_wsi_overlay()