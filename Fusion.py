import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.manifold import TSNE
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
BACKBONE_WEIGHT = "best_deep_graph_model.pth" 

# 输出路径
OUTPUT_DIR = "outputs_viz_hybrid_tsne_umap"
FUSION_WEIGHT_PATH = os.path.join(OUTPUT_DIR, "best_fusion_model.pth")
CACHE_PATH = os.path.join(OUTPUT_DIR, "feature_cache.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
IN_DIM = 1024
HIDDEN_DIM = 256
KNN_K = 12
MAX_NODES = 2500
MIN_PATCHES = 500
EPOCHS = 40      
LR = 1e-3
BATCH_SIZE = 32

# 设置绘图风格
try: plt.style.use('seaborn-v0_8-darkgrid')
except: plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🧹 强制删除缓存
if os.path.exists(CACHE_PATH):
    print(f"🧹 删除旧缓存 {CACHE_PATH}...")
    os.remove(CACHE_PATH)
# ===============================================

class ClinicalUtils:
    @staticmethod
    def norm(s): 
        if s is None: return ""
        return str(s).strip().lower()
    
    @staticmethod
    def to_f(val): 
        try: return float(val) 
        except: return 0.0 # 修改为0.0以防NaN破坏计算
        
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

    # --- 新增的转换函数 ---
    @staticmethod
    def yes_no_to_num(s):
        """将 Yes/No 转换为 1.0/0.0"""
        s = ClinicalUtils.norm(s)
        if "yes" in s: return 1.0
        return 0.0

    @staticmethod
    def residual_to_num(s):
        """将切缘状态转换为数值: R0=0, R1=1, R2=2"""
        s = ClinicalUtils.norm(s)
        if "r2" in s: return 2.0
        if "r1" in s: return 1.0
        return 0.0 # R0 或 RX 默认为 0

def load_clinical_features(json_path):
    print(f"🔄 正在读取临床数据: {json_path}")
    clinical_map = {}
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            if 'data' in data and isinstance(data['data'], list): data = data['data']
            else: data = []

        for case in data:
            # 兼容不同的ID字段
            pid = case.get("case_submitter_id") or case.get("submitter_id") or ""
            pid = pid[:12] # TCGA ID 通常取前12位
            if not pid: continue
            
            diagnoses = case.get("diagnoses", [])
            if not diagnoses: candidates = [case]
            else: candidates = diagnoses

            # 初始化所有特征变量
            final_stage = 0.0
            final_t = 0.0
            final_n = 0.0
            final_m = 0.0
            
            # 新增变量初始化
            final_vasc = 0.0  # 脉管侵犯
            final_lymph_inv = 0.0 # 淋巴管侵犯
            final_peri = 0.0  # 神经侵犯
            final_nodes = 0.0 # 检测淋巴结数
            final_resid = 0.0 # 残留病灶
            
            # 遍历所有诊断记录，选取最严重的(stage最高)作为该病人的代表数据
            # 如果是同一stage，后读取的覆盖前者
            for item in candidates:
                s = ClinicalUtils.stage_to_num(item.get("ajcc_pathologic_stage"))
                
                # 只有当发现更高分期，或者这是第一条记录时更新
                # (这里简化逻辑：只要能读到数据就更新，保留最高分期的那组数据)
                if s >= final_stage:
                    final_stage = s
                    final_t = ClinicalUtils.t_to_num(item.get("ajcc_pathologic_t"))
                    final_n = ClinicalUtils.n_to_num(item.get("ajcc_pathologic_n"))
                    final_m = ClinicalUtils.m_to_num(item.get("ajcc_pathologic_m"))
                    
                    # 提取 Residual Disease (直接在 diagnoses 下)
                    final_resid = ClinicalUtils.residual_to_num(item.get("residual_disease"))
                    
                    # 提取 pathology_details 中的侵犯信息和淋巴结数
                    # pathology_details 是一个列表
                    path_details = item.get("pathology_details", [])
                    if path_details and isinstance(path_details, list):
                        # 通常取第一个 detail，或者遍历查找
                        detail = path_details[0]
                        final_vasc = ClinicalUtils.yes_no_to_num(detail.get("vascular_invasion_present"))
                        final_lymph_inv = ClinicalUtils.yes_no_to_num(detail.get("lymphatic_invasion_present"))
                        final_peri = ClinicalUtils.yes_no_to_num(detail.get("perineural_invasion_present"))
                        
                        # 处理检测淋巴结数量
                        nodes_raw = detail.get("lymph_nodes_tested")
                        # 如果没有记录，默认为12(分期所需的最低标准)还是0？这里保守设为0
                        nodes_val = ClinicalUtils.to_f(nodes_raw) if nodes_raw is not None else 0.0
                        # 归一化：除以100，将其映射到 0~1 左右的范围 (例如 12 -> 0.12)
                        final_nodes = nodes_val / 100.0

            demo = case.get("demographic", {}) or {}
            age = ClinicalUtils.to_f(demo.get("age_at_diagnosis"))
            # 年龄归一化: 转换为岁数再除以100
            age = (age/365.25/100) if age > 0 else 0.6
            
            gender_str = ClinicalUtils.norm(demo.get("gender"))
            gender = 1.0 if "male" in gender_str else 0.0
            
            # 构建新的特征向量 (11维)
            # 顺序: [年龄, 性别, Stage, T, N, M, 脉管侵犯, 淋巴侵犯, 神经侵犯, 淋巴结总数, 切缘]
            new_vec = np.array([
                age, gender, final_stage, final_t, final_n, final_m,
                final_vasc, final_lymph_inv, final_peri, final_nodes, final_resid
            ], dtype=np.float32)
            
            # 简单的去重逻辑：如果已经存在该病人且当前记录Stage更高，则替换
            if pid in clinical_map:
                old_stage = clinical_map[pid][2]
                if final_stage > 0 and final_stage > old_stage:
                    clinical_map[pid] = new_vec
            else:
                clinical_map[pid] = new_vec
            
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        import traceback
        traceback.print_exc()
        return {}, 11

    valid_stage_patients = sum(1 for v in clinical_map.values() if v[2] > 0)
    print(f"最终加载 {len(clinical_map)} 个唯一病人")
    print(f"   -> 有效分期 (Stage > 0): {valid_stage_patients} 人")
    
    # 返回字典和新的特征维度 (11)
    return clinical_map, 11

# --- Backbone & FusionNet ---
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
    actual_k = indices.shape[1]
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, actual_k)
    adj[src, indices] = 1.0
    adj = torch.max(adj, adj.t())
    adj = adj + torch.eye(N, device=coords.device)
    degree = adj.sum(1).clamp(min=1e-6)
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
        if return_embedding: return h_slide
        return self.classifier(h_slide)

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.2))
        self.clin_fc = nn.Sequential(nn.Linear(11, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 1))
        
    def forward(self, img, clin, return_embedding=False):
        h_img = self.img_fc(img)
        h_clin = self.clin_fc(clin)
        combined = torch.cat([h_img, h_clin], 1)
        if return_embedding:
            return combined
        return self.head(combined)

def cox_loss(risk, t, e):
    if e.sum() == 0: return torch.tensor(0.0, requires_grad=True).to(risk.device)
    idx = t.argsort(descending=True)
    risk, e = risk[idx], e[idx]
    return -((risk - torch.log(torch.cumsum(torch.exp(risk), 0))) * e).sum() / (e.sum()+1e-6)

# ================= 🔍 验证模块：使用 Embedding 进行分期预测 =================
def verify_stage_predictability(embeddings, stages, output_path):
    print("\n [验证] 正在评估 Embedding 对 AJCC Stage 的预测能力...")
    
    mask = (stages >= 1) & (stages <= 4)
    X = embeddings[mask]
    y = stages[mask].astype(int)
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print("有效分期类别不足 2 类，跳过验证。")
        return

    print(f"   -> 用于验证的样本数: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = LogisticRegression(max_iter=3000, class_weight='balanced', multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n验证结果 (Accuracy: {acc:.4f})")
    print(report)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Stage {i}' for i in sorted(unique_labels)],
                yticklabels=[f'Stage {i}' for i in sorted(unique_labels)])
    plt.xlabel('Predicted Stage')
    plt.ylabel('True Stage')
    plt.title(f'(Acc: {acc:.2f})')
    plt.savefig(os.path.join(output_path, "stage_verification_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(output_path, "stage_verification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    
    print(f"验证完成，结果已保存至 {output_path}")

# ================= 混合可视化模块 (t-SNE)=================
def generate_visualization(model, dataloader, raw_clin_np, case_ids, device, output_path):
    print("\n开始生成可视化图表 (Risk/Event用t-SNE, Stage用UMAP)...")
    model.eval()
    
    all_risks = []
    all_events = []
    all_times = []
    all_embeddings = []
    
    with torch.no_grad():
        for img, clin, t, e in tqdm(dataloader, desc="Inference"):
            img, clin = img.to(device), clin.to(device)
            risk = model(img, clin, return_embedding=False)
            emb = model(img, clin, return_embedding=True)
            
            all_risks.extend(risk.cpu().numpy().flatten())
            all_events.extend(e.numpy().flatten())
            all_times.extend(t.numpy().flatten())
            all_embeddings.extend(emb.cpu().numpy())
            
    all_risks = np.array(all_risks)
    all_events = np.array(all_events)
    all_times = np.array(all_times)
    all_embeddings = np.array(all_embeddings)
    all_stages = raw_clin_np[:, 2] 
    
    # ========== 🆕 保存 CSV 文件 ==========
    print("   -> 正在保存 coad_msi_labels.csv...")
    
    # 提取临床特征（从 raw_clin_np 中）
    # raw_clin_np 的列顺序: [年龄, 性别, Stage, T, N, M, 脉管侵犯, 淋巴侵犯, 神经侵犯, 淋巴结总数, 切缘]
    clin_age = raw_clin_np[:, 0] * 100  # 恢复到原始年龄范围 (0-1 -> 0-100岁)
    clin_sex = raw_clin_np[:, 1]  # 0=女性, 1=男性
    clin_stage = raw_clin_np[:, 2]  # 1-4
    clin_t = raw_clin_np[:, 3]  # 0-4
    clin_n = raw_clin_np[:, 4]  # 0-3
    clin_m = raw_clin_np[:, 5]  # 0-1
    
    csv_data = {
        'Case_ID': case_ids,
        'Time': all_times,
        'Event': all_events.astype(int),
        'Risk_Score': all_risks,
        'Clin_Age': clin_age,
        'Clin_Sex': clin_sex,
        'Clin_Stage': clin_stage,
        'Clin_T': clin_t,
        'Clin_N': clin_n,
        'Clin_M': clin_m
    }
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_path, "coad_msi_labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV 文件已保存至: {csv_path} (包含 {len(df)} 条记录)")
    # =====================================
    
    # --- 1. KM Curve ---
    print("   -> 绘制 KM 曲线...")
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
    
    # --- 2. 混合降维可视化 ---
    print("   -> 计算降维坐标...")
    if len(all_embeddings) > 30:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 🔵 1 & 2: 使用 t-SNE
        print("      * Running t-SNE for Risk/Event...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        emb_tsne = tsne.fit_transform(all_embeddings)
        
        # Plot 1: Risk (t-SNE)
        sc1 = axes[0].scatter(emb_tsne[:,0], emb_tsne[:,1], c=all_risks, cmap='RdYlGn_r', s=50, alpha=0.7, edgecolors='k', linewidth=0.3)
        plt.colorbar(sc1, ax=axes[0], label='Risk Score')
        axes[0].set_title('(a) t-SNE: Colored by Risk', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        
        # Plot 2: Event (t-SNE)
        colors_evt = ['#3498db' if e==0 else '#e74c3c' for e in all_events]
        axes[1].scatter(emb_tsne[:,0], emb_tsne[:,1], c=colors_evt, s=50, alpha=0.7, edgecolors='k', linewidth=0.3)
        leg_evt = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', label='Censored', markersize=10),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Event', markersize=10)]
        axes[1].legend(handles=leg_evt, loc='best')
        axes[1].set_title('(b) t-SNE: Colored by Event', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "hybrid_viz_combined.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("   ⚠️ 样本过少，跳过降维可视化")
    
    # 3. 执行分期验证
    verify_stage_predictability(all_embeddings, all_stages, output_path)
    
    print(f"所有结果保存在: {output_path}")

# ================= 📦 数据处理与训练 =================
def extract_and_cache_features(backbone, clin_map):
    print("提取特征...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    cache_img, cache_clin, cache_t, cache_e, cache_ids = [], [], [], [], []
    
    with torch.no_grad():
        for fname in tqdm(files):
            pid = os.path.basename(fname)[:12]
            if pid not in clin_map: continue
            try:
                with np.load(fname) as d:
                    if 'feats' not in d or 'time' not in d or 'event' not in d: continue
                    feats = d['feats']
                    coords = d['coords'] if 'coords' in d else np.zeros((feats.shape[0], 2))
                    
                    if feats.shape[0] < MIN_PATCHES: continue
                    if feats.shape[0] > MAX_NODES:
                        idx = np.random.choice(feats.shape[0], MAX_NODES, replace=False)
                        feats = feats[idx]
                        coords = coords[idx]
                        
                    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)
                    
                    feats_t = torch.from_numpy(feats).float().to(DEVICE)
                    coords_t = torch.from_numpy(coords).float().to(DEVICE)
                    
                    try: img_emb = backbone(feats_t, coords_t, return_embedding=True).cpu()
                    except: continue
                    
                    cache_img.append(img_emb.numpy())
                    cache_clin.append(clin_map[pid])
                    cache_t.append(float(d['time']))
                    cache_e.append(float(d['event']))
                    cache_ids.append(pid)  # 🆕 保存 Case ID
            except: continue
            
    if len(cache_img) < 30: raise ValueError("样本过少")
    
    np.savez_compressed(CACHE_PATH, img_features=np.vstack(cache_img), 
                        clin_features=np.vstack(cache_clin), times=np.array(cache_t), 
                        events=np.array(cache_e), case_ids=np.array(cache_ids))  # 🆕 保存 IDs
    return cache_img, cache_clin, cache_t, cache_e, cache_ids

def train_fast_fusion():
    print("🚀 开始训练 (t-SNE/UMAP 混合模式)")
    
    backbone = DeepGraphTransformer(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, k=KNN_K).to(DEVICE)
    if os.path.exists(BACKBONE_WEIGHT):
        backbone.load_state_dict(torch.load(BACKBONE_WEIGHT, map_location=DEVICE), strict=False)
    backbone.eval()
    
    clin_map, _ = load_clinical_features(CLINICAL_JSON_PATH)
    
    if os.path.exists(CACHE_PATH):
        print(f"📦 加载缓存: {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)  # 🆕 添加 allow_pickle=True 以支持字符串数组
        cache_img = [data['img_features'][i:i+1] for i in range(len(data['img_features']))]
        cache_clin = [data['clin_features'][i] for i in range(len(data['clin_features']))]
        cache_t, cache_e = data['times'].tolist(), data['events'].tolist()
        cache_ids = data['case_ids'].tolist() if 'case_ids' in data else [f"UNKNOWN_{i}" for i in range(len(cache_t))]  # 🆕 加载 IDs
    else:
        cache_img, cache_clin, cache_t, cache_e, cache_ids = extract_and_cache_features(backbone, clin_map)
    
    raw_clin_np = np.vstack(cache_clin)
    
    X_clin_norm = torch.from_numpy(raw_clin_np.copy()).float()
    for i in range(X_clin_norm.shape[1]):
        std = X_clin_norm[:, i].std()
        if std > 1e-6: X_clin_norm[:, i] = (X_clin_norm[:, i] - X_clin_norm[:, i].mean()) / std
        
    X_img = torch.cat([torch.from_numpy(x) for x in cache_img], 0)
    Y_t = torch.tensor(cache_t).float()
    Y_e = torch.tensor(cache_e).float()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_c = 0
    best_model_state = None
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_img)):
        print(f"\nFold {fold+1}/5")
        train_ds = TensorDataset(X_img[train_idx], X_clin_norm[train_idx], Y_t[train_idx], Y_e[train_idx])
        test_ds = TensorDataset(X_img[test_idx], X_clin_norm[test_idx], Y_t[test_idx], Y_e[test_idx])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        
        model = FusionNet().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        best_fold_c = 0
        best_fold_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            for img, clin, t, e in train_loader:
                img, clin, t, e = img.to(DEVICE), clin.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
                optimizer.zero_grad()
                loss = cox_loss(model(img, clin).squeeze(), t, e)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                img, clin, t, e = next(iter(test_loader))
                img, clin = img.to(DEVICE), clin.to(DEVICE)
                pred = model(img, clin).squeeze()
                try: c_index = concordance_index(t.numpy(), -pred.cpu().numpy(), e.numpy())
                except: c_index = 0.5
                if c_index > best_fold_c:
                    best_fold_c = c_index
                    best_fold_state = model.state_dict().copy()
        
        print(f"  Best C-Index: {best_fold_c:.4f}")
        if best_fold_c > best_overall_c:
            best_overall_c = best_fold_c
            best_model_state = best_fold_state

    if best_model_state:
        torch.save(best_model_state, FUSION_WEIGHT_PATH)
        print(f"\n最佳模型 (C-Index: {best_overall_c:.4f}) 已保存")
        
        best_model = FusionNet().to(DEVICE)
        best_model.load_state_dict(best_model_state)
        
        full_ds = TensorDataset(X_img, X_clin_norm, Y_t, Y_e)
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        generate_visualization(best_model, full_loader, raw_clin_np, cache_ids, DEVICE, OUTPUT_DIR)
        
        log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
        with open(log_path, 'w') as f:
            f.write(f"Training Log\n")
            f.write(f"Best C-index: {best_overall_c:.4f}\n")
    
        print(f"训练日志已保存到: {log_path}")

if __name__ == "__main__":
    train_fast_fusion()