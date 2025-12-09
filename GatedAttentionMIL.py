import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.cuda.amp as amp
import warnings

warnings.filterwarnings("ignore")

# ================= 🚀 配置区域 =================
DATA_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"
OUTPUT_DIR = "./results_viz"  # 图片保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数
BATCH_SIZE = 1        # MIL 必须为 1
EPOCHS = 20           # 训练轮数
LR = 2e-4             # 学习率
WEIGHT_DECAY = 1e-4
IN_DIM = 1024         
MIN_PATCHES = 1000    # 筛选阈值

# 梯度累积步数
ACCUMULATION_STEPS = 32 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ===============================================

# --- 1. 定义网络 (Gated Attention MIL) ---
class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super(GatedAttentionMIL, self).__init__()
        self.L = hidden_dim
        self.D = in_dim
        self.K = 1 

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.ReLU()
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.L, self.K)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.K)
        )

    # 修改 forward 以适配绘图代码的接口 model(img, clin, return_embedding)
    def forward(self, x, clin=None, return_embedding=False):
        x = x.squeeze(0) 
        H = self.feature_extractor(x) 
        A_V = self.attention_V(H) 
        A_U = self.attention_U(H) 
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0) 
        A = torch.softmax(A, dim=1) 
        M = torch.mm(A, H)  # [1, L] WSI Level Embedding
        
        if return_embedding:
            return M  # 返回特征向量用于 t-SNE
            
        risk = self.classifier(M)
        return risk

# --- 2. 内存数据集 ---
class InMemoryDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 返回: feats, clin(dummy), time, event
        # 你的绘图代码需要 clin，这里即使没有也要返回一个占位符
        return item['feats'], item['clin'], item['time'], item['event']

def cox_loss_func(survtime, censor, hazard_pred):
    if censor.sum() == 0: return torch.tensor(0.0).to(survtime.device)
    
    idx = survtime.argsort(descending=True)
    survtime = survtime[idx]
    censor = censor[idx]
    hazard_pred = hazard_pred[idx]

    risk = torch.exp(hazard_pred)
    risk_cumsum = torch.cumsum(risk, dim=0)
    log_risk = torch.log(risk_cumsum)
    
    uncensored_likelihood = hazard_pred - log_risk
    loss = -uncensored_likelihood[censor.bool()].sum() / (censor.sum() + 1e-8)
    return loss

# --- 3. 补充：分期预测验证函数 (用于绘图代码调用) ---
def verify_stage_predictability(embeddings, stages, output_path):
    """
    使用简单的逻辑回归验证提取的特征是否包含分期信息
    """
    print("      * Verifying Stage Predictability (Logistic Regression)...")
    
    # 过滤掉无效分期 (假设 -1 是无效值)
    valid_idx = np.where(stages != -1)[0]
    if len(valid_idx) < 20:
        print("      ⚠️ 有效分期数据过少，跳过分期验证。")
        return

    X = embeddings[valid_idx]
    y = stages[valid_idx]

    # 简单划分
    split = int(len(X) * 0.7)
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    if len(np.unique(y)) < 2:
        print("      ⚠️ 分期类别单一，无法进行分类验证。")
        return

    clf = LogisticRegression(solver='liblinear', max_iter=100)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], preds)
    
    print(f"      ✅ Stage Prediction Accuracy: {acc:.4f}")
    
    # 保存结果到文本
    with open(os.path.join(output_path, "stage_prediction_log.txt"), "w") as f:
        f.write(f"Stage Prediction Accuracy: {acc:.4f}\n")


# --- 4. 你的绘图代码 (完全保持风格) ---
def generate_visualization(model, dataloader, raw_clin_np, device, output_path):
    print("\n🎨 开始生成可视化图表 (Risk/Event用t-SNE, Stage用UMAP)...")
    model.eval()
    
    all_risks = []
    all_events = []
    all_times = []
    all_embeddings = []
    
    # 注意：这里 dataloader 返回的是 (img, clin, t, e)
    with torch.no_grad():
        for img, clin, t, e in tqdm(dataloader, desc="Inference"):
            img, clin = img.to(device), clin.to(device)
            # 兼容模型 forward
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
    
    # 这里的 raw_clin_np 是所有样本的临床数据，我们假设第2列是 stage
    # 如果没有加载真实的 stage，这里将全是默认值
    all_stages = raw_clin_np[:, 2] 
    
    # --- 1. KM Curve ---
    print("   -> 绘制 KM 曲线...")
    median_risk = np.median(all_risks)
    high_risk = all_risks >= median_risk
    low_risk = all_risks < median_risk
    
    plt.figure(figsize=(10, 6))
    kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
    
    # 避免某一类为空
    if np.sum(high_risk) > 0:
        kmf_h.fit(all_times[high_risk], all_events[high_risk], label='High Risk')
        kmf_h.plot_survival_function(color='#e74c3c', ci_show=True)
    if np.sum(low_risk) > 0:
        kmf_l.fit(all_times[low_risk], all_events[low_risk], label='Low Risk')
        kmf_l.plot_survival_function(color='#2ecc71', ci_show=True)
        
    try:
        res = logrank_test(all_times[high_risk], all_times[low_risk], all_events[high_risk], all_events[low_risk])
        p_val_text = f'p={res.p_value:.4e}'
    except:
        p_val_text = 'p=N/A'
        
    plt.text(0.05, 0.05, p_val_text, 
             fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)

    plt.title("Kaplan-Meier Survival Curve", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
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
    
    print(f"✅ 所有结果保存在: {output_path}")

# --- 5. 数据预加载 ---
def preload_data(data_dir, min_patches):
    print(f"🚀 [1/3] 正在将数据预加载到内存 (RAM)...")
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    loaded_data = []
    dropped = 0
    
    for fpath in tqdm(files, desc="Loading .npz", unit="file"):
        try:
            with np.load(fpath) as data:
                if 'feats' in data:
                    feats = data['feats']
                    if feats.shape[0] < min_patches:
                        dropped += 1
                        continue
                        
                    time = data['time'].item()
                    event = data['event'].item()
                    
                    # 尝试加载 stage，如果不存在则给默认值 -1 (用于绘图代码)
                    if 'stage' in data:
                        stage = data['stage'].item()
                    else:
                        stage = -1 # 未知分期
                    
                    # 构造一个 dummy clin 向量 [dummy, dummy, stage, ...]
                    # 确保 raw_clin_np[:, 2] 能取到 stage
                    clin_dummy = torch.tensor([0, 0, stage, 0]).float()

                    feats_t = torch.from_numpy(feats).float()
                    # 保持 shape 和 device 一致
                    time_t = torch.tensor(time).float()
                    event_t = torch.tensor(event).float()
                    
                    loaded_data.append({
                        'feats': feats_t, 
                        'clin': clin_dummy,
                        'time': time_t,
                        'event': event_t,
                        'path': fpath
                    })
        except Exception as e:
            dropped += 1
            
    print(f"✅ 数据加载完成! 有效样本: {len(loaded_data)} | 丢弃样本: {dropped}")
    return loaded_data

# --- 6. 极速训练主程序 ---
def run_fast_training(data_dir):
    # 1. 预加载
    all_data = preload_data(data_dir, MIN_PATCHES)
    if len(all_data) < 20:
        print("❌ 样本太少，无法训练。")
        return

    # 2. KFold 准备
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    c_indices = []
    
    # 用于收集所有 fold 的验证集结果以进行绘图
    aggregated_test_data = [] 
    aggregated_clin_np = []

    print(f"\n🚀 [2/3] 开始训练 (Device: {DEVICE}, AMP: On)")
    scaler = amp.GradScaler()

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\n🔹 === Fold {fold+1} / 5 ===")
        
        train_subset = [all_data[i] for i in train_idx]
        test_subset = [all_data[i] for i in test_idx]
        
        train_loader = DataLoader(InMemoryDataset(train_subset), batch_size=1, shuffle=True)
        test_loader = DataLoader(InMemoryDataset(test_subset), batch_size=1, shuffle=False)
        
        model = GatedAttentionMIL(in_dim=IN_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # --- Training ---
        epoch_bar = tqdm(range(EPOCHS), desc=f"Fold {fold+1} Training", leave=False)
        for epoch in epoch_bar:
            model.train()
            optimizer.zero_grad()
            batch_loss = 0
            
            # Gradient Accumulation Buffers
            step_preds, step_times, step_events = [], [], []
            
            for i, (feats, clin, t, e) in enumerate(train_loader):
                feats = feats.to(DEVICE)
                t = t.to(DEVICE)
                e = e.to(DEVICE)

                with amp.autocast():
                    pred = model(feats) # clin is ignored in forward
                
                step_preds.append(pred)
                step_times.append(t)
                step_events.append(e)
                
                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    p_stack = torch.cat(step_preds).squeeze()
                    t_stack = torch.stack(step_times).squeeze()
                    e_stack = torch.stack(step_events).squeeze()
                    
                    if p_stack.ndim == 0: p_stack = p_stack.unsqueeze(0)
                    if t_stack.ndim == 0: t_stack = t_stack.unsqueeze(0)
                    if e_stack.ndim == 0: e_stack = e_stack.unsqueeze(0)

                    if e_stack.sum() > 0:
                        loss = cox_loss_func(t_stack, e_stack, p_stack)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        batch_loss = loss.item()
                    
                    step_preds, step_times, step_events = [], [], []
            
            epoch_bar.set_postfix({'loss': f"{batch_loss:.4f}"})

        # --- Evaluation ---
        model.eval()
        test_preds, test_times, test_events = [], [], []
        
        # 我们需要在最后一次fold或者汇总所有fold时绘图
        # 这里我们将测试集数据加入汇总列表
        aggregated_test_data.extend(test_subset)
        
        with torch.no_grad():
            for feats, clin, t, e in test_loader:
                feats = feats.to(DEVICE)
                pred = model(feats)
                test_preds.append(pred.item())
                test_times.append(t.item())
                test_events.append(e.item())
        
        try:
            score = concordance_index(test_times, -np.array(test_preds), test_events)
            print(f"  🏆 Fold {fold+1} C-Index: {score:.4f}")
            c_indices.append(score)
        except:
            print(f"  ⚠️ Fold {fold+1} C-Index 计算失败")

    # --- 最终结果报告 ---
    print("\n" + "="*50)
    print(f"📊 最终结果报告 (Pure Attention MIL)")
    if len(c_indices) > 0:
        mean_c = np.mean(c_indices)
        print(f"Mean C-Index: {mean_c:.4f} (+/- {np.std(c_indices):.4f})")
    
    # --- 统一绘图 (使用所有Fold的验证集数据) ---
    print("\n🚀 [3/3] 生成最终可视化报告 (Aggregated Validation Results)")
    
    # 构建全局验证集的DataLoader
    final_viz_loader = DataLoader(InMemoryDataset(aggregated_test_data), batch_size=1, shuffle=False)
    
    # 构建 raw_clin_np 用于绘图代码的切片操作
    # 从 aggregated_test_data 中提取 clin
    clin_list = [item['clin'].numpy() for item in aggregated_test_data]
    raw_clin_np = np.stack(clin_list) # Shape: [N, 4]
    
    # 使用最后一个 fold 的模型进行特征提取 (或者你可以选择保存最好的模型)
    # 为了绘图连贯性，这里直接复用当前内存中的 model
    generate_visualization(model, final_viz_loader, raw_clin_np, DEVICE, OUTPUT_DIR)
    
    print("="*50)

if __name__ == "__main__":
    print(f"当前使用的计算设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    run_fast_training(DATA_DIR)