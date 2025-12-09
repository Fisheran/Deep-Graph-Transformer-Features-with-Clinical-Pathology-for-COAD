import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= 配置 =================
DATA_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"
SAVE_PATH = "musk_coad_tsne.png"  # 图片保存路径
# =======================================

def plot_feature_distribution(data_dir, save_path):
    print(f"正在读取数据以进行可视化: {data_dir}")
    
    # --- 1. 读取数据 ---
    features = []
    stages = []
    statuses = []
    ids = []

    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    for fname in files:
        try:
            with np.load(os.path.join(data_dir, fname)) as data:
                # 读取特征
                if 'feats' not in data: continue
                feats = data['feats']
                if feats.shape[0] == 0: continue
                
                # Mean Pooling
                slide_emb = np.mean(feats, axis=0)
                
                # 读取标签 (分期 和 状态)
                # 注意：如果某个key不存在，给一个默认值 -1
                stage = data['cov_ajcc_stage_num'].item() if 'cov_ajcc_stage_num' in data else -1
                status = data['event'].item() if 'event' in data else -1
                
                features.append(slide_emb)
                stages.append(stage)
                statuses.append(status)
                ids.append(fname)
                
        except Exception:
            continue

    # 转为 Numpy 数组
    X = np.array(features)
    y_stage = np.array(stages)
    y_status = np.array(statuses)
    
    print(f"有效样本数: {len(X)}")
    print("正在进行 t-SNE 降维 (可能需要几秒钟)...")

    # --- 2. 数据预处理与降维 ---
    # 先做标准化
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # 先用 PCA 降到 50 维 (t-SNE 的推荐预处理步骤)
    pca = PCA(n_components=min(50, len(X)))
    X_pca = pca.fit_transform(X_norm)
    
    # 再用 t-SNE 降到 2 维
    # perplexity 参数通常设为 30，但如果样本少于 30，需要调小
    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca)

    # --- 3. 开始绘图 ---
    plt.figure(figsize=(16, 7))

    # === 子图 1: 按分期 (Stage) 着色 ===
    plt.subplot(1, 2, 1)
    # 过滤掉没有分期数据的点 (值为 -1)
    mask_stage = y_stage != -1
    
    # 使用 Seaborn 绘制散点图
    scatter = sns.scatterplot(
        x=X_embedded[mask_stage, 0], 
        y=X_embedded[mask_stage, 1], 
        hue=y_stage[mask_stage], 
        palette="viridis", # 颜色风格
        style=y_stage[mask_stage], # 不同分期用不同形状
        s=80, # 点的大小
        alpha=0.8
    )
    plt.title(f"MUSK Features by Tumor Stage (N={sum(mask_stage)})", fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Stage (Num)", loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)

    # === 子图 2: 按生存状态 (Status) 着色 ===
    plt.subplot(1, 2, 2)
    mask_status = y_status != -1
    
    # 定义颜色: 0(Alive)=蓝色, 1(Dead)=红色
    custom_palette = {0: "#1f77b4", 1: "#d62728"}
    
    sns.scatterplot(
        x=X_embedded[mask_status, 0], 
        y=X_embedded[mask_status, 1], 
        hue=y_status[mask_status], 
        palette=custom_palette,
        s=80, 
        alpha=0.7
    )
    plt.title(f"MUSK Features by Survival Status (N={sum(mask_status)})", fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    # 手动设置 Legend 标签
    plt.legend(title="Status", labels=["Alive", "Dead"])
    plt.grid(True, linestyle='--', alpha=0.3)

    # --- 保存与展示 ---
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n图片已保存至: {os.path.abspath(save_path)}")
    print("提示: 这是一个 PNG 图片文件，请在您的文件管理器中打开查看。")
    
    # 如果是在 Jupyter 里，下面这行会直接显示
    plt.show()

if __name__ == "__main__":
    plot_feature_distribution(DATA_DIR, SAVE_PATH)