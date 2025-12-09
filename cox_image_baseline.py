import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================
# 请确保路径正确
DATA_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"

# 根据您刚才的 print 结果确认的键名
KEY_FEATS = 'feats'
KEY_TIME  = 'time'
KEY_EVENT = 'event'
# ===========================================

def run_final_baseline(data_dir):
    print("="*50)
    print("启动 MUSK Baseline: 基于 .npz 文件的 Cox 生存分析")
    print("="*50)

    # --- 1. 数据加载 ---
    all_data = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    print(f"扫描到 {len(files)} 个文件，开始读取...")

    valid_count = 0
    empty_count = 0
    
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        try:
            with np.load(fpath) as data:
                # 1.1 获取特征
                if KEY_FEATS not in data: continue
                feats = data[KEY_FEATS] 
                
                # 1.2 关键修复：跳过空切片 (避免 Mean of empty slice 报错)
                if feats.shape[0] == 0:
                    empty_count += 1
                    continue 
                
                # 1.3 Mean Pooling (将 10738x1024 压缩为 1x1024)
                slide_emb = np.mean(feats, axis=0)
                
                # 1.4 获取时间与状态
                if KEY_TIME not in data or KEY_EVENT not in data:
                    continue
                
                t = data[KEY_TIME]
                e = data[KEY_EVENT]
                
                # 处理标量 (0-d array)
                if isinstance(t, np.ndarray): t = t.item()
                if isinstance(e, np.ndarray): e = e.item()
                
                # 1.5 存入列表
                # 结构: [feat_0, ..., feat_1023, time, event]
                row = list(slide_emb) + [t, e]
                all_data.append(row)
                valid_count += 1
                
        except Exception as err:
            print(f"读取出错 {fname}: {err}")

    print(f"\n读取完成报告:")
    print(f"  - 有效病例: {valid_count}")
    print(f"  - 跳过空文件: {empty_count}")

    if valid_count < 50:
        print("错误: 有效样本太少，无法进行 Cox 分析。")
        return

    # --- 2. 转换为 DataFrame ---
    # 自动生成特征列名
    feat_dim = len(all_data[0]) - 2
    col_names = [i for i in range(feat_dim)] + ['time', 'event']
    df = pd.DataFrame(all_data, columns=col_names)

    # --- 3. 数据清洗 ---
    # 3.1 去除 NaN
    df_clean = df.dropna()
    # 3.2 去除无效时间 (time <= 0)
    df_clean = df_clean[df_clean['time'] > 0]
    
    print(f"清洗后最终样本量 (N): {len(df_clean)}")

    # --- 4. 准备矩阵 ---
    X = df_clean[[i for i in range(feat_dim)]].values
    T = df_clean['time'].values
    E = df_clean['event'].values.astype(int) # 确保 Event 是整数

    # --- 5. PCA 降维 (关键步骤) ---
    # 将 1024 维降到 32 维，防止过拟合
    n_components = 32
    print(f"\n[PCA] 正在将特征维度从 {feat_dim} 降维至 {n_components} ...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  - PCA 解释方差比: {np.sum(pca.explained_variance_ratio_):.2%}")

    # --- 6. 5折交叉验证 ---
    print("\n[CV] 开始 5-Fold Cross Validation ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    c_indices = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_pca)):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        E_train, E_test = E[train_idx], E[test_idx]
        
        # 构建 DataFrame
        train_df = pd.DataFrame(X_train, columns=[f'PC{i}' for i in range(n_components)])
        train_df['T'] = T_train
        train_df['E'] = E_train
        
        test_df = pd.DataFrame(X_test, columns=[f'PC{i}' for i in range(n_components)])
        
        # 训练 Cox
        cph = CoxPHFitter(penalizer=0.1) # 加一点正则化
        try:
            cph.fit(train_df, duration_col='T', event_col='E')
            
            # 预测
            preds = cph.predict_partial_hazard(test_df)
            score = concordance_index(T_test, -preds, E_test)
            c_indices.append(score)
            print(f"  Fold {fold+1}: C-Index = {score:.4f}")
            
        except Exception as e:
            print(f"  Fold {fold+1} 失败: {e}")

    # --- 7. 最终结果 ---
    mean_c = np.mean(c_indices)
    std_c = np.std(c_indices)
    
    print("\n" + "="*50)
    print(f"MUSK 特征评估结果 (COAD):")
    print(f"C-Index: {mean_c:.4f} (+/- {std_c:.4f})")
    print("="*50)

    # 自动评价
    if mean_c > 0.62:
        print(">> 评价: 优秀 (Excellent)！特征质量很高。")
    elif mean_c > 0.58:
        print(">> 评价: 良好 (Good)。特征有效，适合做进一步融合。")
    elif mean_c > 0.50:
        print(">> 评价: 一般 (Fair)。包含少量信息，可能需要非线性模型(Attention)来提升。")
    else:
        print(">> 评价: 较差 (Random)。可能需要检查预处理或特征提取过程。")

if __name__ == "__main__":
    run_final_baseline(DATA_DIR)