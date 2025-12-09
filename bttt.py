import os
import pandas as pd
import numpy as np
import subprocess
import time
import glob
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 阈值：至少 20,000 个 Patch
MIN_PATCHES = 20000 

# 2. 路径配置
GDC_CLIENT_TOOL = "/home/student2025/shirx2025/MUSK-surv/TCGA_DATA/gdc-client/gdc-client"
MANIFEST_PATH = "/home/student2025/shirx2025/MUSK-surv/TCGA_DATA/gdc-client/COAD/gdc_manifest.COAD.txt"
NPZ_DIR = "/home/student2025/shirx2025/MUSK-surv/PATIENT_NPZ/COAD"
DOWNLOAD_DIR = "./SVS_HUGE_DOWNLOAD"
# ===============================================

def find_and_download_huge_case():
    # --- Step 1: 读取 Manifest ---
    print("📄 读取 Manifest...")
    try:
        df_manifest = pd.read_csv(MANIFEST_PATH, sep='\t')
    except Exception as e:
        return print(f"❌ Manifest 读取失败: {e}")

    # --- Step 2: 全局扫描 NPZ (寻找超级大样本) ---
    print(f"🔍 正在全盘扫描 NPZ，寻找 Patch >= {MIN_PATCHES} 的样本...")
    npz_files = [f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')]
    
    candidates = [] # 存储 (Patch数, CaseID, NPZ路径)
    
    for fname in tqdm(npz_files):
        fpath = os.path.join(NPZ_DIR, fname)
        try:
            with np.load(fpath) as d:
                if 'feats' not in d: continue
                n = d['feats'].shape[0]
                
                # 只有大于阈值才入选
                if n >= MIN_PATCHES:
                    case_id = fname[:12]
                    candidates.append({
                        'n_patches': n,
                        'case_id': case_id,
                        'path': fpath
                    })
        except: continue

    # --- Step 3: 排序与筛选 ---
    if not candidates:
        print(f"\n❌ 遗憾：没有找到任何 Patch 数超过 {MIN_PATCHES} 的文件。")
        print("建议：请尝试降低阈值 (例如 10000 或 5000)。")
        return

    # 按 Patch 数量从大到小排序
    candidates.sort(key=lambda x: x['n_patches'], reverse=True)
    
    print(f"\n🎉 找到了 {len(candidates)} 个超级样本！")
    print("🏆 Top 3 候选人:")
    for i, c in enumerate(candidates[:3]):
        print(f"   {i+1}. {c['case_id']} | Patches: {c['n_patches']}")

    # --- Step 4: 匹配 Manifest 并下载 ---
    target_info = None
    
    # 依次尝试下载 (从最大的开始，万一最大的那个 Manifest 里没有，就试下一个)
    for cand in candidates:
        case_id = cand['case_id']
        matched = df_manifest[df_manifest['filename'].str.contains(case_id)]
        
        if len(matched) > 0:
            row = matched.iloc[0]
            target_info = {
                'uuid': row['id'],
                'filename': row['filename'],
                'case_id': case_id,
                'patches': cand['n_patches']
            }
            break # 找到了最大的且可下载的，停止
    
    if not target_info:
        print("❌ 所有的候选文件在 Manifest 中都找不到对应的 SVS 下载链接。")
        return

    # --- Step 5: 开始下载 ---
    print("\n" + "="*50)
    print(f"🚀 锁定最终目标 (Patch王): {target_info['case_id']}")
    print(f"💎 Patch 数量: {target_info['patches']}")
    print(f"📂 文件名: {target_info['filename']}")
    print("="*50)
    
    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)
    
    uuid = target_info['uuid']
    filename = target_info['filename']
    final_path = os.path.join(DOWNLOAD_DIR, uuid, filename)
    
    print(f"\n⬇️ 启动下载器 (目标可能会很大，请耐心等待)...")
    
    attempt = 1
    while True:
        if os.path.exists(final_path):
            partials = glob.glob(os.path.join(DOWNLOAD_DIR, uuid, "*.partial"))
            if not partials:
                size_mb = os.path.getsize(final_path) / (1024*1024)
                print(f"\n✅ 下载完成！")
                print(f"📦 文件大小: {size_mb:.2f} MB")
                print(f"📂 路径: {os.path.abspath(final_path)}")
                print(f"🆔 ID: {target_info['case_id']}")
                break
            else:
                print(f"⚠️ 下载未完成，继续第 {attempt} 次重试...")
        
        # 调用 gdc-client
        cmd = [GDC_CLIENT_TOOL, "download", uuid, "-d", DOWNLOAD_DIR]
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Error: {e}")
            
        time.sleep(3)
        attempt += 1

if __name__ == "__main__":
    find_and_download_huge_case()