import pandas as pd
import os

# ================= 配置 =================
# 1. 你刚从 cBioPortal 下载的文件路径
DOWNLOADED_TSV = '/home/student2025/shirx2025/MUSK-surv/MSI_val/coadread_tcga_pan_can_atlas_2018_clinical_data.tsv'
# 2. 我们要保存的目标 CSV 文件名
OUTPUT_CSV = "/home/student2025/shirx2025/MUSK-surv/MSI_val/coad_msi_labels.csv"
# =======================================

def process_cbioportal_data():
    if not os.path.exists(DOWNLOADED_TSV):
        print(f"❌ 找不到文件: {DOWNLOADED_TSV}，请先去 cBioPortal 下载！")
        return

    # 读取 TSV (cBioPortal 默认是 Tab 分隔)
    # 注意：有时候文件前几行是注释，用 comment='#' 跳过
    try:
        df = pd.read_csv(DOWNLOADED_TSV, sep='\t', comment='#')
    except:
        # 如果报错，尝试直接读取
        df = pd.read_csv(DOWNLOADED_TSV, sep='\t')

    print(f"原始数据列名: {df.columns.tolist()}")

    # --- 寻找关键列 ---
    # cBioPortal 的列名可能会变，我们要智能查找
    id_col = None
    msi_col = None

    # 1. 找 ID 列 (通常叫 'Patient ID' 或 'Sample ID')
    for col in df.columns:
        if 'Patient ID' in col or 'Sample ID' in col:
            id_col = col
            break
    
    # 2. 找 MSI 列 (优先找 Subtype，其次找 MSI Status)
    # 在 PanCancer Atlas 中，MSI 状态通常在 'Subtype' 列，值像 'COAD_MSI', 'COAD_MSS'
    for col in df.columns:
        if 'Subtype' in col: 
            msi_col = col
            break
        if 'Microsatellite' in col: # 如果有显式的 MSI Status
            msi_col = col
            break

    if not id_col or not msi_col:
        print("❌ 无法自动找到 ID 或 MSI 列，请手动检查 TSV 文件列名！")
        return

    print(f"✅ 锁定列 -> ID: '{id_col}', MSI: '{msi_col}'")

    # --- 提取并清洗 ---
    cleaned_data = []
    
    for index, row in df.iterrows():
        pid = str(row[id_col])
        status_raw = str(row[msi_col]).upper()
        
        # 统一 ID 格式 (TCGA-XX-XXXX)
        # cBioPortal 有时是 TCGA-A6-2671-01，有时是 TCGA-A6-2671
        # 我们统一截取前 12 位
        pid_clean = pid[:12]
        
        # 统一 MSI 状态
        final_status = 'Unknown'
        if 'MSS' in status_raw or 'STABLE' in status_raw:
            final_status = 'MSS'
        elif 'MSI' in status_raw or 'HIGH' in status_raw: # MSI-H, COAD_MSI
            if 'LOW' in status_raw: # MSI-L 归为 MSS
                final_status = 'MSS'
            else:
                final_status = 'MSI-H'
        
        if final_status != 'Unknown':
            cleaned_data.append({'Case_ID': pid_clean, 'MSI_Status': final_status})

    # 保存
    out_df = pd.DataFrame(cleaned_data)
    # 去重 (防止同一个病人多条记录)
    out_df = out_df.drop_duplicates(subset=['Case_ID'])
    
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"🎉 处理完成！")
    print(f"   总行数: {len(out_df)}")
    print(f"   MSS 数量: {len(out_df[out_df['MSI_Status']=='MSS'])}")
    print(f"   MSI-H 数量: {len(out_df[out_df['MSI_Status']=='MSI-H'])}")
    print(f"📁 已保存至: {OUTPUT_CSV}")
    print("   -> 现在你可以运行之前的亚组分析代码了！")

if __name__ == "__main__":
    process_cbioportal_data()