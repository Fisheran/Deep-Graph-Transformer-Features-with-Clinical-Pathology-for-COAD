"""
Tile Selection Visualization Test
检验组织筛选结果的可视化工具
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import openslide
import cv2
import encoder_lib as enl

# 假设这些是你的函数，根据实际情况导入
# from your_module import tiles_from_wsi, tissue_mask, pick_level_for_mag


def visualize_tile_selection(slide_path, coords, target_lv, tile_size=256, 
                              save_path=None, max_display_tiles=500):
    """
    可视化筛选出的 tiles 在 WSI 上的位置
    
    Parameters:
        slide_path: WSI 文件路径
        coords: 筛选出的坐标 (Level 0 坐标)
        target_lv: 目标层级
        tile_size: tile 大小
        save_path: 保存路径（可选）
        max_display_tiles: 最多显示的 tile 数量
    """
    slide = openslide.OpenSlide(slide_path)
    
    # 获取缩略图层级
    thumb_lv = slide.level_count - 1
    thumb_ds = slide.level_downsamples[thumb_lv]
    w_thumb, h_thumb = slide.level_dimensions[thumb_lv]
    
    # 读取缩略图
    thumbnail = slide.read_region((0, 0), thumb_lv, (w_thumb, h_thumb)).convert("RGB")
    thumbnail = np.array(thumbnail)
    
    # 计算 tile 在缩略图上的大小
    target_ds = slide.level_downsamples[target_lv]
    tile_on_thumb = int(tile_size * target_ds / thumb_ds)
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === 图1: 原始缩略图 ===
    axes[0].imshow(thumbnail)
    axes[0].set_title(f"Original Thumbnail\nLevel {thumb_lv}, Size: {w_thumb}x{h_thumb}")
    axes[0].axis('off')
    
    # === 图2: 组织 mask ===
    mask = enl.tissue_mask(thumbnail)
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"Tissue Mask\nWhite = Tissue")
    axes[1].axis('off')
    
    # === 图3: 筛选的 tiles 叠加在缩略图上 ===
    axes[2].imshow(thumbnail)
    
    # 绘制选中的 tiles
    n_display = min(len(coords), max_display_tiles)
    for i in range(n_display):
        x_L0, y_L0 = coords[i]
        # 转换到缩略图坐标
        x_thumb = int(x_L0 / thumb_ds)
        y_thumb = int(y_L0 / thumb_ds)
        
        rect = patches.Rectangle(
            (x_thumb, y_thumb), tile_on_thumb, tile_on_thumb,
            linewidth=0.5, edgecolor='lime', facecolor='lime', alpha=0.3
        )
        axes[2].add_patch(rect)
    
    axes[2].set_title(f"Selected Tiles\n{len(coords)} tiles (showing {n_display})")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()
    slide.close()


def visualize_sample_tiles(slide_path, coords, target_lv, tile_size=256,
                           n_samples=16, save_path=None):
    """
    随机展示一些筛选出的 tiles
    
    Parameters:
        slide_path: WSI 文件路径
        coords: 筛选出的坐标
        target_lv: 目标层级
        tile_size: tile 大小
        n_samples: 展示数量
        save_path: 保存路径
    """
    slide = openslide.OpenSlide(slide_path)
    
    # 随机选择 tiles
    n_samples = min(n_samples, len(coords))
    indices = np.random.choice(len(coords), n_samples, replace=False)
    
    # 计算网格布局
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        x, y = coords[idx]
        tile = slide.read_region((x, y), target_lv, (tile_size, tile_size)).convert("RGB")
        
        axes[i].imshow(tile)
        axes[i].set_title(f"#{idx}\n({x}, {y})", fontsize=8)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Sample Tiles from {os.path.basename(slide_path)}", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()
    slide.close()


def compute_tile_statistics(slide_path, coords, target_lv, tile_size=256, n_samples=100):
    """
    计算筛选出的 tiles 的统计信息
    
    Parameters:
        slide_path: WSI 文件路径
        coords: 筛选出的坐标
        target_lv: 目标层级
        tile_size: tile 大小
        n_samples: 采样数量（用于计算组织比例）
    """
    slide = openslide.OpenSlide(slide_path)
    
    print("=" * 50)
    print(f"Slide: {os.path.basename(slide_path)}")
    print("=" * 50)
    
    # 基本信息
    print(f"\n[Basic Info]")
    print(f"  Total tiles selected: {len(coords)}")
    print(f"  Target level: {target_lv}")
    print(f"  Tile size: {tile_size}x{tile_size}")
    
    # 坐标范围
    if len(coords) > 0:
        coords = np.array(coords)
        print(f"\n[Coordinate Range]")
        print(f"  X: {coords[:, 0].min()} - {coords[:, 0].max()}")
        print(f"  Y: {coords[:, 1].min()} - {coords[:, 1].max()}")
    
    # 采样计算组织比例
    if len(coords) > 0:
        n_samples = min(n_samples, len(coords))
        indices = np.random.choice(len(coords), n_samples, replace=False)
        
        tissue_ratios = []
        for idx in indices:
            x, y = coords[idx]
            tile = np.array(slide.read_region((x, y), target_lv, (tile_size, tile_size)).convert("RGB"))
            
            # 计算组织比例
            hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
            ratio = (hsv[:, :, 1] > 20).mean()
            tissue_ratios.append(ratio)
        
        tissue_ratios = np.array(tissue_ratios)
        print(f"\n[Tissue Ratio Statistics] (sampled {n_samples} tiles)")
        print(f"  Mean:   {tissue_ratios.mean():.2%}")
        print(f"  Std:    {tissue_ratios.std():.2%}")
        print(f"  Min:    {tissue_ratios.min():.2%}")
        print(f"  Max:    {tissue_ratios.max():.2%}")
        print(f"  Median: {np.median(tissue_ratios):.2%}")
    
    print("=" * 50)
    slide.close()
    
    return tissue_ratios if len(coords) > 0 else np.array([])


def visualize_tissue_ratio_distribution(slide_path, coords, target_lv, 
                                         tile_size=256, n_samples=200, save_path=None):
    """
    可视化 tiles 的组织比例分布
    """
    slide = openslide.OpenSlide(slide_path)
    
    n_samples = min(n_samples, len(coords))
    indices = np.random.choice(len(coords), n_samples, replace=False)
    
    tissue_ratios = []
    for idx in indices:
        x, y = coords[idx]
        tile = np.array(slide.read_region((x, y), target_lv, (tile_size, tile_size)).convert("RGB"))
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        ratio = (hsv[:, :, 1] > 20).mean()
        tissue_ratios.append(ratio)
    
    slide.close()
    
    # 绘制直方图
    plt.figure(figsize=(10, 5))
    plt.hist(tissue_ratios, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(tissue_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(tissue_ratios):.2%}')
    plt.xlabel('Tissue Ratio')
    plt.ylabel('Count')
    plt.title(f'Tissue Ratio Distribution\n{os.path.basename(slide_path)} (n={n_samples})')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


# ============================================================
# 主测试函数
# ============================================================

def test_tile_selection(slide_path, output_dir="./test_output"):
    """
    完整的筛选结果测试
    
    Parameters:
        slide_path: WSI 文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    
    print(f"\n{'='*60}")
    print(f"Testing: {slide_name}")
    print(f"{'='*60}\n")
    
    # 1. 运行筛选
    print("[1/5] Running tile selection...")
    coords, target_lv = enl.tiles_from_wsi(
        slide_path, 
        target_mag=20,      # 根据你的设置调整
        tile=256, 
        step=256, 
        min_tissue=0.5
    )
    print(f"      Found {len(coords)} tiles\n")
    
    if len(coords) == 0:
        print("No tiles found! Check your parameters.")
        return
    
    # 2. 统计信息
    print("[2/5] Computing statistics...")
    tissue_ratios = compute_tile_statistics(slide_path, coords, target_lv)
    
    # 3. 可视化 tile 位置
    print("\n[3/5] Visualizing tile locations...")
    visualize_tile_selection(
        slide_path, coords, target_lv,
        save_path=os.path.join(output_dir, f"{slide_name}_tile_locations.png")
    )
    
    # 4. 展示样本 tiles
    print("\n[4/5] Showing sample tiles...")
    visualize_sample_tiles(
        slide_path, coords, target_lv,
        n_samples=25,
        save_path=os.path.join(output_dir, f"{slide_name}_sample_tiles.png")
    )
    
    # 5. 组织比例分布
    print("\n[5/5] Plotting tissue ratio distribution...")
    visualize_tissue_ratio_distribution(
        slide_path, coords, target_lv,
        save_path=os.path.join(output_dir, f"{slide_name}_tissue_distribution.png")
    )
    
    print(f"\n✅ Test complete! Results saved to: {output_dir}")


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    # 替换为你的 WSI 文件路径
    SLIDE_PATH = "/home/student2025/shirx2025/MUSK-surv/TCGA_DATA/TCGA_COAD/1b2d9c2e-5f7e-44a7-bef5-9c6ffda97429/TCGA-CM-4744-01Z-00-DX1.527ead53-bd55-4321-adea-079bf5e2e8a5.svs"
    OUTPUT_DIR = "./tile_selection_test"
    
    # 确保导入你的函数
    # from your_preprocessing import tiles_from_wsi, tissue_mask
    
    test_tile_selection(SLIDE_PATH, OUTPUT_DIR)