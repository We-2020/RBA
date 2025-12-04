'''
Author: snxy 1113885219@qq.com
Date: 2025-11-27 16:20:35
LastEditors: snxy 1113885219@qq.com
LastEditTime: 2025-12-02 21:55:27
FilePath: /caojiaxiang/brain age/RBA/check_mri_mask_alignment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import lmdb
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch

def check_alignment(lmdb_path, mask_path, shape=(91, 109, 91)):
    # 1. 读取 Mask
    print(f"Loading Mask from: {mask_path}")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    
    # 检查 Mask 维度
    if mask_data.shape != shape:
        print(f"Warning: Mask shape {mask_data.shape} does not match expected {shape}!")
        # 如果维度顺序反了，可能需要 transpose，例如 (91, 109, 91) vs (91, 91, 109)
    
    # 2. 读取 LMDB 中的一个样本并获取统计信息
    print(f"Loading one sample from LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin(write=False) as txn:
        # === 新增逻辑：获取 LMDB 中的样本数量 ===
        try:
            stats = txn.stat()
            total_entries = stats['entries']
            print(f"==========================================")
            print(f"✅ LMDB Total Sample Count: {total_entries}")
            print(f"==========================================")
        except Exception as e:
            print(f"Error getting LMDB statistics: {e}")
            total_entries = 0
            
        if total_entries == 0:
             print("LMDB is empty!")
             return

        # 获取第一个 key
        cursor = txn.cursor()
        if not cursor.next():
            print("LMDB is empty!")
            return
        key, value = cursor.item()
        
        # 解码数据
        img_flat = np.frombuffer(value, dtype=np.float32)
        mri_data = img_flat.reshape(shape)
        
    print(f"Loaded Key: {key.decode()}") # Key 通常是 bytes，需要解码
    print(f"MRI Data Range: {mri_data.min():.2f} to {mri_data.max():.2f}")

    # 3. 可视化叠加 (三个切面)
    # 我们取中间的切片进行观察
    x_mid, y_mid, z_mid = shape[0]//2, shape[1]//2, shape[2]//2

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # --- Axial View (水平面) ---
    # 背景用灰度 MRI
    axes[0].imshow(np.rot90(mri_data[:, :, z_mid]), cmap='gray')
    # 前景用 Mask (半透明)
    mask_slice = mask_data[:, :, z_mid]
    # 使用 np.ma.masked_where 隐藏背景0值，只显示脑区
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[0].imshow(np.rot90(masked_overlay), cmap='jet', alpha=0.5)
    axes[0].set_title(f"Axial Slice (z={z_mid}) - Check L/R & A/P")

    # --- Coronal View (冠状面) ---
    axes[1].imshow(np.rot90(mri_data[:, y_mid, :]), cmap='gray')
    mask_slice = mask_data[:, y_mid, :]
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[1].imshow(np.rot90(masked_overlay), cmap='jet', alpha=0.5)
    axes[1].set_title(f"Coronal Slice (y={y_mid}) - Check Superior/Inferior")

    # --- Sagittal View (矢状面) ---
    axes[2].imshow(np.rot90(mri_data[x_mid, :, :]), cmap='gray')
    mask_slice = mask_data[x_mid, :, :]
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[2].imshow(np.rot90(masked_overlay), cmap='jet', alpha=0.5)
    axes[2].set_title(f"Sagittal Slice (x={x_mid})")

    plt.tight_layout()
    plt.show() # 如果在 Jupyter 里运行
    plt.savefig("alignment_check.png") # 如果在服务器运行，保存图片查看
    
    # 3. 关闭环境
    env.close()

# 配置路径运行
lmdb_path = "/data/MRIlmdb/ba" # 修改为你的 LMDB 路径
mask_path = "/home/caojiaxiang/brain age/Third/ROI_MNI_V4.nii"

# 运行检查
if __name__ == '__main__':
    check_alignment(lmdb_path, mask_path)