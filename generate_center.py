'''
Author: snxy 1113885219@qq.com
Date: 2025-12-04 16:50:42
LastEditors: snxy 1113885219@qq.com
LastEditTime: 2025-12-04 16:51:03
FilePath: /caojiaxiang/brain age/RBA/generate_center.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import nibabel as nib
import numpy as np
import torch
import os

def generate_centroid_tensor(nii_path, region_labels, save_path=None):
    """
    计算 NIfTI 文件中指定 label 的归一化质心坐标
    """
    print(f"Loading NIfTI file from: {nii_path}")
    try:
        img = nib.load(nii_path)
        data = img.get_fdata() # data shape: (91, 109, 91) typically
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    dims = np.array(data.shape)
    print(f"Image dimensions: {dims}")
    
    centroids = []
    valid_labels_count = 0
    
    print("Calculating centroids...")
    
    for label in region_labels:
        # 1. 找到该 label 所有体素的索引 (indices)
        # result is (N, 3) where N is number of voxels
        indices = np.argwhere(data == label)
        
        if len(indices) > 0:
            # 2. 计算几何中心 (mean)
            centroid_voxel = indices.mean(axis=0)
            
            # 3. 归一化到 [0, 1] 范围
            # 这样无论特征图被下采样多少倍，相对位置保持不变
            normalized_centroid = centroid_voxel / dims
            
            centroids.append(normalized_centroid)
            valid_labels_count += 1
        else:
            # 如果某个 Label 在图中不存在 (例如 Label 0 有时代表背景，如果背景全被裁掉了可能就没有)
            # 或者图谱中缺失该区域
            print(f"[Warning] Label {label} not found in image data. Using center [0.5, 0.5, 0.5].")
            centroids.append([0.5, 0.5, 0.5])

    # 转换为 PyTorch Tensor
    # Shape: [num_regions, 3] -> (x, y, z)
    centroids_tensor = torch.tensor(np.array(centroids), dtype=torch.float32)
    
    print(f"Finished. Generated tensor shape: {centroids_tensor.shape}")
    
    if save_path:
        torch.save(centroids_tensor, save_path)
        print(f"Tensor saved to: {save_path}")
        
    return centroids_tensor

if __name__ == '__main__':
    # 1. 配置路径
    nii_file_path = "/home/caojiaxiang/brain age/Third/ROI_MNI_V4.nii"
    output_save_path = "/home/caojiaxiang/brain age/RBA/pths/region_centroids.pt"
    
    # 2. 定义 Labels (你提供的列表)
    region_labels = [
        0, 2001, 2002, 2101, 2102, 2111, 2112, 2201, 2202, 2211, 2212, 2301,
        2302, 2311, 2312, 2321, 2322, 2331, 2332, 2401, 2402, 2501, 2502, 2601,
        2602, 2611, 2612, 2701, 2702, 3001, 3002, 4001, 4002, 4011, 4012, 4021,
        4022, 4101, 4102, 4111, 4112, 4201, 4202, 5001, 5002, 5011, 5012, 5021,
        5022, 5101, 5102, 5201, 5202, 5301, 5302, 5401, 5402, 6001, 6002, 6101,
        6102, 6201, 6202, 6211, 6212, 6221, 6222, 6301, 6302, 6401, 6402, 7001,
        7002, 7011, 7012, 7021, 7022, 7101, 7102, 8101, 8102, 8111, 8112, 8121,
        8122, 8201, 8202, 8211, 8212, 8301, 8302, 9001, 9002, 9011, 9012, 9021,
        9022, 9031, 9032, 9041, 9042, 9051, 9052, 9061, 9062, 9071, 9072, 9081,
        9082, 9100, 9110, 9120, 9130, 9140, 9150, 9160, 9170
    ]

    # 注意：Label 0 通常是背景。
    # 如果你的模型只关心具体的脑区，建议在训练时排除 0，或者确认 0 是否包含了你需要的信息。
    # 这里我们按照你提供的列表完整生成。

    # 3. 执行生成
    centroids = generate_centroid_tensor(nii_file_path, region_labels, output_save_path)
    
    # 4. 简单的验证打印
    if centroids is not None:
        print("\nExample centroids (First 5):")
        print(centroids[:5])