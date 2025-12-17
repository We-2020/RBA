import os
import torch
import numpy as np
from utils import MyDataseT1w  # 替换为您的数据集类
from torch.utils.data import DataLoader
import lmdb
from torch.utils.data import DataLoader
from utils import MyDataseGM,MyDataseT1w,MyDataseGmWmCsf,MyDataseGmWm,MyDataseGmWmCsfT1w,MyDataseT1wRBA,HybridROIDataset,MyDataseT1wRBA
from model.RBA import UNetWithBrainRegionTransformer
import pandas as pd




def analyze_results(df):
    """分析结果：按年龄段和性别分组计算指标"""
    results = {}
    
    # 按10岁年龄段分组计算
    df['age_group'] = pd.cut(df['true_age'], 
                            bins=np.arange(0, 101, 10),
                            right=False)
    
    for age_group, group in df.groupby('age_group'):
        if group.empty:
            continue
            
        mae = np.mean(np.abs(group['pred_age'] - group['true_age']))
        mse = np.mean((group['pred_age'] - group['true_age']) ** 2)
        r2 = 1 - np.sum((group['pred_age'] - group['true_age']) ** 2) / np.sum(
            (group['true_age'] - np.mean(group['true_age'])) ** 2)
        
        group_name = f"{age_group.left}-{age_group.right}"
        results[group_name] = {
            'count': len(group),
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
    
    # 按性别分组计算（如果数据中有性别信息）
    if 'gender' in df.columns:
        for gender, group in df.groupby('gender'):
            mae = np.mean(np.abs(group['pred_age'] - group['true_age']))
            mse = np.mean((group['pred_age'] - group['true_age']) ** 2)
            r2 = 1 - np.sum((group['pred_age'] - group['true_age']) ** 2) / np.sum(
                (group['true_age'] - np.mean(group['true_age'])) ** 2)
            
            results[f"gender_{gender}"] = {
                'count': len(group),
                'mae': mae,
                'mse': mse,
                'r2': r2
            }
    
    return results

def simple_inference(checkpoint_path, data_csv, lmdb_path, output_csv="inference_results.csv"):
    """单进程推理函数，按年龄段和性别分组分析结果"""
    # 1. 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hp = checkpoint['hp']
    
    # 2. 创建模型并加载权重
    model = UNetWithBrainRegionTransformer(
        d_model=hp['embed_dim'],
        nhead=hp['num_heads'],
        num_layers=hp['num_layers'],
        dim_feedforward=hp['hidden_dim'],
        dropout=hp['dropout'],
        num_brain_regions=hp['num_regions'],
        channel=hp['channel']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 3. 准备数据 - 需要读取包含性别信息的完整CSV
    # 假设原始CSV包含三列：subject, age, gender
    # full_df = pd.read_csv(data_csv, header=None, names=['subject', 'true_age'])
    
    # 4. 执行推理
    all_preds = []
    all_subjects = []
    all_ages = []
    
    # 创建数据集 - 这里假设MyDataseT1w只需要subject和age
    # dataset = MyDataseT1w_iD(
    #     data_csv, 
    #     lmdb.open(lmdb_path, readonly=True, lock=False),
    #     train=False,  # 禁用数据增强
    #     shape=hp['reshape']
    # )
    dataset = dataset = MyDataseT1wRBA(data_csv, lmdb.open(lmdb_path, readonly=True, lock=False),train=False,shape=hp['reshape'])
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # 可根据GPU内存调整
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    with torch.no_grad():
        # 初始化收集列表
        all_subjects = []
        all_ages = []
        all_preds = []
        all_rba_preds = []  # 存储所有脑区预测值
        
        for (imgs, age, region, rba_age, _, _) in loader:
            imgs = imgs.float().to(device)
            region = region.float().to(device)
            all_ages_batch = age.float().squeeze().to(device)
            all_rba_true_batch = rba_age.float().to(device)

            # 前向传播
            outputs = model(imgs, region)
            all_preds_batch = outputs['BA'].float().squeeze()
            all_rba_pred_batch = outputs['RBA'].squeeze(-1)  # [bs, num_regions]

            # 收集数据
            # all_subjects.append(subject_ids)  # 使用真实的subject ID
            all_ages.append(all_ages_batch.cpu().numpy())
            all_preds.append(all_preds_batch.cpu().numpy())
            all_rba_preds.extend(all_rba_pred_batch.cpu().numpy())

    # 转换为numpy数组
    all_rba_preds = np.array(all_rba_preds)  # [n_subjects, 117]
    all_ages = np.array(all_ages)            # [n_subjects]
    all_preds = np.array(all_preds)          # [n_subjects]

    # 计算brain age gap (RBA预测值 - 真实年龄)
    brain_age_gaps = all_rba_preds - all_ages[:, np.newaxis]  # [n_subjects, 117]

    # 创建基础DataFrame - 每行是一个样本
    results_df = pd.DataFrame({
        # 'subject': all_subjects,
        'true_age': all_ages,
        'pred_age': all_preds
    })

    # 为每个脑区添加brain age gap列
    for region_idx in range(117):
        region_name = f"region_{region_idx:03d}"  # 替换为真实脑区名称
        results_df[f'gap_{region_name}'] = brain_age_gaps[:, region_idx]

    print(f"结果DataFrame形状: {results_df.shape}")
    print("前几行数据:")
    print(results_df.head())

    # 计算每个脑区的平均brain age gap（按列计算）
    gap_columns = [col for col in results_df.columns if col.startswith('gap_')]
    region_avg_gaps = results_df[gap_columns].mean()

    print("\n各脑区平均Brain Age Gap:")
    for region, avg_gap in region_avg_gaps.items():
        print(f"{region}: {avg_gap:.4f}")

    # 创建脑区统计表 - 每行是一个脑区
    region_stats_df = pd.DataFrame({
        'region': [col.replace('gap_', '') for col in gap_columns],
        'mean_gap': region_avg_gaps.values,
        'std_gap': results_df[gap_columns].std().values,
        'min_gap': results_df[gap_columns].min().values,
        'max_gap': results_df[gap_columns].max().values
    })

    print("\n脑区统计表:")
    print(region_stats_df)
    
    # 6. 保存完整结果
    # results_df.to_csv(output_csv, index=False)
    
    # 7. 按年龄段和性别分组计算指标
    # analysis = analyze_results(results_df)
    
    # # 打印分析结果
    # print("\n按年龄段分析结果:")
    # print("{:<12} {:<8} {:<10} {:<10} {:<10}".format(
    #     "年龄段", "样本数", "MAE", "MSE", "R²"))
    # for key, value in analysis.items():
    #     if key.startswith('gender'):
    #         continue
    #     print("{:<12} {:<8} {:<10.4f} {:<10.4f} {:<10.4f}".format(
    #         key, value['count'], value['mae'], value['mse'], value['r2']))
    
    # # 如果有性别信息
    # if any(key.startswith('gender') for key in analysis):
    #     print("\n按性别分析结果:")
    #     print("{:<12} {:<8} {:<10} {:<10} {:<10}".format(
    #         "性别", "样本数", "MAE", "MSE", "R²"))
    #     for key, value in analysis.items():
    #         if key.startswith('gender'):
    #             gender = key.split('_')[1]
    #             print("{:<12} {:<8} {:<10.4f} {:<10.4f} {:<10.4f}".format(
    #                 gender, value['count'], value['mae'], value['mse'], value['r2']))
    
    # # 计算整体指标
    # mse = np.mean((results_df['pred_age'] - results_df['true_age']) ** 2)
    # mae = np.mean(np.abs(results_df['pred_age'] - results_df['true_age']))
    # r2 = 1 - np.sum((results_df['pred_age'] - results_df['true_age']) ** 2) / np.sum(
    #     (results_df['true_age'] - np.mean(results_df['true_age'])) ** 2)
    
    # print("\n整体指标:")
    # print(f"样本数量: {len(results_df)}")
    # print(f"MSE: {mse:.4f}")
    # print(f"MAE: {mae:.4f}")
    # print(f"R²: {r2:.4f}")
    
    return results_df

if __name__ == '__main__':
    # 使用示例
    checkpoint_path = "/home/caojiaxiang/brain age/RBA/training_logs/checkpoint_T1wRBA.pth"
    data_csv = "/home/caojiaxiang/brain age/cjx/5-fold/val_1.csv"  # 使用验证集
    lmdb_path = "/data/MRIlmdb/ba"
    output_csv = "statistics.csv"
    
    results_df = simple_inference(checkpoint_path, data_csv, lmdb_path, output_csv)
    results_df.to_csv("results.csv")
    # 保存分析结果到CSV
    # analysis_df = pd.DataFrame(analysis).T.reset_index()
    # analysis_df.columns = ['group', 'count', 'mae', 'mse', 'r2']
    # print(analysis_df)
    # analysis_df.to_csv("group_analysis.csv", index=False)