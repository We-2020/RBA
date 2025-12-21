import os
import logging
import pandas as pd
import torch
import lmdb
from torch import nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from scipy import stats
from utils import MyDataseGM,MyDataseT1w,MyDataseGmWmCsf,MyDataseGmWm,MyDataseGmWmCsfT1w,HybridROIDataset,MyDataseT1wRBA
from model.RBA import UNetWithBrainRegionTransformer
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --num_processes=8 RunT1.py
import copy


def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class BrainAgeTrainer:
    def __init__(self, hyperparameters, env, env2=None):
        self.hp = hyperparameters
        self.env = env
        self.env2 = env2
        self.accelerator = Accelerator(split_batches=True)
        self.start_epoch = 0
        self.current_stage = 1  # 初始为第一阶段：联合约束
        self.best_metrics = {
            'mae': float('inf'),
            'loss': float('inf'),
            'r2': 0,
            'epoch': 0
        }
    
        if self.hp.get('resume_from_checkpoint'):
            self._load_checkpoint(self.hp['resume_from_checkpoint'])

        self._setup_logging()
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.L1 = nn.L1Loss()
        
        # 初始化第一阶段优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hp['learning_rate'],
            weight_decay=self.hp['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.hp.get('epochs', 300),
            eta_min=2e-5,
        )

        self.train_loader, self.val_loader = self._prepare_data()
        
        # 准备组件
        (self.model, self.optimizer, self.train_loader, 
         self.val_loader, self.scheduler) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, 
            self.val_loader, self.scheduler
        )

    def _setup_logging(self):
        """配置日志记录"""
        log_dir = self.hp['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{self.hp['experiment_name']}.log")
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            force=True
        )
        self.accelerator.print(f"Logging to {log_file}")

    def _print_hyperparameters(self):
        """打印并记录超参数"""
        hp_str = "\nHyperparameters:\n" + "\n".join(
            [f"{k:20}: {v}" for k, v in self.hp.items()]
        )
        self.accelerator.print(hp_str)
        logging.info(hp_str)

    def _build_model(self):
        """构建模型"""
        return UNetWithBrainRegionTransformer(
            d_model=self.hp['embed_dim'],
            nhead=self.hp['num_heads'],
            num_layers=self.hp['num_layers'],
            dim_feedforward=self.hp['hidden_dim'],
            dropout=self.hp['dropout'],
            num_brain_regions=self.hp['num_regions'],
            channel=self.hp['channel']
        )
        # return ResNet3DRegression()
    def _save_checkpoint(self, epoch):
        """保存训练检查点"""
        if self.accelerator.is_main_process:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.accelerator.get_state_dict(self.model),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_metrics': self.best_metrics,
                'hp': self.hp
            }
            checkpoint_path = os.path.join(
                self.hp['log_dir'], 
                f"checkpoint_{self.hp['experiment_name']}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch}")

    def _load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
        if self.accelerator.is_main_process:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = None

        # 广播检查点到所有进程
        checkpoint = self.accelerator.broadcast(checkpoint, from_process=0)

        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        self.best_metrics = checkpoint['best_metrics']

        if self.accelerator.is_main_process:
            logging.info(f"Resuming training from epoch {self.start_epoch}")

    def _prepare_stage2_optimizer(self):
        """核心策略：阶段2 开启差分学习率，冻结大部分 Backbone 结构"""
        self.accelerator.print(">>> [ALERT] Switching to Stage 2: Fine-tuning for BA Accuracy Peak")
        model_core = self.model.module if hasattr(self.model, "module") else self.model
        
        # 分组参数：Predictor 层需要正常更新，Backbone 层极慢更新保持特征
        predictor_params = [p for n, p in model_core.named_parameters() if any(k in n for k in ['predictor', 'regressor', 'head', 'final'])]
        backbone_params = [p for n, p in model_core.named_parameters() if not any(k in n for k in ['predictor', 'regressor', 'head', 'final'])]

        new_optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.hp['learning_rate'] * 0.1}, 
            {'params': predictor_params, 'lr': self.hp['learning_rate'] * 0.5}
        ], weight_decay=self.hp['weight_decay'])
        
        self.optimizer = self.accelerator.prepare(new_optimizer)

    def _prepare_data(self):
        """准备数据加载器"""
        def _create_loader(df_path, batch_size,aug=False,shuffle=True,drop_last=True):
            dataset = MyDataseT1wRBA(df_path, self.env,train=aug,shape=self.hp['reshape'])
            # dataset = HybridROIDataset(data_list_path=df_path, env=self.env,train=aug,target_shape=self.hp['reshape'],roi_level=1, zoom_factor=1,aal_path="/home/cjx/Im/ROI_MNI_V4.nii")
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.hp['num_workers'],
                pin_memory=True,
                drop_last=True,
                shuffle=shuffle,
                # prefetch_factor=4
            )

        train_loader = _create_loader(
            self.hp['train_csv'], 
            self.hp['batch_size'],
            # self.hp['augmentation']
        )
        val_loader = _create_loader(
            self.hp['val_csv'], 
            self.hp['batch_size'],
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader

    def _calculate_r2(self, preds, labels):
        """计算R平方指标"""
        ss_res = torch.sum((preds - labels) ** 2)
        ss_tot = torch.sum((labels - labels.mean()) ** 2)
        return 1 - ss_res / ss_tot

    def _train_epoch(self, epoch):
        self.model.train()
        metrics_sum = {k: 0.0 for k in ['total', 'main', 'rba', 'cons', 'grad']}
        model_core = self.model.module if hasattr(self.model, "module") else self.model

        for batch_idx, (imgs, age, region, rba_age, noise_img, rand_region_idx) in enumerate(tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process)):
            if torch.isnan(imgs).any(): continue
            self.optimizer.zero_grad()
            
            imgs, region, age, rba_age = imgs.float(), region.float(), age.float().squeeze(), rba_age.float()
            
            # 1. 前向传播
            outputs = self.model(imgs, region)
            preds = outputs['BA'].float().squeeze()
            rba = outputs['RBA'].float().squeeze(-1) # [Batch, 117]
            
            # 2. 核心改进：Loss 量级对齐
            # BA Loss: 均值处理
            main_loss = self.criterion(preds, age)
            
            # RBA Loss: 显式除以区域数量，使其在梯度贡献上与 BA 对等
            # 这样 lambda_rba = 1.0 时，意味着 BA 和 整个RBA向量 的权重是 1:1
            rba_loss_all = self.criterion(rba, rba_age) / self.hp['num_regions']

            # 动态系数获取
            lambda_rba = torch.exp(torch.clamp(model_core.log_lambda_rba, max=5.0))
            
            # 3. 辅助约束计算
            if self.current_stage == 1:
                # --- 一致性 Loss (优化版) ---
                with torch.no_grad():
                    outputs_noise = model_core(noise_img.float(), region, only_rba=True)
                    rba_noise = outputs_noise['RBA'].detach().float().squeeze(-1)
                
                # 排除被修改的脑区
                mask = torch.ones(self.hp['num_regions'], device=rba.device)
                mask[rand_region_idx] = 0
                diff = torch.nan_to_num(rba - rba_noise, nan=0.0) * mask.unsqueeze(0)
                # 一致性也需要对齐量级
                loss_consistency = (diff ** 2).mean() / self.hp['num_regions']
                loss_consistency = loss_consistency * torch.exp(torch.clamp(model_core.log_lambda_consis, max=5.0))

                # --- 梯度 Loss (带 NaN 防护) ---
                e1 = model_core.saved_e1 if not hasattr(self.model, "module") else self.model.module.saved_e1
                grad_loss = torch.tensor(0.0, device=rba.device)
                if e1.requires_grad:
                    grad_outputs = torch.ones_like(outputs['RBA'])
                    # 计算 RBA 对中间特征 e1 的梯度
                    grads = torch.autograd.grad(outputs=outputs['RBA'], inputs=e1, grad_outputs=grad_outputs, 
                                                create_graph=True, retain_graph=True, allow_unused=True)[0]
                    if grads is not None:
                        # 确保梯度集中在目标脑区内
                        region_mask = region[torch.arange(region.size(0)), rand_region_idx].unsqueeze(1)
                        grad_in = grads * region_mask
                        grad_out = grads * (1 - region_mask)
                        # 这里使用 L1 惩罚区域外的梯度，鼓励区域内的梯度
                        grad_loss = (-F.l1_loss(grad_in, torch.zeros_like(grad_in)) + 
                                     0.01 * F.l1_loss(grad_out, torch.zeros_like(grad_out)))

                
                total_loss = main_loss + lambda_rba * rba_loss_all + loss_consistency + grad_loss * torch.exp(torch.clamp(model_core.log_lambda_grad, max=5.0))
            else:
                # 阶段 2：冲刺模式
                # 进一步降低 RBA 干扰，专注于 BA 指标
                loss_consistency = torch.tensor(0.0, device=rba.device)
                grad_loss = torch.tensor(0.0, device=rba.device)
                total_loss = main_loss + (lambda_rba * 0.2) * rba_loss_all + 0.00001*loss_consistency + 0.000001*grad_loss * torch.exp(torch.clamp(model_core.log_lambda_grad, max=5.0))

            # 4. 反向传播
            self.accelerator.backward(total_loss)
            
            # 阶段2增加梯度裁剪，防止最后阶段参数跑飞
            if self.current_stage == 2:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 0.5)
                
            self.optimizer.step()

            # 记录数据
            metrics_sum['total'] += total_loss.item()
            metrics_sum['main'] += main_loss.item()
            metrics_sum['rba'] += rba_loss_all.item() * self.hp['num_regions'] # 记录真实的 RBA MSE
            metrics_sum['cons'] += loss_consistency.item()
            metrics_sum['grad'] += grad_loss.item()

        return {k: v / len(self.train_loader) for k, v in metrics_sum.items()}

    def _validate(self):
        """验证过程"""
        self.model.eval()
        metrics = {
            'loss': 0,
            'mae': 0,
            'r2': 0,
            'r': 0,
            'rba_loss': 0,   # 新增：记录RBA损失（无加权）
        }

        all_preds_list, all_ages_list = [], []

        with torch.no_grad():
            for (imgs, age, region, rba_age, _, _) in self.val_loader:
                imgs = imgs.float()
                region = region.float()
                age = age.float().squeeze()
                rba_age = rba_age.float()

                # 前向传播
                outputs = self.model(imgs, region)
                preds = outputs['BA'].float().squeeze()
                rba = outputs['RBA'].squeeze(-1)  # [bs, num_regions]

                # 收集预测与标签
                all_preds, all_ages = self.accelerator.gather_for_metrics((preds, age))

                # 主任务loss
                metrics['loss'] += self.criterion(all_preds, all_ages).item()
                metrics['mae'] += self.L1(all_preds, all_ages).item()
                metrics['r2'] += self._calculate_r2(all_preds, all_ages).item()

                # ===== 新增：RBA loss（不加权） =====
                # 注意，这里不使用 lambda_rba，仅计算 criterion
                all_rba_pred, all_rba_true = self.accelerator.gather_for_metrics((rba, rba_age))
                metrics['rba_loss'] += self.L1(all_rba_pred, all_rba_true).item()

                # 保存以计算皮尔森相关系数
                all_preds_list.append(all_preds.detach().cpu())
                all_ages_list.append(all_ages.detach().cpu())

        # 汇总指标
        all_preds = torch.cat(all_preds_list).numpy()
        all_ages = torch.cat(all_ages_list).numpy()
        # 在 _validate() 函数中，调用 pearsonr 之前
        if np.any(np.isnan(all_preds)) or np.any(np.isinf(all_preds)):
            print(f"[ERROR] all_preds contains NaN or Inf! Example: {all_preds}")
            # 可选：保存出问题的 batch 或退出
        if np.any(np.isnan(all_ages)) or np.any(np.isinf(all_ages)):
            print(f"[ERROR] all_ages contains NaN or Inf! Example: {all_ages}")
        mask = (
            ~np.isnan(all_preds) & 
            ~np.isnan(all_ages) &
            ~np.isinf(all_preds) & 
            ~np.isinf(all_ages)
        )
        clean_preds = all_preds[mask]
        clean_ages  = all_ages[mask]
        r_value, _ = stats.pearsonr(clean_preds, clean_ages)

        avg_metrics = {k: v / len(self.val_loader) for k, v in metrics.items()}
        avg_metrics['r'] = r_value
        return avg_metrics
    

    def train(self):
        """完整的训练流程：包含两阶段精度冲刺与详细指标记录"""
        stage_switch = self.hp.get('stage_switch_epoch', 230)
        self.accelerator.print(f"🚀 Training started. Total Epochs: {self.hp['epochs']} | Stage Switch Epoch: {stage_switch}")
        
        for epoch in range(self.start_epoch, self.hp['epochs']):
            # 1. 阶段切换检查：进入 Stage 2 开启精度冲刺
            if epoch >= stage_switch and self.current_stage == 1:
                self.current_stage = 2
                self._prepare_stage2_optimizer()
                self.accelerator.print(f"\n{'='*30}\n>> Epoch {epoch+1}: Entering Stage 2 (Precision Peak Phase)\n{'='*30}")

            # 2. 执行训练与验证
            # _train_epoch 内部应返回包含 total_loss, main_loss, consistency_loss, grad_loss, rba_loss 的字典
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate()
            
            # 3. 获取当前各组学习率 (针对差分学习率的情况)
            lr_list = [group['lr'] for group in self.optimizer.param_groups]
            if len(lr_list) == 1:
                lr_str = f"{lr_list[0]:.2e}"
            else:
                # 显示核心骨干与预测头的不同学习率
                lr_str = f"BK:{lr_list[0]:.1e}/HD:{lr_list[1]:.1e}"
            
            # 4. 更新调度器
            self.scheduler.step()

            # 5. 主进程打印详细过程日志
            if self.accelerator.is_main_process:
                # 计算 RBA 对总梯度的实际贡献比 (加权后的 RBA Loss)
                # train_metrics['rba_loss'] 是归一化后的 RBA MSE
                lambda_rba = train_metrics.get('lambda_rba', 1.0)
                rba_weight = lambda_rba * (0.2 if self.current_stage == 2 else 1.0)
                
                log_msg = (
                    f"Epoch {epoch+1:03d}/{self.hp['epochs']} [S{self.current_stage}] | "
                    f"LR: {lr_str} | "
                    f"Train Loss: {train_metrics['total']:.4f} "
                    f"(BA: {train_metrics['main']:.4f}, "
                    f"RBA_w: {train_metrics['rba'] * rba_weight:.4f}, "
                    f"Cons: {train_metrics['cons']:.4f}, "
                    f"Grad: {train_metrics['grad']:.6f}) | "
                    f"Val_BA_MAE: {val_metrics['mae']:.3f} | " 
                    f"Val_RBA_MAE: {val_metrics['rba_loss']:.3f} | "
                    f"R: {val_metrics['r']:.4f} | "
                    f"Best_MAE: {self.best_metrics['mae']:.3f}"
                )
                
                self.accelerator.print(log_msg)
                logging.info(log_msg)

                # 6. 保存最佳模型
                if val_metrics['mae'] < self.best_metrics['mae']:
                    self.best_metrics = {**val_metrics, 'epoch': epoch + 1}
                    self._save_checkpoint(epoch)
                    self.accelerator.print(f" ✨ New Best BA MAE Reached!")

        # 7. 记录最终结果
        final_log = (
            f"\n" + "="*50 +
            f"\nBest Model Results (Epoch {self.best_metrics['epoch']}):\n"
            f"MAE (BA): {self.best_metrics['mae']:.3f}\n"
            f"MSE (BA): {self.best_metrics['loss']:.4f}\n"
            f"MAE (RBA): {self.best_metrics.get('rba_loss', 0.0):.3f}\n"
            f"R²: {self.best_metrics['r2']:.4f}\n"
            f"Pearson R: {self.best_metrics['r']:.4f}\n" +
            "="*50
        )
        self.accelerator.print(final_log)
        logging.info(final_log)

        return self.best_metrics

# 超参数配置示例
HYPERPARAMETERS = {
    # 数据参数
    'train_csv': '/home/caojiaxiang/brain age/cjx/5-fold/train_1.csv',
    'val_csv': '/home/caojiaxiang/brain age/cjx/5-fold/val_1.csv',
#     'train_csv': '/home/sjc/atun1/data/train_1.csv',
#     'val_csv': '/home/sjc/atun1/data/val_1.csv',
    'batch_size': 4,
    'num_workers': 1,
    'channel':32,
    
    # 模型参数
    'embed_dim': 64,
    'num_heads': 8,
    'num_layers': 6,
    'hidden_dim': 512,
    'dropout': 0.1,
    'lambda_consistency':2e-2,
    'lambda_grad':0.1,
    'noise_sigma':0.1,
    'num_regions':117,
    'lambda_rba':1e-3,
    

    # 训练参数
    'epochs': 270,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    
    # 调度器参数
    'scheduler_T0': 200,
    'scheduler_eta_min': 1e-5,
    
    # 日志参数
    'log_dir': './training_logs',
    'experiment_name': 'T1wRBA',
    
    #数据增强
    'augmentation': False,
    'reshape': (91,109,91),
#     'resume_from_checkpoint': '/home/caojiaxiang/afterHW/ATUN/training_logs/checkpoint_GMWMCSF.pth',
    'resume_from_checkpoint': None,
    'num_regions': 117
}

if __name__ == '__main__':
    set_seed(42)
    # 初始化环境
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    lmdb_env = lmdb.open(
        "/data/MRIlmdb/ba",
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )
#     lmdb_env2 = lmdb.open("/data/MRIlmdb/ba", readonly=True, lock=False, readahead=False,
#                     meminit=False)
    
    # 创建训练器并开始训练
    trainer = BrainAgeTrainer(HYPERPARAMETERS, lmdb_env)
    best_metrics = trainer.train()
