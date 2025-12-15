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
from utils import MyDataseGM,MyDataseT1w,MyDataseGmWmCsf,MyDataseGmWm,MyDataseGmWmCsfT1w,MyDataseT1wRBA,HybridROIDataset,MyDataseT1wRBA
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
        self.best_metrics = {
            'mae': float('inf'),
            'loss': float('inf'),
            'r2': 0,
            'epoch': 0
        }
    
        # 添加检查点加载逻辑
        if self.hp.get('resume_from_checkpoint'):
            self._load_checkpoint(self.hp['resume_from_checkpoint'])

        self._setup_logging()
        self._print_hyperparameters()
        
        # 初始化组件
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hp['learning_rate'],
            weight_decay=self.hp['weight_decay']
        )
        # for i, (name, param) in enumerate(self.model.named_parameters()):
        #     if i >= 59 and i <= 70:
        #         print(f"Index {i}: {name}")
        
        
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer, 
#             T_0=self.hp['scheduler_T0'],
#             eta_min=self.hp['scheduler_eta_min']
#         )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=30,      # 第一次周期 epoch 数
        #     T_mult=2,    # 每次重启周期扩大倍数
        #     eta_min=1e-5
        # )
        self.warmup_epochs = 0  # Warmup 的 epoch 数
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=400,
            eta_min=2e-5,
        )

        # 初始化 Warmup 参数
        
        
        # 准备数据
        self.train_loader, self.val_loader = self._prepare_data()
        
        # 使用accelerator准备组件
#         (self.model, self.optimizer, self.train_loader, 
#          self.val_loader, self.scheduler) = self.accelerator.prepare(
#             self.model, self.optimizer, self.train_loader, 
#             self.val_loader, self.scheduler
#         )
        # self.model.to(self.accelerator.device)

        # # 手动包装 DDP，启用 find_unused_parameters
        # self.model = DDP(self.model, device_ids=[self.accelerator.device], find_unused_parameters=True)
        # (self.optimizer, self.train_loader, 
        #  self.val_loader, self.scheduler) = self.accelerator.prepare(
        #     self.optimizer, self.train_loader, 
        #     self.val_loader, self.scheduler
        # )
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
        """训练单个epoch - 修复NaN梯度问题"""
        # torch.autograd.set_detect_anomaly(True)
        # torch.autograd.set_detect_anomaly(True)
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_consistency_loss = 0
        total_grad_loss = 0
        total_rba_loss = 0
        
        # 获取超参数
        # lambda_consistency = self.hp.get('lambda_consistency', 0.1)
        lambda_grad = self.hp.get('lambda_grad', 0.01)
        # lambda_rba = self.hp.get('lambda_rba', 1.0)
        sigma = self.hp.get('noise_sigma', 0.1)
        num_regions = self.hp.get('num_regions', 117)

        model_core = self.model.module if hasattr(self.model, "module") else self.model
        


        for batch_idx, (imgs, age, region, rba_age,noise_img,rand_region_idx) in enumerate(tqdm(self.train_loader, 
                                    disable=not self.accelerator.is_local_main_process)):
            # 清除梯度
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                self.accelerator.print(f"Skipping batch {batch_idx} due to NaN/Inf in input images")
                continue
            self.optimizer.zero_grad()
            
            # 准备数据
            imgs = imgs.float()
            region = region.float()
            age = age.float().squeeze()
            rba_age = rba_age.float()
            
            # 1. 主损失计算 - 使用更稳定的方法
            # 前向传播
            outputs = self.model(imgs, region)
            preds = outputs['BA'].float().squeeze()
            

            # lambda_rba = 0.1
            lambda_rba = torch.exp(torch.clamp(model_core.log_lambda_rba, max=5.0))
            lambda_consistency = torch.exp(torch.clamp(model_core.log_lambda_consis, max=5.0))
            lambda_grad = torch.exp(torch.clamp(model_core.log_lambda_grad, max=5.0))
            
            # 2. RBA损失计算
            rba = outputs['RBA'].float()
            rba_reshaped = rba.squeeze(-1)
            # rba_preds_for_grad = rba[torch.arange(rba.size(0)), rand_region_idx]
            region_mask = region[torch.arange(region.size(0)), rand_region_idx, :, :, :]
            if hasattr(self.model, "module"):
                e1 = self.model.module.saved_e1
            else:
                e1 = self.model.saved_e1
            # print(outputs['e1'].requires_grad, outputs['e1'].grad_fn)
            # print(rba.requires_grad, rba.grad_fn)
            grads = None
            e1 = e1.contiguous()
            rba = rba.contiguous()
            grad_outputs = torch.ones_like(rba).contiguous()
            if e1.requires_grad and rba.requires_grad:
                grad_outputs = torch.ones_like(rba)
                grads = torch.autograd.grad(
                    outputs=rba,
                    inputs=e1,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False
                )[0]
                if grads is not None:
                    grads = grads.contiguous().detach()
                    grads = grads.detach() + 0.01 * e1
            with torch.no_grad():
                outputs_noise = model_core(noise_img, region, only_rba=True)
                rba_noise = outputs_noise['RBA'].detach().float()
                # rba_noise = outputs_noise['RBA'].detach().clone()

            # # 3. 简化训练：暂时移除噪声一致性损失和梯度损失
            # # 先确保基础训练稳定，再逐步添加复杂损失
            mask = torch.ones(117, dtype=torch.bool, device=rba.device)
            mask[rand_region_idx] = 0  # 屏蔽该脑区
            mask = mask.unsqueeze(0).expand(rba.shape[0], -1).unsqueeze(-1)  # [bs, 117, 1]

            # 计算差异
            diff = (rba - rba_noise)

            # ⚠️ NaN/Inf防护
            diff = torch.nan_to_num(diff, nan=0.0, posinf=1e4, neginf=-1e4)

            # 加 mask
            diff = diff * mask

            # 避免 mask 全为 0
            valid_count = mask.sum().clamp(min=1)

            # 安全求平均
            loss_consistency = lambda_consistency * (diff ** 2).sum() / valid_count
            # loss_consistency = torch.tensor(0.0, device=imgs.device)
            # grad_loss = torch.tensor(0.0, device=imgs.device)
            main_loss = self.criterion(preds, age)
            rba_loss_all = self.criterion(rba_reshaped, rba_age)
            
            if grads is not None:
                # 构造脑区mask
                region_mask_grad = region_mask.unsqueeze(1).float()

                # 提取目标脑区梯度与其他脑区梯度
                grad_in_region = grads * region_mask_grad
                grad_out_region = grads * (1 - region_mask_grad)
                # has_nan = torch.any(grads.isnan())

                # # --- 2. 绝对值最大值检测 ---
                # max_abs_value = grads.abs().max()

                # print(f"张量 'grads' 中是否包含 NaN: {has_nan.item()}")
                # print(f"张量 'grads' 的绝对值最大值是: {max_abs_value.item()}")
                # 构造梯度选择性loss
                grad_loss = (
                    - F.l1_loss(grad_in_region, torch.zeros_like(grad_in_region), reduction='mean') +
                    0.1*F.l1_loss(grad_out_region, torch.zeros_like(grad_out_region), reduction='mean')
                )
            else:
                grad_loss = torch.tensor(1.0, device=imgs.device)
            # 4. 组合损失 - 暂时只使用主损失和RBA损失
            # self.accelerator.print(lambda_rba,rba_loss_all,loss_consistency)
            loss = main_loss +  lambda_rba * rba_loss_all + loss_consistency + grad_loss * lambda_grad
            # loss = main_loss + lambda_rba * rba_loss_all
            # print(f"main_loss: {main_loss.item():.6f}, lambda_rba * rba_loss_all: {(lambda_rba * rba_loss_all).item():.6f}, grad_loss: {(grad_loss).item():.6f}")
            # 检查总损失
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Batch {batch_idx}: 总损失为NaN或inf, 主损失: {main_loss.item()},噪声loss: {loss_consistency.item()} RBA损失: {rba_loss_all.item()}")
                continue
            
            # 5. 反向传播和优化

            self.accelerator.backward(loss)
            self.optimizer.step()

            
            # 6. 记录损失
            batch_size = len(imgs)
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_rba_loss += rba_loss_all.item()
            total_consistency_loss += loss_consistency.item()
            # total_grad_loss += grad_loss.item()
            
        #     # 7. 定期清理缓存
        #     if batch_idx % 20 == 0:
        #         torch.cuda.empty_cache()
            
        #     # 8. 日志记录
        #     if batch_idx % self.hp.get('log_interval', 50) == 0:
        #         current_lr = self.optimizer.param_groups[0]['lr']
        #         self.accelerator.print(
        #             f'Train Epoch: {epoch} [{batch_idx * len(imgs)}/{len(self.train_loader.dataset)} '
        #             f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
        #             f'Loss: {loss.item():.6f} | Main: {main_loss.item():.6f} | '
        #             f'RBA: {rba_loss_all.item():.6f} | LR: {current_lr:.6f}'
        #         )
        
        # # 计算平均损失
        # num_samples = len(self.train_loader.dataset)
        # if num_samples == 0:
        #     return {
        #         "total_loss": float('inf'),
        #         "main_loss": float('inf'),
        #         "consistency_loss": float('inf'),
        #         "grad_loss": float('inf'),
        #         "rba_loss": float('inf')
        #     }
        
        avg_loss = total_loss / len(self.train_loader)
        avg_main_loss = total_main_loss / len(self.train_loader)
        avg_consistency_loss = total_consistency_loss / len(self.train_loader)
        avg_grad_loss = total_grad_loss / len(self.train_loader)
        avg_rba_loss = total_rba_loss / len(self.train_loader)
        
        avg_losses = {
            "total_loss": avg_loss,
            "main_loss": avg_main_loss,
            "consistency_loss": avg_consistency_loss,
            "grad_loss": 0,
            "rba_loss": avg_rba_loss,
            "lambda_rba": lambda_rba

        }

        return avg_losses

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
        """完整的训练流程"""
        
        for epoch in range(self.start_epoch, self.hp['epochs']):
            # Warmup 阶段调整学习率
            train_loss = self._train_epoch(self.optimizer)
            val_metrics = self._validate()
            lr = self.optimizer.param_groups[0]['lr']
            # self.scheduler.step(val_metrics['loss'])
            self.scheduler.step()
            
            
            
            # 主进程记录日志
            if self.accelerator.is_main_process:
                
                log_msg = (
                    f"Epoch {epoch+1}/{self.hp['epochs']} | "
                    f"LR: {lr:.2e} | "
                    f"Train Loss: {train_loss['total_loss']:.4f} "
                    f"(Main: {train_loss['main_loss']:.4f}, "
                    f"Cons: {train_loss['consistency_loss']:.4f}, "
                    f"Grad: {train_loss['grad_loss']:.12f}, "
                    f"RBA: {train_loss['rba_loss']:.4f}) | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"MAE: {val_metrics['mae']:.2f} | "
                    f"Val_RBA: {val_metrics['rba_loss']:.2f} | "
                    f"R²: {val_metrics['r2']:.4f} ｜ "
                    f"R: {val_metrics['r']:.4f} | "
                    f"lambda_rba: {train_loss['lambda_rba']}"
                )
                self.accelerator.print(log_msg)
                logging.info(log_msg)

                # 保存最佳模型
                if val_metrics['mae'] < self.best_metrics['mae']:
                    self.best_metrics = val_metrics.copy()
                    self.best_metrics['epoch'] = epoch + 1
#                     torch.save(
#                         self.accelerator.unwrap_model(self.model).state_dict(),
#                         os.path.join(self.hp['log_dir'], f"best_model_{self.hp['experiment_name']}.pth"))
                    self._save_checkpoint(epoch)

        # 记录最终结果
        final_log = (
            f"\nBest Model Results (Epoch {self.best_metrics['epoch']}):\n"
            f"MAE: {self.best_metrics['mae']:.2f} | "
            f"MSE: {self.best_metrics['loss']:.4f} | "
            f"R²: {self.best_metrics['r2']:.4f}"
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
