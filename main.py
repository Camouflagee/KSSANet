from os.path import exists

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import os
import logging
from data import NPZDataset
import numpy as np
from utils import Metric,get_model_size,beijing_time, set_logger
import argparse
# from test import test
import time
from models import KANSR
import sys

sys.path.append('./models/torch_conv_kan')
#todo 24gb memeory of gpu is not enough edit the setting !
parse = argparse.ArgumentParser()
parse.add_argument('--model', type=str,default='KSSANet')
parse.add_argument('--log_out', type=int,default=1)
parse.add_argument('--dataset', type=str,default='CAVE')
parse.add_argument('--check_point', type=str,default=None)
parse.add_argument('--check_step', type=int,default=50)
parse.add_argument('--lr', type=float, default=1e-4)
parse.add_argument('--batch_size', type=int, default=8) # 32
parse.add_argument('--epochs', type=int,default=50)
parse.add_argument('--seed', type=int,default=3407) 
parse.add_argument('--scale', type=int,default=2)
parse.add_argument('--hidden_dim', type=int,default=64) # 128
parse.add_argument('--depth', type=int,default=4) # 8
parse.add_argument('--comments', type=str,default='')
parse.add_argument('--grid_size', type=int,default=5)
parse.add_argument('--spline_order', type=int,default=3)
parse.add_argument('--wandb', type=int,default=1)
parse.add_argument('--gpu', type=int,default=3)
args = parse.parse_args()

    
if args.log_out == 0:
    os.environ['WANDB_MODE'] = 'offline'
    
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
model_name = args.model
model = None
HSI_bands = None
test_dataset_path = None
train_dataset_path = None

if args.dataset == 'CAVE':
    HSI_bands = 31
    if args.scale == 2:
        train_dataset_path = './datasets/CAVE_train_x2.npz'
        test_dataset_path = './datasets/CAVE_test_x2.npz' # 添加测试集路径
    if args.scale == 4:
        train_dataset_path = './datasets/CAVE_train_x4.npz'
        test_dataset_path = './datasets/CAVE_test_x4.npz' # 添加测试集路径
    if args.scale == 8:
        train_dataset_path = './datasets/CAVE_train_x8.npz'
        test_dataset_path = './datasets/CAVE_test_x8.npz' # 添加测试集路径

elif args.dataset == "Chikusei":
    HSI_bands = 128
    if args.scale == 2:
        train_dataset_path = './datasets/Chikusei_train_x2.npz'
        test_dataset_path = './datasets/Chikusei_test_x2.npz' # 添加测试集路径
    if args.scale == 4:
        train_dataset_path = './datasets/Chikusei_train_x4.npz'
        test_dataset_path = './datasets/Chikusei_test_x4.npz' # 添加测试集路径
    if args.scale == 8:
        train_dataset_path = './datasets/Chikusei_train_x8.npz'
        test_dataset_path = './datasets/Chikusei_test_x8.npz' 


if args.model == 'KSSANet':
    model = KANSR.KSSANet(hsi_bands=HSI_bands,
                            scale=args.scale,
                            depth=args.depth,
                            dim=args.hidden_dim)
    
os_id = os.getpid()
model = model.to(device)
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lr=args.lr,params=model.parameters())
scheduler = StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
train_dataset = NPZDataset(train_dataset_path)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
start_epoch = 0 

if args.check_point is not None:
    checkpoint = torch.load(args.check_point)  
    model.load_state_dict(checkpoint['net'],strict=False)  
    optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch']+1 
    scheduler.load_state_dict(checkpoint['scheduler'])
    log_dir,_ = os.path.split(args.check_point)
    print(f'check_point: {args.check_point}')
    
if args.check_point is  None:
    log_dir = f'trained_models/{beijing_time()}_{model_name}_scale{args.scale}_dataset_{args.dataset}'
    if not os.path.exists(log_dir) and args.log_out == 1:
        os.makedirs(log_dir, exist_ok=True)
        
logger = set_logger(model_name, log_dir, args.log_out)
logger.info("".center(39, '-').center(41, '+'))
args.pid = os_id
for _,arg in enumerate(vars(args)):
    logger.info(f" {arg}: {getattr(args,arg)}".ljust(39, ' ').center(41, '|'))
logger.info("".center(39, '-').center(41, '+'))

def train():
    best_loss = float('inf') # 用于跟踪最佳损失，可以选择性地保存最佳模型
    for epoch in range(start_epoch, args.epochs):
        loss_list = []
        start_time = time.time()
        for idx,loader_data in enumerate(train_dataloader):
            LRHSI, GT = loader_data[0].to(device),loader_data[1].to(device)
            pre_hsi = model(LRHSI)
            loss = loss_func(GT,pre_hsi) #todo bug: loss is none
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        epoch_loss = np.mean(loss_list)
        logger.info(f'epoch: {epoch}, loss: {np.mean(loss_list):.5f}, time: {time.time()-start_time:.2f}')
        # print(f'epoch: {epoch}, loss: {np.mean(loss_list)}, time: {time.time()-start_time}')
        # --- 保存检查点逻辑 ---
        if (epoch + 1) % args.check_step == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': epoch_loss,
                'args': args # 保存参数以便后续恢复或查看
            }
            checkpoint_filename = f'model_epoch_{epoch+1}.pth'
            checkpoint_path = os.path.join(log_dir, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

            # 可选：保存最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_filename = 'model_best.pth'
                best_model_path = os.path.join(log_dir, best_model_filename)
                torch.save(checkpoint, best_model_path) # 保存与当前检查点相同的信息
                logger.info(f'Best model saved to {best_model_path} (Loss: {best_loss:.5f})')
    return best_model_path
# if __name__ == "__main__":
best_model_path = train()
print('training done')

# 创建测试 DataLoader
test_dataset = NPZDataset(test_dataset_path)
# 测试时 batch_size 可以设为 1 或其他值，shuffle 通常为 False
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# best_model_path is defined in training procedure.
def test(model, test_loader, device, logger, best_model_path):
    if not exists(best_model_path):
        logger.error(f"最佳模型文件未找到: {best_model_path}")
        print(f"Error: Best model file not found at {best_model_path}")
        return

    logger.info(f"开始测试，加载模型: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device) # 加载到指定设备
    # 严格加载模型权重，如果模型结构完全匹配
    # 如果只想加载权重而不关心其他参数（如优化器状态），可以只加载 'net'
    try:
        model.load_state_dict(checkpoint['net'])
    except RuntimeError as e:
         logger.warning(f"加载模型权重时出现不匹配 (可能由于模型结构更改): {e}")
         # 尝试非严格加载
         model.load_state_dict(checkpoint['net'], strict=False)
         logger.warning("已尝试非严格加载模型权重。")


    model.eval()  # 设置模型为评估模式

    total_psnr, total_rmse, total_sam, total_ergas, total_ssim = 0, 0, 0, 0, 0
    count = 0

    with torch.no_grad(): # 测试时不需要计算梯度
        for idx, loader_data in enumerate(test_loader):
            LRHSI, GT = loader_data[0].to(device), loader_data[1].to(device)
            pre_hsi = model(LRHSI)

            # 计算指标
            # Metric 类期望输入是 tensors，它内部会转换成 numpy
            metrics = Metric(GT, pre_hsi)

            total_psnr += metrics.PSNR
            total_rmse += metrics.RMSE
            total_sam += metrics.SAM
            total_ergas += metrics.ERGAS
            total_ssim += metrics.SSIM
            count += GT.size(0) # 累加样本数量 (通常是 batch_size, 这里是 1)

            # 可以选择性地记录每个样本的指标
            # logger.info(f"  Test Sample {idx+1}: PSNR={metrics.PSNR:.4f}, RMSE={metrics.RMSE:.4f}, SAM={metrics.SAM:.4f}, ERGAS={metrics.ERGAS:.4f}, SSIM={metrics.SSIM:.4f}")


    avg_psnr = total_psnr / count if count > 0 else 0
    avg_rmse = total_rmse / count if count > 0 else 0
    avg_sam = total_sam / count if count > 0 else 0
    avg_ergas = total_ergas / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0

    logger.info("--- 测试结果 ---")
    logger.info(f"平均 PSNR: {avg_psnr:.4f}")
    logger.info(f"平均 RMSE: {avg_rmse:.4f}")
    logger.info(f"平均 SAM: {avg_sam:.4f}")
    logger.info(f"平均 ERGAS: {avg_ergas:.4f}")
    logger.info(f"平均 SSIM: {avg_ssim:.4f}")
    logger.info("--- 测试结束 ---")


# 然后使用训练得到的最佳模型进行测试
if best_model_path and exists(best_model_path):
        test(model, test_dataloader, device, logger, best_model_path)
else:
    logger.error("没有可用的模型进行测试。")

