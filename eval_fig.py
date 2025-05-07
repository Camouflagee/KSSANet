import argparse
import os
from os.path import exists

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # 用于颜色映射和归一化

from data import NPZDataset

from models import KANSR

# 占位符：您需要导入您的KSSANet模型定义
# from models.kssanet import KSSANet # 假设您的模型在 models/kssanet.py 中

# 占位符：您可能需要从您的数据加载器中导入函数
# from data_loader import load_cave_sample_for_visualization # 示例函数名

# 定义用于生成伪彩色图像的波段 (R-29, G-12, B-4)
# 假设波段索引是0-based, 所以 29->28, 12->11, 4->3
PSEUDO_COLOR_BANDS = [28, 11, 3] # R, G, B

def load_model_and_data(model_path, data_path, num_samples_to_load, args): # 参数名修改
    """
    加载预训练的KSSANet模型和数据集中的前N个样本。

    Args:
        model_path (str): 模型权重文件的路径。
        data_path (str): 数据集文件的路径。
        num_samples_to_load (int): 需要加载的数据样本数量。
        args (argparse.Namespace): 命令行参数。

    Returns:
        model (torch.nn.Module): 加载的KSSANet模型。
        lr_hsi_list (list of torch.Tensor): 低分辨率高光谱图像样本列表。
        gt_hsi_list (list of torch.Tensor): 对应的地面真实高光谱图像样本列表。
    """
    print(f"正在加载模型从: {model_path}")
    print(f"正在加载数据集: {data_path}")
    print(f"目标加载样本数量: {num_samples_to_load}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'CAVE':
        HSI_bands = 31
    elif args.dataset == "Chikusei":
        HSI_bands = 128
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
        
    model = KANSR.KSSANet(hsi_bands=HSI_bands,
                            scale=args.scale,
                            depth=args.depth,
                            dim=args.hidden_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    model.eval()

    test_dataset = NPZDataset(data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    lr_hsi_list = []
    gt_hsi_list = []
    
    if num_samples_to_load <= 0:
        raise ValueError("要加载的样本数量必须大于0")
    if num_samples_to_load > len(test_dataset):
        print(f"警告: 请求加载的样本数量 ({num_samples_to_load}) 大于数据集中的总样本数 ({len(test_dataset)}). 将加载所有可用样本。")
        num_samples_to_load = len(test_dataset)

    for idx, loader_data in enumerate(test_dataloader):
        if idx < num_samples_to_load:
            LRHSI, GT = loader_data[0].to(device), loader_data[1].to(device)
            lr_hsi_list.append(LRHSI)
            gt_hsi_list.append(GT)
        else:
            break # 已收集足够数量的样本
    
    if len(lr_hsi_list) != num_samples_to_load:
         # 这种情况可能在dataloader提前结束时发生，尽管上面有长度检查
         print(f"警告: 实际加载的样本数量 ({len(lr_hsi_list)}) 与请求的数量 ({num_samples_to_load}) 不符。")

    return model, lr_hsi_list, gt_hsi_list


def get_pseudo_color_image(hsi_tensor, bands):
    """
    从高光谱图像张量生成伪彩色图像。

    Args:
        hsi_tensor (torch.Tensor or np.ndarray): 高光谱图像，形状为 (C, H, W)。
        bands (list of int): 用于R, G, B通道的波段索引列表。

    Returns:
        np.ndarray: 伪彩色图像，形状为 (H, W, 3)，值在 [0, 1] 范围内。
    """
    if isinstance(hsi_tensor, torch.Tensor):
        hsi_tensor = hsi_tensor.cpu().detach().numpy()

    # 选择指定波段
    pseudo_color_img = hsi_tensor[bands, :, :]
    pseudo_color_img = np.transpose(pseudo_color_img, (1, 2, 0)) # (H, W, 3)

    # 归一化到 [0, 1]
    min_val = np.min(pseudo_color_img)
    max_val = np.max(pseudo_color_img)
    if max_val > min_val:
        pseudo_color_img = (pseudo_color_img - min_val) / (max_val - min_val)
    else:
        pseudo_color_img = np.zeros_like(pseudo_color_img)
    
    return np.clip(pseudo_color_img, 0, 1)


def get_mse_heatmap(pred_hsi, gt_hsi):
    """
    计算预测高光谱图像和地面真实高光谱图像之间的MSE热力图。

    Args:
        pred_hsi (torch.Tensor or np.ndarray): 预测的高光谱图像，形状 (C, H, W)。
        gt_hsi (torch.Tensor or np.ndarray): 地面真实的高光谱图像，形状 (C, H, W)。

    Returns:
        np.ndarray: MSE热力图，形状 (H, W)。
    """
    if isinstance(pred_hsi, torch.Tensor):
        pred_hsi = pred_hsi.cpu().detach().numpy()
    if isinstance(gt_hsi, torch.Tensor):
        gt_hsi = gt_hsi.cpu().detach().numpy()

    squared_error = (pred_hsi - gt_hsi)**2
    mse_map = np.mean(squared_error, axis=0) # 沿波段维度求平均
    return mse_map


if __name__ == "__main__":
    MODEL_PATH = "/data1/lijiansheng/KSSANet/trained_models/2025-05-03_141052_KSSANet_scale8_dataset_CAVE/model_epoch_40.pth"
    dataset_path = "./datasets/CAVE_test_x8.npz"
    num_samples_to_visualize = 3 # 修改为一个整数，代表要处理的样本数量
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='CAVE')
    parse.add_argument('--scale', type=int, default=8)
    parse.add_argument('--hidden_dim', type=int, default=128)
    parse.add_argument('--depth', type=int, default=8)
    parse.add_argument('--gpu', type=int, default=6)
    args = parse.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1. 一次性加载所有模型和数据样本
    print("开始加载模型和所有请求的数据...")
    kssanet_model, all_lr_hsi_samples, all_gt_hsi_samples = load_model_and_data(
        MODEL_PATH, dataset_path, num_samples_to_visualize, args
    )
    print("模型和数据加载完毕。")

    # 遍历预加载的数据
    # current_sample_idx 将是 0-based 索引
    for current_sample_idx, (lr_hsi_sample, gt_hsi_sample) in enumerate(zip(all_lr_hsi_samples, all_gt_hsi_samples)):
        print(f"\n正在处理预加载的样本索引: {current_sample_idx}")

        # lr_hsi_sample 和 gt_hsi_sample 已经是在device上的单个样本 (B=1, C, H, W)
        # 模型也已在device上并处于eval模式

        # 2. 模型推理
        # kssanet_model.eval() # 模型已在load_model_and_data中设为eval
        with torch.no_grad():
            pred_hsi_sr = kssanet_model(lr_hsi_sample)

        # 移除batch维度 (如果存在) 并移动到CPU
        # 输入的 lr_hsi_sample, gt_hsi_sample, 和输出的 pred_hsi_sr 都是 (1, C, H, W)
        pred_hsi_sr_squeezed = pred_hsi_sr.squeeze(0)
        gt_hsi_sample_squeezed = gt_hsi_sample.squeeze(0)
        lr_hsi_sample_squeezed = lr_hsi_sample.squeeze(0)

        pred_hsi_sr_cpu = pred_hsi_sr_squeezed.cpu()
        gt_hsi_cpu = gt_hsi_sample_squeezed.cpu()
        lr_hsi_cpu = lr_hsi_sample_squeezed.cpu()

        # 3. 生成伪彩色图像
        lr_pseudo_color = get_pseudo_color_image(lr_hsi_cpu, PSEUDO_COLOR_BANDS)
        kssanet_pseudo_color = get_pseudo_color_image(pred_hsi_sr_cpu, PSEUDO_COLOR_BANDS)
        gt_pseudo_color = get_pseudo_color_image(gt_hsi_cpu, PSEUDO_COLOR_BANDS)

        # 4. 计算KSSANet预测与GT之间的MSE热力图
        mse_heatmap = get_mse_heatmap(pred_hsi_sr_cpu, gt_hsi_cpu)

        # 5. 可视化结果
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(lr_pseudo_color)
        axes[0].set_title(f"LR HSI Input (Sample Index {current_sample_idx})", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(kssanet_pseudo_color)
        axes[1].set_title(f"KSSANet Output (Sample Index {current_sample_idx})", fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(gt_pseudo_color)
        axes[2].set_title(f"Ground Truth (Sample Index {current_sample_idx})", fontsize=10)
        axes[2].axis('off')

        norm = mcolors.Normalize(vmin=0.0, vmax=0.020)
        im = axes[3].imshow(mse_heatmap, cmap='viridis', norm=norm)
        axes[3].set_title(f"MSE Heatmap (Sample Index {current_sample_idx})", fontsize=10)
        axes[3].axis('off')

        fig.colorbar(im, ax=axes[3], orientation='vertical', fraction=0.046, pad=0.04)
        # plt.tight_layout()
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
        # 6. 保存图像
        base_save_dir = "fig_res"
        dataset_save_dir = os.path.join(base_save_dir, args.dataset)
        os.makedirs(dataset_save_dir, exist_ok=True)

        fig_filename = f"sample_idx_{current_sample_idx}_scale_x{args.scale}.png" # 文件名使用索引
        full_save_path = os.path.join(dataset_save_dir, fig_filename)

        plt.savefig(full_save_path)
        print(f"图像已保存到: {full_save_path}")
        
        plt.show()

