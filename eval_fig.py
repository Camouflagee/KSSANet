from os.path import exists

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # 用于颜色映射和归一化

from main import logger
from models import KANSR

# 占位符：您需要导入您的KSSANet模型定义
# from models.kssanet import KSSANet # 假设您的模型在 models/kssanet.py 中

# 占位符：您可能需要从您的数据加载器中导入函数
# from data_loader import load_cave_sample_for_visualization # 示例函数名

# 定义用于生成伪彩色图像的波段 (R-29, G-12, B-4)
# 假设波段索引是0-based, 所以 29->28, 12->11, 4->3
PSEUDO_COLOR_BANDS = [28, 11, 3] # R, G, B

def load_model_and_data(model_path, data_path):
    """
    占位符函数：加载预训练的KSSANet模型和CAVE数据集中的一个样本。
    您需要根据您的实际情况实现这个函数。

    Args:
        model_path (str): 模型权重文件的路径。
        data_id (any): 用于指定加载哪个数据样本的标识符。

    Returns:
        model (torch.nn.Module): 加载的KSSANet模型。
        lr_hsi (torch.Tensor): 低分辨率高光谱图像样本 (作为模型输入)。
        gt_hsi (torch.Tensor): 对应的地面真实高光谱图像样本。
    """
    print(f"正在加载模型从: {model_path}")
    print(f"正在加载数据样本: {data_path}")

    # --- 在此填充您的模型加载逻辑 ---
    # model = KSSANet(...) # 初始化您的模型
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # ---------------------------------
    best_model_path = model_path
    if not exists(best_model_path):
        logger.error(f"最佳模型文件未找到: {best_model_path}")
        print(f"Error: Best model file not found at {best_model_path}")
        return
    if args.dataset == 'CAVE':
        HSI_bands = 31
    else args.dataset == "Chikusei":
        HSI_bands = 128
    model = KANSR.KSSANet(hsi_bands=HSI_bands,
                            scale=args.scale,
                            depth=args.depth,
                            dim=args.hidden_dim)
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
    test_dataset_path = data_path   
    test_dataset = NPZDataset(test_dataset_path)
    # 测试时 batch_size 可以设为 1 或其他值，shuffle 通常为 False
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # --- 在此填充您的数据加载逻辑 ---
    # 示例：lr_hsi, gt_hsi = load_cave_sample_for_visualization(data_id)
    # lr_hsi = lr_hsi.unsqueeze(0).to(device) # 添加batch维度并移动到设备
    # gt_hsi = gt_hsi.unsqueeze(0).to(device) # 添加batch维度并移动到设备
    # 确保 lr_hsi 和 gt_hsi 是 (B, C, H, W) 或 (C, H, W) 格式的张量
    # ---------------------------------
    # 创建一些虚拟数据作为占位符 (C, H, W)
    # CAVE 数据集通常有31个波段
    for idx, loader_data in enumerate(test_dataloader):
        LRHSI, GT = loader_data[0].to(device), loader_data[1].to(device)
        lr_hsi_sample=LRHSI
        gt_hsi_sample=GT
        break    

    
    lr_hsi = lr_hsi_sample
    gt_hsi = gt_hsi_sample
    
    return model, lr_hsi, gt_hsi


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
    # --- 配置 ---
    MODEL_PATH = "path/to/your/kssanet_model_weights.pth"  # 修改为您的模型权重路径
    DATA_SAMPLE_ID = "cave_sample_01"  # 修改为您想可视化的数据样本标识

    try:
        # 1. 加载模型和数据样本
        # 注意：您需要实现 load_model_and_data 函数
        kssanet_model, lr_hsi_sample, gt_hsi_sample = load_model_and_data(MODEL_PATH, DATA_SAMPLE_ID)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kssanet_model.to(device)
        lr_hsi_sample = lr_hsi_sample.to(device)
        # gt_hsi_sample 不需要移动到device，因为MSE计算和伪彩图生成在CPU上用numpy

        # 2. 模型推理
        kssanet_model.eval()
        with torch.no_grad():
            pred_hsi_sr = kssanet_model(lr_hsi_sample) # SR: Super-Resolved

        # 移除batch维度 (如果存在) 并移动到CPU
        # 假设输出是 (1, C, H, W)
        if pred_hsi_sr.ndim == 4 and pred_hsi_sr.shape[0] == 1:
            pred_hsi_sr = pred_hsi_sr.squeeze(0) 
        if gt_hsi_sample.ndim == 4 and gt_hsi_sample.shape[0] == 1:
            gt_hsi_sample = gt_hsi_sample.squeeze(0)
            
        pred_hsi_sr_cpu = pred_hsi_sr.cpu()
        gt_hsi_cpu = gt_hsi_sample.cpu()


        # 3. 生成KSSANet的伪彩色预测图像
        kssanet_pseudo_color = get_pseudo_color_image(pred_hsi_sr_cpu, PSEUDO_COLOR_BANDS)

        # 4. 计算KSSANet预测与GT之间的MSE热力图
        mse_heatmap = get_mse_heatmap(pred_hsi_sr_cpu, gt_hsi_cpu)

        # 5. 可视化结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 显示KSSANet的伪彩色输出
        axes[0].imshow(kssanet_pseudo_color)
        axes[0].set_title("KSSANet Output (Pseudo-color)")
        axes[0].axis('off')

        # 显示MSE热力图
        # 使用与图中相似的颜色范围 [0, 0.020]
        norm = mcolors.Normalize(vmin=0.0, vmax=0.020)
        im = axes[1].imshow(mse_heatmap, cmap='viridis', norm=norm) # 'viridis' 是一个常用的感知均匀色图
        axes[1].set_title("MSE Heatmap (KSSANet vs GT)")
        axes[1].axis('off')

        # 添加颜色条
        fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    except NotImplementedError as e:
        print(f"错误: {e}")
        print("请确保您已在 'main.py' 中正确实现 'load_model_and_data' 函数，")
        print("并提供了正确的模型路径和数据加载逻辑。")
    except Exception as e:
        print(f"运行中发生错误: {e}")
