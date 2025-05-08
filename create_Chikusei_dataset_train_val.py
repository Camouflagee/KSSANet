import numpy as np
import os
import cv2
import scipy.io
import h5py # <--- 添加导入
from tqdm import tqdm # 用于显示进度条

def crop_patches(image_gt, image_lr, patch_size, stride, scale):
    """
    将高分辨率和低分辨率图像裁剪成对应的 patch 对。
    """
    gt_patches = []
    lr_patches = []
    h_gt, w_gt, _ = image_gt.shape
    h_lr, w_lr, _ = image_lr.shape
    lr_patch_size = patch_size // scale
    lr_stride = stride // scale

    # 确保 LR 图像尺寸与 GT 图像尺寸和 scale 因子匹配
    assert h_lr == h_gt // scale and w_lr == w_gt // scale, \
        f"LR image size ({h_lr},{w_lr}) does not match GT size ({h_gt},{w_gt}) with scale {scale}"

    for y in range(0, h_gt - patch_size + 1, stride):
        for x in range(0, w_gt - patch_size + 1, stride):
            # 提取 GT patch
            gt_patch = image_gt[y : y + patch_size, x : x + patch_size, :]

            # 计算对应的 LR patch 坐标
            lr_y, lr_x = y // scale, x // scale
            # 提取 LR patch
            lr_patch = image_lr[lr_y : lr_y + lr_patch_size, lr_x : lr_x + lr_patch_size, :]

            # 转换维度为 (C, H, W) 并添加到列表
            gt_patches.append(gt_patch.transpose(2, 0, 1))
            lr_patches.append(lr_patch.transpose(2, 0, 1))

    return np.array(gt_patches), np.array(lr_patches)

def create_chikusei_dataset(data_mat_path, save_path, scale, patch_size=64, stride=32):
    """
    创建 Chikusei 数据集的 npz 文件 (训练集为 patches, 测试集为完整图像)。

    Args:
        data_mat_path: 原始 Chikusei .mat 文件路径。
        save_path: 保存 .npz 文件的目录。
        scale: 下采样倍数。
        patch_size: 训练集 GT patch 的边长。
        stride: 训练集 patch 的步长。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Scale x{scale}:")
    print(f"  加载 Chikusei 数据集: {data_mat_path}...")
    # --- 修改开始 ---
    # 使用 h5py 加载 v7.3 mat 文件
    with h5py.File(data_mat_path, 'r') as f:
        # 尝试查找包含高光谱数据的键
        data_key = None
        for key in f.keys():
            # 检查是否是数据集并且维度符合 HSI (通常是 3D)
            # 注意：h5py 数据集对象没有 ndim 属性，需要检查 shape
            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                # 假设数据是 (C, H, W) 或 (W, H, C) 或 (H, W, C)
                # Chikusei 通常是 (H, W, C)
                # 检查尺寸是否合理，例如大于 100x100x100
                if all(s > 100 for s in f[key].shape):
                        data_key = key
                        break
        if data_key is None:
            print(f"错误：在 {data_mat_path} 中未自动找到合适的高光谱数据键。请检查 .mat 文件结构。")
            print(f"文件中的顶层键: {list(f.keys())}")
            return

        # 加载数据并确保是 numpy 数组
        # h5py 读取的数据需要显式转换为 numpy 数组
        # 注意：MATLAB 保存时可能是 (W, H, C)，需要转置为 (H, W, C)
        # 或者直接是 (H, W, C)。这里先假设是 (H, W, C)
        img_data = f[data_key][()] # [()] 读取整个数据集到内存
        # 如果加载后的维度不是 (H, W, C)，可能需要调整
        # 例如，如果加载后是 (C, H, W)，则需要 img = img_data.transpose(1, 2, 0)
        # 例如，如果加载后是 (W, H, C)，则需要 img = img_data.transpose(1, 0, 2)
        # 根据你实际 .mat 文件存储方式判断
        # 检查维度顺序，假设是 H, W, C

        img = np.array(img_data).transpose(2,1,0) # 确保是 numpy array

        # --- 修改结束 ---
        print(f"  成功加载数据，原始尺寸: {img.shape}")


    # --- 1. 预处理和分割 ---
    print("  进行预处理 (裁剪和归一化)...")
    # 中心裁剪 (MATLAB: 107:2410, 144:2191) -> Python: 106:2410, 143:2191
    img = img[106:2410, 143:2191, :]
    # 归一化
    img = img.astype(np.float32) / np.max(img)
    print(f"  裁剪后尺寸: {img.shape}")

    # 分割测试集和训练集 (按行分割)
    test_img_size = 512
    test_data_full = img[:test_img_size, :, :]
    train_data_full = img[test_img_size:, :, :]
    print(f"  测试集原始尺寸: {test_data_full.shape}")
    print(f"  训练集原始尺寸: {train_data_full.shape}")

    # --- 2. 处理测试集 ---
    print("  处理测试集...")
    LRHSI_test_list = []
    GT_test_list = []
    h_test, w_test, c_test = test_data_full.shape
    num_test_scenes = w_test // test_img_size

    for i in tqdm(range(num_test_scenes), desc="  生成测试场景"):
        left = i * test_img_size
        right = left + test_img_size
        scene_gt = test_data_full[:, left:right, :] # (H, W, C)

        # 生成低分辨率图像
        lr_h, lr_w = h_test // scale, test_img_size // scale
        scene_lr = cv2.resize(scene_gt, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC) # (lr_H, lr_W, C)

        # 转换维度 (C, H, W)
        scene_gt_t = scene_gt.transpose(2, 0, 1)
        scene_lr_t = scene_lr.transpose(2, 0, 1)

        LRHSI_test_list.append(scene_lr_t)
        GT_test_list.append(scene_gt_t)

    # 转换为 numpy 数组并保存测试集
    if LRHSI_test_list and GT_test_list:
        LRHSI_test_array = np.array(LRHSI_test_list)
        GT_test_array = np.array(GT_test_list)
        save_name_test = f'Chikusei_test_x{scale}.npz'
        np.savez(os.path.join(save_path, save_name_test),
                 LRHSI=LRHSI_test_array,
                 GT=GT_test_array)
        print(f'  测试集已保存: {save_name_test}')
        print(f'  测试集 LRHSI shape: {LRHSI_test_array.shape}')
        print(f'  测试集 GT shape: {GT_test_array.shape}')
    else:
        print("  未能生成测试集数据。")

    # --- 3. 处理训练集 ---
    print("  处理训练集 (生成 Patches)...")
    # 生成整个训练集的低分辨率版本
    h_train, w_train, c_train = train_data_full.shape
    lr_h_train, lr_w_train = h_train // scale, w_train // scale
    print(f"  生成训练集 LR 图像 (尺寸: {lr_h_train}x{lr_w_train})...")
    train_data_lr_full = cv2.resize(train_data_full, (lr_w_train, lr_h_train), interpolation=cv2.INTER_CUBIC)

    # 裁剪 Patches
    print(f"  裁剪训练集 Patches (GT: {patch_size}x{patch_size}, LR: {patch_size//scale}x{patch_size//scale}, Stride: {stride})...")
    GT_train_patches, LRHSI_train_patches = crop_patches(train_data_full, train_data_lr_full, patch_size, stride, scale)

    # 保存训练集
    if LRHSI_train_patches.size > 0 and GT_train_patches.size > 0:
        save_name_train = f'Chikusei_train_x{scale}.npz'
        np.savez(os.path.join(save_path, save_name_train),
                 LRHSI=LRHSI_train_patches,
                 GT=GT_train_patches)
        print(f'  训练集已保存: {save_name_train}')
        print(f'  训练集 LRHSI shape: {LRHSI_train_patches.shape}')
        print(f'  训练集 GT shape: {GT_train_patches.shape}')
    else:
        print("  未能生成训练集数据 (Patches)。")

    print("-" * 30)


if __name__ == '__main__':
    # --- 配置参数 ---
    # !! 重要 !!: 请将此路径修改为你本地 Chikusei .mat 文件的实际路径
    # 例如: 'D:/Datasets/Chikusei/HyperspecVNIR_Chikusei_20140729.mat'
    # 或者 './Chikusei/Hyperspec_Chikusei_ENVI/Hyperspec_Chikusei_MATLAB/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat'
    # 请确保路径正确无误
    chikusei_mat_file_path = 'Chikusei\Hyperspec_Chikusei_ENVI\Hyperspec_Chikusei_MATLAB\Chikusei_MATLAB\HyperspecVNIR_Chikusei_20140729.mat'

    save_directory = './datasets' # 保存 .npz 文件的目录
    scales_to_generate = [2,4,8] # 需要生成的下采样倍数列表，例如 [2, 4, 8]
    train_patch_size = 64    # 训练集 GT patch 的边长
    train_patch_stride = 32  # 训练集 patch 的步长
    # --- 配置结束 ---

    if not os.path.exists(chikusei_mat_file_path):
         print(f"错误：无法找到 Chikusei 数据文件 '{chikusei_mat_file_path}'")
         print("请在脚本中修改 'chikusei_mat_file_path' 变量为正确的路径。")
    else:
        # 创建不同尺度的数据集
        for scale_factor in scales_to_generate:
            create_chikusei_dataset(chikusei_mat_file_path,
                                    save_directory,
                                    scale_factor,
                                    patch_size=train_patch_size,
                                    stride=train_patch_stride)
        print("所有数据集处理完成。")