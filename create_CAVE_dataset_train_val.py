# author: LI JIANSHENG
# download the raw data from https://cave.cs.columbia.edu/repository/Multispectral
# ensure the dataset in the path 'KSSANET/complete_ms_data'
# running this code to produce the .npz file
import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt # 导入 matplotlib
import random # 导入 random 库

def create_cave_dataset(data_path, save_path, scale):
    """
    创建CAVE数据集的npz文件，并划分为训练集和测试集
    Args:
        data_path: 原始数据路径
        save_path: 保存路径
        scale: 下采样倍数
    """
    if not os.path.exists(save_path): # 检查保存路径是否存在
        os.makedirs(save_path)

    # 定义训练集和测试集的数量
    num_train = 24
    num_test = 7

    # 获取所有场景路径，并排除 watercolors_ms
    all_scene_paths = glob(os.path.join(data_path, '*_ms'))
    valid_scene_paths = [p for p in all_scene_paths if 'watercolors_ms' not in p]

    # 检查是否有足够的场景
    if len(valid_scene_paths) < num_train + num_test:
        print(f"错误：有效场景数量 ({len(valid_scene_paths)}) 不足以划分训练集 ({num_train}) 和测试集 ({num_test})。")
        return

    # 随机打乱场景列表
    random.shuffle(valid_scene_paths)

    # 分割训练集和测试集场景
    train_scenes = valid_scene_paths[:num_train]
    test_scenes = valid_scene_paths[num_train : num_train + num_test]

    print(f"Scale x{scale}:")
    print(f"  总有效场景数: {len(valid_scene_paths)}")
    print(f"  训练场景数: {len(train_scenes)}")
    print(f"  测试场景数: {len(test_scenes)}")

    # --- 处理训练集 ---
    LRHSI_train_list = []
    GT_train_list = []
    print("  处理训练集...")
    for scene_path in train_scenes:
        pathlist = os.path.split(scene_path)
        scene_name = pathlist[-1]
        scene_folder_path = os.path.join(scene_path, scene_name) # 修正场景文件夹路径
        # 读取场景中的所有波段
        bands = sorted(glob(os.path.join(scene_folder_path, '*.png'))) # 使用修正后的路径
        if len(bands) != 31:  # CAVE数据集应该有31个波段
            print(f"    跳过场景 {scene_name} (波段数: {len(bands)} != 31)")
            continue

        # 读取并堆叠所有波段
        scene_data = []
        for band in bands:
            img = cv2.imread(band, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"    警告：无法读取图像 {band}")
                continue
            scene_data.append(img)

        if not scene_data: # 如果该场景没有成功读取任何波段
             print(f"    跳过场景 {scene_name} (无法读取任何波段)")
             continue

        scene_data = np.stack(scene_data, axis=0)  # (31, H, W)
        scene_data = scene_data.astype(np.float32) / 65535.0
        # 生成低分辨率图像
        h, w = scene_data.shape[1:]
        lr_h, lr_w = h//scale, w//scale
        lr_data = cv2.resize(scene_data.transpose(1,2,0), (lr_w, lr_h),
                           interpolation=cv2.INTER_CUBIC).transpose(2,0,1)

        LRHSI_train_list.append(lr_data)
        GT_train_list.append(scene_data)

    # 转换为numpy数组并保存训练集
    if LRHSI_train_list and GT_train_list:
        LRHSI_train_array = np.array(LRHSI_train_list)
        GT_train_array = np.array(GT_train_list)
        save_name_train = f'CAVE_train_x{scale}.npz'
        np.savez(os.path.join(save_path, save_name_train),
                 LRHSI=LRHSI_train_array,
                 GT=GT_train_array)
        print(f'  训练集已保存: {save_name_train}')
        print(f'  训练集 LRHSI shape: {LRHSI_train_array.shape}')
        print(f'  训练集 GT shape: {GT_train_array.shape}')
    else:
        print("  未能生成训练集数据。")


    # --- 处理测试集 ---
    LRHSI_test_list = []
    GT_test_list = []
    print("  处理测试集...")
    for scene_path in test_scenes:
        pathlist = os.path.split(scene_path)
        scene_name = pathlist[-1]
        scene_folder_path = os.path.join(scene_path, scene_name) # 修正场景文件夹路径
        # 读取场景中的所有波段
        bands = sorted(glob(os.path.join(scene_folder_path, '*.png'))) # 使用修正后的路径
        if len(bands) != 31:  # CAVE数据集应该有31个波段
            print(f"    跳过场景 {scene_name} (波段数: {len(bands)} != 31)")
            continue

        # 读取并堆叠所有波段
        scene_data = []
        for band in bands:
            img = cv2.imread(band, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"    警告：无法读取图像 {band}")
                continue
            scene_data.append(img)

        if not scene_data: # 如果该场景没有成功读取任何波段
             print(f"    跳过场景 {scene_name} (无法读取任何波段)")
             continue

        scene_data = np.stack(scene_data, axis=0)  # (31, H, W)
        scene_data = scene_data.astype(np.float32) / 65535.0
        # 生成低分辨率图像
        h, w = scene_data.shape[1:]
        lr_h, lr_w = h//scale, w//scale
        lr_data = cv2.resize(scene_data.transpose(1,2,0), (lr_w, lr_h),
                           interpolation=cv2.INTER_CUBIC).transpose(2,0,1)

        LRHSI_test_list.append(lr_data)
        GT_test_list.append(scene_data)

    # 转换为numpy数组并保存测试集
    if LRHSI_test_list and GT_test_list:
        LRHSI_test_array = np.array(LRHSI_test_list)
        GT_test_array = np.array(GT_test_list)
        save_name_test = f'CAVE_test_x{scale}.npz'
        np.savez(os.path.join(save_path, save_name_test),
                 LRHSI=LRHSI_test_array,
                 GT=GT_test_array)
        print(f'  测试集已保存: {save_name_test}')
        print(f'  测试集 LRHSI shape: {LRHSI_test_array.shape}')
        print(f'  测试集 GT shape: {GT_test_array.shape}')
    else:
        print("  未能生成测试集数据。")
    print("-" * 30)


if __name__ == '__main__':
    # 注意：请确保这里的 data_path 指向 CAVE 数据集的 'complete_ms_data' 文件夹的 **父** 目录
    # 例如，如果你的结构是 CAVE/complete_ms_data/beads_ms/...
    # 那么 data_path 应该是 'CAVE/complete_ms_data'
    data_path = 'CAVE/complete_ms_data'
    save_path = './datasets' # 保存路径保持不变

    # 创建不同尺度的数据集
    for scale in [2, 4, 8]:
        create_cave_dataset(data_path, save_path, scale)
