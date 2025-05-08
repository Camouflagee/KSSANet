# author: LI JIANSHENG
# download the raw data from https://cave.cs.columbia.edu/repository/Multispectral
# ensure the dataset in the path 'KSSANET/complete_ms_data'
# running this code to produce the .npz file
import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt # 导入 matplotlib

def create_cave_dataset(data_path, save_path, scale):
    """
    创建CAVE数据集的npz文件
    Args:
        data_path: 原始数据路径
        save_path: 保存路径
        scale: 下采样倍数
    """
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
        
    scene_list = glob(os.path.join(data_path, '*_ms'))
    LRHSI_list = []
    GT_list = []
    
    for scene_path in scene_list[:-1]: #scene_list[:-1] means we don't load the folder: watercolors_ms cuz images inside this folder has three dimensions (512,512,4). others only have two dimensions (512,512)
        pathlist = os.path.split(scene_path)
        scene_name = pathlist[-1]
        scene_path = os.path.join(scene_path, scene_name)
        # 读取场景中的所有波段
        bands = sorted(glob(os.path.join(scene_path, '*.png')))
        if len(bands) != 31:  # CAVE数据集应该有31个波段
            continue
            
        # 读取并堆叠所有波段
        scene_data = []
        for band in bands:
            img = cv2.imread(band, cv2.IMREAD_UNCHANGED)
            scene_data.append(img)
        scene_data = np.stack(scene_data, axis=0)  # (31, H, W)
        scene_data = scene_data.astype(np.float32) / 65535.0
        # 生成低分辨率图像
        h, w = scene_data.shape[1:]
        lr_h, lr_w = h//scale, w//scale
        lr_data = cv2.resize(scene_data.transpose(1,2,0), (lr_w, lr_h), 
                           interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        #         # debugg --- 开始可视化代码 ---
        # # 选择一个波段进行可视化，例如中间波段 (索引 15)
        # band_index_to_show = 15
        # original_band = scene_data[band_index_to_show]
        # lr_band = lr_data[band_index_to_show]
        
        # plt.figure(figsize=(10, 5))
        
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_band, cmap='gray')
        # plt.title(f'Original Band {band_index_to_show} (Shape: {original_band.shape})')
        # plt.axis('off')
        
        # plt.subplot(1, 2, 2)
        # plt.imshow(lr_band, cmap='gray')
        # plt.title(f'LR Band {band_index_to_show} (Shape: {lr_band.shape}, Scale: x{scale})')
        # plt.axis('off')
        
        # plt.suptitle(f'Scene: {scene_name}')
        # plt.tight_layout()
        # plt.show() 
        # # --- 可视化代码结束 ---
        LRHSI_list.append(lr_data)
        GT_list.append(scene_data)
    
    # 转换为numpy数组并保存
    LRHSI_array = np.array(LRHSI_list)
    GT_array = np.array(GT_list)
    
    # 保存为npz文件
    save_name = f'CAVE_train_x{scale}.npz'
    np.savez(os.path.join(save_path, save_name),
             LRHSI=LRHSI_array,
             GT=GT_array)
    
    print(f'Dataset saved: {save_name}')
    print(f'LRHSI shape: {LRHSI_array.shape}')
    print(f'GT shape: {GT_array.shape}')

if __name__ == '__main__':
    data_path = 'CAVE/complete_ms_data'
    save_path = './datasets'
    
    # 创建不同尺度的数据集
    for scale in [2, 4, 8]:
        create_cave_dataset(data_path, save_path, scale)