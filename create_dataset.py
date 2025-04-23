# author: LI JIANSHENG
# download the raw data from https://cave.cs.columbia.edu/repository/Multispectral
# ensure the dataset in the path 'KSSANET/complete_ms_data'
# running this code to produce the .npz file
import numpy as np
import os
from glob import glob
import cv2

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
    
    for scene_path in scene_list:
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
            img = cv2.imread(band, cv2.IMREAD_GRAYSCALE)
            scene_data.append(img)
        scene_data = np.stack(scene_data, axis=0)  # (31, H, W)
        
        # 生成低分辨率图像
        h, w = scene_data.shape[1:]
        lr_h, lr_w = h//scale, w//scale
        lr_data = cv2.resize(scene_data.transpose(1,2,0), (lr_w, lr_h), 
                           interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        
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
    data_path = './complete_ms_data'
    save_path = './datasets'
    
    # 创建不同尺度的数据集
    for scale in [2, 4, 8]:
        create_cave_dataset(data_path, save_path, scale)