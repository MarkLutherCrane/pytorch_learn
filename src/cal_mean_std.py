import os
import numpy as np
from PIL import Image
from torchvision import transforms

# 数据集根目录
data_dir = "../hymenoptera_data/train"

# 初始化统计变量 RGB
channel_sum = np.zeros(3) # 每个通道像素值和
channel_sq_sum = np.zeros(3) # 平方和
total_pixels = 0  # 像素点个数

# ToTensor 归一化到0~1
to_tensor = transforms.ToTensor()

img_cnt = 0
# 递归遍历子文件夹
for root, dirs, files in os.walk(data_dir):
    # 遍历当前文件夹下的所有文件
    for filename in files:
        # 只处理图像文件
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            # 构建完整路径
            img_path = os.path.join(root, filename)

            try:
                # 打开图像并转换为RGB模式 （避免灰度图或RGBA格式影响）
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    # 转为Tensor (形状:[C,H,W] 值0~1)
                    img_tensor = to_tensor(img_rgb)

                    # 累加每个通道的像素和
                    channel_sum += img_tensor.sum(dim=[1,2]).numpy()
                    # 累加每个通道的像素值平方和
                    channel_sq_sum += (img_tensor **2).sum(dim=[1,2]).numpy()
                    # 累加总像素 H*W
                    total_pixels += img_tensor.shape[1] * img_tensor.shape[2]


            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue

mean = channel_sum / total_pixels
std = np.sqrt(channel_sq_sum / total_pixels - mean** 2)
print(f"数据集均值 (RGB): {mean}")
print(f"数据集标准差 (RGB): {std}")