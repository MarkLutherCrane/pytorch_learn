from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")
# 1. 向tensorboard添加图片
image_path = "hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
image_array = np.array(img_PIL)
writer.add_image("Test", image_array, 1, dataformats="HWC")
#global_step 是指图片变化的步骤
# 2. 向tensorboard里绘制图像
# y = x
# for i in range(100):
    # writer.add_scalar("y=x^2", i*i, i)

writer.close()