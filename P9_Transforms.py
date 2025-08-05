from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
"""
tensor数据类型

通过transforms.ToTensor去解决俩个问题
1、transforms该如何使用
2、Tensor数据类型
"""

img_path = "hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()


print(tensor_img)