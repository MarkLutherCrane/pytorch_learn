from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx] # 图片的名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 组装img路径
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        """
        Get total number of samples in dataset

        Returns:
            int: 数据集中的图像数量
                 Number of images in the dataset
        """
        return len(self.img_path)

root_dir = "../hymenoptera_data/train"
ant_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ant_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset

print(len(train_dataset)) # 打印俩个数据集的大小

# 展示一张图片
img, label = train_dataset[124]
img.show()