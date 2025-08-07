import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)



input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1],])
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # ceil=True会补齐不够的位置

    def forward(self, x):
        x = self.maxpool1(x)
        return x

myNet = MyNet()
# output = myNet(input)
# print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = myNet(imgs)
    writer.add_images("maxpool_input", imgs, step)
    writer.add_images("maxpool_output", output, step)

    step = step + 1

writer.close()