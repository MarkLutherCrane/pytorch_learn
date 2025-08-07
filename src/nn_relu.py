import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3],])
input = torch.reshape(input, (-1,1,2,2))
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid1(x)
        return x


myNet = MyNet()

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("relu_input", imgs, step)
    output = myNet(imgs)
    writer.add_images("relu_output", output, step)
    step = step + 1
writer.close()

