from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")
img = Image.open("hymenoptera_data/train/bees/342758693_c56b89b6b6.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
print(img_tensor[0][0][1])
print(img_tensor[0][0][2])
print("=====================")
trans_norm = transforms.Normalize([0.51798377, 0.47618317, 0.34957992],  [0.27797324, 0.25808253, 0.28743273])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
print(img_norm[0][0][1])
print(img_norm[0][0][2])
writer.add_image("Normalize", img_norm, 3)

writer.close()