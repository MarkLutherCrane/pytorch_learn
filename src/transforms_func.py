from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("../logs")
img = Image.open("../hymenoptera_data/train/bees/342758693_c56b89b6b6.jpg")
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


# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))  # 调整原图大小
img_resize = trans_resize(img)
# img(PIL) -> totensor -> img_resize(tensor)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)


# Compose
# PIL -> PIL -> tensor
trans_resize_2 = transforms.Resize(512) # 短边=512, 长边=max(H,W)/min(H,W)*512
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)



# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop((50,100));
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i+10)

writer.close()