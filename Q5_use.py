import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import torch.nn as nn
import os
import csv
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 遍历文件夹并进行预测
folder_path = "./Attachment/Attachment 3"  # 替换为你的图片文件夹路径
results = []

model = resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 假设你有5个类别
model.load_state_dict(torch.load('./model/model_weights.pth'))
model.eval()  # 设置为评估模式

for img_file in tqdm(os.listdir(folder_path)):
    img_path = os.path.join(folder_path, img_file)
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 增加一个批次维度

    # 进行预测
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()

    results.append([img_file, predicted_label])

# 将结果保存到 CSV 文件
with open('./result/prediction_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Label"])
    writer.writerows(results)
