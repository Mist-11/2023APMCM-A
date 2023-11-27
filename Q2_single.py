from ultralytics import YOLO
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

model = YOLO('./model/finetune2.pt')
image_paths = './Attachment/Attachment 1/70.jpg'
results = model(image_paths)
center_points = []

for box in results[0].boxes.xyxy:
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), 185 - int(y_min), int(x_max), 185 - int(y_max)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_points.append((center_x, center_y))

center_x, center_y = zip(*center_points)
color = (30 / 255, 144 / 255, 255 / 255)  # 点颜色
sns.scatterplot(x=center_x, y=center_y, color=color)
# 添加标签和标题
plt.xlabel('X axis pixel points')
plt.ylabel('Y axis pixel points')
plt.title('Position of the apples')
plt.xlim(0, 270)
plt.ylim(1, 185)
plt.savefig('./result/image/single_scatter.png')
# 显示散点图
plt.show()
