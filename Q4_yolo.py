from ultralytics import YOLO
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from way import getpath, get_m, openimage

model = YOLO('./model/finetune2.pt')

image_paths = './Attachment/Attachment 1/'
path = getpath(image_paths)
results = model(path)
m_list = []

for r in results:
    for box in r.boxes.xyxy:
        x_min, y_min, x_max, y_max = box
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        l = abs(x_max - x_min)
        w = abs(y_max - y_min)
        l, w = openimage(l, w)
        m = get_m(l, w)
        m_list.append(m)



# 使用Seaborn的样式
sns.set(style="whitegrid")

# 绘制直方图，并调整柱子的边框宽度和颜色
sns.histplot(m_list, bins=range(min(m_list), max(m_list) + 1), kde=False,
             color='skyblue', alpha=0.7, edgecolor='slateblue', linewidth=1.5)

# 设置标签和标题
plt.xlabel('The quality of apples', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Histogram of apple quality distribution', fontsize=14, fontweight='bold')
# 设置轴刻度大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# 增加网格线
plt.grid(True, linestyle='--', alpha=0.5)
# 去除上方和右侧的轴线
sns.despine()
plt.savefig('./result/image/Apple quality distribution map.png')
# 显示图表
plt.show()
