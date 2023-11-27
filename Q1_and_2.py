from ultralytics import YOLO
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from way import getpath

# 全局参数设置
TRAIN = False
PRE = True
markers = False
tailor = False
point = True
count = False

apple_list = []
center_points = []
apple_num = 0

if TRAIN:
    model = YOLO('yolov8n.pt')  # 加载预训练模型（用于训练）

    # 训练模型
    results = model.train(data='./apple/data.yaml', epochs=50, imgsz=640)
    model.export(format='onnx')

if PRE:
    model = YOLO('./model/finetune2.pt')
    f1_path = './Attachment/Attachment 1/'
    image_paths = getpath(f1_path)
    # 打印图像路径列表
    # print(image_paths)
    results = model(image_paths)
    # print(results)
    # 处理结果列表
    a = 1
    for r in results:
        # print(r.boxes.xyxy)  # 打印包含检测边界框的Boxes对象
        path = './Attachment/Attachment 1/' + str(a) + '.jpg'
        image = cv2.imread(path)
        num = 0
        for box in r.boxes.xyxy:
            num += 1
            apple_num += 1
            x_min, y_min, x_max, y_max = box
            # 将坐标转换为整数
            x_min, y_min, x_max, y_max = int(x_min), 185 - int(y_min), int(x_max), 185 - int(y_max)
            if markers:
                # 在图像上绘制边界框
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                output_path = './result/count/' + str(a) + '.jpg'  # 设置保存图像的路径
                cv2.imwrite(output_path, image)  # 保存图像
            if tailor:
                cropped_image = image[y_min:y_max, x_min:x_max]
                output_path = './result/target/' + str(apple_num) + '.jpg'
                cv2.imwrite(output_path, cropped_image)
            if point:
                # 中心点计算
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                center_points.append((center_x, center_y))
        if count:
            apple_list.append(num)
        a += 1

if count:
    # 频率分布直方图
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(apple_list, bins=range(min(apple_list), max(apple_list) + 1), kde=False, color='skyblue',
                 edgecolor='black')
    # Add labels and title with improved font sizes for better readability
    plt.xlabel('The number of apples', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of apple distribution', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('./result/image/Histogram.png')
    plt.show()
if point:
    # 散点图
    center_x, center_y = zip(*center_points)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    color = (30 / 255, 144 / 255, 255 / 255)  # 点颜色
    sns.scatterplot(x=center_x, y=center_y, color=color, s=10, edgecolor='w', linewidth=1)
    plt.xlabel('X axis pixel points', fontsize=14)
    plt.ylabel('Y axis pixel points', fontsize=14)
    plt.title('Position of the apples', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('./result/image/scatter.png')
    # 显示散点图
    plt.show()
