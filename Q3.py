import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from way import getMaturity
import numpy as np

show = True
level = True

if show:
    high_image_path = './result/target/57.jpg'
    middle_image_path = './result/target/1269.jpg'
    low_image_path = './result/target/780.jpg'

    high_image = cv2.imread(high_image_path)
    middle_image = cv2.imread(middle_image_path)
    low_image = cv2.imread(low_image_path)

    high_hsv = cv2.cvtColor(high_image, cv2.COLOR_BGR2HSV)
    middle_hsv = cv2.cvtColor(middle_image, cv2.COLOR_BGR2HSV)
    low_hsv = cv2.cvtColor(low_image, cv2.COLOR_BGR2HSV)

    high_hist = cv2.calcHist([high_hsv], [0], None, [180], [0, 180])
    middle_hist = cv2.calcHist([middle_hsv], [0], None, [180], [0, 180])
    low_hist = cv2.calcHist([low_hsv], [0], None, [180], [0, 180])

    # 转换直方图数据为一维数组
    high_hist = high_hist.flatten()
    middle_hist = middle_hist.flatten()
    low_hist = low_hist.flatten()
    # 创建 x 轴数据
    x_data = np.arange(len(high_hist))  # 假设所有直方图都有相同的长度
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", 3)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x_data, y=high_hist, color=palette[0], label='High Maturity')
    sns.lineplot(x=x_data, y=middle_hist, color=palette[1], label='Middle Maturity')
    sns.lineplot(x=x_data, y=low_hist, color=palette[2], label='Low Maturity')
    plt.title('Hue Histograms for Different Maturity Levels')
    plt.xlabel('Hue')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('./result/image/Hue Histograms for Different Maturity Levels.png')
    plt.show()

if level:
    Maturity = []
    for i in range(1, 2795):
        image_path = './result/target/{}.jpg'.format(i)
        Maturity.append(getMaturity(image_path))
    # 频率分布直方图
    plt.figure(figsize=(8, 8))
    sns.histplot(Maturity, bins=[0.5, 1.5, 2.5, 3.5, 4.5], kde=False)
    plt.xlabel('Maturity')
    plt.ylabel('Frequency')
    plt.xticks([1, 2, 3, 4],
               ['Level 1', 'Level 2', 'Level 3', 'Level 4'])
    plt.title('Histogram of apple maturity')
    plt.savefig('./result/image/maturity.png')
    plt.show()
