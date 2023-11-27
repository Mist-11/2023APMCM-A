import cv2
import numpy as np
import os


def callback(x):
    pass


def getMaturity(path):
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    red_data = hist[(hist >= 0) & (hist <= 20) | (hist >= 170) & (hist <= 180)]
    percentage = round(len(red_data) / len(hist), 2)
    if percentage > 0.85:
        return 1  # 生理成熟
    elif 0.75 <= percentage <= 0.85:
        return 2  # 食品成熟
    elif 0.65 <= percentage < 0.75:
        return 3  # 可采成熟
    else:
        return 4  # 未成熟


# 红色HSV阈值
lower_red = np.array([0, 109, 72])
upper_red = np.array([179, 255, 255])
# 绿色HSV阈值
lower_green = np.array([25, 52, 72])
upper_green = np.array([102, 255, 255])


def getmask(image):  # 返回红色/绿色掩码
    red_mask = cv2.inRange(image, lower_red, upper_red)
    green_mask = cv2.inRange(image, lower_green, upper_green)
    # 计算每个掩膜的面积
    green_area = cv2.countNonZero(green_mask)
    red_area = cv2.countNonZero(red_mask)
    diff = abs(green_area - red_area) / max(green_area, red_area)
    if diff > 0.2:
        if green_area > red_area:
            return green_mask
        else:
            return red_mask
    else:  # 返回并集
        mask = cv2.bitwise_or(green_mask, red_mask)
        return mask


def nolight(image_cv):
    hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]
    _, highlights_mask = cv2.threshold(v_channel, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    highlights_mask = cv2.morphologyEx(highlights_mask, cv2.MORPH_CLOSE, kernel)
    highlights_mask = cv2.morphologyEx(highlights_mask, cv2.MORPH_OPEN, kernel)
    inpaint_radius = 3  # inpaint处理的半径
    image_cv = cv2.inpaint(image_cv, highlights_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return image_cv


def getpath(path):
    image_paths = []
    # 遍历1.jpg到200.jpg图像文件
    for i in range(1, 10000):
        file_path = os.path.join(path, f"{i}.jpg")  # 构建文件路径
        if os.path.isfile(file_path):  # 确认文件存在
            image_paths.append(file_path)  # 将文件路径添加到列表中
        else:
            break
    return image_paths


def openimage(width, height, limit=75):
    scale_factor = min(limit / width, limit / height)
    return round(width * scale_factor, 0), round(height * scale_factor)


def get_m(l, w):
    R = (l + w) / 20
    v = 0.75 * np.pi * R ** 3
    m = v * 0.82
    return int(m/3)
