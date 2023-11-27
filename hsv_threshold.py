import cv2
import numpy as np
from PIL import Image
from way import callback

# 读取图像
original_image = Image.open('./result/target/9.jpg')
target_size = (153, 150)  # 将图片调整为目标尺寸（158x140像素）
resized_image = original_image.resize(target_size, Image.ANTIALIAS)

# CV处理
image_np = np.array(resized_image)
image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 600, 600)

# 创建滑动条
cv2.createTrackbar('Hue Min', 'Result', 0, 179, callback)
cv2.createTrackbar('Hue Max', 'Result', 179, 179, callback)
cv2.createTrackbar('Sat Min', 'Result', 0, 255, callback)
cv2.createTrackbar('Sat Max', 'Result', 255, 255, callback)
cv2.createTrackbar('Val Min', 'Result', 0, 255, callback)
cv2.createTrackbar('Val Max', 'Result', 255, 255, callback)

while (True):
    # 获取滑动条的值
    h_min = cv2.getTrackbarPos('Hue Min', 'Result')
    h_max = cv2.getTrackbarPos('Hue Max', 'Result')
    s_min = cv2.getTrackbarPos('Sat Min', 'Result')
    s_max = cv2.getTrackbarPos('Sat Max', 'Result')
    v_min = cv2.getTrackbarPos('Val Min', 'Result')
    v_max = cv2.getTrackbarPos('Val Max', 'Result')

    # 设置HSV阈值
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # 颜色分割
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 显示结果
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
