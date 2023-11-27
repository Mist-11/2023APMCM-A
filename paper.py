from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('./model/finetune2.pt')

image_paths = './Attachment/Attachment 1/70.jpg'
results = model(image_paths)
center_points = []
area_list = []
m_list = []

for box in results[0].boxes.xyxy:
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    l = abs(x_max - x_min)
    w = abs(y_max - y_min)
    area = l * w
    center_points.append(((x_min + x_max) / 2, (y_min + y_max) / 2))
    area_list.append(area)

radii = [np.sqrt(a / np.pi) for a in area_list]
image = cv2.imread('./Attachment/Attachment 1/70.jpg')
cv2.imwrite('./result/面积绘制（前）.jpg', image)
for center, radius in zip(center_points, radii):
    center = (int(center[0]), int(center[1]))
    cv2.circle(image, center, int(radius), (0, 0, 255), 2)  # 红色圆形轮廓
cv2.imwrite('./result/image/面积绘制（后）.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

path = './Attachment/Attachment 1/70.jpg'
center_points = []
ellipse_axes = []
results = model(path)

for box in results[0].boxes.xyxy:
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    l = abs(x_max - x_min)
    w = abs(y_max - y_min)
    center_points.append(((x_min + x_max) / 2, (y_min + y_max) / 2))
    ellipse_axes.append((l // 2, w // 2))  # 整数除法，因为像素值必须是整数

image = cv2.imread('./Attachment/Attachment 1/70.jpg')
for center, axes in zip(center_points, ellipse_axes):
    center = (int(center[0]), int(center[1]))
    cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 255), 2)

cv2.imshow('Circles on Image', image)
cv2.imwrite('./result/image/S70.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
