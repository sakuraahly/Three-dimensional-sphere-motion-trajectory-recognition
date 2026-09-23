import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图片（替换为你自己的图片路径）
image = cv2.imread("test.jpg")

# 2. 转为灰度图（Canny 需要单通道输入）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 高斯模糊降噪（5x5 核），减少噪点对边缘的干扰
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Canny 边缘检测，双阈值：低阈值 50，高阈值 150
edges = cv2.Canny(blurred, 50, 150)

# 5. 用 matplotlib 并排显示原图与边缘图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.tight_layout()
plt.show()

# 6. 保存边缘检测结果
cv2.imwrite("edges_result.jpg", edges)
print("边缘检测结果已保存为 edges_result.jpg")
