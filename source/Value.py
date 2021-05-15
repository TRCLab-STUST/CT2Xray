import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖檔
img = cv2.imread('result1.png')
img_xray = cv2.imread('6412776.png')
# 轉為灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_xray = cv2.cvtColor(img_xray, cv2.COLOR_BGR2GRAY)
# 畫出直方圖
plt.hist(gray.ravel(), 256, [0, 256])
#plt.hist(gray_xray.ravel(), 256, [0, 256])
plt.show()
plt.hist(gray_xray.ravel(), 256, [0, 256])
plt.show()