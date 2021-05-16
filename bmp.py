import cv2

image=cv2.imread('./home/j203-1/CT2XRAY_LINUX/drive-download-20210516T183116Z-001/RibFrac421-image-0-front.bmp')
cv2.imwrite('img1.png',image)