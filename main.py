import os
import numpy as np
from PIL import Image
from glob2 import glob as glob
import matplotlib.pyplot as plt

import time

# # 2020/12/20 - 10:13
# + Add OpenCV Lib
import cv2


def scale_range(array_2d, min_v, max_v, threshold_low=0.0, threshold_high=1.0):#compress image form p view to ont view
    np_min = np.min(array_2d)
    np_max = np.max(array_2d)
    for h in range(len(array_2d)):
        for w in range(len(array_2d[0])):
            if array_2d[h][w] <= int(threshold_low * (np_max - np_min)):
                array_2d[h][w] = 0
                continue
            if array_2d[h][w] >= int(threshold_high * (np_max - np_min)):
                array_2d[h][w] = 255
                continue
            array_2d[h][w] -= np_min
            array_2d[h][w] /= np_max / (max_v - min_v)
            array_2d[h][w] += min_v

        print("\rSuccessful Max: {}, Min {}.".format(array_2d.max(), array_2d.min()),
              end='')
    return array_2d


def scale_range_a(array_2d, min_v, max_v):
    np_min = np.min(array_2d)
    np_max = np.max(array_2d)
    for w in range(len(array_2d)):
        array_2d[w] -= np_min
        array_2d[w] /= np_max / (max_v - min_v)
        array_2d[w] += min_v

    print("\rSuccessful Max: {}, Min {}.".format(array_2d.max(), array_2d.min()),
          end='')

    return array_2d


def main():
    IMAGES_CT_FILENAME = []

    # print(len(os.listdir("E:\ct2xray\CT2Xray\ct\x1")))
    for i in range(len(os.listdir("/home/j203-1/CT2XRAY_LINUX/ct/mask"))):##iamge read
        IMAGES_CT_FILENAME.append("ct/mask/RibFrac424-image-{}-top.jpg".format(i))

    # print(IMAGES_CT_FILENAME)
    # print("----------------------")

    # IMAGES_CT_FILENAME_1 = glob(os.path.join("ct/X2", "*.jpg"), recursive=True)
    # IMAGES_CT_FILENAME.sort()

    assert len(IMAGES_CT_FILENAME) > 0, 'Must have image(s) in the resources directory!'

    ref_image = Image.open(IMAGES_CT_FILENAME[0]) ##image read






    WIDTH = ref_image.width#read image width

    LENGTH = len(IMAGES_CT_FILENAME)##read how many img

    ref_image.close()

    # # # # #
    # ! Test Use
    # img = Image.open(IMAGES_CT_FILENAME[0]).convert('L')
    # img_2 = Image.open(IMAGES_CT_FILENAME[100]).convert('L')
    # arr = np.array(img, dtype=np.uint8)
    # arr_2 = np.array(img, dtype=np.uint8)
    # for i in np.sum(arr, axis=0):
    #     for j in np.sum(arr_2, axis=0):
    #         if i != j:
    #             print("{} not eq {}".format(i, j))
    # print(np.sum(arr, axis=0))
    # print(np.sum(arr_2, axis=0))
    # plt.imsave("arr_save.png", arr, vmin=0, vmax=255, cmap="gray")
    # # # # #

    # # # # #
    # Calculate
    result_image_data = np.zeros((LENGTH, WIDTH), np.uint)#make class to 0


    Y = 0
    for ct_image in IMAGES_CT_FILENAME:
        # # 2020/12/20 10:18
        # + Add OpenCV read method
        img_orig = cv2.imread(ct_image)
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY) ##gray imge

        # 畫出直方圖

        # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯模糊
        # img_gray = cv2.medianBlur(img_gray, 9)##中值滤波
        # cv2.imshow( "result123.png", np.array(img_gray))# show gra    y image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return 0
        # cv2.imwrite("result31.png",img_gray)

        _, binary =cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)##Binarization
        # _, binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
        #                                   3), cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                                             cv2.THRESH_BINARY, 11, 3    ) 自適應二值化
        #
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img_gray, contours, -1, 0, cv2.FILLED)

        # --------------jast show image for now
        # cv2.imshow( "result123.png", np.array(img_gray))# show gray image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return 0
        # --------------

        # return 0
        # # 2020/12/20 10:15
        # - Change image read method
        # img = Image.open(ct_image).convert("L")
        data = np.array(img_gray, dtype=np.uint8)
        # +print(data)
        add_data = np.sum(data, axis=0)
        # print(add_data)
        # # - Deprecated
        # for w in range(WIDTH):
        #     for h in range(HEIGHT):
        #         result_image_data[Y][w] += data[h][w]
        # for i in range(1):
        print(len(add_data))
        result_image_data[Y] = add_data
        # result_image_data[Y] = scale_range_a(add_data, 0, 255)

        Y += 1
        print("\r{} - {}/{}, MAX: {}".format(ct_image, Y, LENGTH, np.max(add_data)), end='')
    # # 2020/12/20 10:22
    # + save
    print("\narray value normalizing!")
    img_arr = scale_range(result_image_data, 0, 255)
    image = Image.fromarray(np.uint8(img_arr))
    cv2.imwrite("result_orig.png", np.array(image))

    # img_xray_hist=cv2.imread('/home/j203-1/CT2XRAY_LINUX/source/5973453.png')
    # img_ct2xray_hist = cv2.imread('result.png')
    # # 轉為灰階圖片
    # gray_xray=cv2.cvtColor(img_xray_hist,cv2.COLOR_BGR2GRAY)
    # gray_ct2xray = cv2.cvtColor(img_ct2xray_hist, cv2.COLOR_BGR2GRAY)
    # # 畫出直方圖ct2xray
    # plt.hist(gray_xray.ravel(), 256, [0, 256])
    # plt.savefig('hist_xray.png')
    # plt.show()
    # plt.hist(gray_ct2xray.ravel(), 256, [0, 256])
    # plt.savefig('hist_ct2xray.png')
    # plt.show()

# ----------------------
# -----------------------

# # # # #
# - Use to Find best Threshold value
# for thres in range(1, 35, 1):
#     th = thres * 0.01
#     print("array Max: {}, Min {}. Threshold: {}".format(result_image_data.max(), result_image_data.min(), th))
#     print("array value normalizing!")
#     scale_range(result_image_data, 0, 255, threshold_low=th)
#     print("Successful Max: {}, Min {}.".format(result_image_data.max(), result_image_data.min()))
#     plt.imsave("output-threshold:{}.png".format(th), result_image_data, vmin=0, vmax=255, cmap="gray")

# # END
# RESULT = Image.fromarray(result_image_data, 'RGB')
# RESULT.save("test.png")
# RESULT.show()


if __name__ == '__main__':
    main()
