import pdb
from pprint import pprint
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import os
import shutil
from numpy.core.records import array
from numpy.core.shape_base import block
import time


def img_y_shadow(img: cv2.Mat):
    (h, w) = img.shape
    acc_y = [0 for _ in range(0, h, 1)]
    for r in range(0, h, 1):
        for c in range(0, w, 1):
            if img[r][c] == 255:
                acc_y[r] += 1

    return acc_y


def img_x_shadow(img: cv2.Mat):
    (h, w) = img.shape
    acc_w = [0 for _ in range(0, w, 1)]
    for c in range(0, w, 1):
        for r in range(0, h, 1):
            if img[r][c] == 255:
                acc_w[c] += 1

    return acc_w


def show_shadow(img: cv2.Mat,  direction: str = 'x'):
    pass


def img_show_gray(img: cv2.Mat):
    # 展示图片，路径展示方式
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.show()


def img_show_array(a):
    plt.imshow(a)
    plt.show()


def format_img(file_path: str):
    img = cv2.imread(file_path, 0)
    thresh = 200
    # 二值化并且反色
    ret, img_b = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    return img_b


# 展示投影图， 输入参数arr是图片的二维数组，direction是x,y轴
def show_shadow(arr, direction='x'):

    a_max = max(arr)
    if direction == 'x':  # x轴方向的投影
        a_shadow = np.zeros((a_max, len(arr)), dtype=int)
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                a_shadow[j][i] = 255
    elif direction == 'y':  # y轴方向的投影
        a_shadow = np.zeros((len(arr), a_max), dtype=int)
        for i in range(0, len(arr)):  # 遍历行
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                a_shadow[i][j] = 255

    img_show_array(a_shadow)


def main():
    reversed_img = format_img('./practice/imgs/example1.png')
    img_y_shadow_a = img_y_shadow(reversed_img)
    img_x_shadow_a = img_x_shadow(reversed_img)
    show_shadow(img_y_shadow_a, 'y')
    show_shadow(img_x_shadow_a, 'x')


if __name__ == '__main__':
    main()
