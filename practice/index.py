import os
import sys
sys.path.append(os.path.join(os.getcwd()))

from training import predict
import shutil
from pprint import pprint
from typing import Literal, TypeAlias, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math


Position: TypeAlias = tuple[int, int, int, int]


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


def img_show_gray(img: cv2.Mat):
    # 展示图片，路径展示方式
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.show()


def img_show_array(a: cv2.Mat):
    plt.imshow(a)
    plt.show()


def to_binary_img(img: cv2.Mat) -> cv2.Mat:
    thresh = 200
    # 二值化并且反色
    ret, img_b = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    return img_b


# 展示投影图， 输入参数arr是图片的二维数组，direction是x,y轴
def show_shadow(arr: list[int], direction: Literal['x', 'y'] = 'y'):
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
            for j in range(0, arr[i]):
                a_shadow[i][j] = 255

    img_show_array(a_shadow)


def split_img_rows(shadow_y: list, dimension: tuple[int, int]) -> list[Position]:
    """
    按行切割图片
    """
    height, width = dimension
    inLine = False
    startRow = 0
    mark_boxs = []
    for (r, pixels) in enumerate(shadow_y):
        if inLine == False and pixels > 10:
            inLine = True
            startRow = r
        elif inLine == True and pixels < 10 and (r - startRow) > 5:
            # 大于 5 且 小于 10 时，重新开始计数，避免多余笔划像素干扰
            inLine = False
            if (r - startRow) > 10:
                top = max(startRow - 1, 0)
                bottom = min(height, r + 1)
                bounding = (
                    0, top, width, bottom)  # left top, right bottom
                mark_boxs.append(bounding)

    return mark_boxs


def split_img_columns(shadow_x: list, dimension: tuple[int, int], thresh=5) -> list[Position]:
    """
    按列切割图片
    """
    height, width = dimension
    inColumn = False
    startColum = 0
    mark_boxs = []
    for (c, pixels) in enumerate(shadow_x):
        if inColumn == False and pixels > thresh:
            inColumn = True
            startColum = c
        elif inColumn == True and pixels < thresh and (c - startColum) > 5:
            # 大于 5 且 小于 10 时，重新开始计数，避免多余笔划像素干扰
            inColumn = False
            left = max(startColum - 1, 0)
            right = min(width, c + 1)
            bounding = (
                left, 0, right, height)  # left top, right bottom
            mark_boxs.append(bounding)

    return mark_boxs


def get_square_img(image: cv2.Mat):
    x, y, w, h = cv2.boundingRect(image)
    image = image[y:y+h, x:x+w]

    max_size = 18
    max_size_and_border = 24

    if w > max_size or h > max_size:  # 有超过宽高的情况
        if w >= h:  # 宽比高长，压缩宽
            times = max_size/w
            w = max_size
            h = int(h*times)
        else:  # 高比宽长，压缩高
            times = max_size/h
            h = max_size
            w = int(w*times)
        # 保存图片大小
        image = cv2.resize(image, (w, h))

    xw = image.shape[0]
    xh = image.shape[1]
    xwLeftNum = int((max_size_and_border-xw)/2)
    xwRightNum = (max_size_and_border-xw) - xwLeftNum

    xhLeftNum = int((max_size_and_border-xh)/2)
    xhRightNum = (max_size_and_border-xh) - xhLeftNum

    img_large = np.pad(image, ((xwLeftNum, xwRightNum),
                       (xhLeftNum, xhRightNum)), 'constant', constant_values=(0, 0))

    return img_large


def cut_imgs(img: cv2.Mat, mark_boxs: list[Position], is_square: bool = False) -> list[cv2.Mat]:
    img_items = []  # 存放裁剪好的图片
    for bounding in mark_boxs:
        left, top, right, bottom = bounding
        img_item = img[top:bottom, left:right]
        if is_square:
            img_item = get_square_img(img_item)
        img_items.append(img_item)
    return img_items

# 裁剪图片,img 图片数组， mark_boxs 区域标记


def save_imgs(dir: str, img_list: list) -> list[str]:
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    img_paths = []
    for (index, img) in enumerate(img_list):
        if len(img) == 0:
            continue
        filepath = '{dir}/part_{index}.jpg'.format(dir=dir, index=index)
        cv2.imwrite(filepath, img)
        img_paths.append(filepath)
    return img_paths


def dilate_img(img: cv2.Mat) -> cv2.Mat:
    kernel = np.ones((3, 3), np.uint8)  # 膨胀核大小
    return cv2.dilate(img, kernel, iterations=6)  # 图像膨胀6次


def get_position_of_char(row_position: Position, block_position: Position, char_position: Position) -> Position:
    """
    计算某字符在原图中的绝对位置
    """
    return (row_position[0] + block_position[0] + char_position[0], 
            row_position[1] + block_position[1] + char_position[1], 
            row_position[0] + block_position[0] + char_position[2],
            row_position[1] + block_position[1] + char_position[3], 
            )


def split_chars():
    ori_img = cv2.imread('./practice/imgs/question.png', 0)
    reversed_img = to_binary_img(ori_img)
    img_y_shadow_a = img_y_shadow(reversed_img)
    img_x_shadow_a = img_x_shadow(dilate_img(reversed_img))
    # 切分行
    img_rows_position = split_img_rows(img_y_shadow_a, reversed_img.shape)
    row_imgs = cut_imgs(ori_img, img_rows_position)
    all_char_imgs: list[list[list[cv2.Mat]]] = []  # 记录所有字符
    all_mark_boxs: list[list[list[Position]]] = []  # 记录所有字符在大图中的绝对位置
    # 从每行切分块（按列切分）
    for row_index, row_img in enumerate(row_imgs):
        binary_row_img = to_binary_img(row_img)
        enlarge_row_img = dilate_img(binary_row_img)
        x_shadow_of_row = img_x_shadow(enlarge_row_img)
        blocks_of_row_positions = split_img_columns(
            x_shadow_of_row, enlarge_row_img.shape)
        blocks_of_row = cut_imgs(row_img, blocks_of_row_positions)
        # save_imgs('./split/row_{}'.format(row_index), blocks_of_row)
        # 存储每行的字符块图片
        char_img_of_row: list[list[cv2.Mat]] = []
        mark_boxs_of_row: list[list[Position]] = []
        # 切分每块的每个字符（按列切分）
        for block_index, block_img in enumerate(blocks_of_row):
            binary_block_img = to_binary_img(block_img)
            shadow_of_block = img_x_shadow(binary_block_img)
            chars_of_block_position = split_img_columns(
                shadow_of_block, block_img.shape, thresh=2)
            chars_of_block = cut_imgs(
                binary_block_img, chars_of_block_position, True)
            # save_imgs('./split/row_{}/block_{}'.format(row_index,
            #           block_index), chars_of_block)
            char_img_of_row.append(chars_of_block)
            mark_boxs_of_row.append([get_position_of_char(img_rows_position[row_index],
                                                          blocks_of_row_positions[block_index], char_position)for char_position in chars_of_block_position])
        all_char_imgs.append(char_img_of_row)
        all_mark_boxs.append(mark_boxs_of_row)
    return (all_char_imgs, all_mark_boxs, ori_img)

# 绘制文本
def cv2ImgAddText(img: cv2.Mat, text: str, left:int, top:int, textColor, textSize):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("./sample/fonts/AppleSDGothicNeo.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

Color: TypeAlias = tuple[int, int, int]
def calculate(chars: list[str]) -> tuple[str, Color]:
    """
    计算运算结果（√、×、正确答案）
    """
    chars_str = ''.join(chars)
    if '=' not in chars_str:
        return ('', (0,0,0))

    equal_index = chars_str.find('=')
    expression = chars_str[0:equal_index]
    initial_result = chars_str[equal_index + 1:]

    if '÷' in expression:
        expression = expression.replace('÷', '/')
    elif '×' in expression:
        expression = expression.replace('×', '*')

    res:str
    try:
        res = str(math.trunc(eval(expression)))
    except Exception as e:
        print("Exception",e)
        return ('', (0,0,0))

    # 没有写答案
    if not initial_result:
        return (res, (192, 192,192))

    if res == initial_result:
        return ('√', (0, 255, 0))
    else:
        return ('×', (255, 0, 0))
    

def main():
    all_char_imgs, all_mark_boxs, ori_img = split_chars()
    for (row_index, char_of_row) in enumerate( all_char_imgs):
        for (block_index, char_of_block) in enumerate(char_of_row):
            # 预测块内的多个字符
            block_chars = predict.predict_imgs(char_of_block)
            result, color = calculate(block_chars)
            (left, top, right, bottom) = all_mark_boxs[row_index][block_index][-1]
            iw = right - left
            ih = bottom - top
            textSize = max(iw, ih)
            ori_img = cv2ImgAddText(ori_img, result, right + 3, top, color, textSize)

    # 将写满结果的原图保存
    cv2.imwrite('result.jpg', ori_img)



if __name__ == '__main__':
    main()
