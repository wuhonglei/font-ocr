import os
import re
import shutil
from pprint import pprint

from PIL import Image, ImageDraw, ImageFont

chars = ['0', '1', '2', '3', '4', '5', '6',
         '7', '8', '9', '=', '+', '-', '×', '÷']


def mkdir_dataset():
    """
    文本对应的文件夹，给每一个分类建一个文件
    """
    dir = "dataset"
    if os.path.isdir(dir):
        shutil.rmtree(dir)

    for char in chars:
        train_images_dir = dir + "/"+str(char)
        os.makedirs(train_images_dir)  # mkdir directory recursively


# %% 生成图片
def make_image(chars: list, font_path, width=24, height=24, rotate=0):

    # 从字典中取出键值对
    for char in chars:
        img = Image.new('RGB', (width, height), "black")
        draw = ImageDraw.Draw(img)
        # 加载一种字体,字体大小是图片宽度的90%
        font = ImageFont.truetype(font_path, int(width))
        # 获取字体的宽高
        # 计算字体绘制的x,y坐标，主要是让文字画在图标中心
        bounding = draw.textbbox(xy=(0, 0), text=char, font=font)
        (left, top, right, bottom) = bounding
        font_width = right - left
        font_height = bottom - top
        x = (width - font_width - left * 2)/2
        y = (height - font_height - top * 2)/2
        draw.text((x, y), char,
                  fill=(255, 255, 255), font=font)
        # 设置图片倾斜角度
        img = img.rotate(rotate)
        # 命名文件保存，命名规则：dataset/编号/img-编号_r-选择角度_字体.png
        font_name: str = os.path.basename(font_path).lower()
        normalize_font_name = re.sub('(\s|\.)', '_', font_name)
        img_path = "./dataset/{}/img-{}_r-{}_{}.png".format(
            char, char, rotate, normalize_font_name)
        img.save(img_path)


# %% 存放字体的路径


def make_diff_images():
    font_dir = "./sample/fonts"
    for font_name in os.listdir(font_dir):
        # 把每种字体都取出来，每种字体都生成一批图片
        path_font_file = os.path.join(font_dir, font_name)
        # 倾斜角度从 -10 到 10 度，每个角度都生成一批图片
        for rotate in range(-3, 3, 1):
            # 每个字符都生成图片
            make_image(chars, path_font_file, rotate=rotate)


def main():
    pprint('--[sample] start generate samples of numbers--')
    mkdir_dataset()
    make_diff_images()
    pprint('--[sample] finish generate samples of numbers--')


if __name__ == '__main__':
    main()
