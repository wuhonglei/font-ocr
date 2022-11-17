import cv2
import numpy as np

from pprint import pprint

from training import helper

def predict_single_img(img: cv2.Mat) -> None:
    imgs = np.array([img])
     # 构建模型
    model = helper.create_model()
    # 加载前期训练好的权重
    model.load_weights('checkpoint/char_checkpoint')
    # 读出图片分类
    class_name = np.load('class_name.npy')
    # 预测图片，获取预测值
    predicts = model.predict(imgs)
    for predict in predicts:
        index = np.argmax(predict)  # 寻找最大值
        result = class_name[index]  # 取出字符
        return result


def predict_imgs(img_list: list[cv2.Mat]) -> list[str]:
    """
    设置待识别的图片
    """
    imgs = np.array(img_list)

    # 构建模型
    model = helper.create_model()
    # 加载前期训练好的权重
    model.load_weights('checkpoint/char_checkpoint')
    # 读出图片分类
    class_name = np.load('class_name.npy')
    # 预测图片，获取预测值
    predicts = model.predict(imgs)
    results = []  # 保存结果的数组
    for predict in predicts:  # 遍历每一个预测结果
        index = np.argmax(predict)  # 寻找最大值
        result = class_name[index]  # 取出字符
        results.append(result)

    return results


if __name__ == '__main__':
    # img1 = cv2.imread('./split/row_0/block_2/part_1.jpg', 0)
    # img2 = cv2.imread('./split/row_3/block_0/part_1.jpg', 0)
    img3 = cv2.imread('./split/row_0/block_1/part_1.jpg', 0)
    # img4 = cv2.imread('./split/row_0/block_1/part_3.jpg', 0)
    # + ÷ - =
    pprint(predict_imgs([img3]))
