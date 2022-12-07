import numpy as np
import cv2
import os
import torchvision.transforms.functional as TF
from PIL import Image

# 数据增强后的图片保存路劲

save_path = 'D://good_dataset/train/0.normal/'

for info in os.listdir(r'D://good_dataset/train/0.normal'):
    domain = os.path.abspath(
        r'D://good_dataset/train/0.normal')  # 获取文件夹的路径，此处其实没必要这么写，目的是为了熟悉os的文件夹操作
    info1 = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
    img = cv2.imread(info1)
    img = Image.fromarray(np.uint8(img))
    # img = Image.open(info1)
    info = info.split(".")[0]

    # 亮度变换
    brightness_img_2 = np.asarray(TF.adjust_brightness(img, 0.2))
    brightness_img_4 = np.asarray(TF.adjust_brightness(img, 0.6))
    brightness_img_6 = np.asarray(TF.adjust_brightness(img, 1.5))
    brightness_img_8 = np.asarray(TF.adjust_brightness(img, 2.0))
    cv2.imwrite(save_path + info + '_brightness_img_2.jpg', brightness_img_2)
    cv2.imwrite(save_path + info + '_brightness_img_4.jpg', brightness_img_4)
    cv2.imwrite(save_path + info + '_brightness_img_6.jpg', brightness_img_6)
    cv2.imwrite(save_path + info + '_brightness_img_8.jpg', brightness_img_8)

    # 对比度
    contrast_img_2 = np.asarray(TF.adjust_contrast(img, 0.2))
    contrast_img_4 = np.asarray(TF.adjust_contrast(img, 0.6))
    contrast_img_6 = np.asarray(TF.adjust_contrast(img, 1.5))
    contrast_img_8 = np.asarray(TF.adjust_contrast(img, 2.0))
    cv2.imwrite(save_path + info + '_contrast_img_2.jpg', contrast_img_2)
    cv2.imwrite(save_path + info + '_contrast_img_4.jpg', contrast_img_4)
    cv2.imwrite(save_path + info + '_contrast_img_6.jpg', contrast_img_6)
    cv2.imwrite(save_path + info + '_contrast_img_8.jpg', contrast_img_8)

    # 饱和度
    saturation_img_2 = np.asarray(TF.adjust_saturation(img, -0.2))
    saturation_img_4 = np.asarray(TF.adjust_saturation(img, 0))
    saturation_img_6 = np.asarray(TF.adjust_saturation(img, 0.4))
    saturation_img_8 = np.asarray(TF.adjust_saturation(img, 2))
    cv2.imwrite(save_path + info + '_saturation_img_2.jpg', saturation_img_2)
    cv2.imwrite(save_path + info + '_saturation_img_4.jpg', saturation_img_4)
    cv2.imwrite(save_path + info + '_saturation_img_6.jpg', saturation_img_6)
    cv2.imwrite(save_path + info + '_saturation_img_8.jpg', saturation_img_8)