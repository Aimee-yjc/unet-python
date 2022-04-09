# -*- coding: UTF-8 -*-

"""
@Project ：unet 图像分割
@File    ：dataset.py
@IDE     ：PyCharm
@Author  ：Aimee
@Date    ：2022/4/19
"""
import glob
import  SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

if __name__ == '__main__':
    flair_train = glob.glob(r'F:/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/*/*flair.nii.gz')
    seg_train = glob.glob(r'F:/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/*/*seg.nii.gz')

    flair_test = glob.glob(r'F:/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/*/*flair.nii.gz')
    seg_test = glob.glob(r'F:/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/*/*seg.nii.gz')

    print('begin')
    for i in range(len(flair_train)):
        print(i)
        img1 = (read_img(flair_train[i])[100]).astype(np.uint8)
        img2 = (read_img(seg_train[i])[100]).astype(np.uint8)
        dir = 'data/train/'
        filename1 = dir + format(str(i), '0>3s') + '.png'
        plt.imshow(img1)
        plt.axis('off')
        plt.savefig(filename1, bbox_inches='tight',pad_inches=0)
        filename2 = dir + format(str(i), '0>3s') + '_mask.png'
        plt.imshow(img2)
        plt.axis('off')
        plt.savefig(filename2, bbox_inches='tight',pad_inches=0)
    print('over')



