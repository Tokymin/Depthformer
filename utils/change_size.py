# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image
import os.path

"""修改图片文件大小jpgfile：图片文件；savedir：修改后要保存的路径"""


def convertjpg(jpgfile, savedir, width=320, height=320):
    img = Image.open(jpgfile)
    new_img = img.resize((width, height), Image.BILINEAR)
    new_img.save(os.path.join(savedir, os.path.basename(jpgfile)))


"""查找给定路径下图片文件，并修改其大小"""


def modifyjpgSize(file, saveDir):
    for jpgfile in glob.glob(file):
        convertjpg(jpgfile, saveDir)


if __name__ == '__main__':
    # 测试代码
    # for foler in range(17, 151):
    file_path = r"/home/toky/Datasets/Phantom_dataset/part3/"
    poses = []
    saveDir = r'/home/toky/Datasets/Phantom_dataset/part3/'
    for file in os.listdir(file_path):
        if len(file.split(".")) == 1 or file.split(".")[1] != "jpg":  # 排除“gray”的文件夹和其他文件
            continue
        else:
            modifyjpgSize(file_path + "/" + file, saveDir)
