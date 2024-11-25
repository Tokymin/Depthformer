import os
from PIL import Image  # 或直接import Image


def get_path(path):
    '''返回目录中所有PNG图像的文件名列表'''
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


if __name__ == '__main__':

    pth = get_path('/home/toky/Datasets/Phantom_dataset/0408phantom-1/')  # get_path(path)中path参数是自己的文件夹目录，例如('../mask_dir_Class)
    for file in pth:
        im = Image.open(file)
        img = im.convert('L')
        save_path = "/home/toky/Datasets/Phantom_dataset/0408phantom-1/gray/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img.save(save_path + file.split('/')[-1])
