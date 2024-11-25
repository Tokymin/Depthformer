# 处理数据集
import os


def sort_photo_files(path):
    # 针对形如 frame_000044.jpg 格式的图片
    filelists = os.listdir(path)
    sort_num_first = []
    for file in filelists:
        f = file.split(".")[0]
        g = f.split("_")[1]
        # sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.append(int(g))
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in filelists:
            if sort_num == int(file.split(".")[0].split("_")[1]):
                sorted_file.append(file)
    return sorted_file


def sort_photo_files_2(path):
    # 针对形如 000044.jpg 格式的图片
    filelists = os.listdir(path)
    sort_num_first = []
    for file in filelists:
        f = file.split(".")[0]
        sort_num_first.append(int(f))
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in filelists:
            if sort_num == int(file.split(".")[0]):
                sorted_file.append(file)
    return sorted_file


def sort_depth_files(path):
    # 针对形如 aov_image_0500 格式的深度图片
    filelists = os.listdir(path)
    sort_num_first = []
    for file in filelists:
        f = file.split(".")[0]
        g = f.split("_")[2]
        # sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.append(int(g))
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in filelists:
            if sort_num == int(file.split(".")[0].split("_")[2]):
                sorted_file.append(file)
    return sorted_file


def make_photo_test_file_txt():
    with open(r'/home/toky/Datasets/Endo_colon_unity/clinical_data/test_c.txt', 'a',
              encoding='utf-8') as f:
        alllist = sort_photo_files_2(u"/home/toky/Datasets/Endo_colon_unity/clinical_data/photo/")
        for name in alllist:
            text = 'photo/' + name + '\n'
            f.write(text)


def make_depth_test_file_txt():
    with open(r'/home/toky/Datasets/Endo_colon_unity/test_depth_c.txt', 'a',
              encoding='utf-8') as f:
        alllist = sort_photo_files_2(u"/home/toky/Datasets/Endo_colon_unity/test_dataset/depth/")
        for name in alllist:
            text = 'depth/' + name + '\n'
            f.write(text)


if __name__ == '__main__':
    # make_photo_test_file_txt()  # 读取一个文件夹下的所有彩色图片，排序，并写入txt文件
    make_depth_test_file_txt()
