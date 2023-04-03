import os
import imagehash
from PIL import Image
from tqdm import tqdm
import time
import shutil

start = time.time()


def get_image_fingerprint(filename):
    image = Image.open(filename)
    image.resize((8, 8), resample=Image.BILINEAR)
    # Convert the image to grayscale
    image = image.convert("L")
    # Calculate the perceptual hash
    fingerprint = imagehash.phash(image)
    return fingerprint


def repeat_list_in_fold(folder_path: str, threshold=5):
    filename_list = []
    hash_dict = {}
    del_list = []
    print('calculate the image fingerprint:')
    for file_name in tqdm(os.listdir(folder_path)):
        if not file_name.endswith('.jpg') and not file_name.endswith('.jpeg') and not file_name.endswith('.png'):
            continue
        filename = f'{folder_path}/{file_name}'
        try:
            hash_dict[file_name] = get_image_fingerprint(filename)
            filename_list.append(file_name)
        except:
            del_list.append(file_name)
            continue

    repeat_list = list()
    repeat_list_total = list()
    filename_list_total = filename_list


    while len(filename_list) != 0:
        value = hash_dict[filename_list[0]]
        del hash_dict[filename_list[0]]

        for key in hash_dict.copy().keys():
            if value - hash_dict[key] <= threshold:
                repeat_list.append(key)
                del hash_dict[key]

        if len(repeat_list) != 0:
            for i in repeat_list:
                filename_list.remove(i)
            repeat_list.append(filename_list[0])
            repeat_list_total.append(repeat_list)
            repeat_list = []
        filename_list.pop(0)
    for repeat in repeat_list_total:
        del_list += repeat[1:]
    return repeat_list_total, del_list


def output_repeat_pic(repeat_list: list, source_path: str, output_path: str):
    for n in range(len(repeat_list)):
        if os.path.exists(f'{output_path}/{n}') is False:
            os.makedirs(f'{output_path}/{n}')

        for pic_id in repeat_list[n]:
            raw = f'{source_path}/{pic_id}'
            target = f'{output_path}/{n}/{pic_id}'
            shutil.copyfile(raw, target)


def delet_repeat_pic(source_path: str, delet_list: list):
    for id in delet_list:
        img_path = f'{source_path}/{id}'
        os.remove(img_path)


if __name__ == '__main__':
    folder_path = '/home/ubuntu/sda_8T/codespace/new_lei/Dataset/LAION/clip_retrieval/creative-advertisment/images'
    output_path = './crawl_result/clipsubset__creative-advertisment/repeat_outputs3'
    repeat_list, del_list = repeat_list_in_fold(folder_path=folder_path, threshold=8)

    # output_repeat_pic(repeat_list, source_path=folder_path, output_path=output_path)    # 输出重复的图片
    delet_repeat_pic(folder_path, del_list)        # 删除重复的图片

