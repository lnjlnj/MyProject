import json
import logging
import random
import os
import pickle
import pandas as pd
import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pyarrow as pa
import pyarrow.parquet as pq
import math


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def processor_the_data(original_json_path: str, img_path: str, save_path: str):
    with open(original_json_path, 'r') as f:
        data = json.load(f)

    split_num = 10000
    part_num = math.ceil(len(data) / split_num)
    random.shuffle(data)
    data_total = []

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for n in range(part_num):
        if split_num * n < len(data):
            data_total.append(data[n*split_num:n*split_num+split_num])
        else:
            data_total.append(data[n*split_num:])
    i = 0
    for data in data_total:
        i += 1
        dataset = MyDataset(data)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True)
        datas = []

        print(f'-----数据处理进度(part {i}/{len(data_total)})-----')
        for batch in tqdm.tqdm(dataloader):
            images = []
            batch_data = []
            for pic in batch['pic_id']:
                Image.MAX_IMAGE_PIXELS = 89478485
                try:
                    image = Image.open(f'{img_path}/{pic}').convert('RGB')
                    images.append(image)
                except:
                    continue

            inputs = processor(images=images, return_tensors="pt")
            for n in range(len(images)):
                batch_data.append({'image': inputs['pixel_values'][n].numpy().reshape(3 * 224 * 224),
                                   'label': batch['label'][n].numpy().reshape(1)})

            datas += batch_data

        temp_file = f'{save_path}/part{i}.parquet'
        df = pd.DataFrame(datas)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, temp_file)
        datas = []
        df = None

    return print(f'-----数据预处理完成, 文件保存至{save_path}-----')


if __name__ == '__main__':
    json_path = '/home/leiningjie/PycharmProjects/dataset/advertisement_flickr30k_binary/train_binary.json'
    img_path = '/home/leiningjie/PycharmProjects/dataset/advertisement_flickr30k_binary/total'
    model_path = '/home/leiningjie/PycharmProjects/model/vit/vit-base/original'
    processor = ViTImageProcessor.from_pretrained(model_path)
    save_path = '/home/leiningjie/PycharmProjects/dataset/advertisement_flickr30k_binary/train'

    processor_the_data(json_path, img_path, save_path)

