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



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def processor_the_data(original_json_path:str, img_path:str, save_path:str):
    with open(original_json_path, 'r') as f:
        data = json.load(f)
    dataset = MyDataset(data)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True)

    datas = []
    print('-----数据处理进度-----')
    for batch in tqdm.tqdm(dataloader):
        images = []
        batch_data = []
        for pic in batch['pic_id']:
            image = Image.open(f'{img_path}/{pic}').convert('RGB')
            images.append(image)




        inputs = processor(images=images, return_tensors="pt")
        for n in range(len(images)):
            batch_data.append({'image': inputs['pixel_values'][n], 'label': batch['label'][n]})
        datas += batch_data

    with open(save_path, 'wb') as f:
        pickle.dump(datas, f)

    return print(f'-----数据预处理完成, 文件保存至{save_path}-----')

def processor_the_data2(original_json_path:str, img_path:str, save_path:str):
    with open(original_json_path, 'r') as f:
        data = json.load(f)
    dataset = MyDataset(data)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True)

    datas = []
    print('-----数据处理进度-----')
    for batch in tqdm.tqdm(dataloader):
        images = []
        batch_data = []
        for pic in batch['pic_id']:
            image = Image.open(f'{img_path}/{pic}').convert('RGB')
            images.append(image)

        inputs = processor(images=images, return_tensors="pt")
        for n in range(len(images)):
            batch_data.append({'image': inputs['pixel_values'][n].numpy().reshape(3*224*224), 'label': batch['label'][n].numpy().reshape(1)})
        datas += batch_data

    # 将数据保存为Arrow文件
    columns = []
    for key in datas[0].keys():
        column = pa.array([item[key] for item in datas])
        columns.append((key, column))

    df = pd.DataFrame(datas)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, './example.parquet')
    table = pa.Table.from_arrays(columns)

    with pa.OSFile(save_path, 'wb') as f:
        with pa.RecordBatchFileWriter(f, table.schema) as writer:
            writer.write_table(table)

    return print(f'-----数据预处理完成, 文件保存至{save_path}-----')


if __name__ == '__main__':

    json_path = '/media/lei/sda_2T/MyGithub/dataset/test_dataset/label.json'

    img_path = '/media/lei/sda_2T/MyGithub/dataset/test_dataset/images'
    with open(json_path, 'r') as f:
        data = json.load(f)
    for n in data:
        if n['pic_id'] == '5516.jpg':
            data.remove(n)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    model_path = '/media/lei/sda_2T/MyGithub/model/vit-base/original'
    processor = ViTImageProcessor.from_pretrained(model_path)
    save_path = './test.pkl'

    processor_the_data(json_path, img_path, save_path)


