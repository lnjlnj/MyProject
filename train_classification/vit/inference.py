import json
import pandas as pd
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import requests
from classification_trainer import Trainer
import pyarrow.parquet as pq
import os
import argparse
from tqdm import tqdm


# device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'
model_path = '/home/leiningjie/PycharmProjects/model/vit/vit-base/original'
#
processor = ViTImageProcessor.from_pretrained(model_path)
#

# # model
model = ViTForImageClassification.from_pretrained(model_path)
model.classifier = nn.Linear(768, 2)
model.to(device)


# Dataloader
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def create_parquet_data(parquet_path:str):
    pq_data = pq.read_table(parquet_path)
    df = pq_data.to_pandas()
    data = []
    for index, rows in df.iterrows():
        image = rows['image'].reshape(3, 224, 224)
        label = rows['label']
        data.append({'image': torch.from_numpy(image), 'label': torch.from_numpy(label)})

    train_dataset = MyDataset(data)

    return train_dataset


def create_data(json_path, img_path):
    with open(json_path) as f:
        data = json.load(f)
    for n in data:
        pic_id = n['pic_id']
        n['pic_id'] = f"{img_path}/{pic_id}"
    dataset = MyDataset(data)

    return dataset


def create_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(n=10000, ignore_index=True)
    data = df['image_abs_path'].tolist()
    dataset = MyDataset(data)

    return dataset



if __name__ == '__main__':

    inference_csv_path = '/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/0/record.csv'
    result_path = '/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/0/vit_binary_result.csv'

    test_dataset = create_data_from_csv(inference_csv_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        shuffle=True)

    trainer = Trainer(model=model, use_gpu=True, processor=processor,
                      train_path=None, eval_dataset=test_dataset)

    trainer.inference(batch_size=1500, model_path='./checkpoint/acc_0.934.pt', result_path=result_path)
