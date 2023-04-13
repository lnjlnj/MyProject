import json

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


if __name__ == '__main__':

    img_path = '/home/leiningjie/PycharmProjects/dataset/metaphor/total'

    train_path = '/home/leiningjie/PycharmProjects/dataset/metaphor/train'
    test_parquet = '/home/leiningjie/PycharmProjects/dataset/metaphor/test_binary.json'

    train_parquet_list = os.listdir(train_path)
    test_dataset = create_data(test_parquet, img_path=img_path)

    trainer = Trainer(model=model, use_gpu=True, processor=processor,
                      train_path=train_path, eval_dataset=test_dataset)

    trainer.train_with_parquet(eval_epoch=3, batch_size=128, total_epoches=100)
