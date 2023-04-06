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

device = 'cuda'
model_path = '/media/lei/sda_2T/MyGithub/model/vit-base/original'
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


pq_data = pq.read_table('./test.parquet')
df = pq_data.to_pandas()

data = []
for index, rows in df.iterrows():
    image = rows['image'].reshape(3, 224, 224)
    label = rows['label']
    data.append({'image': torch.from_numpy(image), 'label': torch.from_numpy(label)})

train_dataset = MyDataset(data)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=10)


trainer = Trainer(model=model, use_gpu=True, processor=processor,
                  train_dataset=train_dataset[:-100], eval_dataset=train_dataset[-100:])

trainer.train(eval_epoch=5)
