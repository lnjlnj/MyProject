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


with open('/media/lei/sda_2T/MyGithub/dataset/test_dataset/label.json', 'r') as f:
    data_1 = json.load(f)
for n in data_1:
    pic_id = n['pic_id']
    n['pic_id'] = f"/media/lei/sda_2T/MyGithub/dataset/test_dataset/images/{pic_id}"
dataset = MyDataset(data_1)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# for batch_data in dataloader:
#     images = []
#     for pic in batch_data['pic_id']:
#         image = Image.open(pic).convert('RGB')
#         images.append(image)
#
#     inputs = processor(images=images, return_tensors="pt")
#     inputs.to(device)
#     outputs = model(**inputs)

trainer = Trainer(model=model, use_gpu=True, processor=processor,
                  train_dataset=dataset, eval_dataset=dataset)

trainer.train()
print(1)



# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
