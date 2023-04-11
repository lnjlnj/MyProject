import json
import numpy as np
import pandas
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import os
import pyarrow as pa
import pyarrow.parquet as pq
import math
from tqdm import tqdm
import random
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    device = 'cuda'
    result_path = '/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/0/record.csv'
    json_path = '/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/0/image_abs_paths.json'
    with open('/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/0/image_abs_paths.json', 'r') as f:
        images_abs_path = json.load(f)


    dataset = MyDataset(images_abs_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=800,
        shuffle=False)

    model_path = '/home/leiningjie/PycharmProjects/model/clip/clip-large/original'
    model = CLIPModel.from_pretrained(model_path)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(model_path)

    with torch.no_grad():
        batch_num = 0
        for batch in tqdm(dataloader):
            batch_num += 1
            images = []
            new_batch = []

            for abs_path in batch:
                try:
                    image = Image.open(abs_path)
                    img_array = np.array(image)
                    if img_array.shape[-1] == 3 and len(img_array.shape) == 3:
                        np_content = img_array[0][0][0]
                        if np.all(img_array == np_content):
                            continue
                        else:
                            images.append(image)
                            new_batch.append(abs_path)
                    elif img_array.shape[-1] == 4 and len(img_array.shape) == 3:
                        image = Image.open(abs_path).convert('RGB')
                        img_array = np.array(image)
                        np_content = img_array[0][0][0]
                        if np.all(img_array == np_content):
                            continue
                        else:
                            images.append(image)
                            new_batch.append(abs_path)
                    elif len(img_array.shape) != 3:
                        continue
                except:
                    continue

            inputs = processor(text=["creative advertisement", "normal photo"], images=images, return_tensors="pt", padding=True)
            inputs.to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities  0000
            predict = list(torch.argmax(probs, dim=-1).to('cpu').numpy())

            df = pd.DataFrame({'image_abs_path':new_batch,
                                   'predict':predict})
            if os.path.exists(result_path) is False:
                df.to_csv(result_path, index=False)
            elif os.path.exists(result_path) is True:
                df.to_csv(result_path, mode='a', index=False, header=False)

