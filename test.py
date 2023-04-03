import json
import random
import os
import pickle
import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

model_path = 'google/vit-base-patch16-224'
#
processor = ViTImageProcessor.from_pretrained(model_path)

pic_id_list = os.listdir('/home/ubuntu/sda_8T/codespace/new_lei/Dataset/LAION/clip_retrieval/creative-advertisment/images')
data = []
image_path = '/home/ubuntu/sda_8T/codespace/new_lei/Dataset/LAION/clip_retrieval/creative-advertisment/images'
for id in pic_id_list:
    cate = random.randint(0, 1)
    data.append({'pic_id':f'{image_path}/{id}', 'label':cate})

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

dataset = MyDataset(data[:100])
dataloader = DataLoader(
            dataset=dataset,
            batch_size=20,
            shuffle=True)

datas = []
for batch in tqdm.tqdm(dataloader):
    images = []
    batch_data = []
    for pic in batch['pic_id']:
        image = Image.open(pic).convert('RGB')
        images.append(image)

    inputs = processor(images=images, return_tensors="pt")
    for n in range(len(images)):
        batch_data.append({'image':inputs['pixel_values'][n], 'label':batch['label'][n]})
    datas += batch_data
with open('./test.pkl', 'wb') as f:
    pickle.dump(datas, f)

with open('./test.pkl', 'rb') as f:
    data_1 = pickle.load(f)

# df = pd.DataFrame(datas)
# # tes = [pa.array(n) for n in df['image']]
# # table = pa.Table.from_arrays(tes, names=['image'])
# table = pa.Table.from_arrays(df['image'], names=['image'])



print(1)