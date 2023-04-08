"""
create binary label
"""
import pyarrow.parquet as pq
import os
import pandas as pd
import torch

father_dir = '/home/leiningjie/PycharmProjects/dataset/advertisement_flickr30k_binary/temp'
parquet_list = os.listdir(father_dir)
df_list = []
data = []
for file in parquet_list:
    parquet_path = f'{father_dir}/{file}'
    pq_data = pq.read_table(parquet_path)
    df = pq_data.to_pandas()

    for index, rows in df.iterrows():
        image = rows['image'].reshape(3, 224, 224)
        label = rows['label']
        data.append({'image': torch.from_numpy(image), 'label': torch.from_numpy(label)})
    print(1)

for file in parquet_list:

    file_path = f'{father_dir}/{file}'
    df = pd.read_parquet(file_path)
    df_list.append(df)


print(1)