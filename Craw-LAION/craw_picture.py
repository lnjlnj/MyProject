import json
import aiohttp
import asyncio
import pyarrow.parquet as pq
import pandas as pd
import random
from tqdm import tqdm
import math

json_path = './clipsubset_metaphor-photo.json'
with open(json_path, 'r') as f:
    data = json.load(f)

random_sample = []
for sample in data:
    random_sample.append((sample['url'], sample['caption']))



headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}


async def download_image(sample:tuple, n, record, save_path):
    image_url = sample[0]
    image_caption = sample[1]
    file = image_url.split('/')[-1]
    if 'png' in file:
        file_name = f'{n}.png'
    else:
        file_name = f'{n}.jpg'
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, headers=headers) as response:
                if response.status == 200:
                    record.append({'pic_id':file_name, 'caption':image_caption})
                    with open(f'{save_path}/{file_name}', 'wb') as f:
                        f.write(await response.read())
    except:
        pass


async def main(concurrency:int, image_save_path:str, json_path:str):

    record = []
    num = math.ceil(len(random_sample))
    for asyncio_n in range(num):
        start = asyncio_n*concurrency
        if start+concurrency < len(random_sample):
            end = start + concurrency
        else:
            end = -1
        tasks = []
        for i, sample in enumerate(random_sample[start:end]):
            task = asyncio.create_task(download_image(sample, i+start, record, save_path=image_save_path))
            tasks.append(task)

        [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]

    with open(json_path, 'w') as f:
        json.dump(record, f)

if __name__ == "__main__":
    type_list = ['metaphor-photo', 'psas']
    for type in type_list:
        image_path = f'./{type}/images'
        json_path = f'./{type}/caption.json'
        asyncio.run(main(concurrency=2000, image_save_path=image_path,
                         json_path=json_path))

