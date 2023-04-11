import json
import aiohttp
import asyncio
import os
import pandas as pd
from tqdm import tqdm
import math
import argparse

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/89.0.4389.82 Safari/537.36"
}


async def download_image(sample:tuple, save_path):
    image_url = sample[1]
    image_id = sample[0]
    file = image_url.split('/')[-1]
    if 'png' in file:
        file_name = f'{image_id}.png'
    else:
        file_name = f'{image_id}.jpg'
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, headers=headers) as response:
                if response.status == 200:
                    with open(f'{save_path}/{file_name}', 'wb') as f:
                        f.write(await response.read())
    except:
        pass


async def main(concurrency:int, data, image_save_path:str):
    num = math.ceil(len(data))
    for asyncio_n in range(num):
        start = asyncio_n*concurrency
        if start+concurrency < len(data):
            end = start + concurrency
        else:
            end = -1
        tasks = []
        for i, sample in enumerate(data[start:end]):
            task = asyncio.create_task(download_image(sample, save_path=image_save_path))
            tasks.append(task)

        [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__":

    num = '0'

    fold = f'/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/{num}'
    all_file = os.listdir(f'{fold}/split')
    image_save_total_path = f'{fold}/images'
    if os.path.exists(image_save_total_path) is False:
        os.makedirs(image_save_total_path)

    for file in all_file:
        crawled_list = os.listdir(image_save_total_path)
        file_name = file.split('.')[0]

        if file_name in crawled_list:
            continue

        parquet_path = f'{fold}/split/{file}'
        image_save_path = f"{image_save_total_path}/{file_name}"

        if os.path.exists(image_save_path) is False:
            os.makedirs(image_save_path)

        df = pd.read_parquet(parquet_path)

        data = []
        for index, rows in df.iterrows():
            data.append(rows)

        print(f'正在下载---{file}----')
        asyncio.run(main(concurrency=4000, data=data, image_save_path=image_save_path))
        print(f'下载完成---{file}----')

