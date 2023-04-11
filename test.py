import json
import pandas
import os
import shutil
from PIL import Image
import numpy as np

import pandas as pd

path = '/home/leiningjie/PycharmProjects/dataset/metaphor'

fold_1 = 'images'
fold_2 = 'creative-adv'
fold_3 = 'non_metaphor'

fold_total = [fold_1, fold_2, fold_3]

record = []
if os.path.exists(f'{path}/total') is False:
    os.makedirs(f'{path}/total')

# for fold in fold_total:
#
#     file_list = os.listdir(f'{path}/{fold}')
#     for file in file_list:
#         pic_id = f'{fold}_{file}'
#         src = f'{path}/{fold}/{file}'
#         try:
#             image = Image.open(src)
#             img_array = np.array(image)
#             if img_array.shape[-1] == 3 and len(img_array.shape) == 3:
#                 np_content = img_array[0][0][0]
#                 if np.all(img_array == np_content):
#                     os.remove(src)
#
#             elif img_array.shape[-1] == 4 and len(img_array.shape) == 3:
#                 image = Image.open(path).convert('RGB')
#                 img_array = np.array(image)
#                 np_content = img_array[0][0][0]
#                 if np.all(img_array == np_content):
#                     os.remove(src)
#
#             elif len(img_array.shape) != 3:
#                 os.remove(src)
#         except:
#             os.remove(src)

for fold in fold_total:
    if fold != 'non_metaphor':
        file_list = os.listdir(f'{path}/{fold}')
        for file in file_list[:-125]:
            pic_id = f'{fold}_{file}'
            src = f'{path}/{fold}/{file}'
            try:
                image = Image.open(src)
                record.append({'pic_id': f'{fold}_{file}', 'label': 1})
                tgt = f'{path}/total/{pic_id}'
                shutil.copyfile(src, tgt)
            except:
                continue
    else:
        file_list = os.listdir(f'{path}/{fold}')[:5000]
        for file in file_list[:-250]:
            pic_id = f'{fold}_{file}'
            src = f'{path}/{fold}/{file}'
            try:
                image = Image.open(src)
                record.append({'pic_id': f'{fold}_{file}', 'label': 0})
                tgt = f'{path}/total/{pic_id}'
                shutil.copyfile(src, tgt)
            except:
                continue

with open(f'{path}/train_binary.json', 'w') as f:
    json.dump(record, f)









print(1)