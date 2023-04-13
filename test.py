import json
import pandas
import os
import shutil
from PIL import Image
import numpy as np

import pandas as pd

path = '/home/leiningjie/PycharmProjects/dataset/metaphor'

fold_1 = 'train_binary.json'
fold_2 = 'test_binary.json'
fold_3 = 'laion.json'

with open(f'{path}/{fold_1}', 'r') as f:
    train = json.load(f)

with open(f'{path}/{fold_2}', 'r') as f:
    test = json.load(f)

with open(f'{path}/{fold_3}', 'r') as f:
    data = json.load(f)

for n in data[1900:]:
    train.append({'pic_id':n, 'label':0})
for n in data[:1900]:
    test.append({'pic_id': n, 'label': 0})

with open(f'{path}/{fold_1}', 'w') as f:
    json.dump(train, f)
with open(f'{path}/{fold_2}', 'w') as f:
    json.dump(test, f)
print()