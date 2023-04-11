import os.path
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq


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

class Trainer:
    def __init__(self, model, use_gpu, *args, **kwargs):
        self.model = model
        self.use_gpu = use_gpu
        self.train_path = kwargs['train_path']
        self.eval_dataset = kwargs['eval_dataset']
        self.processor = kwargs['processor']

        if torch.cuda.is_available():
            if self.use_gpu:
                self.device = 'cuda'
                logging.info('训练将在GPU上进行...')

            else:
                self.device = 'cpu'
                logging.info('训练将在CPU上进行...')
        elif torch.cuda.is_available() is False:
            self.device = 'cpu'
            logging.info('找不到可用GPU,训练将在CPU上进行...')

    def train_with_parquet(self,
                           criterion=torch.nn.CrossEntropyLoss(),
                           eval_epoch=None,
                           total_epoches=10,
                           batch_size=16,
                           accumulation_steps=1,
                           learning_rate=1e-4,
                           warmup_ratio=0.1,
                           weight_decay=0.1,
                           eps=1e-06,
                           loss_log_freq=40):
        model = self.model
        model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        train_parquet_list = os.listdir(self.train_path)

        eval_record = []

        for epoch in range(total_epoches):
            n = random.randint(0, len(train_parquet_list)-1)
            train_dataset = create_parquet_data(f'{self.train_path}/{train_parquet_list[n]}')
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True)
            model.train()
            print(f'\nepoch {epoch}----training in {train_parquet_list[n]}, {len(train_dataset)} samples in total ')
            for index, batch in enumerate(tqdm(train_loader)):
                image_tensor = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                inputs = {'pixel_values': image_tensor}
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, label.squeeze(1))

                optimizer.step()

                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                loss.backward()
                if (index + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if eval_epoch is not None:
                if epoch % eval_epoch == 0:
                    predicts = []
                    labels = []
                    with torch.no_grad():
                        model.eval()
                        test_loader = DataLoader(
                            dataset=self.eval_dataset,
                            batch_size=batch_size,
                            shuffle=True)
                        for index, batch in enumerate(tqdm(test_loader)):
                            label = batch['label'].to(self.device)
                            images = []
                            for pic in batch['pic_id']:
                                image = Image.open(pic).convert('RGB')
                                images.append(image)
                            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                            outputs = self.model(**inputs)
                            predict = torch.argmax(outputs.logits, dim=-1)
                            y_pred = predict.to('cpu').numpy().tolist()
                            y_true = label.to('cpu').numpy().tolist()
                            predicts += y_pred
                            labels += y_true
                        accuracy = accuracy_score(labels, predicts)
                        eval_record.append(accuracy)
                        if max(eval_record) == accuracy:
                            if os.path.exists(f'./checkpoint') is False:
                                os.mkdir(f'./checkpoint')
                            model_save_path = f'./checkpoint/epoch{epoch}_acc_{accuracy}.pt'
                            torch.save(model.state_dict(), model_save_path)
                            print(f'\nepoch{epoch}达到测试集最高准确度{accuracy}，模型保存至{model_save_path}')

    def eval(self, batch_size=10):
        self.model.to(self.device)
        predicts = []
        labels = []
        with torch.no_grad():
            self.model.eval()
            test_loader = DataLoader(
                dataset=self.eval_dataset,
                batch_size=batch_size,
                shuffle=True)
            for index, batch in enumerate(tqdm(test_loader)):
                image_tensor = batch['image'].to(self.device)
                label = batch['label'].squeeze(1).to(self.device)
                inputs = {'pixel_values': image_tensor}
                outputs = self.model(**inputs)
                predict = torch.argmax(outputs.logits, dim=-1)
                y_pred = predict.to('cpu').numpy().tolist()
                y_true = label.to('cpu').numpy().tolist()
                predicts += y_pred
                labels += y_true

            accuracy = accuracy_score(labels, predicts)
            print(f'eval_dataset的准确率为{accuracy}')
