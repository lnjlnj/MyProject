import os.path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model, use_gpu, *args, **kwargs):
        self.model = model
        self.use_gpu = use_gpu
        self.train_dataset = kwargs['train_dataset']
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

    def train(self,
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
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        eval_record = []
        for epoch in range(total_epoches):
            model.train()
            print(f'\nepoch {epoch}')
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
