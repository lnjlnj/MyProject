import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


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
              total_epoches=10,
              batch_size=16,
              accumulation_steps=1,
              learning_rate=1e-4,
              warmup_ratio=0.1,
              weight_decay=0.1,
              eps=1e-06,
              loss_log_freq=40):
        self.model.to(self.device)
        self.model.train()
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(total_epoches):

            for index, batch in enumerate(tqdm(train_loader)):
                images = []
                for pic in batch['pic_id']:
                    image = Image.open(pic).convert('RGB')
                    images.append(image)
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                batch['label'].to(self.device)
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, batch['label'].to(self.device))

                optimizer.step()

                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                loss.backward()



