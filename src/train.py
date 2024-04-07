import argparse
import configparser
import sys
import os
import traceback

from logger import Logger
SHOW_LOG = True

from predict import Predictor

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.nn as nn
import evaluate

class Train():
    def __init__(self,
                 params: dict) -> None:
        
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        self.config.read("config.ini", encoding="utf-8")

        if params != None:
            assert isinstance(params, dict)
            keys = list(params.keys())
            assert (len(params) == 2 and
                    'train' in keys and
                    'val' in keys and
                    'batch_size' in keys and
                    'lr' in keys and
                    'num_epochs' in keys and
                    'class_weights' in keys and
                    'has_sample_weights' in keys and
                    'save_path' in keys)
            self.params = params
        else:
            self.params = {'train': os.path.join(os.getcwd(), self.config['SPLIT_DATA']['unit_test_train']),
                           'val': os.path.join(os.getcwd(), self.config['SPLIT_DATA']['unit_test_train']),
                           'batch_size': 1,
                           'lr': 5e-5,
                           'num_epochs': 1,
                           'class_weights': None,
                           'has_sample_weights': False,
                           'save_path': os.path.join(os.getcwd(), 'weights', 'unit_test_save')}
            
        try:
            self.train_data = pd.read_csv(self.params['train'])
            self.val_data = pd.read_csv(self.params['val'])
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if not self.params['has_sample_weights']:
           self.train_data['samp_weight'] = 1
           self.val_data['samp_weight'] = 1

        if not os.path.exists(self.params['save_path']):
            try:
                os.mkdir(self.params['save_path'])
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)

        self._Predictor = Predictor(params={'mode': 'train', 'tests': 'none'})

        train_tokens = self._Predictor.tokenize_inputs(self.train_data['text'].tolist())
        val_tokens = self._Predictor.tokenize_inputs(self.val_data['text'].tolist())

        try:
            self.train_dataset = TensorDataset(
                train_tokens['input_ids'],
                train_tokens['attention_mask'],
                torch.tensor(self.train_data['label'].tolist(), dtype=torch.long),
                torch.tensor(self.train_data['samp_weight'].to_list())
            )

            self.val_dataset = TensorDataset(
                val_tokens['input_ids'],
                val_tokens['attention_mask'],
                torch.tensor(self.val_data['label'].tolist(), dtype=torch.long),
                torch.tensor(self.val_data['samp_weight'].to_list())
            )
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        try:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.params['batch_size'], shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.params['batch_size'], shuffle=False)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if self.params['class_weights'] == None:
            weights = [1., 1.]
        else:
            weights = self.params['class_weights']
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(self._Predictor.device))
        self.optimizer = optim.AdamW(self._Predictor.model.parameters(), lr=self.params['lr'])

        self._mcc = evaluate.load('src/evaluate/matthews_correlation/matthews_correlation.py') #os.path.join('srs','evaluate','matthews_correlation'))
        self._f1 = evaluate.load('src/evaluate/f1/f1.py') #os.path.join('srs','evaluate','f1'))
        self.log.info('class Train instance ready')

        
    def run_model_on_loader(self,
                            loader: DataLoader,
                            epoch: int,
                            num_epochs: int,
                            mode: str) -> None:
        assert mode in ['Training', 'Validation']
        total_loss = 0
        epoch_mcc = 0
        epoch_f1 = 0
        cumul_mcc = 0
        cumul_f1 = 0
        dummy = {
            0 : [1, 0], 
            1 : [0, 1]
        }
        
        for input_ids, attention_mask, labels, samp_weights in loader:
        
            input_ids, attention_mask, labels = input_ids.to(self._Predictor.device), attention_mask.to(self._Predictor.device), labels.to(self._Predictor.device)

            outputs = self._Predictor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        

            oh_dum = []
            for el in labels:
                oh_dum.append(dummy[el.item()])
            labels_oh = torch.FloatTensor(oh_dum).to(self._Predictor.device)
            outputs_loss = self.criterion(outputs.logits, labels_oh)
            total_loss += outputs_loss.item()
        
            if mode == 'Training':
                self.optimizer.zero_grad()
                outputs_loss.backward()
                self.optimizer.step()
        
            gt = torch.argmax(labels_oh, dim=-1)
            results = torch.argmax(outputs.logits, dim=-1)
            res_f1 = self._f1.compute(references=gt, predictions=results, average='weighted', sample_weight=samp_weights.tolist())
            res_mcc = self._mcc.compute(references=gt, predictions=results, sample_weight=samp_weights.tolist())
        
            cumul_mcc += res_mcc['matthews_correlation']
            cumul_f1 += res_f1['f1']

        loss = total_loss / len(loader)
        epoch_mcc = cumul_mcc / len(loader)
        epoch_f1 = cumul_f1 / len(loader)

        self.log.info(f'{mode} epoch {epoch + 1}/{num_epochs}: --Loss: {loss:.4f} --Mathews Correlation: {epoch_mcc:.4f} --F1: {epoch_f1:.4f}\n')

    def train_model(self) -> bool:
        for epoch in range(self.params['num_epochs']):
            self._Predictor.model_train()

            self.run_model_on_loader(self.train_loader, epoch, self.params['num_epochs'], mode='Training')

            self._Predictor.model_eval()

            with torch.no_grad():
                self.run_model_on_loader(self.val_loader, epoch, self.params['num_epochs'], mode='Validation')
            
        return True

    def save_model(self) -> bool:
        self._Predictor.model.save_pretrained(self.params['save_path'])
        return True
        
        

        

            
