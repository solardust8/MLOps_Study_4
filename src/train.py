import argparse
import configparser
import sys
import os
import traceback

from logger import Logger
SHOW_LOG = True

from predict import Predictor

import pandas as pd
import json
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
        self.config.read("config.ini")

        if params != None:
            assert isinstance(params, dict)
            keys = list(params.keys())
            assert (len(params) == 2 and
                    'train' in keys and
                    'val' in keys and
                    'batch_size' in keys and
                    'lr' in keys and
                    'class_weights' in keys and
                    'save_path' in keys)
            self.params = params
        else:
            self.params = {'train': os.path.join(os.getcwd(), self.config['SPLIT_DATA']['unit_test_train']),
                           'val': os.path.join(os.getcwd(), self.config['SPLIT_DATA']['unit_test_train']),
                           'batch_size': 1,
                           'lr': 5e-5,
                           'class_weights': None,
                           'save_path': os.path.join(os.getcwd(), 'weights', 'unit_test_save')}
            
        try:
            self.train_data = pd.read_csv(self.params['train'])
            self.val_data = pd.read_csv(self.params['val'])
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
            
        if not os.path.exists(self.params['save_path']):
            try:
                os.mkdir(self.params['save_path'])
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)

        self._Predictor = Predictor(**{'mode': 'train', 'tests': 'none'})

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

        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.params['class_weights']).to(self._Predictor.device))
        self.optimizer = optim.AdamW(self._Predictor.model.parameters(), lr=self.params['lr'])

        self._mcc = evaluate.load('matthews_correlation')
        self._f1 = evaluate.load('f1')
        
        

        

            
