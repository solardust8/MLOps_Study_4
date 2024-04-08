import argparse
import configparser
import sys
import os
import traceback

import json
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


from src.logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self,
                 params: dict) -> None:
        if params != None:
            assert isinstance(params, dict)
            keys = list(params.keys())
            assert (len(params) == 2 and
                    'mode' in keys and
                    'tests' in keys)
            assert (params['mode'] in ["infere", "train"] and
                    params['tests'] in ["func", "none"])
            self.args = params
        else:
            self.args = {'mode': 'infere', 'tests': 'none'}

        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini", encoding="utf-8")
        
        self.root_dir = self.config['ROOT']['root_dir']
    
        self.device = self.config['PARAMETERS']['device']
        self.seq_max_length = int(self.config['PARAMETERS']['tokenizer_seq_max_length'])
    
        assert (isinstance(self.device, str) and
                self.device in ['cpu', 'cuda:0'])
        assert (isinstance(self.seq_max_length, int) and
                self.seq_max_length > 0 and
                self.seq_max_length <= 512)

    
        try:
            if self.args['mode'] == 'train':
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.config['PARAMETERS']['base_weights'], local_files_only=True).to(self.device)
            else:
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.config['PARAMETERS']['finetuned_weights'], local_files_only=True).to(self.device)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                self.config['PARAMETERS']['tokenizer'], local_files_only=True)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        self.id2label = ['ham', 'spam']
        self.log.info("Predictor is ready")
        
    def tokenize_inputs(self, prompts: list[str]):
        try:
            tokens = self.tokenizer(
                prompts,
                max_length=self.seq_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        return tokens
    
    def model_eval(self):
        return self.model.eval()
    
    def model_train(self):
        return self.model.train()
    
    def get_model_output(self, prompts: list[str]) -> list[dict]:
        tokens = self.tokenize_inputs(prompts)
        self.model.eval()
        result = []
        for i in range(len(prompts)):
            input_ids, attention_mask = tokens['input_ids'][i].unsqueeze(0).to(self.device), tokens['attention_mask'][i].unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            softmaxed = torch.nn.functional.softmax(outputs.logits, dim=-1)
            res_idx = torch.argmax(softmaxed, dim=-1)
            result.append({'label': self.id2label[res_idx], 'score': softmaxed.squeeze(0)[res_idx].item()})
        return result
          

    def run_tests(self) -> bool:
        if self.args['tests'] == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        result = self.get_model_output(prompts=data["text"])
                        print(result)
                        accuracy = 0.
                        for res, gt in zip(result, data["label"]):
                            accuracy += int(res['label'] == gt)
                        print(f'Predictor has {accuracy/len(result)} accuracy score')
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                    self.log.info(
                        f'Predictor passed func test {f.name}')        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor")
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        help="Select mode",
                        required=False,
                        default="infere",
                        const="infere",
                        nargs="?",
                        choices=["infere", "train"])
    parser.add_argument("-t",
                        "--tests",
                        type=str,
                        help="Select tests",
                        required=False,
                        default="none",
                        const="none",
                        nargs="?",
                        choices=["func", "none"])
    try:
        args = parser.parse_args()
    except Exception:
        print(traceback.format_exc())
        sys.exit(1)
    params = vars(args)
    predictor = Predictor(params=params)
    if params['tests'] != None:
        predictor.run_tests()