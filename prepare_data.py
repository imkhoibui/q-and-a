import config as cfg
import torch
from typing import Dict, List, Optional

def process_e2e_qg(example):
    source_text = f"generating questions: {example['context'].strip()}"
    questions = example['question']
    target_text = " {sep_token} " + questions
    target_text = f"{target_text} {{sep_token}}"
    example['source_text'] = source_text
    example['target_text'] = target_text
    return example

class DataProcessor():
    def __init__(self, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.sep_token = cfg.SEP_TOKEN

    def process(self, dataset):
        dataset = dataset.map(process_e2e_qg)
        dataset = dataset.map(self.add_eos_to_example)
        dataset = dataset.map(self.add_special_tokens)
        dataset = dataset.map(self.convert_to_features, batched=True)
        return dataset 
    
    def add_eos_to_example(self, example):
        example["source_text"] = example["source_text"] + "</s>"
        example["target_text"] = example["target_text"] + "</s>"
        return example
    
    def add_special_tokens(self, example):
        example['target_text'] = example['target_text'].replace('{sep_token}', self.sep_token)
        return example

    def convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        encodings = {
            'source_ids': source_encoding['input_ids'],
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings

class T2TDataCollator():
    def __init__(self, tokenizer, model_type="t5", mode="training", using_tpu=False):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mode = mode
        self.using_tpu = using_tpu
    
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([example['source_ids'] for example in batch])
        target_ids = torch.stack([example['target_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])

        pad_token_id = self.tokenizer.pad_token_id
        lm_labels = target_ids.clone()
        decoder_input_ids = self._shift_right_t5(lm_labels)
        if self.mode == 'training':
            lm_labels[lm_labels[:, :] == pad_token_id] = -100

        params =  {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids
        }

        return params

    def _shift_right_t5(self, input_ids):
        decoder_start_token_id = self.tokenizer.pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        assert (
            decoder_start_token_id is not None
        )
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item()

        return shifted_input_ids