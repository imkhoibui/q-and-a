import logging
from typing import Optional, Dict, Union

from nltk import sent_tokenize
import torch

class E2EQGPipeline:
    def __init__(self, model, tokenizer, use_cuda):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        self.model_type = "t5"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        input_length = inputs["input_ids"].shape[-1]
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
def pipeline(model, tokenizer):
    return E2EQGPipeline(model, tokenizer, use_cuda=True)