from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm
from prepare_data import DataProcessor, T2TDataCollator
from eval import eval

import config as cfg
import logging
import evaluate
import collections
import torch
import numpy as np

logger = logging.getLogger(__name__)

raw_datasets = load_dataset("squad")
model_checkpoint = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens([cfg.SEP_TOKEN])
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

train_dataset = load_dataset("squad", split="train[:1000]")
valid_dataset = load_dataset("squad", split="validation[:1000]")

processor = DataProcessor(
    tokenizer=tokenizer,
    max_source_length = cfg.MAX_SOURCE_LENGTH,
    max_target_length = cfg.MAX_TARGET_LENGTH,
)

train_dataset = processor.process(train_dataset)
validation_dataset = processor.process(valid_dataset)

print("Checkpoint 1: loading data done")

columns = ["source_ids", "target_ids", "attention_mask"]
train_dataset.set_format(type="torch", columns=columns)
validation_dataset.set_format(type='torch', columns=columns)

torch.save(train_dataset, f"{cfg.TRAIN_DATA_PATH}")
torch.save(validation_dataset, f"{cfg.VALIDATION_DATA_PATH}")

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model.resize_token_embeddings(len(tokenizer))

data_collator = T2TDataCollator(
    tokenizer=tokenizer,
    mode="training",
)

train_dataset = torch.load("dataset/train_dataset.pt")
validation_dataset = torch.load("dataset/validation_dataset.pt")

print("Checkpoint 2: finish data creation")
print("Sample format of training data:", train_dataset[0])

args = TrainingArguments(
    "t5-tuned",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("t5-tuned")
tokenizer.save_pretrained("t5-tuned")
print("Checkpoint 3: finish training")

## Evaluation
eval()

print("Checkpoint 4: finish evaluating")
# results = {}

# logger.info("*** Evaluate ***")
# eval_output = trainer.evaluate()

# output_eval_file = "results/eval_results.txt"

# with open(output_eval_file, "w") as writer:
#     logger.info("*** Eval results ***")
#     for key in sorted(eval_output.keys()):
#         logger.info("  %s = %s", key, str(eval_output[key]))
#         writer.write("%s = %s\n" % (key, str(eval_output[key])))
#     results.update(eval_output)

## Get predictions
from pipelines import pipeline
print("Checkpoint 5: testing")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-tuned", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("t5-tuned", local_files_only=True)
nlp = pipeline(model, tokenizer)

with open("data/text_data.txt", "r") as files:
    text = files.read()
print(nlp(text))