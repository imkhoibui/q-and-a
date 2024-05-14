# Project Overview

This is a repo for my course CS312 - Natural Language Processing.
Objective: a fine-tuned model to perform question generation task given a text.

## Introduction

- The model chosen is Google's [t5-base](https://huggingface.co/docs/transformers/model_doc/t5)
fine-tuned on [SQuAD's 1.1 dataset](https://rajpurkar.github.io/SQuAD-explorer/).

- t5-base (Text-to-Text Transfer Transformer) proved to be a prominent model in various text 
generation tasks thanks to its efficient pre-training capabilities.
- Here we adapt transformer-based end-to-end question generation to restructure the inputs,
allowing a better data adaptation to the task.

## Prepare Data
- The dataset has the following format: 
`["id" : id_value, "title" : title_value, "context" : context_value, "question" : question_value, "answer", answer_dict]`

- For the question generation task, we would not need the use of answer as usual, 
so the end-to-end architecture proposed a reframing of inputs into utilizing just 
context-question pairs.

- The inputs are being changed to allow for all questions per line (AQPL) format, 
such that context-question and question-question are separated with a special
token `<sep>`.

## Project structure
- `requirements.txt` contains all the packages required to run the program at the time being.

- `train.py` contains 5 checkpoints, data-loading, data-preprocessing, training, evaluating & testing.

- `pipelines.py` is utilized to run the program seemlessly.

- `config.py` can be changed to meet your training arguments.


## Results

We trained with 3 epochs, for the first 1000 examples from squad dataset.

We then evaluate and received the following results:
- `epoch = 3.0`
- `eval_loss = 5.103535175323486`
- `eval_runtime = 695.7682`
- `eval_samples_per_second = 1.437`
- `eval_steps_per_second = 0.046`

We then tested our model on a short paragraph and get the following results:

## References
