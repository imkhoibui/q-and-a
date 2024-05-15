# Project Overview - t5 end-to-end question generation

This is a repo for my course CS312 - Natural Language Processing.
Objective: a fine-tuned model to perform a question-generation task given a text.

## Introduction

- The model chosen is Google's [t5-base](https://huggingface.co/docs/transformers/model_doc/t5)
fine-tuned on [SQuAD's 1.1 dataset](https://rajpurkar.github.io/SQuAD-explorer/).

- Thanks to its efficient pre-training capabilities, the t5-base (Text-to-Text Transfer Transformer)
proved a prominent model in various text-to-text tasks.

![t5-model-architecture](https://miro.medium.com/v2/resize:fit:1200/1*sKB0j5FCGQcR-QqIAM-LWA.png)

- Here we adapt transformer-based end-to-end question generation to restructure the inputs,
allowing a better data adaptation to the task. End-to-end is an approach where the model
attempts to generate questions from context without using the answer. Therefore, we
had to modify the SQuAD dataset to use only the context-question pairs.

## Prepare Data
- The dataset has the following format: 
`["id" : id_value, "title" : title_value, "context" : context_value, "question" : question_value, "answer", answer_dict]`

- For the question generation task, we would not need the use of answers as usual, 
so the end-to-end architecture proposed reframing of inputs into utilizing just 
context-question pairs.

- The inputs are being changed to allow for all questions per line (AQPL) format, 
such that context-question and question-question are separated with a special
token `<sep>`.

## Project structure
- `requirements.txt` contains all the packages required to run the program at the time being.

- `train.py` contains 5 checkpoints, data-loading, data-preprocessing, training, evaluating & testing.

- `pipelines.py` is utilized to run the program after training.

- `config.py` can be changed to meet your training arguments.


## Results

We trained with 3 epochs with examples gathered from the SQuAD dataset.

We then evaluated and received the following results:
- `epoch = 3.0`
- `eval_loss = 5.103535175323486`
- `eval_runtime = 695.7682`
- `eval_samples_per_second = 1.437`
- `eval_steps_per_second = 0.046`

Future work should attempt to train the model with much more extensive data from the dataset to generate results.

## References
Lopez, Luis Enrico, et al. "Transformer-based end-to-end question generation." arXiv preprint arXiv:2005.01107 4 (2020).
