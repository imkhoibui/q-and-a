import torch
import config as cfg
import logging
from torch.utils.data import DataLoader
from prepare_data import T2TDataCollator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available else 'cpu'

def get_predictions(model, tokenizer, data_loader, num_beams=4, max_length=32, length_penalty=1):
    model.to(device)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outs = model.generate(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device),
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
            )

            prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            predictions.extend(prediction)

    return predictions

def eval():
    tokenizer = AutoTokenizer.from_pretrained("t5-tuned", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-tuned", local_files_only=True)

    valid_dataset = torch.load(cfg.VALIDATION_DATA_PATH)

    collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type="t5",
        mode="inference"
    )

    loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collator)

    predictions = get_predictions(
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        num_beams=4,
        max_length=32,
    )

    with open("results/eval_result.txt", 'w') as f:
        f.write("\n".join(predictions))
    
    logging.info(f"Output saved at results/eval_results.txt")