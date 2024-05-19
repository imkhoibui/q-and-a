import torch

class QGPipeline:
    """
        Question generation pipeline
    """
    def __init__(self, model, tokenizer):
        """
            This function initializes the model pipeline
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.generate_kwargs = {"max_length": 256, "num_beams": 4, "length_penalty": 1.5,
            "no_repeat_ngram_size": 3, "early_stopping": True}

    def __call__(self, context: str, **generate_kwargs):
        inputs = self.prepare_inputs(context)
        outs = self.model.generate(input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), **generate_kwargs
        )
        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions

    def prepare_inputs(self, context):
        """
            This function prepares the context (source text) for predictions
        """
        source_text = f"generate questions: {context}"
        source_text = source_text + " </s>"
        inputs = self.tokenize([source_text], padding=False)
        return inputs
    
    def tokenize(self, inputs, padding=True, truncation=True, 
                  add_special_tokens=True, max_length=512):
        
        inputs = self.tokenizer.batch_encode_plus(inputs, max_length=max_length,
            add_special_tokens=add_special_tokens, truncation=truncation,
            padding="max_length" if padding else False, pad_to_max_length=padding,
            return_tensors="pt")
        return inputs
    
class QAPipeline:
    """
        Question answering pipeline
    """
    def __init__(self, model, tokenizer):
        """
            This function initializes the model pipeline
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, question: str, context: str):
        inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
        return answer
        
def pipeline(task, model, tokenizer):
    """
        This function returns a model pipeline instance
    """
    if task == "qg":
        return QGPipeline(model, tokenizer)
    elif task == "qa":
        return QAPipeline(model, tokenizer)
        