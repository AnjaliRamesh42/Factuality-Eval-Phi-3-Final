from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.classifier_model import load_classifier_model
import torch.nn.functional as F
import torch


    
def preprocess_question(question, tokenizer):
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    # inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device as the model
    return inputs

def classify_question(question, model, tokenizer):
    inputs = preprocess_question(question, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    confidence_open_ended = probs[:, 1].item()
    confidence_not_open_ended = probs[:, 0].item()
    return confidence_open_ended, confidence_not_open_ended

def interpret_classification(query, model, tokenizer):
    confidence_open_ended, confidence_not_open_ended = classify_question(query, model, tokenizer)
    if confidence_open_ended > confidence_not_open_ended:
        classification = 'Open-ended'
    else:
        classification = 'Not open-ended'
    return {
        'question': query,
        'classification': classification,
        'confidence_open_ended': confidence_open_ended,
        'confidence_not_open_ended': confidence_not_open_ended
    }