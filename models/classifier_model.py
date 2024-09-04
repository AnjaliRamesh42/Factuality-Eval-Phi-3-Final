import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import io
from data.paths import get_model_save_path

def load_classifier_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_path = get_model_save_path()

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

    #contents = pickle.load(f) becomes...
    with open(data_path, 'rb') as f:
        model = CPU_Unpickler(f).load()

    # model.eval()
    return model, tokenizer