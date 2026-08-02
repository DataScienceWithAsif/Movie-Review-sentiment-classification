from dotenv import load_dotenv
from utils.model_loader import ModelLoader

import torch
load_dotenv()

class Predictor:
    def __init__(self):
        self.modol_loader = ModelLoader()
        self.tokenizer = self.modol_loader.get_tokenizer()
        self.model = self.modol_loader.get_model()
 
    def get_predicted_sentiment(self, review: str):
        inputs = self.tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
 
        with torch.no_grad():
            outputs = self.model(**inputs)
 
        prediction = torch.argmax(outputs.logits, dim=1).item()
 
        return self.model.config.id2label[prediction]
 
    def get_predicted_sentiment_with_confidence(self, review: str, device: str = "cpu"):
        """
        Same as get_predicted_sentiment, but also returns confidence.
 
        `device` param added for ZeroGPU support: ZeroGPU only makes a GPU
        available inside a function wrapped with @spaces.GPU. Outside that
        window there IS no GPU, so the model/inputs must default to "cpu"
        and only move to "cuda" when explicitly called from inside that
        decorated window (see app.py).
        """
        inputs = self.tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
 
        if str(self.model.device) != device:
            self.model.to(device)
 
        with torch.no_grad():
            outputs = self.model(**inputs)
 
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[prediction].item()
        label = self.model.config.id2label[prediction]
 
        return label, confidence