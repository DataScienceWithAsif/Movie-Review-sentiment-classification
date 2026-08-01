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



