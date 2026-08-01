from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch

load_dotenv()

class ModelLoader:
    def __init__(self):
        self.Model_name = "A-Asif/movie-review-sentiment-distilbert-FT"

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.Model_name)
        return tokenizer

    def get_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.Model_name)

        return model