from dotenv import load_dotenv
from src.model.predictor import Predictor

load_dotenv()

review = "This movie was absolutely amazing. I loved every minute of it."

predictor = Predictor()

prediction = predictor.get_predicted_sentiment(review=review)

print(f"Predicted review sentiment is: {prediction}")