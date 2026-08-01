from dotenv import load_dotenv
from src.model.predictor import Predictor

load_dotenv()

predictor = Predictor()


def main(review):
    prediction = predictor.get_predicted_sentiment(review=review)

    print(f"Predicted review sentiment is: {prediction}")

if __name__ == "__main__":
    review = "This movie was absolutely amazing. I loved every minute of it."
    main(review=review)