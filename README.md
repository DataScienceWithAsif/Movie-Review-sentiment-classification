# 🎬 Movie Review Sentiment Classification

A modern NLP project that classifies movie reviews as **Positive** or **Negative** using a fine-tuned **DistilBERT** model hosted on Hugging Face. Includes an interactive Streamlit app for single-review predictions as well as full batch evaluation with accuracy, precision, recall, F1-score, and a confusion matrix.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-NLP-yellow)](https://huggingface.co/docs/transformers/index)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Highlights

- ✅ Fine-tuned transformer model for sentiment classification
- ✅ Clean Python inference pipeline (`Predictor` + `ModelLoader`)
- ✅ Stylish Streamlit web app — "The Critic's Desk" — with two modes:
  - **Single Review**: paste a review, get an instant verdict with a confidence chart
  - **Batch Evaluation**: upload a labeled CSV and get accuracy/precision/recall/F1 + confusion matrix, computed in-app
- ✅ CLI scripts for quick terminal testing and evaluation
- ✅ 96% accuracy on a 200-review held-out test set

---

## 🧠 Model Information

- **Base architecture:** DistilBERT
- **Task:** Binary sentiment classification (Positive / Negative)
- **Model hub repo:** `A-Asif/movie-review-sentiment-distilbert-FT`
- **Dataset:** IMDB movie reviews — 2,000 labeled reviews, nearly perfectly balanced (1,005 positive / 995 negative, ~50.2% / 49.8%)

### Training

The model was fine-tuned for 10 epochs. Validation performance across training:

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 |
|------:|---------------:|-----------------:|---------:|----------:|-------:|-----:|
| 1 | 1.1311 | 0.1920 | 92.0% | 90.4% | 94.0% | 92.2% |
| 2 | 0.7179 | 0.2486 | 93.5% | 91.4% | 96.0% | 93.7% |
| 5 | 0.0714 | 0.3900 | 92.0% | 94.7% | 89.0% | 91.8% |
| 6 | 0.0088 | 0.4211 | 94.0% | 93.1% | 95.0% | 94.1% |
| 10 | 0.0006 | 0.4908 | 93.5% | 93.1% | 94.0% | 93.5% |

*(Full per-epoch log available in `Experiments/`.)* Training loss drops sharply after epoch 5 while validation loss slowly rises — a sign of mild overfitting in later epochs, though validation accuracy stays stable in the 92–94% range throughout.

### Evaluation Results

Evaluated on a held-out, labeled test set of 200 reviews (115 negative / 85 positive):

| Metric | Score |
|---|---|
| **Accuracy** | 96.00% |
| **Precision** | 96.39% |
| **Recall** | 94.12% |
| **F1-score** | 95.24% |

**Confusion Matrix:**

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| **Actual Negative** | 112 | 3 |
| **Actual Positive** | 5 | 80 |

Per-class breakdown:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.96 | 0.97 | 0.97 | 115 |
| Positive | 0.96 | 0.94 | 0.95 | 85 |

The model performs strongly and evenly across both classes. Most misclassifications occur on reviews with mixed or sarcastic sentiment (e.g., a review that is broadly complimentary but contains criticism partway through).

---

## 📁 Project Structure

```text
Movie-Review-sentiment-classification/
├── app.py                      # Streamlit UI app (Single Review + Batch Evaluation tabs)
├── main.py                     # CLI entry script
├── test.py                     # Simple prediction smoke script
├── evaluate.py                 # CLI evaluation script (accuracy/F1/confusion matrix)
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── data/
│   ├── ReviewClassification.csv
│   └── test.csv
├── Experiments/
│   └── data_preprocessing.ipynb
└── src/
    ├── model/
    │   └── predictor.py        # Prediction logic
    └── utils/
        └── model_loader.py     # Loads tokenizer/model from HF Hub
```

---

## 🚀 Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/DataScienceWithAsif/Movie-Review-sentiment-classification.git
cd Movie-Review-sentiment-classification
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
```

> For Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## ▶️ Usage

### Option A — Run from terminal (CLI)

```bash
python main.py
```

Expected output (example):

```text
Predicted review sentiment is: POSITIVE
```

### Option B — Launch Streamlit app (recommended)

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

The app has two tabs:

- **🎬 Single Review** — paste a review (or click one of the example buttons) and click **Deliver Verdict**. You'll see the predicted label, a confidence percentage, a confidence bar chart comparing Positive vs. Negative, and a running session history of everything you've tried.
- **📊 Batch Evaluation** — upload a labeled CSV (e.g. `data/test.csv`), specify which columns hold the review text and the true label, and click **Run Evaluation**. The app runs every row through the model and displays accuracy, precision, recall, F1-score, a full classification report, and a confusion matrix — plus a downloadable CSV of per-row predictions.

---

## 📊 Model Evaluation (CLI)

To evaluate the model from the terminal instead of the app:

```bash
python evaluate.py --data data/test.csv --text-col Review --label-col sentiment
```

This will:
- Print accuracy, precision, recall, and F1-score to the terminal
- Save a confusion matrix plot to `confusion_matrix.png`
- Save a full metrics report to `metrics.txt`

> Adjust `--text-col` and `--label-col` to match the actual column names in your CSV — check with `df.columns` if unsure.

---

## ⚙️ How It Works

1. `ModelLoader` downloads the tokenizer and model from Hugging Face.
2. `Predictor` tokenizes review text and runs inference using PyTorch.
3. The highest logit score decides the sentiment label, and softmax gives a confidence score.
4. The Streamlit UI displays the prediction, confidence chart, and (in Batch Evaluation) aggregate metrics and a confusion matrix.

---

## 🧪 Quick Test Script

You can also run:

```bash
python test.py
```

This performs a basic end-to-end inference check with a sample review.

---

## 🛠 Troubleshooting

- **Import errors (`ModuleNotFoundError`)**
  Ensure dependencies are installed and run:
  ```bash
  pip install -e .
  ```

- **Model download issues**
  Check internet access and verify the Hugging Face model repo is public and available.

- **Slow first run**
  First inference may take longer due to model download and cache warm-up.

- **"Columns not found" error in Batch Evaluation**
  The app shows you the actual column names in your uploaded CSV if the ones you typed don't match — double-check capitalization (e.g. `Review` vs `review`).

---

## 🔮 Improvement Ideas

- Add automated unit/integration tests
- Add Docker support for one-command deployment
- Add batch prediction endpoint (FastAPI/Flask)
- Investigate mild overfitting in later training epochs (e.g. earlier stopping, regularization)
- Add error analysis for misclassified reviews (sarcasm/mixed-sentiment cases)

---

## 🤝 Contributing

Contributions are welcome.
Open an issue first to discuss major changes, then submit a PR.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
