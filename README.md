# 🎬 Movie Review Sentiment Classification

A modern NLP project that classifies movie reviews as **Positive** or **Negative** using a fine-tuned **DistilBERT** model hosted on Hugging Face.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-NLP-yellow)](https://huggingface.co/docs/transformers/index)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Highlights

- ✅ Fine-tuned transformer model for sentiment classification  
- ✅ Clean Python inference pipeline (`Predictor` + `ModelLoader`)  
- ✅ Stylish Streamlit web app for real-time predictions  
- ✅ CLI script for quick terminal testing  

---

## 🧠 Model Information

- **Base architecture:** DistilBERT  
- **Task:** Binary sentiment classification (Positive / Negative)  
- **Model hub repo:** `A-Asif/movie-review-sentiment-distilbert-FT`  
- **Dataset context:** IMDB movie reviews (project datasets included in `data/` and root)  

---

## 📁 Project Structure

```text
Movie-Review-sentiment-classification/
├── app.py                      # Streamlit UI app
├── main.py                     # CLI entry script
├── test.py                     # Simple prediction smoke script
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

Then open the local URL shown in terminal (usually `http://localhost:8501`), paste a review, and click **Deliver Verdict**.

---

## ⚙️ How It Works

1. `ModelLoader` downloads tokenizer and model from Hugging Face.  
2. `Predictor` tokenizes the review text and runs inference using PyTorch.  
3. The highest logit score decides the sentiment label.  
4. Streamlit UI also displays prediction confidence.  

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

---

## 🔮 Improvement Ideas

- Add automated unit/integration tests
- Add Docker support for one-command deployment
- Add batch prediction endpoint (FastAPI/Flask)
- Add model evaluation dashboard and metrics tracking

---

## 🤝 Contributing

Contributions are welcome.  
Open an issue first to discuss major changes, then submit a PR.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).