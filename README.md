# 📝 T5 Text Summarizer

An NLP-based text summarization application built using a fine-tuned **T5-Small Transformer model**. The trained model is integrated with a **FastAPI backend** and an HTML frontend to generate concise summaries from input text.

## 🚀 Project Overview

This project implements an end-to-end **Text Summarization** pipeline using the **T5 (Text-to-Text Transfer Transformer)** architecture.

The T5-Small model is fine-tuned on a dialogue summarization dataset and integrated with a FastAPI application for generating summaries through a web interface.

### 🔄 Workflow

```text
Input Dialogue / Text
        ↓
Text Preprocessing
        ↓
T5 Tokenization
        ↓
Fine-Tuned T5-Small Model
        ↓
Beam Search
        ↓
Generated Summary
        ↓
FastAPI Backend
        ↓
HTML Web Interface
```

## ✨ Features

- 🤗 Fine-tuned T5-Small Transformer model
- 📝 Dialogue and text summarization
- 🧹 Text preprocessing and cleaning
- 🔤 T5 tokenization
- 🧠 Beam Search for summary generation
- ⚡ FastAPI backend
- 🌐 HTML frontend
- 📡 REST API endpoint
- 💻 Local deployment using Uvicorn

## 🛠️ Technologies

**Python | PyTorch | Hugging Face Transformers | T5 | FastAPI | Pydantic | Jinja2 | HTML | Uvicorn**

**Domain:** Natural Language Processing (NLP) | Deep Learning | Generative AI

## 🧠 Model

This project uses **T5-Small**, a Transformer-based encoder-decoder model developed by Google.

T5 treats different NLP tasks as a **text-to-text problem**. For this project, T5-Small was fine-tuned specifically for **text summarization**.

## 📊 Model Training

The training pipeline includes:

1. Dataset loading
2. Data preprocessing
3. Text cleaning
4. Tokenization using the T5 Tokenizer
5. T5-Small model initialization
6. Model fine-tuning
7. Validation
8. Summary generation

## 📂 Project Structure

```text
T5_Text_Summarizer/
│
├── Application/
│   ├── app.py
│   ├── index.html
│   └── t5_text_summarizer.ipynb
│
├── Output/
│   └── webpage_glimpse.jpg
│
├── requirements.txt
├── .gitignore
└── README.md
```

### 📄 File Description

| File / Folder | Description |
|---|---|
| `Application/app.py` | FastAPI backend and T5 inference pipeline |
| `Application/index.html` | Frontend interface for the summarization application |
| `Application/t5_text_summarizer.ipynb` | Model training, fine-tuning, and evaluation notebook |
| `Output/webpage_glimpse.jpg` | Screenshot of the working application |
| `requirements.txt` | Python dependencies required to run the application |
| `.gitignore` | Files and folders excluded from GitHub |
| `README.md` | Project documentation |

> **Note:** The trained `saved_summary_model/` folder is not included in the GitHub repository because model weights can be large. The trained model is kept locally and backed up separately.

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd T5_Text_Summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Trained Model

Place the trained model inside the `Application` folder:

```text
Application/
├── app.py
├── index.html
├── t5_text_summarizer.ipynb
└── saved_summary_model/
```

The `saved_summary_model` directory should contain the trained model and tokenizer files.

## ▶️ Running the Application

Navigate to the `Application` directory:

```bash
cd Application
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000/
```

### 📚 FastAPI Documentation

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 🔌 API Endpoint

### `POST /summarize/`

Accepts input dialogue/text and returns a generated summary.

### Request

```json
{
    "dialogue": "Your input text goes here."
}
```

### Response

```json
{
    "summary": "Generated summary goes here."
}
```

## 🔮 Future Improvements

- 🚀 Deploy the application online
- 🤗 Host the trained model on Hugging Face
- 🎨 Improve the frontend UI
- ⚡ Optimize inference speed
- 📚 Train on a larger dataset
- 🧠 Experiment with T5-Base and larger Transformer models
- 📊 Add additional evaluation metrics
- 🔐 Add production-ready API configuration

## 👩‍💻 Author

**Heer Shah**
