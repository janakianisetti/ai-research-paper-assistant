
# AI Research Paper Synthesizer 🧠

An intelligent web application that uses advanced Machine Learning (Hugging Face Transformers) to read academic PDFs and extract key insights instantly.

This is a perfect beginner-friendly AI project designed to demonstrate practical skills in Python, backend development, and NLP integration.

![Screenshot mock]({{{app_screenshot_mock}}}) 

## Features
- **PDF Parsing**: Automatically extracts text from academic PDF documents.
- **AI Synthesis**: Uses a powerful local inference model (`google/flan-t5-base`) to interpret the text.
- **Multi-layered Output**:
  - Abstract Summary
  - Key Findings
  - ELI5 (Explain Like I'm 5) Simple Explanation
- **Interactive Chatbot**: Ask questions about the paper using a dedicated `question-answering` model (`deepset/roberta-base-squad2`).
- **Suggested Questions**: The AI analyzes the paper and automatically suggests 5-8 relevant questions to help you explore the document.
- **Aesthetic UI**: A modern, responsive glass-morphism frontend design built without external frameworks.

## Tech Stack
- **Backend:** Python, Flask server
- **AI/ML:** Hugging फेस `transformers`, PyTorch
- **Data Extractor:** `PyPDF2`
- **Frontend:** Vanilla HTML/CSS/JS (Google Fonts: Inter)

## How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed. 

### 2. Setup Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
When running off a fresh install, the app will download the T5 ML model from Hugging Face for the first time. This may take a minute and consume about 1GB of storage space.

### 5. Access
Open your browser and navigate to: `http://127.0.0.1:5000`

## How the AI AI Works
Under the hood, `summarizer.py` utilizes the Hugging Face `pipeline` API to instantiate two models:
1. `text2text-generation` (`google/flan-t5-base`): Uniquely trained to follow direct prompts. Used for Summarization, Key Findings, ELI5 explanation, and generating Suggested Questions.
2. `question-answering` (`deepset/roberta-base-squad2`): Used strictly for the Chatbot functionality to pinpoint specific answers to user questions using the paper as context.

When a PDF is uploaded, PyPDF2 reads the text, cleaning it up. We then pass a bounded chunk of this text alongside specific prompts (e.g. "What are the key findings of this text?") into the T5 model. The outputs are generated locally on your processor (or GPU) and relayed back to the frontend.

When asking a question via the interactive Chatbot, the stored text chunk and user's query are passed to the QA model to extract precise answers interactively.



