# Final Project — Computer Vision & Natural Language Processing

**Student:** Othmar Casilla
**Course:** Introduction to Computer Vision & Introduction to Natural Language Processing
**Submitted:** April 2026

This archive contains two deliverables:

1. **`home-estimator/`** — the main combined CV + NLP final project.
2. **`react_agent.ipynb`** — a supplementary LangChain ReAct-agent notebook demonstrating tool use.

The project report is in `home-estimator/report/final_report.pdf` (PDF), with the source markdown alongside it.

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or newer
- ~3 GB free disk for dependencies (PyTorch, Whisper, spaCy)

### 2. Set up a virtual environment

```bash
cd home-estimator
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> The `spacy download` line is required — `en_core_web_sm` is not on PyPI.

### 4. Configure API keys

Copy the example env file and fill in your own OpenAI key:

```bash
cp .env.example .env
# then edit .env and replace `your_openai_api_key_here` with your real key
```

The OpenAI key is only needed for:
- The "Smart Analysis" tab in the Streamlit app (`app.py`)
- The chat / Whisper voice features
- The supplementary `react_agent.ipynb` (also needs `SERPER_API_KEY` for web search)

The notebooks (`01_eda`, `02_image_model`, `03_text_model`) and the rule-based parts of the app **do not** require any API keys.

---

## Running the Project

### Streamlit Web App

```bash
streamlit run app.py
```

The app loads the pre-trained models from `models/` and exposes four tabs:
- **Get Estimate** — upload a photo + describe a job to get a cost estimate
- **Chat with AI** — follow-up Q&A (requires OpenAI key)
- **How It Works** — architecture overview
- **Model Performance** — confusion matrices and training curves

### Notebooks

Run them in order from `home-estimator/notebooks/`:

| # | Notebook | Purpose | API key needed? |
|---|---|---|---|
| 01 | `01_eda.ipynb` | Exploratory data analysis (text + images + pricing) | No |
| 02 | `02_image_model.ipynb` | Train MobileNetV2 image classifier | No |
| 03 | `03_text_model.ipynb` | Train TF-IDF + Logistic Regression text classifiers | No |

All three notebooks have stored outputs so results are visible without re-running. Re-running `02` and `03` will regenerate the model files in `models/`.

### Supplementary ReAct Agent

```bash
cd ..        # back to project root
jupyter notebook react_agent.ipynb
```

This is a standalone notebook unrelated to the home-estimator project. It demonstrates LangChain's ReAct framework and requires both `OPENAI_API_KEY` and `SERPER_API_KEY` (free tier available at serper.dev).

---

## Project Structure

```
class NLP-submission/
├── README.md                  ← you are here
├── .env.example               ← template for ReAct agent keys
├── react_agent.ipynb          ← LangChain ReAct demo
├── class-project-details      ← course rubric
└── home-estimator/            ← main project
    ├── .env.example
    ├── requirements.txt
    ├── app.py                 ← Streamlit web application
    ├── notebooks/             ← EDA, image model, text model (executed)
    ├── src/                   ← reusable pipeline modules
    │   ├── cv_pipeline.py
    │   ├── nlp_pipeline.py
    │   ├── chat_pipeline.py
    │   ├── voice_pipeline.py
    │   ├── estimator.py
    │   └── utils.py
    ├── data/
    │   ├── images/            ← 90 web-crawled training images (15 per class)
    │   ├── texts/             ← labeled job descriptions
    │   └── pricing/           ← pricing reference table
    ├── models/                ← pre-trained models (so the app runs out of the box)
    ├── report/
    │   ├── final_report.md
    │   ├── final_report.pdf   ← graded report
    │   └── figures/           ← all charts referenced in the report
    ├── download_images.py     ← optional: regenerate the image dataset
    ├── download_images_bing.py
    └── scrape_reddit.py       ← optional: collect additional Reddit job descriptions
```

---

## Notes for the Grader

- **Pre-trained models are included** under `home-estimator/models/`, so `app.py` runs without retraining.
- **All notebooks have stored outputs**; cells can be inspected without re-execution.
- **No live API key is included** in this archive. Provide your own in `.env` to enable the OpenAI-backed features. The rule-based estimator works without any key.
- **Datasets are included** under `home-estimator/data/`. The image dataset was assembled via Bing Image Search using `download_images_bing.py`.

If you run into any setup issue, the most common cause is a missing `en_core_web_sm` — re-run `python -m spacy download en_core_web_sm` inside the activated virtual environment.
