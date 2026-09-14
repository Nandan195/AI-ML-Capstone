# AI-ML Capstone Project

## Overview

This repository contains an end-to-end AI/ML capstone project comprising three connected modules: a web-scraping data pipeline, an analytics and machine learning suite, and an intelligent Support Assistant built with RAG.

## Modules

### 1. Data Pipeline

- Scrapes book data from `books.toscrape.com` using `requests` and `BeautifulSoup`.
- Cleans and transforms raw data, converting GBP to INR strictly using the rate `1 GBP = 105.50 INR`.
- Stores the processed data in a normalized SQLite database with a primary key / foreign key relationship between categories and books.
- Executes SQL queries validated against Pandas DataFrames to answer analytical questions, outputting results to CSV.

### 2. Analytics & Machine Learning

- Loads the standard Titanic dataset securely.
- Performs extensive Exploratory Data Analysis (EDA), missing-value handling, and visualizations.
- Evaluates classification models (Logistic Regression, Decision Trees, Random Forest), tuning Random Forest via Grid Search.
- Addresses class imbalance using SMOTE and compares performance metrics.
- Computes a regression side-task on passenger fare.
- Saves the complete `sklearn` data preprocessing and modeling pipeline as a joblib artifact.

### 3. Support Assistant

- Analyzes an 8-document Zepto policy corpus.
- Chunks and natively embeds documents utilizing `all-MiniLM-L6-v2` via `sentence-transformers`.
- Populates and queries a local `ChromaDB` vector store using cosine similarity.
- Orchestrates conditional routing logic using `LangGraph` to dynamically intercept intent (policy vs general queries).
- Exposes a `FastAPI` endpoint (`POST /ask`).
- Features a deterministic, fully-offline Mock Mode, alongside an optional Gemini path.
- Includes a structured `Dockerfile`.

## Repository Structure

```text
|-- .gitignore
|-- README.md
|-- analytics/
|   |-- 01_eda.py
|   |-- 02_modeling.py
|   |-- README.md
|   |-- artifacts/
|   |   |-- best_pipeline.joblib
|   |-- outputs/
|   |   |-- plots/
|   |   |-- reports/
|   |-- titanic.csv
|-- data_pipeline/
|   |-- README.md
|   |-- database.py
|   |-- database/
|   |   |-- zepto_books.db
|   |-- outputs/
|   |-- queries.py
|   |-- run_pipeline.py
|   |-- scrape_pipeline.py
|-- requirements.txt
|-- support_assistant/
|   |-- Dockerfile
|   |-- README.md
|   |-- data/
|   |-- docs/
|   |-- embeddings.py
|   |-- gemini_client.py
|   |-- graph.py
|   |-- ingestion.py
|   |-- main.py
|   |-- models.py
|   |-- outputs/
|   |-- prompts.py
|   |-- vector_store.py
```

## Setup

Create and activate the Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(Note: Do not commit the `.venv` directory to version control.)*

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Project

### Data Pipeline

Execute the complete data pipeline sequence:

```powershell
python data_pipeline/run_pipeline.py
```

### Analytics

Run the Exploratory Data Analysis script:

```powershell
python analytics/01_eda.py
```

Run the Modeling script:

```powershell
python analytics/02_modeling.py
```

### Support Assistant

First, run the deterministic ingestion script to build the local ChromaDB index:

```powershell
python -m support_assistant.ingestion
```

Then, boot the FastAPI server using Uvicorn:

```powershell
python -m uvicorn support_assistant.main:app --port 8000
```

- **MOCK_LLM=1 (or unset):** The required offline mode. The assistant intercepts queries natively without making external API calls.
- **MOCK_LLM=0:** The optional Gemini mode. Requires exporting `GEMINI_API_KEY` to the environment to process external LLM generations.

## Docker

A `Dockerfile` is provided inside `support_assistant/` for containerization. Docker execution was not tested because the Docker daemon was unavailable on the development machine.

## Git / Submission

The repository's Git history contains the required branching structure. All developmental commits were recorded on the `feature/capstone-complete` feature branch, which was merged securely back into `main` using an explicit merge commit to preserve the complete history.

## Final Project Status

All three graded modules (Data Pipeline, Analytics, and Support Assistant) have been successfully implemented and tested locally. The Docker runtime remains formally untested because the Docker engine was unavailable on the host. The project is finalized and complete.
