# Fake Job Posting Detection System (V2)

**An End-to-End Machine Learning System that identifies fraudulent job postings using a Hybrid Architecture.**


## Project Overview
Employment scams are on the rise, with sophisticated fraudsters copying legitimate listings to steal candidate data. This project provides a **ML system** designed to filter out these scams.

This system uses a **Hybrid Approach**:
1.  **Layer 1 (The Brain):** A Random Forest model (trained on 17,000+ records) and analyzes text patterns using TF-IDF.
2.  **Layer 2 (The Guardrails):** A Rule-Based Logic engine flags high-risk metadata anomalies (e.g., missing logos in remote jobs) that might trick the AI.

## Key Features
* **Automated Data Pipeline:** Ingestion, Transformation (TF-IDF + OneHot), and Training scripts.
* **High Performance:** Achieved **0.72 F1-Score** (Class 1) and **about 100% Precision** (~Zero False Positives).
* **Interactive Web App:** A Streamlit UI for real-time risk assessment.
* **Smart Risk Logic:** Detects "Content Spoofing" where scammers copy real text but fail metadata checks.
* **Scalable Architecture:** Modular code structure (`src` components) ready for deployment.

## Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn, Pandas, NumPy, Imbalanced-Learn
* **Web Framework:** Streamlit
* **Environment Management:** uv (modern Python package manager)
* **Data Processing:** TF-IDF Vectorization, One-Hot Encoding

## Project Structure
```text
Job-Authentication-V2/
│
├── artifacts/              # Saved Models & Preprocessors (.pkl)
├── data/                   # Raw and Processed Datasets
├── src/
│   ├── components/         # Core ML Modules
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/           # Inference Pipelines
│   │   └── predict_pipeline.py
│   └── utils.py            # Utility functions (save/load objects)
│
├── app.py                  # Streamlit Web Application
├── run_pipeline.py         # Script to trigger full training
├── requirements.txt        # Dependencies
└── README.md               # Project Documentation
