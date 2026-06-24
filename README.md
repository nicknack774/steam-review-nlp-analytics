# Steam Review NLP Analytics

Comparative evaluation of NLP models (BERT, VADER, TextBlob) and storage strategies (PostgreSQL, Parquet, SQLite) for large-scale Steam review analysis (~991K reviews).

## Setup
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Download dataset: [link or instructions]
4. `python app.py` → http://127.0.0.1:8080

## Structure
- `frontend/` — Flask templates and static assets
- `notebooks/` — model training, SHAP analysis, benchmarking
- `figures/` — generated charts for analysis
