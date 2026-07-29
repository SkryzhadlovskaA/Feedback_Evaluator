# Feedback Evaluator

NLP pipeline for analyzing Erasmus+ participant feedback.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

Edit `INPUT_PATH` in `run_evaluator_all_steps.py`, then:

```bash
python run_evaluator_all_steps.py
```

Output is saved to `outputs/<project_name>_final_output.json`.

## Web UI (optional)

```bash
pip install streamlit
streamlit run app.py
```

In the browser you can:
- Pick an existing CSV from `data/raw/` or upload a new one
- Run the full pipeline with one click
- View strengths, improvements, and learning outcomes
- Download the JSON report

## Pipeline steps

1. Preprocess CSV → split into sentences
2. Sentiment analysis (VADER + rules, question-aware)
3. Learning sentence detection
4. Erasmus+ outcome matching (embeddings)
5. Cluster strengths and improvements
6. Extract representative quotes
7. Build final JSON report

## Input CSV format

Required columns: `project_id`, `response_id`, `question`, `text`

Semicolon-separated files (European Excel export) are supported.

## Learning outcomes (Erasmus + project topic)

Each learning sentence is matched against **11 categories**:

- 10 official Erasmus+ outcomes (defined in `src/taxonomy.py`)
- 1 **project topic** outcome (defined in `PROJECT_TOPICS` by `project_id`)
- If nothing matches clearly (similarity below 0.40), the label **Other** is assigned

### Adding a new project

Before running the pipeline, add an entry to `PROJECT_TOPICS` in `src/taxonomy.py`:

```python
PROJECT_TOPICS = {
    "Your_Project_Id": {  # must match project_id in the CSV
        "label": "Project topic: Your theme here",
        "description": "keywords describing the project theme",
        "examples": [
            "Example participant sentence 1",
            "Example participant sentence 2",
        ]
    },
}
```

The `project_id` column in your CSV must match the key exactly.

## Step-by-step scripts

- `run_step1.py` – preprocessing
- `run_step2.py` – sentiment
- `run_step3.py` – learning detection
- `run_step4.py` – outcome matching
- `run_step5.py` – clustering
- `run_step6.py` – quote extraction
- `run_step7.py` – final JSON
