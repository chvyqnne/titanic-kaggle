# Titanic - Machine Learning from Disaster

A machine learning project for the [Titanic Kaggle competition](https://www.kaggle.com/competitions/titanic/overview), predicting passenger survival using ensemble and stacking models with engineered features and model tuning.

## Directory Structure

``` bash
titanic-kaggle/
├── data/ # Raw train and test datasets 
├── notebooks/ # EDA and tuning notebooks
├── outputs/ # Submission
├── src/ # Core project logic (EDA, models, pipeline, stacking, and tuning)
├── main.py
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repo:

```bash
git clone https://github.com/your-username/titanic-kaggle.git
cd titanic-kaggle
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python src/pipeline.py
```
This will preprocess data, tune and evaluate models, train the stacking classifier, and generate a `submission.csv` in `outputs/`

## Notebooks
- `notebooks/eda.ipynb`: Exploratory Data Analysis and feature insights
- `notebooks/tuning.ipynb`: Grid search, CV, model comparison with visualizations

## Model Performance (Stacking)

| Metric | Score |
| ------ | ----- |
| Accuracy | 0.838 |
| Precision | 0.836 |
| Recall | 0.757 |
| F1 Score | 0.794 |
| Kaggle Public Score | **0.77990** |
