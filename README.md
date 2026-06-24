# Telecom Customer Churn Prediction

This repository contains an end-to-end machine learning project for predicting customer churn in the telecom industry. The dataset is sourced from Kaggle and is used to explore customer behavior, build predictive models, and demonstrate a complete production-style ML workflow.

## What This Project Is About

Customer churn is a critical business problem because retaining existing customers is usually more cost-effective than acquiring new ones. This project shows how to turn raw telecom customer data into a practical churn prediction solution using machine learning.

It covers the full pipeline that recruiters often look for in data science and ML projects:

- ML model building
- feature engineering
- classification
- model evaluation
- FastAPI deployment
- Docker
- MLflow tracking
- GitHub Actions CI
- Evidently AI monitoring
- API testing
- business interpretation

## Overview

This project aims to:

- analyze churn patterns in customer data
- identify features that influence churn
- build and compare machine learning classifiers
- provide an end-to-end notebook workflow for exploration and prediction

## Dataset

The project uses the Telco Customer Churn dataset stored in [WA_Fn-UseC_-Telco-Customer-Churn.csv](WA_Fn-UseC_-Telco-Customer-Churn.csv).

The dataset includes customer information such as:

- demographics
- service usage
- contract details
- billing and payment information
- churn label

## Project Files

- [customer-churn-prediction.ipynb](customer-churn-prediction.ipynb) - Jupyter notebook containing data analysis, preprocessing, visualization, and model training
- [WA_Fn-UseC_-Telco-Customer-Churn.csv](WA_Fn-UseC_-Telco-Customer-Churn.csv) - raw dataset used in the project
- [README.md](README.md) - project overview and usage instructions

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- XGBoost

## Setup

How to Run API

1. Train model:
python src/train.py

2. Start API:
uvicorn app.main:app --reload

3. Open Swagger:
http://127.0.0.1:8000/docs

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. Open [customer-churn-prediction.ipynb](customer-churn-prediction.ipynb) in Jupyter Notebook or JupyterLab.
2. Run the cells in order to:
   - load the dataset
   - explore the data
   - preprocess features
   - train and evaluate machine learning models

## Notebook Workflow

The notebook includes the following steps:

- data loading and inspection
- exploratory data analysis
- data cleaning and preprocessing
- feature engineering
- model training and evaluation
- churn prediction insights

## Model Types Explored

The notebook compares multiple classification models, including:

- K-Nearest Neighbors
- Support Vector Classifier
- Random Forest
- Logistic Regression
- Decision Tree
- AdaBoost
- Gradient Boosting
- Voting Classifier

## Contributing

Contributions are welcome. If you would like to improve the analysis or add new models, feel free to open a pull request.

## License

This project is intended for educational and personal use. If you plan to reuse it publicly, consider adding an appropriate open-source license.

