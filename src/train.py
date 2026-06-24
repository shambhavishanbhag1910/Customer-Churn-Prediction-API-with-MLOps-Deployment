from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def get_project_root() -> Path:
    """
    Returns project root directory.
    If this file is src/train.py, then project root is one level above src.
    """
    return Path(__file__).resolve().parents[1]


def find_dataset(project_root: Path) -> Path:
    """
    Finds the Telco Churn dataset in common locations.
    """
    possible_paths = [
        project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset not found. Keep 'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
        "in project root, data/, or data/raw/ folder."
    )


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Loads dataset and cleans column names.
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Telco Churn dataset safely.

    Fixes:
    - TotalCharges object/string issue
    - Churn Yes/No conversion
    - duplicate rows
    - target NaN issue
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()

    if "Churn" not in df.columns:
        raise KeyError("Churn column not found in dataset.")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Safe Churn conversion
    # This avoids converting 0/1 into NaN if code is re-run.
    churn_values = set(df["Churn"].dropna().unique())

    if churn_values <= {"Yes", "No"}:
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    elif churn_values <= {0, 1}:
        df["Churn"] = df["Churn"].astype(int)
    else:
        raise ValueError(f"Unexpected Churn values found: {churn_values}")

    if df["Churn"].isna().any():
        raise ValueError("Churn contains NaN after conversion. Check original Churn values.")

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separates X and y.
    Drops customerID because it is only an identifier.
    """
    X = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()
    y = df["Churn"].copy()

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds preprocessing pipeline:
    - Numeric columns: median imputation + scaling
    - Categorical columns: most frequent imputation + one-hot encoding
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # sklearn version compatibility
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    return preprocessor


def get_models():
    """
    Returns candidate models for comparison.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=11,
        ),
    }

    return models


def evaluate_model(model_pipeline, X_test, y_test) -> dict:
    """
    Evaluates model and returns metrics.
    """
    y_pred = model_pipeline.predict(X_test)

    if hasattr(model_pipeline.named_steps["model"], "predict_proba"):
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }

    return metrics


def train_and_compare_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Trains multiple models and selects best model based on F1 score.
    """
    models = get_models()

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        result = {
            "model_name": model_name,
            **metrics,
        }

        results.append(result)
        trained_pipelines[model_name] = pipeline

        print(
            f"{model_name} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC_AUC: {metrics['roc_auc']:.4f}"
            if metrics["roc_auc"] is not None
            else f"{model_name} | F1: {metrics['f1']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)

    best_model_name = results_df.iloc[0]["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    return best_model_name, best_pipeline, results_df


def save_artifacts(project_root: Path, best_pipeline, results_df, best_model_name, X_test, y_test):
    """
    Saves:
    - best model pipeline
    - model metrics CSV
    - model metrics JSON
    - classification report
    """
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"

    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_churn_model_pipeline.joblib"
    metrics_csv_path = reports_dir / "model_metrics.csv"
    metrics_json_path = reports_dir / "model_metrics.json"
    classification_report_path = reports_dir / "classification_report.txt"

    joblib.dump(best_pipeline, model_path)

    results_df.to_csv(metrics_csv_path, index=False)

    best_pred = best_pipeline.predict(X_test)

    report = classification_report(
        y_test,
        best_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )

    with open(classification_report_path, "w", encoding="utf-8") as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(report)

    best_metrics = results_df.iloc[0].to_dict()

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=4)

    print("\nArtifacts saved:")
    print("Model:", model_path)
    print("Metrics CSV:", metrics_csv_path)
    print("Metrics JSON:", metrics_json_path)
    print("Classification Report:", classification_report_path)


def main():
    project_root = get_project_root()

    data_path = find_dataset(project_root)
    print("Dataset path:", data_path)

    df_raw = load_data(data_path)
    print("Raw data shape:", df_raw.shape)

    df = clean_data(df_raw)
    print("Cleaned data shape:", df.shape)
    print("Churn distribution:")
    print(df["Churn"].value_counts())

    X, y = split_features_target(df)

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    best_model_name, best_pipeline, results_df = train_and_compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    print("\nModel comparison:")
    print(results_df)

    print("\nBest model:", best_model_name)

    save_artifacts(
        project_root=project_root,
        best_pipeline=best_pipeline,
        results_df=results_df,
        best_model_name=best_model_name,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()