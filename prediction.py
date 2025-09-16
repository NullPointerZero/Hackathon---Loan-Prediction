"""Loan approval prediction script using a feature engineering and model pipeline.

This script trains a model on the provided Kaggle-like loan approval dataset
and generates a submission file for the test set. Compared to the original
approach, the new pipeline handles missing values more robustly, applies proper
categorical encoding, scales numeric features, evaluates the model with cross
validation, and uses a stronger classifier (RandomForest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_PATH = "train_ctrUa4K.csv"
TEST_PATH = "test_lAUu6dG.csv"
SUBMISSION_TEMPLATE_PATH = "sample_submission_file.csv"


def load_datasets(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    submission_template_path: str = SUBMISSION_TEMPLATE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, test and submission template CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(submission_template_path)
    return train_df, test_df, submission_df


def build_model_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Create the preprocessing and modeling pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return model


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> None:
    """Evaluate the model with stratified cross validation and print the scores."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, features, target, cv=cv, scoring="accuracy")
    print(
        "Cross-validation accuracy: "
        f"{scores.mean():.4f} ± {scores.std():.4f}"
    )


def main() -> None:
    train_df, test_df, submission_df = load_datasets()

    features = train_df.drop(columns=["Loan_ID", "Loan_Status"])
    target = train_df["Loan_Status"].map({"Y": 1, "N": 0}).astype(int)

    test_features = test_df.drop(columns=["Loan_ID"])

    categorical_features = features.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [col for col in features.columns if col not in categorical_features]

    model = build_model_pipeline(categorical_features, numeric_features)

    evaluate_model(model, features, target)

    model.fit(features, target)
    predictions = model.predict(test_features)

    submission = submission_df.copy()
    submission["Loan_ID"] = test_df["Loan_ID"]
    submission["Loan_Status"] = np.where(predictions == 1, "Y", "N")
    submission.to_csv("my_submission.csv", index=False)
    print("Saved predictions to my_submission.csv")


if __name__ == "__main__":
    main()
