"""Train a simple baseline match result classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

USE_CLASS_BALANCING = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "match_training_dataset.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "match_result_logistic_regression.pkl"

TARGET_COLUMN = "result"

# TODO: Choose which features you want the model to use.
# Start small, inspect the results, then edit this list manually.
FEATURE_COLUMNS = [
    "recent_form_diff",
    "goals_for_last_10_diff",
    "goals_against_last_10_diff",
    "strength_gap",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Available columns:")
    for column in df.columns:
        print(f"- {column}")

    print(f"\nClass distribution for '{TARGET_COLUMN}':")
    print(df[TARGET_COLUMN].value_counts())

    missing_feature_columns = [
        column for column in FEATURE_COLUMNS if column not in df.columns
    ]
    if missing_feature_columns:
        raise ValueError(f"Missing feature columns: {missing_feature_columns}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

        # TODO: Choose whether class balancing helps this project.
    # True = model cares more about draws.
    # False = model may get higher accuracy but ignore draws.
    class_weight = "balanced" if USE_CLASS_BALANCING else None

    # First baseline model: Logistic Regression inside a pipeline.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    classifier = model.named_steps["classifier"]

    # TODO: Choose which evaluation metrics matter most for this project.
    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nLog loss:")
    print(log_loss(y_test, y_pred_proba, labels=classifier.classes_))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion matrix:")
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=classifier.classes_),
        index=classifier.classes_,
        columns=classifier.classes_,
    )
    print(confusion_matrix_df)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nSaved model to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()