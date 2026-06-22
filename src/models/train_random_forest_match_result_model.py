"""Train and compare Random Forest match result classifiers."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "match_training_dataset.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "match_result_random_forest.pkl"

TARGET_COLUMN = "result"

FEATURE_COLUMNS = [
    "recent_form_diff",
    "goals_for_last_10_diff",
    "goals_against_last_10_diff",
    "goal_diff_last_10_diff",
    "strength_gap",
    "host_advantage_diff",
]

CLASS_LABELS = ["draw", "team_a_win", "team_b_win"]

MODEL_CONFIGS = [
    {
        "model_name": "baseline_random_forest",
        "params": {
            "n_estimators": 300,
            "random_state": 42,
            "max_depth": None,
        },
    },
    {
        "model_name": "balanced_random_forest",
        "params": {
            "n_estimators": 300,
            "random_state": 42,
            "max_depth": None,
            "class_weight": "balanced",
        },
    },
    {
        "model_name": "limited_depth_random_forest",
        "params": {
            "n_estimators": 300,
            "random_state": 42,
            "max_depth": 8,
            "min_samples_leaf": 5,
        },
    },
]


def validate_columns(df: pd.DataFrame) -> None:
    missing_feature_columns = [
        column for column in FEATURE_COLUMNS if column not in df.columns
    ]
    if missing_feature_columns:
        raise ValueError(f"Missing feature columns: {missing_feature_columns}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")


def train_and_evaluate(
    model_name: str,
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba, labels=model.classes_)
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 72)
    print(f"Model name: {model_name}")
    print("=" * 72)
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features used: {FEATURE_COLUMNS}")

    print("\nAccuracy:")
    print(accuracy)

    print("\nLog loss:")
    print(loss)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=CLASS_LABELS, zero_division=0))

    print("\nConfusion matrix:")
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=CLASS_LABELS),
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )
    print(confusion_matrix_df)

    print("\nRecall summary:")
    print(f"draw recall: {report['draw']['recall']}")
    print(f"team_a_win recall: {report['team_a_win']['recall']}")
    print(f"team_b_win recall: {report['team_b_win']['recall']}")

    print("\nFeature importances:")
    feature_importances_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(feature_importances_df)

    return {
        "model_name": model_name,
        "model": model,
        "accuracy": accuracy,
        "log_loss": loss,
        "draw_recall": report["draw"]["recall"],
        "team_a_win_recall": report["team_a_win"]["recall"],
        "team_b_win_recall": report["team_b_win"]["recall"],
    }


def print_comparison_summary(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("Final comparison summary")
    print("=" * 72)

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "model_name": result["model_name"],
                "accuracy": result["accuracy"],
                "log_loss": result["log_loss"],
                "draw_recall": result["draw_recall"],
                "team_a_win_recall": result["team_a_win_recall"],
                "team_b_win_recall": result["team_b_win_recall"],
            }
        )

    print(pd.DataFrame(summary_rows))


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Available columns:")
    for column in df.columns:
        print(f"- {column}")

    print(f"\nClass distribution for '{TARGET_COLUMN}':")
    print(df[TARGET_COLUMN].value_counts())

    validate_columns(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    for config in MODEL_CONFIGS:
        model = RandomForestClassifier(**config["params"])
        results.append(
            train_and_evaluate(
                model_name=config["model_name"],
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
        )

    print_comparison_summary(results)

    best_result = min(results, key=lambda result: result["log_loss"])

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result["model"], MODEL_OUTPUT_PATH)

    print(f"\nSaved best Random Forest model: {best_result['model_name']}")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
