"""Inspect saved match result model predictions for two teams."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "match_result_logistic_regression.pkl"
TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "team_stats.csv"

FEATURE_COLUMNS = [
    "recent_form_diff",
    "goals_for_last_10_diff",
    "goals_against_last_10_diff",
    "goal_diff_last_10_diff",
    "strength_gap",
    "host_advantage_diff",
]

REQUIRED_TEAM_STATS_COLUMNS = [
    "team",
    "recent_form",
    "goals_for_last_10",
    "goals_against_last_10",
]

CLASS_LABELS = ["draw", "team_a_win", "team_b_win"]


def load_team_stats() -> pd.DataFrame:
    team_stats = pd.read_csv(TEAM_STATS_PATH)

    missing_columns = [
        column for column in REQUIRED_TEAM_STATS_COLUMNS if column not in team_stats.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing team stats columns: {missing_columns}")

    return team_stats


def get_team_row(team_name: str, team_stats: pd.DataFrame) -> pd.Series:
    matches = team_stats[
        team_stats["team"].astype(str).str.lower() == team_name.lower()
    ]

    if matches.empty:
        available_teams = ", ".join(sorted(team_stats["team"].astype(str).tolist()))
        raise ValueError(f"Team '{team_name}' not found. Available teams: {available_teams}")

    return matches.iloc[0]


def build_feature_row(team_a_row: pd.Series, team_b_row: pd.Series) -> pd.DataFrame:
    recent_form_diff = float(team_a_row["recent_form"]) - float(
        team_b_row["recent_form"]
    )
    goals_for_last_10_diff = float(team_a_row["goals_for_last_10"]) - float(
        team_b_row["goals_for_last_10"]
    )
    goals_against_last_10_diff = float(
        team_b_row["goals_against_last_10"]
    ) - float(team_a_row["goals_against_last_10"])

    team_a_goal_diff_last_10 = float(team_a_row["goals_for_last_10"]) - float(
        team_a_row["goals_against_last_10"]
    )
    team_b_goal_diff_last_10 = float(team_b_row["goals_for_last_10"]) - float(
        team_b_row["goals_against_last_10"]
    )
    goal_diff_last_10_diff = team_a_goal_diff_last_10 - team_b_goal_diff_last_10

    strength_gap = (
        abs(recent_form_diff)
        + abs(goals_for_last_10_diff)
        + abs(goals_against_last_10_diff)
    )

    team_a_host_advantage = 0.0
    team_b_host_advantage = 0.0
    if "host_advantage" in team_a_row.index:
        team_a_host_advantage = float(team_a_row["host_advantage"])
        team_b_host_advantage = float(team_b_row["host_advantage"])

    host_advantage_diff = team_a_host_advantage - team_b_host_advantage

    feature_values = {
        "recent_form_diff": recent_form_diff,
        "goals_for_last_10_diff": goals_for_last_10_diff,
        "goals_against_last_10_diff": goals_against_last_10_diff,
        "goal_diff_last_10_diff": goal_diff_last_10_diff,
        "strength_gap": strength_gap,
        "host_advantage_diff": host_advantage_diff,
    }

    missing_features = [
        column for column in FEATURE_COLUMNS if column not in feature_values
    ]
    if missing_features:
        raise ValueError(f"Missing calculated feature columns: {missing_features}")

    return pd.DataFrame([[feature_values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = joblib.load(MODEL_PATH)
    team_stats = load_team_stats()

    team_a_row = get_team_row(args.team_a, team_stats)
    team_b_row = get_team_row(args.team_b, team_stats)

    X = build_feature_row(team_a_row, team_b_row)

    predicted_class = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    probabilities_by_class = dict(zip(model.classes_, probabilities))

    print(f"team_a: {args.team_a}")
    print(f"team_b: {args.team_b}")

    print("\nFeature values:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}: {X.iloc[0][column]}")

    print(f"\nPredicted class: {predicted_class}")

    print("\nPredicted probabilities:")
    for label in CLASS_LABELS:
        print(f"- {label}: {probabilities_by_class.get(label, 0.0):.4f}")


if __name__ == "__main__":
    main()
