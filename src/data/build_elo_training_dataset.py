"""Build an Elo/FIFA-rank match training dataset for experimentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "international_results_2023_onward.csv"
TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "combined_team_stats.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "elo_match_training_dataset.csv"

RESULT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]

TEAM_STATS_COLUMNS = [
    "team",
    "recent_form",
    "goals_for_last_10",
    "goals_against_last_10",
    "matches_played",
    "elo_rating",
    "fifa_rank",
    "host_advantage",
]

OUTPUT_COLUMNS = [
    "date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "tournament",
    "neutral",
    "elo_diff",
    "fifa_rank_diff",
    "recent_form_diff",
    "goals_for_last_10_diff",
    "goals_against_last_10_diff",
    "goal_diff_last_10_diff",
    "strength_gap",
    "host_advantage_diff",
    "result",
    "team_a_win",
    "draw",
    "team_b_win",
]


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing_columns}"
        )


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find historical results file at: {path}")

    results = pd.read_csv(path)
    validate_columns(results, RESULT_COLUMNS, "historical results")
    return results


def load_team_stats(path: Path = TEAM_STATS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Could not find combined team stats file at: "
            f"{path}. Run src/data/build_combined_team_stats.py first."
        )

    team_stats = pd.read_csv(path)
    validate_columns(team_stats, TEAM_STATS_COLUMNS, "combined team stats")
    team_stats = team_stats.copy()
    team_stats["team_key"] = team_stats["team"].astype(str).str.lower()
    return team_stats


def get_team_features(
    team_name: str,
    team_stats_df: pd.DataFrame,
) -> pd.Series | None:
    matches = team_stats_df[
        team_stats_df["team_key"] == str(team_name).lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def has_elo_and_rank(team_features: pd.Series) -> bool:
    return (
        pd.notna(team_features["elo_rating"])
        and pd.notna(team_features["fifa_rank"])
    )


def get_result_label(team_a_goals: int, team_b_goals: int) -> str:
    if team_a_goals > team_b_goals:
        return "team_a_win"

    if team_a_goals == team_b_goals:
        return "draw"

    return "team_b_win"


def build_match_row(
    match_row: pd.Series,
    team_stats_df: pd.DataFrame,
) -> tuple[dict[str, Any] | None, str | None]:
    team_a = str(match_row["home_team"])
    team_b = str(match_row["away_team"])

    team_a_features = get_team_features(team_a, team_stats_df)
    team_b_features = get_team_features(team_b, team_stats_df)

    if team_a_features is None or team_b_features is None:
        return None, "team_missing"

    if not has_elo_and_rank(team_a_features) or not has_elo_and_rank(team_b_features):
        return None, "elo_or_fifa_rank_missing"

    team_a_goals = int(match_row["home_score"])
    team_b_goals = int(match_row["away_score"])
    result = get_result_label(team_a_goals, team_b_goals)

    elo_diff = float(team_a_features["elo_rating"]) - float(
        team_b_features["elo_rating"]
    )
    fifa_rank_diff = float(team_b_features["fifa_rank"]) - float(
        team_a_features["fifa_rank"]
    )
    recent_form_diff = float(team_a_features["recent_form"]) - float(
        team_b_features["recent_form"]
    )
    goals_for_last_10_diff = float(team_a_features["goals_for_last_10"]) - float(
        team_b_features["goals_for_last_10"]
    )
    goals_against_last_10_diff = float(
        team_b_features["goals_against_last_10"]
    ) - float(team_a_features["goals_against_last_10"])

    team_a_goal_diff_last_10 = float(team_a_features["goals_for_last_10"]) - float(
        team_a_features["goals_against_last_10"]
    )
    team_b_goal_diff_last_10 = float(team_b_features["goals_for_last_10"]) - float(
        team_b_features["goals_against_last_10"]
    )
    goal_diff_last_10_diff = team_a_goal_diff_last_10 - team_b_goal_diff_last_10

    strength_gap = (
        abs(recent_form_diff)
        + abs(goals_for_last_10_diff)
        + abs(goals_against_last_10_diff)
    )
    host_advantage_diff = float(team_a_features["host_advantage"]) - float(
        team_b_features["host_advantage"]
    )

    return (
        {
            "date": match_row["date"],
            "team_a": team_a,
            "team_b": team_b,
            "team_a_goals": team_a_goals,
            "team_b_goals": team_b_goals,
            "tournament": match_row["tournament"],
            "neutral": match_row["neutral"],
            "elo_diff": elo_diff,
            "fifa_rank_diff": fifa_rank_diff,
            "recent_form_diff": recent_form_diff,
            "goals_for_last_10_diff": goals_for_last_10_diff,
            "goals_against_last_10_diff": goals_against_last_10_diff,
            "goal_diff_last_10_diff": goal_diff_last_10_diff,
            "strength_gap": strength_gap,
            "host_advantage_diff": host_advantage_diff,
            "result": result,
            "team_a_win": int(result == "team_a_win"),
            "draw": int(result == "draw"),
            "team_b_win": int(result == "team_b_win"),
        },
        None,
    )


def build_elo_training_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    results = load_results()
    team_stats = load_team_stats()

    training_rows = []
    skipped_team_missing = 0
    skipped_elo_or_fifa_rank_missing = 0

    for _, match_row in results.iterrows():
        training_row, skip_reason = build_match_row(match_row, team_stats)

        if training_row is None:
            if skip_reason == "team_missing":
                skipped_team_missing += 1
            elif skip_reason == "elo_or_fifa_rank_missing":
                skipped_elo_or_fifa_rank_missing += 1
            continue

        training_rows.append(training_row)

    training_dataset = pd.DataFrame(training_rows, columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_matches_loaded": len(results),
        "rows_created": len(training_dataset),
        "skipped_team_missing": skipped_team_missing,
        "skipped_elo_or_fifa_rank_missing": skipped_elo_or_fifa_rank_missing,
        "output_path": OUTPUT_PATH,
        "class_distribution": training_dataset["result"].value_counts(),
    }

    return training_dataset, summary


def main() -> None:
    _, summary = build_elo_training_dataset()

    print(f"Total historical matches loaded: {summary['total_matches_loaded']}")
    print(f"Rows created: {summary['rows_created']}")
    print(
        "Matches skipped because team missing: "
        f"{summary['skipped_team_missing']}"
    )
    print(
        "Matches skipped because Elo/FIFA rank missing: "
        f"{summary['skipped_elo_or_fifa_rank_missing']}"
    )
    print(f"Output path: {summary['output_path']}")
    print("Class distribution:")
    print(summary["class_distribution"])


if __name__ == "__main__":
    main()
