"""Build a historical match training dataset for future model training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "international_results_2023_onward.csv"
TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "all_team_stats.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "match_training_dataset.csv"

RESULT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

TEAM_STATS_COLUMNS = [
    "team",
    "recent_form",
    "goals_for_last_10",
    "goals_against_last_10",
    "matches_played",
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
    "strength_gap",
    "host_advantage_diff",
    "result",
    "team_a_win",
    "draw",
    "team_b_win",
]


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Load historical international match results."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find historical results file at: {path}")

    results = pd.read_csv(path)
    validate_columns(results, RESULT_COLUMNS, "historical results")
    return results


def load_team_stats(path: Path = TEAM_STATS_PATH) -> pd.DataFrame:
    """Load processed team stats used for feature lookup."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find team stats file at: {path}")

    team_stats = pd.read_csv(path)
    validate_columns(team_stats, TEAM_STATS_COLUMNS, "team stats")
    return team_stats


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when an input dataset is missing required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing_columns}"
        )


def get_team_features(
    team_name: str,
    team_stats_df: pd.DataFrame,
) -> pd.Series | None:
    """Return one team's feature row, or None when the team is not available."""
    matches = team_stats_df[
        team_stats_df["team"].astype(str).str.lower() == str(team_name).lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def get_result_label(team_a_goals: int, team_b_goals: int) -> str:
    """Create the categorical match result label from the goals scored."""
    if team_a_goals > team_b_goals:
        return "team_a_win"

    if team_a_goals == team_b_goals:
        return "draw"

    return "team_b_win"


def build_match_row(
    match_row: pd.Series,
    team_stats_df: pd.DataFrame,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Build one training row from one historical match.

    Returns:
    - the output row, or None when either team is missing
    - missing team names encountered for warning summaries
    """
    team_a = str(match_row["home_team"])
    team_b = str(match_row["away_team"])

    team_a_features = get_team_features(team_a, team_stats_df)
    team_b_features = get_team_features(team_b, team_stats_df)

    missing_teams = []

    if team_a_features is None:
        missing_teams.append(team_a)

    if team_b_features is None:
        missing_teams.append(team_b)

    if missing_teams:
        return None, missing_teams

    team_a_goals = int(match_row["home_score"])
    team_b_goals = int(match_row["away_score"])
    result = get_result_label(team_a_goals, team_b_goals)

    recent_form_diff = float(team_a_features["recent_form"]) - float(
        team_b_features["recent_form"]
    )
    goals_for_last_10_diff = float(team_a_features["goals_for_last_10"]) - float(
        team_b_features["goals_for_last_10"]
    )
    goals_against_last_10_diff = float(
        team_b_features["goals_against_last_10"]
    ) - float(team_a_features["goals_against_last_10"])
    strength_gap = (
        recent_form_diff
        + goals_for_last_10_diff * 0.1
        + goals_against_last_10_diff * 0.1
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
            "elo_diff": 0.0,
            "fifa_rank_diff": 0.0,
            "recent_form_diff": recent_form_diff,
            "goals_for_last_10_diff": goals_for_last_10_diff,
            "goals_against_last_10_diff": goals_against_last_10_diff,
            "strength_gap": strength_gap,
            "host_advantage_diff": 0.0,
            "result": result,
            "team_a_win": int(result == "team_a_win"),
            "draw": int(result == "draw"),
            "team_b_win": int(result == "team_b_win"),
        },
        [],
    )


def build_training_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and save the historical match training dataset."""
    results = load_results()
    team_stats = load_team_stats()

    training_rows = []
    missing_team_counts: Counter[str] = Counter()
    skipped_matches = 0

    for _, match_row in results.iterrows():
        training_row, missing_teams = build_match_row(match_row, team_stats)

        if training_row is None:
            skipped_matches += 1
            missing_team_counts.update(missing_teams)
            continue

        training_rows.append(training_row)

    training_dataset = pd.DataFrame(training_rows, columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_matches_loaded": len(results),
        "training_rows_created": len(training_dataset),
        "matches_skipped": skipped_matches,
        "missing_team_counts": missing_team_counts,
        "output_path": OUTPUT_PATH,
    }

    return training_dataset, summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact dataset build summary."""
    print("Historical Match Training Dataset Summary")
    print(f"Total historical matches loaded: {summary['total_matches_loaded']}")
    print(f"Training rows created: {summary['training_rows_created']}")
    print(f"Matches skipped: {summary['matches_skipped']}")
    print("Top 20 missing team names:")

    missing_team_counts = summary["missing_team_counts"]

    if missing_team_counts:
        for team_name, count in missing_team_counts.most_common(20):
            print(f"- {team_name}: {count}")
    else:
        print("- None")

    print(f"Output path: {summary['output_path']}")


def main() -> None:
    """Run the dataset builder."""
    try:
        _, summary = build_training_dataset()
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        print(
            "Add the required raw CSV with columns: "
            f"{', '.join(RESULT_COLUMNS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print_summary(summary)


if __name__ == "__main__":
    main()
