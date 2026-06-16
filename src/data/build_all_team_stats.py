"""Build all-team aggregate stats from historical international results."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "international_results_2023_onward.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "all_team_stats.csv"

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

OUTPUT_COLUMNS = [
    "team",
    "recent_form",
    "goals_for_last_10",
    "goals_against_last_10",
    "matches_played",
]


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


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Load historical international match results."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find historical results file at: {path}")

    results = pd.read_csv(path)
    validate_columns(results, RESULT_COLUMNS, "historical results")

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="raise")

    return results


def get_unique_teams(results: pd.DataFrame) -> list[str]:
    """Return every unique team appearing as home or away team."""
    teams = pd.concat([results["home_team"], results["away_team"]])
    return sorted(teams.dropna().astype(str).unique().tolist())


def get_team_matches(team_name: str, results: pd.DataFrame) -> pd.DataFrame:
    """Return all matches involving one team, sorted from newest to oldest."""
    team_matches = results[
        (results["home_team"] == team_name) | (results["away_team"] == team_name)
    ].copy()

    return team_matches.sort_values("date", ascending=False)


def get_points_for_team(team_name: str, match_row: pd.Series) -> int:
    """Calculate points earned by one team in one match."""
    is_home = match_row["home_team"] == team_name

    goals_for = int(match_row["home_score"] if is_home else match_row["away_score"])
    goals_against = int(
        match_row["away_score"] if is_home else match_row["home_score"]
    )

    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


def get_goals_for_team(team_name: str, match_row: pd.Series) -> tuple[int, int]:
    """Return goals for and goals against for one team in one match."""
    is_home = match_row["home_team"] == team_name

    if is_home:
        return int(match_row["home_score"]), int(match_row["away_score"])

    return int(match_row["away_score"]), int(match_row["home_score"])


def build_team_stats_row(team_name: str, results: pd.DataFrame) -> dict[str, Any]:
    """Build one all-team stats row using the team's most recent 10 matches."""
    team_matches = get_team_matches(team_name, results)
    recent_matches = team_matches.head(10)

    total_points = 0
    goals_for = 0
    goals_against = 0

    for _, match_row in recent_matches.iterrows():
        match_goals_for, match_goals_against = get_goals_for_team(team_name, match_row)

        goals_for += match_goals_for
        goals_against += match_goals_against
        total_points += get_points_for_team(team_name, match_row)

    matches_used = len(recent_matches)
    recent_form = total_points / matches_used if matches_used else 0.0

    return {
        "team": team_name,
        "recent_form": recent_form,
        "goals_for_last_10": goals_for,
        "goals_against_last_10": goals_against,
        "matches_played": len(team_matches),
    }


def build_all_team_stats() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and save all-team stats from historical results."""
    results = load_results()
    teams = get_unique_teams(results)

    rows = [build_team_stats_row(team, results) for team in teams]
    all_team_stats = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    all_team_stats = all_team_stats[all_team_stats["matches_played"] >= 10]

    all_team_stats = all_team_stats.sort_values(
        ["recent_form", "goals_for_last_10", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_team_stats.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_unique_teams_found": len(teams),
        "output_rows_created": len(all_team_stats),
        "output_path": OUTPUT_PATH,
        "top_teams": all_team_stats.head(10),
        "bottom_teams": all_team_stats.tail(10).sort_values(
            ["recent_form", "goals_for_last_10", "team"],
            ascending=[True, True, True],
        ),
    }

    return all_team_stats, summary


def print_team_summary(title: str, teams: pd.DataFrame) -> None:
    """Print a compact ranking table for recent form summaries."""
    print(title)

    if teams.empty:
        print("- None")
        return

    for _, row in teams.iterrows():
        print(
            f"- {row['team']}: "
            f"recent_form={row['recent_form']:.2f}, "
            f"GF={row['goals_for_last_10']}, "
            f"GA={row['goals_against_last_10']}, "
            f"matches_played={row['matches_played']}"
        )


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact all-team stats build summary."""
    print("All-Team Stats Summary")
    print(f"Total unique teams found: {summary['total_unique_teams_found']}")
    print(f"Output rows created: {summary['output_rows_created']}")
    print(f"Output path: {summary['output_path']}")
    print_team_summary("Top 10 teams by recent_form:", summary["top_teams"])
    print_team_summary("Bottom 10 teams by recent_form:", summary["bottom_teams"])


def main() -> None:
    """Run the all-team stats builder."""
    try:
        _, summary = build_all_team_stats()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print_summary(summary)


if __name__ == "__main__":
    main()
