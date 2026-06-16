"""Utilities for loading raw and processed datasets."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "team_stats.csv"


def load_team_stats() -> pd.DataFrame:
    """
    Loads processed team-level stats.
    """
    if not TEAM_STATS_PATH.exists():
        raise FileNotFoundError(f"Could not find team stats file at: {TEAM_STATS_PATH}")

    df = pd.read_csv(TEAM_STATS_PATH)

    required_columns = [
        "team",
        "elo_rating",
        "fifa_rank",
        "recent_form",
        "goals_for_last_10",
        "goals_against_last_10",
        "host_advantage",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def get_team_row(team_name: str, team_stats: pd.DataFrame) -> pd.Series:
    """
    Gets one team's stats from the team stats dataframe.
    """
    matches = team_stats[team_stats["team"].str.lower() == team_name.lower()]

    if matches.empty:
        available_teams = ", ".join(sorted(team_stats["team"].tolist()))
        raise ValueError(
            f"Team '{team_name}' not found. Available teams: {available_teams}"
        )

    return matches.iloc[0]

def get_teams_by_group(group_name: str, team_stats: pd.DataFrame | None = None) -> list[str]:
    """
    Returns all teams in a given group using the group column from team_stats.csv.
    Example: get_teams_by_group("C") -> ["Brazil", "Morocco", "Haiti", "Scotland"]
    """

    if team_stats is None:
        team_stats = load_team_stats()

    if "group" not in team_stats.columns:
        raise ValueError("team_stats.csv must contain a 'group' column.")

    group_teams = team_stats[team_stats["group"].str.upper() == group_name.upper()]

    if group_teams.empty:
        available_groups = ", ".join(sorted(team_stats["group"].unique()))
        raise ValueError(
            f"Group '{group_name}' not found. Available groups: {available_groups}"
        )

    return group_teams["team"].tolist()