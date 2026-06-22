"""Build combined team stats for ML training."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALL_TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "all_team_stats.csv"
WORLD_CUP_TEAM_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "team_stats.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "combined_team_stats.csv"

ALL_TEAM_COLUMNS = [
    "team",
    "recent_form",
    "goals_for_last_10",
    "goals_against_last_10",
    "matches_played",
]

WORLD_CUP_COLUMNS = [
    "team",
    "elo_rating",
    "fifa_rank",
    "host_advantage",
]

OUTPUT_COLUMNS = [
    "team",
    "recent_form",
    "goals_for_last_10",
    "goals_against_last_10",
    "matches_played",
    "elo_rating",
    "fifa_rank",
    "host_advantage",
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


def main() -> None:
    all_team_stats = pd.read_csv(ALL_TEAM_STATS_PATH)
    world_cup_team_stats = pd.read_csv(WORLD_CUP_TEAM_STATS_PATH)

    validate_columns(all_team_stats, ALL_TEAM_COLUMNS, "all_team_stats")
    validate_columns(world_cup_team_stats, WORLD_CUP_COLUMNS, "team_stats")

    all_team_stats = all_team_stats[ALL_TEAM_COLUMNS].copy()
    world_cup_team_stats = world_cup_team_stats[WORLD_CUP_COLUMNS].copy()

    all_team_stats["team_key"] = all_team_stats["team"].astype(str).str.lower()
    world_cup_team_stats["team_key"] = (
        world_cup_team_stats["team"].astype(str).str.lower()
    )

    combined_team_stats = all_team_stats.merge(
        world_cup_team_stats[["team_key", "elo_rating", "fifa_rank", "host_advantage"]],
        on="team_key",
        how="left",
    )

    combined_team_stats["host_advantage"] = (
        combined_team_stats["host_advantage"].fillna(0)
    )
    combined_team_stats = combined_team_stats[OUTPUT_COLUMNS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_team_stats.to_csv(OUTPUT_PATH, index=False)

    print(f"Total teams in all_team_stats: {len(all_team_stats)}")
    print(
        "Teams that received elo_rating: "
        f"{combined_team_stats['elo_rating'].notna().sum()}"
    )
    print(
        "Teams that received fifa_rank: "
        f"{combined_team_stats['fifa_rank'].notna().sum()}"
    )
    print(
        "Teams missing elo_rating: "
        f"{combined_team_stats['elo_rating'].isna().sum()}"
    )
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
