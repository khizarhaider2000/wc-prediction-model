from src.data.load_data import load_team_stats, get_teams_by_group
from src.simulation.simulate_group import simulate_group


def simulate_full_group_stage_once() -> dict:
    """
    Simulates all World Cup groups once.

    Returns:
    - group_winners
    - runners_up
    - third_place_teams
    - qualified_teams
    - group_tables
    """

    team_stats = load_team_stats()
    groups = sorted(team_stats["group"].unique())

    group_winners = []
    runners_up = []
    third_place_teams = []
    qualified_teams = []
    group_tables = {}

    for group_name in groups:
        teams = get_teams_by_group(group_name, team_stats)

        _, final_table = simulate_group(teams)

        group_tables[group_name] = final_table

        winner = final_table[0]
        runner_up = final_table[1]
        third_place = final_table[2]

        group_winners.append(winner["team"])
        runners_up.append(runner_up["team"])

        qualified_teams.append(winner["team"])
        qualified_teams.append(runner_up["team"])

        third_place_teams.append(
            {
                "group": group_name,
                "team": third_place["team"],
                "points": third_place["points"],
                "goal_difference": third_place["goal_difference"],
                "goals_for": third_place["goals_for"],
            }
        )

    best_third_place = sorted(
        third_place_teams,
        key=lambda row: (
            row["points"],
            row["goal_difference"],
            row["goals_for"],
        ),
        reverse=True,
    )[:8]

    for row in best_third_place:
        qualified_teams.append(row["team"])

    return {
        "group_winners": group_winners,
        "runners_up": runners_up,
        "third_place_teams": third_place_teams,
        "best_third_place": best_third_place,
        "qualified_teams": qualified_teams,
        "group_tables": group_tables,
    }


def simulate_group_stage_many_times(num_simulations: int = 1_000) -> list[dict]:
    """
    Simulates the full group stage many times and calculates:
    - win group probability
    - top 2 probability
    - third-place qualification probability
    - total qualification probability
    """

    team_stats = load_team_stats()

    summary = {}

    for _, row in team_stats.iterrows():
        team = row["team"]
        summary[team] = {
            "team": team,
            "group": row["group"],
            "group_wins": 0,
            "top_two_finishes": 0,
            "third_place_qualifications": 0,
            "total_qualifications": 0,
        }

    for _ in range(num_simulations):
        simulation = simulate_full_group_stage_once()

        group_winners = set(simulation["group_winners"])
        runners_up = set(simulation["runners_up"])
        best_third_place = {row["team"] for row in simulation["best_third_place"]}
        qualified_teams = set(simulation["qualified_teams"])

        for team in summary:
            if team in group_winners:
                summary[team]["group_wins"] += 1

            if team in group_winners or team in runners_up:
                summary[team]["top_two_finishes"] += 1

            if team in best_third_place:
                summary[team]["third_place_qualifications"] += 1

            if team in qualified_teams:
                summary[team]["total_qualifications"] += 1

    results = []

    for team, data in summary.items():
        results.append(
            {
                "group": data["group"],
                "team": team,
                "win_group_prob": data["group_wins"] / num_simulations,
                "top_two_prob": data["top_two_finishes"] / num_simulations,
                "third_place_qualify_prob": data["third_place_qualifications"] / num_simulations,
                "total_qualify_prob": data["total_qualifications"] / num_simulations,
            }
        )

    results = sorted(
        results,
        key=lambda row: (
            row["group"],
            row["total_qualify_prob"],
            row["win_group_prob"],
        ),
        reverse=False,
    )

    return results

def get_round_of_32_teams_once() -> list[str]:
    """
    Simulates one full group stage and returns the 32 qualified teams.

    Qualification format:
    - 12 group winners
    - 12 runners-up
    - 8 best third-place teams
    """

    simulation = simulate_full_group_stage_once()

    qualified_teams = simulation["qualified_teams"]

    return qualified_teams