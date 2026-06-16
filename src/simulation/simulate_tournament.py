from src.data.load_data import load_team_stats
from src.simulation.bracket_mapping import create_round_of_32_matchups
from src.simulation.simulate_group_stage import simulate_full_group_stage_once
from src.simulation.simulate_knockout import simulate_knockout_bracket_from_matchups


def simulate_tournament_once() -> dict:
    """
    Simulates one full tournament:
    - group stage
    - Round of 32
    - Round of 16
    - quarterfinals
    - semifinals
    - final
    """

    group_stage = simulate_full_group_stage_once()
    round_of_32_matchups = create_round_of_32_matchups(group_stage)

    bracket = simulate_knockout_bracket_from_matchups(round_of_32_matchups)

    return {
        "group_stage": group_stage,
        "bracket": bracket,
        "champion": bracket["champion"],
    }


def simulate_tournament_many_times(num_simulations: int = 1_000) -> list[dict]:
    """
    Simulates the full tournament many times and calculates:
    - qualify probability
    - Round of 16 probability
    - quarterfinal probability
    - semifinal probability
    - final probability
    - champion probability
    """

    team_stats = load_team_stats()

    summary = {}

    for _, row in team_stats.iterrows():
        team = row["team"]
        summary[team] = {
            "team": team,
            "group": row["group"],
            "qualifications": 0,
            "round_of_16": 0,
            "quarterfinals": 0,
            "semifinals": 0,
            "finals": 0,
            "championships": 0,
        }

    for _ in range(num_simulations):
        tournament = simulate_tournament_once()

        group_stage = tournament["group_stage"]
        bracket = tournament["bracket"]
        champion = tournament["champion"]

        qualified_teams = set(group_stage["qualified_teams"])
        round_of_16 = set(bracket["round_of_16"])
        quarterfinalists = set(bracket["quarterfinalists"])
        semifinalists = set(bracket["semifinalists"])
        finalists = set(bracket["finalists"])

        for team in summary:
            if team in qualified_teams:
                summary[team]["qualifications"] += 1

            if team in round_of_16:
                summary[team]["round_of_16"] += 1

            if team in quarterfinalists:
                summary[team]["quarterfinals"] += 1

            if team in semifinalists:
                summary[team]["semifinals"] += 1

            if team in finalists:
                summary[team]["finals"] += 1

            if team == champion:
                summary[team]["championships"] += 1

    results = []

    for team, data in summary.items():
        results.append(
            {
                "group": data["group"],
                "team": team,
                "qualify_prob": data["qualifications"] / num_simulations,
                "round_of_16_prob": data["round_of_16"] / num_simulations,
                "quarterfinal_prob": data["quarterfinals"] / num_simulations,
                "semifinal_prob": data["semifinals"] / num_simulations,
                "final_prob": data["finals"] / num_simulations,
                "champion_prob": data["championships"] / num_simulations,
            }
        )

    results = sorted(
        results,
        key=lambda row: row["champion_prob"],
        reverse=True,
    )

    return results
