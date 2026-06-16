import random

from src.models.predict_match import predict_match_from_teams


def simulate_knockout_match(team_a: str, team_b: str) -> str:
    """
    Simulates a knockout match and returns the winner.
    """

    prediction = predict_match_from_teams(team_a, team_b)

    outcome = random.choices(
        population=["team_a_win", "draw", "team_b_win"],
        weights=[
            prediction.team_a_win_prob,
            prediction.draw_prob,
            prediction.team_b_win_prob,
        ],
        k=1,
    )[0]

    if outcome == "team_a_win":
        return team_a

    if outcome == "team_b_win":
        return team_b

    total_strength = prediction.team_a_win_prob + prediction.team_b_win_prob
    team_a_advancement_prob = prediction.team_a_win_prob / total_strength

    winner = random.choices(
        population=[team_a, team_b],
        weights=[team_a_advancement_prob, 1 - team_a_advancement_prob],
        k=1,
    )[0]

    return winner


def simulate_knockout_round(teams: list[str]) -> list[str]:
    """
    Simulates one knockout round.
    """

    if len(teams) % 2 != 0:
        raise ValueError("Knockout round must have an even number of teams.")

    winners = []

    for i in range(0, len(teams), 2):
        team_a = teams[i]
        team_b = teams[i + 1]

        winner = simulate_knockout_match(team_a, team_b)
        winners.append(winner)

    return winners


def simulate_knockout_bracket(round_of_32_teams: list[str]) -> dict:
    """
    Simulates the full knockout bracket from Round of 32 to champion.
    """

    if len(round_of_32_teams) != 32:
        raise ValueError(
            f"Round of 32 must have exactly 32 teams. Got {len(round_of_32_teams)}."
        )

    round_of_16 = simulate_knockout_round(round_of_32_teams)
    quarterfinalists = simulate_knockout_round(round_of_16)
    semifinalists = simulate_knockout_round(quarterfinalists)
    finalists = simulate_knockout_round(semifinalists)
    champion = simulate_knockout_round(finalists)[0]

    return {
        "round_of_32": round_of_32_teams,
        "round_of_16": round_of_16,
        "quarterfinalists": quarterfinalists,
        "semifinalists": semifinalists,
        "finalists": finalists,
        "champion": champion,
    }


def simulate_knockout_bracket_from_matchups(
    round_of_32_matchups: list[tuple[str, str]],
) -> dict:
    """
    Simulates the full knockout bracket from explicit Round of 32 matchups.
    """

    if len(round_of_32_matchups) != 16:
        raise ValueError(
            "Round of 32 must have exactly 16 matchups. "
            f"Got {len(round_of_32_matchups)}."
        )

    round_of_32 = []
    round_of_16 = []

    for team_a, team_b in round_of_32_matchups:
        round_of_32.extend([team_a, team_b])
        round_of_16.append(simulate_knockout_match(team_a, team_b))

    quarterfinalists = simulate_knockout_round(round_of_16)
    semifinalists = simulate_knockout_round(quarterfinalists)
    finalists = simulate_knockout_round(semifinalists)
    champion = simulate_knockout_round(finalists)[0]

    return {
        "round_of_32_matchups": round_of_32_matchups,
        "round_of_32": round_of_32,
        "round_of_16": round_of_16,
        "quarterfinalists": quarterfinalists,
        "semifinalists": semifinalists,
        "finalists": finalists,
        "champion": champion,
    }
