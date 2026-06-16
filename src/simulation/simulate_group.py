"""Simulate World Cup group stage matches."""


import random
from itertools import combinations

import numpy as np

from src.models.predict_match import predict_match_from_teams


def initialize_group_table(teams: list[str]) -> dict:
    """
    Creates an empty standings table for a group.
    """
    table = {}

    for team in teams:
        table[team] = {
            "team": team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }

    return table


def sample_match_result(prediction):
    """
    Randomly selects win/draw/loss based on predicted probabilities.
    """
    outcome = random.choices(
        population=["team_a_win", "draw", "team_b_win"],
        weights=[
            prediction.team_a_win_prob,
            prediction.draw_prob,
            prediction.team_b_win_prob,
        ],
        k=1,
    )[0]

    return outcome


def generate_scoreline(prediction, outcome: str) -> tuple[int, int]:
    """
    Generates a soccer scoreline using a Poisson distribution.

    This is more realistic than Gaussian randomness because soccer goals
    are low-count events.
    """

    max_attempts = 100

    for _ in range(max_attempts):
        team_a_goals = np.random.poisson(prediction.team_a_expected_goals)
        team_b_goals = np.random.poisson(prediction.team_b_expected_goals)

        if outcome == "team_a_win" and team_a_goals > team_b_goals:
            return int(team_a_goals), int(team_b_goals)

        if outcome == "team_b_win" and team_b_goals > team_a_goals:
            return int(team_a_goals), int(team_b_goals)

        if outcome == "draw" and team_a_goals == team_b_goals:
            return int(team_a_goals), int(team_b_goals)

    # Fallback in case the sampled goals do not match the chosen outcome
    if outcome == "team_a_win":
        team_b_goals = np.random.poisson(prediction.team_b_expected_goals)
        team_a_goals = team_b_goals + 1
        return int(team_a_goals), int(team_b_goals)

    if outcome == "team_b_win":
        team_a_goals = np.random.poisson(prediction.team_a_expected_goals)
        team_b_goals = team_a_goals + 1
        return int(team_a_goals), int(team_b_goals)

    draw_goals = round(
        (prediction.team_a_expected_goals + prediction.team_b_expected_goals) / 2
    )
    return int(draw_goals), int(draw_goals)


def update_table(
    table: dict,
    team_a: str,
    team_b: str,
    team_a_goals: int,
    team_b_goals: int,
) -> None:
    """
    Updates group standings after one match.
    """
    table[team_a]["played"] += 1
    table[team_b]["played"] += 1

    table[team_a]["goals_for"] += team_a_goals
    table[team_a]["goals_against"] += team_b_goals

    table[team_b]["goals_for"] += team_b_goals
    table[team_b]["goals_against"] += team_a_goals

    if team_a_goals > team_b_goals:
        table[team_a]["wins"] += 1
        table[team_b]["losses"] += 1
        table[team_a]["points"] += 3

    elif team_b_goals > team_a_goals:
        table[team_b]["wins"] += 1
        table[team_a]["losses"] += 1
        table[team_b]["points"] += 3

    else:
        table[team_a]["draws"] += 1
        table[team_b]["draws"] += 1
        table[team_a]["points"] += 1
        table[team_b]["points"] += 1

    table[team_a]["goal_difference"] = (
        table[team_a]["goals_for"] - table[team_a]["goals_against"]
    )

    table[team_b]["goal_difference"] = (
        table[team_b]["goals_for"] - table[team_b]["goals_against"]
    )


def sort_group_table(table: dict) -> list[dict]:
    """
    Sorts table using common group-stage tiebreakers:
    points, goal difference, goals for.
    """
    return sorted(
        table.values(),
        key=lambda row: (
            row["points"],
            row["goal_difference"],
            row["goals_for"],
        ),
        reverse=True,
    )


def simulate_group(teams: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Simulates every match in a 4-team group.

    Returns:
    - match_results
    - final_table
    """
    table = initialize_group_table(teams)
    match_results = []

    for team_a, team_b in combinations(teams, 2):
        prediction = predict_match_from_teams(team_a, team_b)
        outcome = sample_match_result(prediction)
        team_a_goals, team_b_goals = generate_scoreline(prediction, outcome)

        update_table(
            table=table,
            team_a=team_a,
            team_b=team_b,
            team_a_goals=team_a_goals,
            team_b_goals=team_b_goals,
        )

        match_results.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "team_a_win_prob": prediction.team_a_win_prob,
                "draw_prob": prediction.draw_prob,
                "team_b_win_prob": prediction.team_b_win_prob,
            }
        )

    final_table = sort_group_table(table)

    return match_results, final_table

def simulate_group_many_times(
    teams: list[str],
    num_simulations: int = 10_000,
) -> list[dict]:
    """
    Simulates a group many times and calculates:
    - group winner probability
    - top 2 probability
    - average points
    - average goal difference
    """

    summary = {}

    for team in teams:
        summary[team] = {
            "team": team,
            "group_wins": 0,
            "top_two_finishes": 0,
            "total_points": 0,
            "total_goal_difference": 0,
        }

    for _ in range(num_simulations):
        _, final_table = simulate_group(teams)

        for position, row in enumerate(final_table):
            team = row["team"]

            if position == 0:
                summary[team]["group_wins"] += 1

            if position < 2:
                summary[team]["top_two_finishes"] += 1

            summary[team]["total_points"] += row["points"]
            summary[team]["total_goal_difference"] += row["goal_difference"]

    results = []

    for team, data in summary.items():
        results.append(
            {
                "team": team,
                "win_group_prob": data["group_wins"] / num_simulations,
                "top_two_prob": data["top_two_finishes"] / num_simulations,
                "avg_points": data["total_points"] / num_simulations,
                "avg_goal_difference": data["total_goal_difference"] / num_simulations,
            }
        )

    results = sorted(
        results,
        key=lambda row: (
            row["win_group_prob"],
            row["top_two_prob"],
            row["avg_points"],
        ),
        reverse=True,
    )

    return results