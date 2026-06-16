"""Predict individual match outcomes."""


from dataclasses import dataclass

from src.data.load_data import get_team_row, load_team_stats


@dataclass
class MatchPrediction:
    team_a: str
    team_b: str

    team_a_win_prob: float
    draw_prob: float
    team_b_win_prob: float

    team_a_expected_goals: float
    team_b_expected_goals: float


def predict_match(
    team_a: str,
    team_b: str,
    team_a_rating: float,
    team_b_rating: float,
) -> MatchPrediction:
    """
    Placeholder match predictor using rating difference.
    Later, this will be replaced by a trained ML model.
    """

    rating_diff = team_a_rating - team_b_rating

    team_a_win_prob = 0.40 + rating_diff * 0.001
    team_b_win_prob = 0.40 - rating_diff * 0.001
    draw_prob = 0.20

    team_a_win_prob = max(0.10, min(0.75, team_a_win_prob))
    team_b_win_prob = max(0.10, min(0.75, team_b_win_prob))

    total = team_a_win_prob + draw_prob + team_b_win_prob
    team_a_win_prob /= total
    draw_prob /= total
    team_b_win_prob /= total

    team_a_expected_goals = 1.25 + rating_diff * 0.003
    team_b_expected_goals = 1.25 - rating_diff * 0.003

    team_a_expected_goals = max(0.30, team_a_expected_goals)
    team_b_expected_goals = max(0.30, team_b_expected_goals)

    return MatchPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_win_prob=team_a_win_prob,
        draw_prob=draw_prob,
        team_b_win_prob=team_b_win_prob,
        team_a_expected_goals=team_a_expected_goals,
        team_b_expected_goals=team_b_expected_goals,
    )

def calculate_strength_diff(team_a_row, team_b_row) -> float:
    """
    Calculates a non-ML strength difference between two teams.

    Positive = advantage Team A
    Negative = advantage Team B

    This is a heuristic baseline, not machine learning.
    """

    elo_diff = float(team_a_row["elo_rating"]) - float(team_b_row["elo_rating"])

    # Lower FIFA rank is better, so reverse the subtraction
    fifa_rank_diff = float(team_b_row["fifa_rank"]) - float(team_a_row["fifa_rank"])

    recent_form_diff = float(team_a_row["recent_form"]) - float(team_b_row["recent_form"])

    goals_for_diff = (
        float(team_a_row["goals_for_last_10"])
        - float(team_b_row["goals_for_last_10"])
    )

    # Fewer goals conceded is better
    goals_against_diff = (
        float(team_b_row["goals_against_last_10"])
        - float(team_a_row["goals_against_last_10"])
    )

    host_advantage_diff = (
        float(team_a_row["host_advantage"])
        - float(team_b_row["host_advantage"])
    )

    strength_diff = (
        elo_diff * 0.0008
        + fifa_rank_diff * 0.0010
        + recent_form_diff * 0.0300
        + goals_for_diff * 0.0045
        + goals_against_diff * 0.0045
        + host_advantage_diff * 0.0500
)

    return strength_diff

def predict_match_from_teams(team_a: str, team_b: str) -> MatchPrediction:
    """
    Loads team stats and predicts a match using team names only.
    """

    team_stats = load_team_stats()

    team_a_row = get_team_row(team_a, team_stats)
    team_b_row = get_team_row(team_b, team_stats)

    strength_diff = calculate_strength_diff(team_a_row, team_b_row)

    return predict_match_from_strength_diff(
        team_a=team_a,
        team_b=team_b,
        strength_diff=strength_diff,
    )




def predict_match_from_strength_diff(
    team_a: str,
    team_b: str,
    strength_diff: float,
) -> MatchPrediction:
    """
    Converts a heuristic strength difference into win/draw/loss probabilities
    and expected goals.

    This is a conservative non-ML baseline. It softens the raw strength_diff
    so favorites do not become too overpowered.
    """

    adjusted_diff = strength_diff * 0.45

    team_a_win_prob = 0.39 + adjusted_diff
    team_b_win_prob = 0.39 - adjusted_diff

    # Draw probability decreases slightly when teams are far apart
    draw_prob = 0.22 - min(abs(adjusted_diff) * 0.06, 0.04)

    team_a_win_prob = max(0.06, min(0.78, team_a_win_prob))
    team_b_win_prob = max(0.06, min(0.78, team_b_win_prob))
    draw_prob = max(0.18, min(0.28, draw_prob))

    total = team_a_win_prob + draw_prob + team_b_win_prob

    team_a_win_prob /= total
    draw_prob /= total
    team_b_win_prob /= total

    team_a_expected_goals = 1.25 + adjusted_diff * 1.4
    team_b_expected_goals = 1.25 - adjusted_diff * 1.4

    team_a_expected_goals = max(0.35, min(3.50, team_a_expected_goals))
    team_b_expected_goals = max(0.35, min(3.50, team_b_expected_goals))

    return MatchPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_win_prob=team_a_win_prob,
        draw_prob=draw_prob,
        team_b_win_prob=team_b_win_prob,
        team_a_expected_goals=team_a_expected_goals,
        team_b_expected_goals=team_b_expected_goals,
    )