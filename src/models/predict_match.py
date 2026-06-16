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


def predict_match_from_teams(team_a: str, team_b: str) -> MatchPrediction:
    """
    Loads team stats and predicts a match using team names only.
    """

    team_stats = load_team_stats()

    team_a_row = get_team_row(team_a, team_stats)
    team_b_row = get_team_row(team_b, team_stats)

    team_a_rating = float(team_a_row["elo_rating"])
    team_b_rating = float(team_b_row["elo_rating"])

    return predict_match(
        team_a=team_a,
        team_b=team_b,
        team_a_rating=team_a_rating,
        team_b_rating=team_b_rating,
    )