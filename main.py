import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.predict_match import predict_match_from_teams
from src.simulation.simulate_group_stage import simulate_full_group_stage_once
from src.simulation.simulate_tournament import simulate_tournament_many_times


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "tournament_probabilities.csv"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_match_prediction(team_a: str, team_b: str) -> None:
    prediction = predict_match_from_teams(team_a, team_b)

    print(f"\n{prediction.team_a} vs {prediction.team_b}\n")

    print(f"{prediction.team_a} win: {format_percent(prediction.team_a_win_prob)}")
    print(f"Draw: {format_percent(prediction.draw_prob)}")
    print(f"{prediction.team_b} win: {format_percent(prediction.team_b_win_prob)}")

    print("\nExpected goals:")
    print(f"{prediction.team_a}: {prediction.team_a_expected_goals:.2f}")
    print(f"{prediction.team_b}: {prediction.team_b_expected_goals:.2f}")


def print_debug_bracket() -> None:
    from src.simulation.bracket_mapping import create_round_of_32_matchups

    group_stage = simulate_full_group_stage_once()
    matchups = create_round_of_32_matchups(group_stage)

    print("\nRound of 32 Matchups\n")

    for team_a, team_b in matchups:
        print(f"{team_a} vs {team_b}")


def print_tournament_summary(results: list[dict]) -> None:
    print("\nTournament Simulation Summary\n")

    print(
        f"{'Team':<25} "
        f"{'Qualify':<10} "
        f"{'R16':<10} "
        f"{'QF':<10} "
        f"{'SF':<10} "
        f"{'Final':<10} "
        f"{'Champion':<10}"
    )

    for row in results:
        print(
            f"{row['team']:<25} "
            f"{format_percent(row['qualify_prob']):<10} "
            f"{format_percent(row['round_of_16_prob']):<10} "
            f"{format_percent(row['quarterfinal_prob']):<10} "
            f"{format_percent(row['semifinal_prob']):<10} "
            f"{format_percent(row['final_prob']):<10} "
            f"{format_percent(row['champion_prob']):<10}"
        )


def run_simulation(num_simulations: int) -> None:
    results = simulate_tournament_many_times(num_simulations=num_simulations)

    print_tournament_summary(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved tournament probabilities to: {OUTPUT_PATH}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["match", "debug", "simulate"],
        default="simulate",
    )

    parser.add_argument("--team-a", type=str)
    parser.add_argument("--team-b", type=str)

    parser.add_argument("--num-simulations", type=int, default=1000)

    return parser.parse_args()


def main():
    random.seed(42)
    np.random.seed(42)

    args = parse_args()

    if args.mode == "match":
        if not args.team_a or not args.team_b:
            raise ValueError("--team-a and --team-b are required for match mode.")

        print_match_prediction(args.team_a, args.team_b)
        return

    if args.mode == "debug":
        print_debug_bracket()
        return

    if args.mode == "simulate":
        run_simulation(args.num_simulations)
        return


if __name__ == "__main__":
    main()