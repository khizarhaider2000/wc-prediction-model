import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.simulate_tournament import simulate_tournament_many_times
from src.simulation.bracket_mapping import debug_print_round_of_32_bracket


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "tournament_probabilities.csv"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def main():
    random.seed(42)
    np.random.seed(42)

    debug_print_round_of_32_bracket()

    results = simulate_tournament_many_times(
        num_simulations=1_000,
    )

    print_tournament_summary(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved tournament probabilities to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
