"""Feature construction for match prediction models."""

import pandas as pd


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features from cleaned match data."""
    return matches.copy()
