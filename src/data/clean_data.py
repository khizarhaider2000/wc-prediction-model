"""Data cleaning routines for match and team datasets."""

import pandas as pd


def clean_match_data(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of match-level data."""
    return matches.copy()
