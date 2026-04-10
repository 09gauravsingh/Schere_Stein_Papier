from __future__ import annotations

import pandas as pd


def slice_by_band(df: pd.DataFrame, value_col: str, bins: list[float], labels: list[str], metric_col: str) -> pd.DataFrame:
    """
    Create slice metrics by value bands (e.g., ambient temp bands).
    """
    out = df.copy()
    out["band"] = pd.cut(out[value_col], bins=bins, labels=labels, include_lowest=True)
    return out.groupby("band")[metric_col].agg(["count", "mean", "median"]).reset_index()
