from __future__ import annotations

from pathlib import Path
from typing import Union

import dask.dataframe as dd
import pandas as pd

__all__ = ["pyxa"]


def _validate_columns(df: Union[pd.DataFrame, dd.DataFrame], required: set[str], file_name: str) -> None:
    """Raise a clear ``ValueError`` naming the file and any missing required column(s)."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} is missing required column(s): {sorted(missing)}")
