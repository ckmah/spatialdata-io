from __future__ import annotations

from pathlib import Path
from typing import Union

import dask.dataframe as dd
import pandas as pd

from spatialdata_io._constants._constants import PyxaKeys

__all__ = ["pyxa"]


def _validate_columns(df: Union[pd.DataFrame, dd.DataFrame], required: set[str], file_name: str) -> None:
    """Raise a clear ``ValueError`` naming the file and any missing required column(s)."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} is missing required column(s): {sorted(missing)}")


def _get_points(path: Path) -> dd.DataFrame:
    ddf = dd.read_csv(path, dtype={PyxaKeys.CELL_ID.value: str})
    _validate_columns(
        ddf,
        {
            PyxaKeys.CELL_ID.value,
            PyxaKeys.GENE.value,
            PyxaKeys.X_UM.value,
            PyxaKeys.Y_UM.value,
            PyxaKeys.Z_UM.value,
        },
        path.name,
    )
    ddf[PyxaKeys.ASSIGNED.value] = ~ddf[PyxaKeys.CELL_ID.value].str.endswith(PyxaKeys.UNASSIGNED_SUFFIX.value)
    return ddf
