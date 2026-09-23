from __future__ import annotations

from pathlib import Path
from typing import Union

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely

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


def _get_table(cell_by_gene_path: Path, cell_metadata_path: Path) -> ad.AnnData:
    by_gene = pd.read_csv(cell_by_gene_path, index_col=PyxaKeys.CELL_ID.value, dtype={PyxaKeys.CELL_ID.value: str})
    metadata = pd.read_csv(cell_metadata_path, index_col=PyxaKeys.CELL_ID.value, dtype={PyxaKeys.CELL_ID.value: str})

    _validate_columns(
        metadata,
        {PyxaKeys.X_UM.value, PyxaKeys.Y_UM.value, PyxaKeys.Z_UM.value},
        cell_metadata_path.name,
    )

    metadata = metadata.loc[by_gene.index]
    spatial_cols = [PyxaKeys.X_UM.value, PyxaKeys.Y_UM.value, PyxaKeys.Z_UM.value]
    adata = ad.AnnData(by_gene, obs=metadata.drop(columns=spatial_cols))
    adata.obsm["spatial"] = metadata[spatial_cols].values
    adata.obs[PyxaKeys.REGION_KEY.value] = pd.Series(PyxaKeys.REGION.value, index=adata.obs_names, dtype="category")
    adata.obs[PyxaKeys.CELL_ID.value] = adata.obs_names
    return adata


def _get_shapes(path: Path) -> gpd.GeoDataFrame:
    parquet_file = pq.ParquetFile(path)
    chunks = []
    for batch in parquet_file.iter_batches():
        chunk = batch.to_pandas()
        chunk["geometry"] = shapely.from_wkb(chunk["geometry"])
        chunks.append(gpd.GeoDataFrame(chunk, geometry="geometry"))
    gdf = pd.concat(chunks, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    _validate_columns(gdf, {PyxaKeys.CELL_ID.value, PyxaKeys.Z_INDEX.value}, path.name)

    gdf[PyxaKeys.CELL_ID.value] = gdf[PyxaKeys.CELL_ID.value].astype(str)
    gdf = gdf[gdf.geometry.is_valid]
    gdf.index = gdf[PyxaKeys.CELL_ID.value]
    return gdf
