from __future__ import annotations

from pathlib import Path
from typing import Union

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel

from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io._docs import inject_docs

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


@inject_docs(px=PyxaKeys)
def pyxa(path: str | Path, dataset_id: str = "pyxa") -> SpatialData:
    """
    Read *Pyxa* (Stellaromics/Meteor-APA pipeline) analysis-group output.

    This function reads the following files:

        - ``{px.CELL_ASSIGNED_GENE_FILE!r}``: Transcript-level gene assignments.
        - ``{px.CELL_BY_GENE_FILE!r}``: Per-cell gene expression counts.
        - ``{px.CELL_METADATA_FILE!r}``: Per-cell metadata (volume, spatial coordinates).
        - ``{px.SEGMENTATION_GEOMETRIES_FILE!r}``: Per-cell segmentation polygons.

    Only analysis-group (AG) level output is supported. No public specification
    exists for this format at the time of writing; this reader is derived from
    internal documentation and validated against real analysis-group output.
    Unassigned transcripts (``cell_id`` ending in ``"_-1"``) are kept in the
    points table, flagged via an ``assigned`` column, rather than dropped.

    Parameters
    ----------
    path
        Path to the directory containing the 4 Pyxa output files.
    dataset_id
        Dataset identifier, currently unused for element naming (reserved for
        future multi-sample support).

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    assigned_gene_path = path / PyxaKeys.CELL_ASSIGNED_GENE_FILE.value
    by_gene_path = path / PyxaKeys.CELL_BY_GENE_FILE.value
    metadata_path = path / PyxaKeys.CELL_METADATA_FILE.value
    geometries_path = path / PyxaKeys.SEGMENTATION_GEOMETRIES_FILE.value

    for p in (assigned_gene_path, by_gene_path, metadata_path, geometries_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected Pyxa output file not found: {p}")

    points = PointsModel.parse(
        _get_points(assigned_gene_path),
        coordinates={"x": PyxaKeys.X_UM.value, "y": PyxaKeys.Y_UM.value, "z": PyxaKeys.Z_UM.value},
        feature_key=PyxaKeys.GENE.value,
        instance_key=PyxaKeys.CELL_ID.value,
    )

    shapes = ShapesModel.parse(_get_shapes(geometries_path))

    table = TableModel.parse(
        _get_table(by_gene_path, metadata_path),
        region=PyxaKeys.REGION.value,
        region_key=PyxaKeys.REGION_KEY.value,
        instance_key=PyxaKeys.INSTANCE_KEY.value,
    )

    return SpatialData(
        points={"transcripts": points},
        shapes={PyxaKeys.REGION.value: shapes},
        tables={"rna": table},
    )
