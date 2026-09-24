from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anndata as ad
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely
import zarr
from spatialdata import SpatialData
from spatialdata.models import Image3DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Scale, Sequence, Translation
from xarray import DataArray

from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io._docs import inject_docs

__all__ = ["pyxa"]


def _validate_columns(df: pd.DataFrame | dd.DataFrame, required: set[str], file_name: str) -> None:
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


def _get_image(path: Path) -> DataArray:
    """Load the full-resolution level of an OME-Zarr (OME-NGFF v0.5) mosaic image.

    Only the highest-resolution dataset (index 0 in the multiscale metadata,
    conventionally ``scale0``) is read; the reader does not (yet) reuse the
    precomputed lower-resolution pyramid levels also present in the store.
    """
    group = zarr.open_group(store=str(path), mode="r")
    ome = cast("dict[str, Any]", group.attrs.asdict()["ome"])
    multiscale = ome["multiscales"][0]
    dataset0 = multiscale["datasets"][0]

    axes = tuple(a["name"] for a in multiscale["axes"])
    array = da.from_zarr(str(path), component=dataset0["path"])

    # drop the singleton "t" axis, which spatialdata's image models don't model
    t_index = axes.index("t")
    array = da.squeeze(array, axis=t_index)
    axes = tuple(a for a in axes if a != "t")

    coordinate_transformations = {ct["type"]: ct for ct in dataset0["coordinateTransformations"]}
    scale_values = [
        v
        for v, a in zip(coordinate_transformations["scale"]["scale"], multiscale["axes"], strict=True)
        if a["name"] != "t"
    ]
    translation_values = [
        v
        for v, a in zip(coordinate_transformations["translation"]["translation"], multiscale["axes"], strict=True)
        if a["name"] != "t"
    ]
    transformation = Sequence(
        [
            Scale(scale_values, axes=axes),
            Translation(translation_values, axes=axes),
        ]
    )

    channel_labels = [c.get("label") for c in ome.get("omero", {}).get("channels", [])]
    c_coords = channel_labels if len(channel_labels) == array.shape[axes.index("c")] else None

    return Image3DModel.parse(array, dims=axes, c_coords=c_coords, transformations={"global": transformation})


@inject_docs(px=PyxaKeys)
def pyxa(path: str | Path, dataset_id: str = "pyxa", image_path: str | Path | None = None) -> SpatialData:
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
    image_path
        Optional path to a mosaic OME-Zarr (OME-NGFF v0.5) directory, e.g. a
        DAPI mosaic. Not colocated with the other 4 files in Pyxa's output
        layout, so it must be given explicitly. Only the full-resolution
        level is read (see :func:`_get_image`). If ``None``, no image is
        included in the returned :class:`~spatialdata.SpatialData`.

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

    images = {}
    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Expected Pyxa mosaic image not found: {image_path}")
        images[PyxaKeys.MOSAIC_IMAGE.value] = _get_image(image_path)

    return SpatialData(
        points={"transcripts": points},
        shapes={PyxaKeys.REGION.value: shapes},
        tables={"rna": table},
        images=images,
    )
