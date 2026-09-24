from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anndata as ad
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import zarr
from spatialdata import SpatialData
from spatialdata._logging import logger
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
    # PointsModel needs known feature categories; computing them here is one pass over the gene column
    ddf[PyxaKeys.GENE.value] = ddf[PyxaKeys.GENE.value].astype("category").cat.as_known()
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


def _get_voxel_size(cell_metadata_path: Path, n_rows: int = 10_000) -> tuple[float, float]:
    """Infer the (xy, z) voxel size in um of the segmentation polygons from the per-cell metadata.

    Polygons are stored in pixel coordinates (xy) with a z-plane index (``ZIndex``), while points
    and the table use micrometers. The metadata carries each cell centroid in both units, related
    by a pure scale per axis (no offset), so each size is a least-squares fit through the origin.
    A pure scale is fully determined by a few cells, so only the first ``n_rows`` cells are read;
    the fit is then checked to reproduce every sampled cell, and an error is raised if it does not.
    """
    columns = [
        PyxaKeys.X_UM.value,
        PyxaKeys.Y_UM.value,
        PyxaKeys.Z_UM.value,
        PyxaKeys.X_PIXELS.value,
        PyxaKeys.Y_PIXELS.value,
        PyxaKeys.Z_PIXELS.value,
    ]
    metadata = pd.read_csv(cell_metadata_path, usecols=lambda c: c in columns, nrows=n_rows)
    _validate_columns(metadata, set(columns), cell_metadata_path.name)

    def fit(um_cols: list[str], px_cols: list[str]) -> float:
        um = metadata[um_cols].to_numpy().ravel()
        px = metadata[px_cols].to_numpy().ravel()
        size = float(np.dot(um, px) / np.dot(px, px))
        residual = np.abs(um - size * px).max()
        if residual > 1e-6 * max(np.abs(um).max(), 1.0):
            raise ValueError(
                f"{cell_metadata_path.name}: {um_cols} and {px_cols} are not related by a pure scale "
                f"(max residual {residual:.3g} um with a fitted size of {size:.6g} um/pixel)"
            )
        return size

    xy = fit([PyxaKeys.X_UM.value, PyxaKeys.Y_UM.value], [PyxaKeys.X_PIXELS.value, PyxaKeys.Y_PIXELS.value])
    z = fit([PyxaKeys.Z_UM.value], [PyxaKeys.Z_PIXELS.value])
    return xy, z


def _polygonal_part(geometry: shapely.Geometry) -> shapely.Geometry:
    """Keep only the (multi)polygonal part of a geometry, dropping any lines or points."""
    if isinstance(geometry, shapely.Polygon | shapely.MultiPolygon):
        return geometry
    polygons = [p for p in shapely.get_parts(geometry) if isinstance(p, shapely.Polygon)]
    return shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0] if polygons else shapely.Polygon()


def _make_polygonal_valid(geometries: np.ndarray) -> np.ndarray:
    """Repair invalid geometries in place of dropping them, keeping each one a (Multi)Polygon.

    Valid geometries are returned untouched. Invalid ones are repaired with
    :func:`shapely.make_valid` (``"structure"`` method), which rebuilds the polygon from its
    rings, and any zero-area parts produced by the repair (lines, points) are discarded.
    """
    geometries = geometries.copy()
    invalid = ~shapely.is_valid(geometries)
    if invalid.any():
        repaired = shapely.make_valid(geometries[invalid], method="structure", keep_collapsed=False)
        geometries[invalid] = [_polygonal_part(g) for g in repaired]
    return geometries


def _get_shapes(path: Path, xy_size: float, z_size: float) -> gpd.GeoDataFrame:
    """Read the per-cell, per-z-plane segmentation polygons and convert them to micrometers.

    xy coordinates are scaled from pixels by ``xy_size``. Since shapes are 2D in spatialdata, z is
    stored as a ``Z_um`` column: plane ``k`` spans ``[k, k + 1)`` in ``Z_pixels`` units, so its
    centre sits at ``(k + 0.5) * z_size``. Scaling can turn polygons that touch themselves at a
    single vertex into self-intersecting ones through floating point rounding, so the scaled
    geometries are repaired and then validated.
    """
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
    gdf[PyxaKeys.Z_UM.value] = (gdf[PyxaKeys.Z_INDEX.value] + 0.5) * z_size

    scaled = shapely.transform(gdf.geometry.to_numpy(), lambda coords: coords * xy_size)
    n_invalid = int((~shapely.is_valid(scaled)).sum())
    fixed = _make_polygonal_valid(scaled)
    if n_invalid:
        area_change = np.abs(shapely.area(fixed) - shapely.area(scaled)) / np.maximum(shapely.area(scaled), 1e-12)
        logger.info(
            f"{path.name}: repaired {n_invalid} invalid polygon(s) after scaling to micrometers "
            f"(max relative area change {area_change.max():.2g})"
        )
    gdf = gdf.set_geometry(fixed)

    empty = gdf.geometry.is_empty.to_numpy()
    if empty.any():
        logger.warning(f"{path.name}: dropping {int(empty.sum())} polygon(s) with no area left after repair")
        gdf = gdf[~empty]
    if not gdf.geometry.is_valid.all() or not set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{path.name}: segmentation polygons are still invalid or non-polygonal after repair")

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
    Read *Pyxa* (Stellaromics) output.

    This function reads the following files:

        - ``{px.CELL_ASSIGNED_GENE_FILE!r}``: Transcript-level gene assignments.
        - ``{px.CELL_BY_GENE_FILE!r}``: Per-cell gene expression counts.
        - ``{px.CELL_METADATA_FILE!r}``: Per-cell metadata (volume, spatial coordinates).
        - ``{px.SEGMENTATION_GEOMETRIES_FILE!r}``: Per-cell segmentation polygons.

    No public specification exists for this format at the time of writing; this
    reader is validated against the public demo dataset at
    https://huggingface.co/datasets/Stellaromics/demo.

    All elements are returned in micrometers in the ``global`` coordinate
    system. Segmentation polygons are stored on disk in pixel units, one polygon
    per cell per z-plane (``ZIndex``); the reader converts them to micrometers
    (repairing any polygon that the conversion makes invalid) and adds their z
    as a ``Z_um`` column (the centre of the z-plane), since shapes are 2D in
    spatialdata. Both voxel sizes are inferred from ``{px.CELL_METADATA_FILE!r}``,
    whose per-cell centroids are the area-weighted centroids of each cell's
    polygons in both units. Unassigned
    transcripts (``cell_id`` ending in ``"_-1"``) are kept in the points
    table, flagged via an ``assigned`` column, rather than dropped.

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

    xy_size, z_size = _get_voxel_size(metadata_path)
    shapes = ShapesModel.parse(_get_shapes(geometries_path, xy_size, z_size))

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
