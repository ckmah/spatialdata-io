import math
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from click.testing import CliRunner
from spatialdata import get_extent, read_zarr

from spatialdata_io.__main__ import pyxa_wrapper
from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io.readers.pyxa import (
    _get_image,
    _get_points,
    _get_shapes,
    _get_table,
    _validate_columns,
    pyxa,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "pyxa_test"


def _make_tiny_ome_zarr(path: Path) -> None:
    """Build a minimal single-scale OME-NGFF v0.5 store, shape (t=1, c=1, z=2, y=4, x=4)."""
    data = np.arange(2 * 4 * 4, dtype="uint8").reshape(1, 1, 2, 4, 4)
    group = zarr.open_group(store=str(path), mode="w")
    array = group.create_array("scale0/image", shape=data.shape, dtype=data.dtype, dimension_names=["t", "c", "z", "y", "x"])
    array[:] = data
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "scale0/image",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.2, 0.2]},
                            {"type": "translation", "translation": [0.0, 0.0, 1.0, 2.0, 3.0]},
                        ],
                    },
                ],
                "name": "image",
            }
        ],
        "omero": {"channels": [{"label": "DAPI"}]},
    }


def test_pyxa_keys_filenames() -> None:
    assert PyxaKeys.CELL_ASSIGNED_GENE_FILE == "cell_assigned_gene_v1.csv"
    assert PyxaKeys.CELL_BY_GENE_FILE == "cell_by_gene_v1.csv"
    assert PyxaKeys.CELL_METADATA_FILE == "cell_metadata_v1.csv"
    assert PyxaKeys.SEGMENTATION_GEOMETRIES_FILE == "segmentation_geometries_v1.parquet"


def test_pyxa_keys_columns() -> None:
    assert PyxaKeys.CELL_ID == "cell_id"
    assert PyxaKeys.GENE == "Gene"
    assert PyxaKeys.X_UM == "X_um"
    assert PyxaKeys.Y_UM == "Y_um"
    assert PyxaKeys.Z_UM == "Z_um"
    assert PyxaKeys.VOLUME_UM3 == "Volume_um3"
    assert PyxaKeys.ROI == "ROI"
    assert PyxaKeys.Z_INDEX == "ZIndex"
    assert PyxaKeys.BORDER == "Border"
    assert PyxaKeys.FOV == "FOV"
    assert PyxaKeys.UNASSIGNED_SUFFIX == "_-1"
    assert PyxaKeys.REGION_KEY == "region"
    assert PyxaKeys.REGION == "cell_shapes"
    assert PyxaKeys.INSTANCE_KEY == "cell_id"
    assert PyxaKeys.ASSIGNED == "assigned"


def test_validate_columns_passes_when_present() -> None:
    df = pd.DataFrame({"cell_id": [1], "Gene": ["A"]})
    _validate_columns(df, {"cell_id", "Gene"}, "test_file.csv")


def test_validate_columns_raises_when_missing() -> None:
    df = pd.DataFrame({"cell_id": [1]})
    with pytest.raises(ValueError, match=r"test_file\.csv is missing required column\(s\): \['Gene'\]"):
        _validate_columns(df, {"cell_id", "Gene"}, "test_file.csv")


def test_get_points_keeps_unassigned_transcripts() -> None:
    points = _get_points(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    assert isinstance(points, dd.DataFrame)
    computed = points.compute()
    assert "assigned" in computed.columns
    assert (~computed["assigned"]).sum() > 0
    assert computed[~computed["assigned"]]["cell_id"].str.endswith("_-1").all()
    assert computed["assigned"].sum() > 0


def test_get_points_has_required_coordinate_columns() -> None:
    points = _get_points(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    computed = points.compute()
    for col in ("X_um", "Y_um", "Z_um", "Gene", "cell_id"):
        assert col in computed.columns


def test_get_table_matches_raw_values() -> None:
    adata = _get_table(
        FIXTURE_DIR / "cell_by_gene_v1.csv",
        FIXTURE_DIR / "cell_metadata_v1.csv",
    )
    raw_by_gene = pd.read_csv(FIXTURE_DIR / "cell_by_gene_v1.csv", index_col="cell_id")
    raw_metadata = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", index_col="cell_id")

    assert adata.n_obs == len(raw_by_gene)
    sample_cell = raw_by_gene.index[0]
    sample_gene = raw_by_gene.columns[0]
    assert adata[sample_cell, sample_gene].X[0, 0] == raw_by_gene.loc[sample_cell, sample_gene]

    assert list(adata.obsm["spatial"][0]) == list(raw_metadata.loc[sample_cell, ["X_um", "Y_um", "Z_um"]])
    assert (adata.obs["region"] == "cell_shapes").all()


def test_get_shapes_matches_raw_row_count() -> None:
    gdf = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet")
    raw = gpd.read_parquet(FIXTURE_DIR / "segmentation_geometries_v1.parquet")
    assert len(gdf) == len(raw)
    assert all(isinstance(c, str) for c in gdf["cell_id"])
    assert gdf.geometry.is_valid.all()


def test_pyxa_reader_builds_valid_sdata() -> None:
    sdata = pyxa(FIXTURE_DIR)

    assert "transcripts" in sdata.points
    assert "cell_shapes" in sdata.shapes
    assert "rna" in sdata.tables

    raw_transcripts = pd.read_csv(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    extent = get_extent(sdata["transcripts"])
    assert math.floor(extent["x"][0]) <= math.floor(raw_transcripts["X_um"].min())
    assert math.ceil(extent["x"][1]) >= math.ceil(raw_transcripts["X_um"].max())


def test_pyxa_reader_missing_file_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            pyxa(Path(tmpdir))


def test_get_image_loads_full_resolution_level() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "tiny.ome.zarr"
        _make_tiny_ome_zarr(zarr_path)

        image = _get_image(zarr_path)
        assert image.dims == ("c", "z", "y", "x")
        assert image.shape == (1, 2, 4, 4)
        assert list(image.coords["c"].values) == ["DAPI"]
        np.testing.assert_array_equal(image.values, np.arange(2 * 4 * 4, dtype="uint8").reshape(1, 2, 4, 4))


def test_pyxa_reader_includes_image_when_given() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "tiny.ome.zarr"
        _make_tiny_ome_zarr(zarr_path)

        sdata = pyxa(FIXTURE_DIR, image_path=zarr_path)
        assert "mosaic_image" in sdata.images
        assert sdata["mosaic_image"].shape == (1, 2, 4, 4)


def test_pyxa_reader_missing_image_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            pyxa(FIXTURE_DIR, image_path=Path(tmpdir) / "does_not_exist.ome.zarr")


def test_cli_pyxa() -> None:
    runner = CliRunner()
    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            pyxa_wrapper,
            ["--input", str(FIXTURE_DIR), "--output", str(output_zarr)],
        )
        assert result.exit_code == 0, result.output
        sdata = read_zarr(output_zarr)
        assert "transcripts" in sdata.points
