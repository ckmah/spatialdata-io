import math
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import zarr
from click.testing import CliRunner
from spatialdata import get_extent, read_zarr
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.__main__ import pyxa_wrapper
from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io.readers.pyxa import (
    _get_image,
    _get_points,
    _get_shapes,
    _get_table,
    _get_voxel_size,
    _make_polygonal_valid,
    _validate_columns,
    pyxa,
)

# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml for instructions on
# how to download and place the data on disk
FIXTURE_DIR = Path("./data") / "pyxa_xsmall"
MOSAIC_DIR = FIXTURE_DIR / "mosaic_3d.ome.zarr"


def _make_tiny_ome_zarr(path: Path) -> None:
    """Build a minimal single-scale OME-NGFF v0.5 store, shape (t=1, c=1, z=2, y=4, x=4)."""
    data = np.arange(2 * 4 * 4, dtype="uint8").reshape(1, 1, 2, 4, 4)
    group = zarr.open_group(store=str(path), mode="w")
    array = group.create_array(
        "scale0/image", shape=data.shape, dtype=data.dtype, dimension_names=["t", "c", "z", "y", "x"]
    )
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


def test_pyxa_reader_gene_categories_are_known(caplog: pytest.LogCaptureFixture) -> None:
    points = pyxa(FIXTURE_DIR)["transcripts"]
    # PointsModel.parse warns (and computes them itself) when the feature categories are unknown
    assert "unknown categories" not in caplog.text
    assert points["Gene"].cat.known
    raw = pd.read_csv(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    assert set(points["Gene"].cat.categories) == set(raw["Gene"])


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
    assert adata[sample_cell, sample_gene].to_df().iloc[0, 0] == raw_by_gene.loc[sample_cell, sample_gene]

    assert list(adata.obsm["spatial"][0]) == list(raw_metadata.loc[sample_cell, ["X_um", "Y_um", "Z_um"]])
    assert (adata.obs["region"] == "cell_shapes").all()


def test_get_shapes_matches_raw_row_count() -> None:
    gdf = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size=0.114984751, z_size=0.5)
    raw = gpd.read_parquet(FIXTURE_DIR / "segmentation_geometries_v1.parquet")
    assert len(gdf) == len(raw)
    assert all(isinstance(c, str) for c in gdf["cell_id"])
    assert gdf.geometry.is_valid.all()
    assert set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}


def test_get_shapes_converts_to_um() -> None:
    gdf = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size=0.114984751, z_size=0.5)
    raw = gpd.read_parquet(FIXTURE_DIR / "segmentation_geometries_v1.parquet")
    np.testing.assert_allclose(gdf.total_bounds, raw.total_bounds * 0.114984751)
    np.testing.assert_allclose(gdf["Z_um"], (raw["ZIndex"] + 0.5) * 0.5)


def test_make_polygonal_valid_fixes_self_intersection() -> None:
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    square = shapely.box(0, 0, 1, 1)
    fixed = _make_polygonal_valid(np.array([bowtie, square]))
    assert shapely.is_valid(fixed).all()
    assert set(shapely.get_type_id(fixed)) <= {shapely.GeometryType.POLYGON, shapely.GeometryType.MULTIPOLYGON}
    assert shapely.area(fixed[0]) == pytest.approx(2.0)
    assert fixed[1] is square  # valid geometries are passed through untouched


def test_make_polygonal_valid_drops_non_polygonal_parts() -> None:
    # a polygon with a zero-width spike: make_valid returns the square plus a dangling line
    spiky = shapely.Polygon([(0, 0), (1, 0), (1, 1), (1, 2), (1, 1), (0, 1)])
    (fixed,) = _make_polygonal_valid(np.array([spiky]))
    assert fixed.is_valid
    assert fixed.geom_type in {"Polygon", "MultiPolygon"}
    assert fixed.area == pytest.approx(1.0)


def test_get_voxel_size() -> None:
    xy, z = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    assert xy == pytest.approx(0.114984751)
    assert z == pytest.approx(0.5)


def _write_metadata(path: Path, n: int, xy: float, z: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    px = pd.DataFrame(rng.uniform(1, 1000, size=(n, 3)), columns=["X_pixels", "Y_pixels", "Z_pixels"])
    metadata = px.assign(X_um=px["X_pixels"] * xy, Y_um=px["Y_pixels"] * xy, Z_um=px["Z_pixels"] * z)
    metadata.to_csv(path, index=False)
    return metadata


def test_get_voxel_size_reads_only_a_sample(tmp_path: Path) -> None:
    path = tmp_path / "cell_metadata_v1.csv"
    metadata = _write_metadata(path, n=5_000, xy=0.25, z=1.5)
    # corrupt everything past the fitted sample: the fit must not read these rows
    metadata.loc[2_000:, ["X_um", "Y_um", "Z_um"]] = 1e6
    metadata.to_csv(path, index=False)
    assert _get_voxel_size(path, n_rows=1_000) == pytest.approx((0.25, 1.5))


def test_get_voxel_size_raises_when_not_a_pure_scale(tmp_path: Path) -> None:
    path = tmp_path / "cell_metadata_v1.csv"
    metadata = _write_metadata(path, n=100, xy=0.25, z=1.5)
    metadata["X_um"] += 10.0  # an offset between pixel and um coordinates
    metadata.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not related by a pure scale"):
        _get_voxel_size(path)


def _area_weighted_centroids(shapes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-cell centroid of the polygon stack, weighting each z-plane polygon by its area."""
    centroids = shapes.geometry.centroid
    weighted = (
        pd.DataFrame(
            {
                "cell_id": shapes["cell_id"].to_numpy(),
                "area": shapes.geometry.area.to_numpy(),
                "x": (centroids.x * shapes.geometry.area).to_numpy(),
                "y": (centroids.y * shapes.geometry.area).to_numpy(),
                "z": (shapes["Z_um"] * shapes.geometry.area).to_numpy(),
            }
        )
        .groupby("cell_id")[["area", "x", "y", "z"]]
        .sum()
    )
    return weighted[["x", "y", "z"]].div(weighted["area"], axis=0)


def test_pyxa_reader_shapes_in_um_with_identity_transform() -> None:
    shapes = pyxa(FIXTURE_DIR)["cell_shapes"]
    assert isinstance(get_transformation(shapes, to_coordinate_system="global"), Identity)
    assert shapes.geometry.is_valid.all()


def test_pyxa_reader_shapes_aligned_with_cell_metadata() -> None:
    shapes = pyxa(FIXTURE_DIR)["cell_shapes"]
    metadata = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", index_col="cell_id")
    # the per-cell metadata centroid is exactly the area-weighted centroid of the cell's polygon stack
    centroids = _area_weighted_centroids(shapes)
    expected = metadata.loc[centroids.index, ["X_um", "Y_um", "Z_um"]].to_numpy()
    np.testing.assert_allclose(centroids.to_numpy(), expected, atol=1e-6)


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


def test_pyxa_reader_example_mosaic() -> None:
    sdata = pyxa(FIXTURE_DIR, image_path=MOSAIC_DIR)
    image = sdata["mosaic_image"]
    assert image.dims == ("c", "z", "y", "x")
    assert list(image.coords["c"].values) == ["DAPI"]
    # the mosaic is cropped to the same 100 um cube as the cells
    extent = get_extent(image)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert extent == {"z": (20, 121), "y": (-5140, -5039), "x": (900, 1001)}


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
