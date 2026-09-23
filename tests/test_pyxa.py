from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest

from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io.readers.pyxa import _get_points, _validate_columns

FIXTURE_DIR = Path(__file__).parent / "data" / "pyxa_test"


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
