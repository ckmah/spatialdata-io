import pandas as pd
import pytest

from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io.readers.pyxa import _validate_columns


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
