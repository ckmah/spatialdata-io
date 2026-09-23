"""Generate the small Pyxa test fixture checked in under tests/data/pyxa_test/.

The source data is internal Stellaromics/Meteor-APA analysis-group output and
is not publicly hosted, so (unlike most spatialdata-io readers) there is no
download.py -- this script instead subsamples a local copy down to the
100kB-10MB range required by the contribution guidelines, and its output is
what gets committed.

Usage:
    python scripts/generate_pyxa_test_data.py <path-to-local-ag_output-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

N_CELLS = 150
N_UNASSIGNED_TRANSCRIPTS = 500

CELL_ASSIGNED_GENE_FILE = "cell_assigned_gene_v1.csv"
CELL_BY_GENE_FILE = "cell_by_gene_v1.csv"
CELL_METADATA_FILE = "cell_metadata_v1.csv"
SEGMENTATION_GEOMETRIES_FILE = "segmentation_geometries_v1.parquet"


def main(source_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(source_dir / CELL_METADATA_FILE)
    subset_ids = metadata["cell_id"].iloc[:N_CELLS].tolist()
    metadata_subset = metadata[metadata["cell_id"].isin(subset_ids)]
    metadata_subset.to_csv(out_dir / CELL_METADATA_FILE, index=False)

    by_gene = pd.read_csv(source_dir / CELL_BY_GENE_FILE)
    by_gene_subset = by_gene[by_gene["cell_id"].isin(subset_ids)]
    by_gene_subset.to_csv(out_dir / CELL_BY_GENE_FILE, index=False)

    assigned_gene = pd.read_csv(source_dir / CELL_ASSIGNED_GENE_FILE)
    assigned_subset = assigned_gene[assigned_gene["cell_id"].isin(subset_ids)]
    unassigned = assigned_gene[assigned_gene["cell_id"].str.endswith("_-1")]
    unassigned_subset = unassigned.sample(n=min(N_UNASSIGNED_TRANSCRIPTS, len(unassigned)), random_state=0)
    pd.concat([assigned_subset, unassigned_subset]).to_csv(out_dir / CELL_ASSIGNED_GENE_FILE, index=False)

    table = pq.read_table(
        source_dir / SEGMENTATION_GEOMETRIES_FILE,
        filters=[("cell_id", "in", subset_ids)],
    )
    pq.write_table(table, out_dir / SEGMENTATION_GEOMETRIES_FILE)

    print(f"Wrote fixture to {out_dir}")
    for f in out_dir.iterdir():
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-local-ag_output-dir>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(__file__).parent.parent / "tests" / "data" / "pyxa_test")
