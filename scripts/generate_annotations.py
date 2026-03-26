"""Generate pyInfinityFlow annotation CSV files from metadata.

Creates:
- data/annotations/backbone_annotation.csv (shared across all donors)
- data/annotations/infinity_marker_annotation_{donor}.csv (one per donor)
"""

import os
import csv
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(ROOT, "data", "annotations")
METADATA_DIR = os.path.join(ROOT, "metadata")

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Backbone channels: fluorescence channels excluding PE (YL1-A) and viability (VL2-A)
# Format: reference_channel, query_channel, display_name
BACKBONE = [
    ("BL1-A", "BL1-A", "CD4"),
    ("YL3-A", "YL3-A", "CD56"),
    ("RL1-A", "RL1-A", "CD19"),
    ("RL3-A", "RL3-A", "CD8"),
    ("VL1-A", "VL1-A", "CD11c"),
    ("VL5-A", "VL5-A", "CD3"),
]

DONORS = ["D004", "D005", "D006", "D007"]
PE_CHANNEL = "YL1-A"


def write_backbone_annotation():
    path = os.path.join(ANNOTATIONS_DIR, "backbone_annotation.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in BACKBONE:
            writer.writerow(row)
    print(f"Wrote backbone_annotation.csv ({len(BACKBONE)} channels)")


def write_infinity_marker_annotations():
    platemap = pd.read_csv(os.path.join(METADATA_DIR, "platemap.csv"))
    manifest = pd.read_csv(os.path.join(METADATA_DIR, "fcs_manifest.csv"))

    # Build well_id -> (target, is_isotype) mapping
    well_info = {
        row.well_id: (row.target, row.is_isotype_control)
        for _, row in platemap.iterrows()
    }

    for donor in DONORS:
        donor_files = manifest[
            (manifest.donor_id == donor) & (manifest.file_type == "sample")
        ].sort_values("well_id")

        rows = []
        for _, frow in donor_files.iterrows():
            target = well_info[frow.well_id][0]
            # 3-column format: filename, channel, marker_name
            # (pyInfinityFlow has a bug with 4-column isotype format when
            # empty strings are read as NaN by pandas)
            rows.append([frow.filename, PE_CHANNEL, target])

        path = os.path.join(
            ANNOTATIONS_DIR, f"infinity_marker_annotation_{donor}.csv"
        )
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        print(f"Wrote infinity_marker_annotation_{donor}.csv ({len(rows)} markers)")


if __name__ == "__main__":
    write_backbone_annotation()
    write_infinity_marker_annotations()
    print("\nDone!")
