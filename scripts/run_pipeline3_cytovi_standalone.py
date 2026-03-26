"""Pipeline 3: cytoVI standalone — joint imputation and batch correction.

Reads raw FCS files, builds per-well AnnData objects with backbone + PE marker,
uses cytoVI's multi-panel support (nan_layer) to impute missing markers
and batch-correct across donors simultaneously.

Each well is treated as a separate "panel" with its unique PE marker.

Usage:
    uv run python scripts/run_pipeline3_cytovi_standalone.py
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import fcsparser
from scvi.external import cytovi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONORS = ["D004", "D005", "D006", "D007"]
OUTPUT_PATH = os.path.join(ROOT, "data", "processed", "adata_cytovi_standalone.h5ad")

# Backbone channels: fcsparser uses $PnS (long) names
BACKBONE_CHANNELS = {
    "CD4-FITC-A": "CD4",
    "CD56-PE-Cy7-A": "CD56",
    "CD19-APC-A": "CD19",
    "CD8-APC-Cy7-A": "CD8",
    "CD11c-Brilliant Violet 421-A": "CD11c",
    "CD3-Brilliant Violet 711-A": "CD3",
}
PE_CHANNEL = "Infinity Marker-R-PE-A"

ARCSINH_COFACTOR = 2000
# Subsample to this many cells per well to keep total size manageable
CELLS_PER_WELL = 1000


def load_well_adata(fcs_path: str, donor: str, well_id: str, target: str):
    """Load a single FCS file as AnnData with backbone + this well's PE marker."""
    _, data = fcsparser.parse(fcs_path, reformat_meta=True)

    # Subsample if needed
    n_cells = len(data)
    if n_cells > CELLS_PER_WELL:
        idx = np.random.RandomState(42).choice(n_cells, CELLS_PER_WELL, replace=False)
        data = data.iloc[idx].reset_index(drop=True)
        n_cells = CELLS_PER_WELL

    # Build expression matrix: backbone columns + PE marker column
    backbone_names = list(BACKBONE_CHANNELS.values())
    col_names = backbone_names + [target]

    expr = np.zeros((n_cells, len(col_names)))
    for fcs_col, display_name in BACKBONE_CHANNELS.items():
        col_idx = col_names.index(display_name)
        expr[:, col_idx] = data[fcs_col].values

    # PE marker
    expr[:, -1] = data[PE_CHANNEL].values

    adata = ad.AnnData(
        X=expr,
        var=pd.DataFrame(index=col_names),
        obs=pd.DataFrame(
            {"donor": donor, "well_id": well_id, "target": target},
            index=[f"{donor}_{well_id}_{i}" for i in range(n_cells)],
        ),
    )
    return adata


def main():
    print("Pipeline 3: cytoVI standalone (imputation + integration)")
    print("=" * 60)

    manifest = pd.read_csv(os.path.join(ROOT, "metadata", "fcs_manifest.csv"))
    platemap = pd.read_csv(os.path.join(ROOT, "metadata", "platemap.csv"))

    # Build isotype set for filtering
    isotype_wells = set(
        platemap[platemap.is_isotype_control == True]["well_id"].tolist()
    )
    well_to_target = dict(zip(platemap.well_id, platemap.target))

    # Load all wells across all donors as separate AnnDatas
    all_adatas = []
    total_cells = 0
    for donor in DONORS:
        donor_files = manifest[
            (manifest.donor_id == donor) & (manifest.file_type == "sample")
        ]
        n_wells = 0
        for _, row in donor_files.iterrows():
            if row.well_id in isotype_wells:
                continue
            target = well_to_target.get(row.well_id)
            if target is None:
                continue

            fcs_path = os.path.join(ROOT, row.relative_path)
            a = load_well_adata(fcs_path, donor, row.well_id, target)

            # Store raw and preprocess
            a.layers["raw"] = a.X.copy()
            cytovi.transform_arcsinh(a, global_scaling_factor=ARCSINH_COFACTOR)
            cytovi.scale(a)

            all_adatas.append(a)
            total_cells += a.n_obs
            n_wells += 1
        print(f"  {donor}: {n_wells} wells loaded")

    print(f"\nTotal: {len(all_adatas)} well-panels, {total_cells} cells")

    # Merge all panels — cytoVI handles missing markers via nan_layer
    print("Merging panels with cytoVI...")
    adata = cytovi.merge_batches(all_adatas, batch_key="panel")
    print(f"  Combined: {adata.n_obs} cells x {adata.n_vars} features")

    # Set up and train cytoVI
    print("Setting up cytoVI model...")
    cytovi.CYTOVI.setup_anndata(adata, layer="scaled", batch_key="panel")

    model = cytovi.CYTOVI(adata)
    print("Training cytoVI model...")
    model.train(n_epochs_kl_warmup=50)

    # Extract results
    print("Extracting corrected representations...")
    adata.obsm["X_CytoVI"] = model.get_latent_representation()
    adata.layers["imputed"] = model.get_normalized_expression()

    # UMAP on corrected latent space
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata, use_rep="X_CytoVI")
    sc.tl.umap(adata)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    adata.write(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {adata.n_obs} cells x {adata.n_vars} features")
    print(f"  Donors: {adata.obs['donor'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
