"""Pipeline 1: Merge pyInfinityFlow outputs and batch-correct with Harmony.

Requires pyInfinityFlow to have been run for all donors first.

Usage:
    uv run python scripts/run_pipeline1_harmony.py
"""

import os
import glob
import numpy as np
import anndata as ad
import scanpy as sc
from harmonypy import run_harmony

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONORS = ["D004", "D005", "D006", "D007"]
OUTPUT_PATH = os.path.join(ROOT, "data", "processed", "adata_infinityflow_harmony.h5ad")


def find_h5ad(donor: str) -> str:
    """Find the pyInfinityFlow output h5ad for a donor."""
    donor_dir = os.path.join(ROOT, "data", "processed", f"infinityflow_{donor}")
    h5ad_files = glob.glob(os.path.join(donor_dir, "**", "*.h5ad"), recursive=True)
    if not h5ad_files:
        raise FileNotFoundError(
            f"No h5ad found for {donor} in {donor_dir}. "
            "Run run_infinityflow.py --all first."
        )
    return h5ad_files[0]


def main():
    print("Pipeline 1: pyInfinityFlow + Harmony")
    print("=" * 50)

    # Load per-donor InfinityFlow outputs
    adatas = []
    for donor in DONORS:
        path = find_h5ad(donor)
        print(f"Loading {donor}: {path}")
        a = ad.read_h5ad(path)
        a.obs["donor"] = donor
        adatas.append(a)

    # Merge
    print("\nMerging donors...")
    adata = ad.concat(adatas, label="donor", keys=DONORS)
    adata.obs_names_make_unique()
    print(f"  Combined: {adata.n_obs} cells x {adata.n_vars} features")

    # PCA
    print("Running PCA...")
    sc.pp.pca(adata)

    # Harmony batch correction (using harmonypy directly to avoid scanpy wrapper bug)
    print("Running Harmony integration...")
    harmony_out = run_harmony(adata.obsm["X_pca"], adata.obs, "donor")
    z_corr = harmony_out.Z_corr
    if hasattr(z_corr, "cpu"):
        z_corr = z_corr.cpu().numpy()
    z_corr = np.asarray(z_corr)
    # Z_corr is (n_cells, n_pcs) — no transpose needed
    if z_corr.shape[0] != adata.n_obs:
        z_corr = z_corr.T
    adata.obsm["X_pca_harmony"] = z_corr

    # Neighbors + UMAP on corrected embedding
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    adata.write(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {adata.n_obs} cells x {adata.n_vars} features")
    print(f"  Donors: {adata.obs['donor'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
