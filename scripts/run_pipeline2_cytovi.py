"""Pipeline 2: Merge pyInfinityFlow outputs and batch-correct with cytoVI.

Requires pyInfinityFlow to have been run for all donors first.

Usage:
    uv run python scripts/run_pipeline2_cytovi.py
"""

import os
import glob
import anndata as ad
import scanpy as sc
from scvi.external import cytovi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONORS = ["D004", "D005", "D006", "D007"]
OUTPUT_PATH = os.path.join(ROOT, "data", "processed", "adata_infinityflow_cytovi.h5ad")

# Arcsinh cofactor: 2000 for full-spectrum, 100 for conventional flow
ARCSINH_COFACTOR = 2000


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
    print("Pipeline 2: pyInfinityFlow + cytoVI")
    print("=" * 50)

    # Load per-donor InfinityFlow outputs
    adatas = []
    for donor in DONORS:
        path = find_h5ad(donor)
        print(f"Loading {donor}: {path}")
        a = ad.read_h5ad(path)
        a.obs["donor"] = donor

        # cytoVI expects a "raw" layer — store current X as raw
        a.layers["raw"] = a.X.copy()

        # cytoVI preprocessing: arcsinh transform + min-max scaling
        cytovi.transform_arcsinh(a, global_scaling_factor=ARCSINH_COFACTOR)
        cytovi.scale(a)
        adatas.append(a)

    # Merge with cytoVI's batch-aware merge
    print("\nMerging donors with cytoVI...")
    adata = cytovi.merge_batches(adatas, batch_key="donor")
    print(f"  Combined: {adata.n_obs} cells x {adata.n_vars} features")

    # Set up and train cytoVI model
    print("Setting up cytoVI model...")
    cytovi.CYTOVI.setup_anndata(adata, layer="scaled", batch_key="donor")

    model = cytovi.CYTOVI(adata)
    print("Training cytoVI model...")
    model.train(n_epochs_kl_warmup=50)

    # Extract batch-corrected representations
    print("Extracting corrected representations...")
    adata.obsm["X_CytoVI"] = model.get_latent_representation()
    adata.layers["imputed"] = model.get_normalized_expression()

    # Compute UMAP on corrected latent space
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
