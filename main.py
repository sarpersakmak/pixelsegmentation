
"""
Title:       main.py
Description:
    - Entry point that orchestrates the full CMPE 442 Assignment-3 pipeline.
    - STEP 1: Loads all 4 images and extracts pixel feature matrices
              via image_utils.load_images() and image_utils.pixels_to_features().
    - STEP 2: Saves original images to report/ via image_utils.save_original().
    - STEP 3: Runs K-Means segmentation for all images and all K values
              via kmeans.KMeans.fit(); saves 16 segmentation maps to report/.
    - STEP 4: Runs GMM segmentation for all images and all K values
              via gmm.GMM.fit(); saves 16 segmentation maps to report/.
    - STEP 5: Prints a formatted summary table of inertia (K-Means) and
              log-likelihood (GMM) for every image × K combination.
"""

import os
import numpy as np  # type: ignore

from chromaseg import image_utils
from chromaseg.kmeans import KMeans  # type: ignore
from chromaseg.gmm import GMM     # type: ignore

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 14008175400 % 200   # seed = 0
np.random.seed(SEED)

# ── Hyper-parameters ──────────────────────────────────────────────────────
K_VALUES   = [2, 4, 8, 16]
REPORT_DIR = "report"
IMAGES_DIR = "images"
RESIZE_TO  = (120, 120)    # resize images before processing for speed


# ── STEP 5 helper: metrics table ─────────────────────────────────────────
def print_metrics_table(image_names: list,
                        k_values: list,
                        km_metrics: dict,
                        gmm_metrics: dict) -> None:
    """
    STEP 5 helper: Print a formatted table of inertia (K-Means) and
                   log-likelihood (GMM) for each image × K combination.

    Parameters
    ----------
    image_names : list of str  – bare filenames of the 4 images
    k_values    : list of int  – K values used
    km_metrics  : dict         – {(image_name, k): inertia}
    gmm_metrics : dict         – {(image_name, k): log_likelihood}

    Returns
    -------
    None – prints to stdout.
    """
    col_img = 14
    col_k   = 6
    col_val = 18

    header_km  = " | ".join([f"KM  K={k:>2}" for k in k_values])
    header_gmm = " | ".join([f"GMM K={k:>2}" for k in k_values])

    sep = "─" * (col_img + 2 + len(header_km) + 3 + len(header_gmm) + 4)

    print("\n[main] ─── Final Segmentation Metric Summary " + "─" * 30)
    print(f"[main]  {'Image':<{col_img}}  {header_km}  ║  {header_gmm}")
    print(f"[main]  {sep}")

    for name in image_names:
        km_vals  = [f"{km_metrics.get((name, k), float('nan')):>10.2f}"
                    for k in k_values]
        gmm_vals = [f"{gmm_metrics.get((name, k), float('nan')):>10.2f}"
                    for k in k_values]
        km_str   = " | ".join(km_vals)
        gmm_str  = " | ".join(gmm_vals)
        print(f"[main]  {name:<{col_img}}  {km_str}  ║  {gmm_str}")

    print(f"[main]  {sep}")
    print("[main]  Units: K-Means = Inertia (lower is better) | "
          "GMM = Log-Likelihood (higher is better)\n")


# ── Main pipeline ─────────────────────────────────────────────────────────
def main() -> None:
    """
    STEP 1–5: Full pipeline entry point.

    Loads images, runs K-Means and GMM for every (image, K) pair,
    saves all 32 + 4 PNG outputs to report/, and prints the final
    metric summary table.
    """
    print("=" * 70)
    print("  CMPE 442 Programming Assignment-3")
    print("  Author  : Sarper Sakmak  |  ID: 14008175400  |  Section: 2")
    print("=" * 70)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── STEP 1: Load images ───────────────────────────────────────────────
    print("\n[main] ── STEP 1: Loading images ──")
    images = image_utils.load_images(images_dir=IMAGES_DIR,
                                     resize_to=RESIZE_TO)
    image_names = sorted(images.keys())

    # Build feature matrices once (reused across all K values)
    features = {}
    for name in image_names:
        features[name] = image_utils.pixels_to_features(images[name])
        print(f"[main]   '{name}' feature matrix shape: {features[name].shape}")

    # ── STEP 2: Save original images ──────────────────────────────────────
    print("\n[main] ── STEP 2: Saving original images ──")
    for idx, name in enumerate(image_names, start=1):
        save_path = os.path.join(REPORT_DIR, f"original_image{idx}.png")
        image_utils.save_original(images[name], save_path)

    # ── STEP 3: K-Means segmentation ──────────────────────────────────────
    print("\n[main] ── STEP 3: Running K-Means segmentation ──")
    km_metrics  = {}   # {(name, k): inertia}
    saved_files = []

    for idx, name in enumerate(image_names, start=1):
        X = features[name]
        print(f"\n[main]  ── Image {idx}: '{name}' ──")
        for k in K_VALUES:
            print(f"[main]   K = {k}")
            km = KMeans(k=k, n_init=5, max_iter=300, tol=1e-4, seed=SEED)
            km.fit(X)
            km_metrics[(name, k)] = km.inertia_

            seg_map = image_utils.labels_to_segmentation_map(
                km.labels_, images[name]
            )
            title     = f"K-Means  |  {name}  |  K={k}"
            save_path = os.path.join(
                REPORT_DIR, f"kmeans_image{idx}_k{k}.png"
            )
            image_utils.save_segmentation_map(
                images[name], seg_map, title, save_path
            )
            saved_files.append(save_path)

    # ── STEP 4: GMM segmentation ───────────────────────────────────────────
    print("\n[main] ── STEP 4: Running GMM segmentation ──")
    gmm_metrics = {}   # {(name, k): log_likelihood}

    for idx, name in enumerate(image_names, start=1):
        X = features[name]
        print(f"\n[main]  ── Image {idx}: '{name}' ──")
        for k in K_VALUES:
            print(f"[main]   K = {k}")
            gmm = GMM(k=k, n_init=3, max_iter=100, tol=1e-4, seed=SEED)
            gmm.fit(X)
            gmm_metrics[(name, k)] = gmm.log_likelihood_

            seg_map = image_utils.labels_to_segmentation_map(
                gmm.labels_, images[name]
            )
            title     = f"GMM  |  {name}  |  K={k}"
            save_path = os.path.join(
                REPORT_DIR, f"gmm_image{idx}_k{k}.png"
            )
            image_utils.save_segmentation_map(
                images[name], seg_map, title, save_path
            )
            saved_files.append(save_path)

    # ── STEP 5: Print summary table ────────────────────────────────────────
    print("\n[main] ── STEP 5: Final Metric Summary ──")
    print_metrics_table(image_names, K_VALUES, km_metrics, gmm_metrics)

    # ── Completion message ─────────────────────────────────────────────────
    print("=" * 70)
    print("[main] Pipeline complete. All outputs saved to report/")
    print(f"[main] Total files saved: {4 + len(saved_files)}")
    print("=" * 70)
    print("\n[main] Saved files:")
    for idx, name in enumerate(image_names, start=1):
        print(f"  report/original_image{idx}.png")
    for f in saved_files:
        print(f"  {f}")
    print()


if __name__ == "__main__":
    main()
