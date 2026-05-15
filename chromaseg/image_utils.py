"""
Title:       image_utils.py
Description:
    - Provides image I/O and segmentation-map utilities for the pipeline.
    - STEP 1: Loads all JPEG/PNG images from a given directory and
              optionally resizes them for faster processing
              (load_images).
    - STEP 2: Reshapes an (H, W, 3) image into an (H*W, 3) feature matrix
              and normalises pixel values to [0, 1]
              (pixels_to_features).
    - STEP 3: Converts a flat label array back into an (H, W, 3) colour
              segmentation map by replacing each pixel with its cluster
              mean colour (labels_to_segmentation_map).
    - STEP 4: Saves a side-by-side matplotlib figure showing the original
              image next to the segmentation map as a PNG file
              (save_segmentation_map).
"""

import os
import glob

import numpy as np          # type: ignore
import matplotlib           # type: ignore
matplotlib.use("Agg")       # non-interactive backend for file saving
import matplotlib.pyplot as plt  # type: ignore
from PIL import Image            # type: ignore

SEED = 14008175400 % 200   # seed = 0


# ── STEP 1: Load images ───────────────────────────────────────────────────
def load_images(images_dir: str = "images",
                resize_to: tuple = (150, 150)) -> dict:
    """
    Load all JPEG/PNG images from *images_dir*, optionally resize them,
    and return a dictionary mapping filename → numpy array.

    Parameters
    ----------
    images_dir : str   – relative path to the folder containing images
                         (default "images")
    resize_to  : tuple – (width, height) to resize every image before
                         processing; set to None to skip resizing
                         (default (150, 150))

    Returns
    -------
    images : dict {str: np.ndarray (H, W, 3), dtype uint8}
             Keys are bare filenames (e.g. "image1.jpg").
    """
    pattern  = os.path.join(images_dir, "*")
    paths    = sorted(glob.glob(pattern))
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    images = {}
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext not in valid_ext:
            continue
        fname = os.path.basename(path)
        img   = Image.open(path).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, Image.LANCZOS)  # type: ignore
        arr = np.array(img, dtype=np.uint8)
        images[fname] = arr
        print(f"[image_utils] Loaded '{fname}' → shape {arr.shape}")

    print(f"[image_utils] Total images loaded: {len(images)}\n")
    return images


# ── STEP 2: Pixel feature extraction ─────────────────────────────────────
def pixels_to_features(image: np.ndarray) -> np.ndarray:
    """
    Reshape an (H, W, 3) image to (H*W, 3) and normalise to [0, 1].

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    X : np.ndarray, shape (H*W, 3), dtype float64 – normalised pixel features
    """
    H, W, C = image.shape
    X = image.reshape(H * W, C).astype(np.float64) / 255.0
    return X


# ── STEP 3: Segmentation map creation ────────────────────────────────────
def labels_to_segmentation_map(labels: np.ndarray,
                                image: np.ndarray) -> np.ndarray:
    """
    Convert a flat cluster-label array into a colour segmentation map.

    For each cluster k the mean RGB colour of all pixels assigned to k
    is computed from the *original* image; every pixel in the map is then
    filled with that mean colour.

    Parameters
    ----------
    labels : np.ndarray, shape (H*W,) – cluster label for each pixel
    image  : np.ndarray, shape (H, W, 3), dtype uint8 – original image

    Returns
    -------
    seg_map : np.ndarray, shape (H, W, 3), dtype uint8 – colour map
    """
    H, W, C   = image.shape
    N         = H * W
    pixels    = image.reshape(N, C).astype(np.float64)
    seg_flat  = np.zeros((N, C), dtype=np.float64)

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask            = labels == lab
        mean_color      = pixels[mask].mean(axis=0)
        seg_flat[mask]  = mean_color

    seg_map = seg_flat.reshape(H, W, C).clip(0, 255).astype(np.uint8)
    return seg_map


# ── STEP 4: Save segmentation map ─────────────────────────────────────────
def save_segmentation_map(original: np.ndarray,
                          seg_map: np.ndarray,
                          title: str,
                          save_path: str) -> None:
    """
    Save a side-by-side matplotlib figure: original image | segmentation map.

    Parameters
    ----------
    original  : np.ndarray, shape (H, W, 3) – original RGB image (uint8)
    seg_map   : np.ndarray, shape (H, W, 3) – segmented colour map (uint8)
    title     : str – figure suptitle (e.g. "K-Means | image1 | K=4")
    save_path : str – full output file path (e.g. "report/kmeans_image1_k4.png")

    Returns
    -------
    None – saves the PNG file to *save_path*.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(seg_map)
    axes[1].set_title("Segmentation Map", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[image_utils] Saved → {save_path}")


# ── Utility: save original image ─────────────────────────────────────────
def save_original(image: np.ndarray, save_path: str) -> None:
    """
    Save the original image as a standalone PNG file.

    Parameters
    ----------
    image     : np.ndarray, shape (H, W, 3) – RGB image (uint8)
    save_path : str – output file path (e.g. "report/original_image1.png")

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Original", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[image_utils] Saved → {save_path}")
