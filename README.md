# ChromaSeg

Unsupervised image segmentation using **K-Means** and **Gaussian Mixture Models (GMM)**, both implemented from scratch with NumPy — no scikit-learn or any ML framework.

---

## Results

| Original | K-Means (K=8) | GMM (K=8) |
|----------|--------------|-----------|
| ![](examples/outputs/original_image1.png) | ![](examples/outputs/kmeans_image1_k8.png) | ![](examples/outputs/gmm_image1_k8.png) |
| ![](examples/outputs/original_image3.png) | ![](examples/outputs/kmeans_image3_k8.png) | ![](examples/outputs/gmm_image3_k8.png) |

---

## Features

- **K-Means** — Lloyd's algorithm with vectorised distance computation via NumPy broadcasting, multi-restart strategy, and empty-cluster re-seeding
- **GMM (EM)** — Full-covariance Gaussian Mixture Model with Cholesky-based log-determinant, log-sum-exp E-step, covariance regularisation, and K-Means warm-start
- Supports any number of clusters `K` and any RGB image
- Saves side-by-side original / segmentation map figures as PNG

---

## Project Structure

```
chromaseg/
├── chromaseg/
│   ├── __init__.py
│   ├── kmeans.py        # KMeans class
│   ├── gmm.py           # GaussianMixture class
│   ├── image_utils.py   # I/O and segmentation map utilities
│   └── pipeline.py      # End-to-end segmentation pipeline
├── examples/
│   └── images/          # Sample input images
├── outputs/             # Saved segmentation maps (gitignored)
├── run.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/chromaseg.git
cd chromaseg
pip install -r requirements.txt
```

---

## Usage

**Run on the example images:**
```bash
python run.py
```

**Custom images and cluster counts:**
```bash
python run.py --images path/to/images --k 2 4 8 16 --output outputs/
```

**Use the classes directly:**
```python
import numpy as np
from chromaseg.kmeans import KMeans
from chromaseg.gmm import GaussianMixture

X = image.reshape(-1, 3) / 255.0          # (H*W, 3) normalised pixels

km = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X)
print(km.labels_, km.inertia_)

gm = GaussianMixture(n_components=8, n_init=5, random_state=42).fit(X)
print(gm.labels_, gm.lower_bound_)
```

---

## Algorithm Details

### K-Means
Lloyd's algorithm with:
- Random initialisation (sampling k distinct data points)
- Vectorised `(N, 1, D) − (1, K, D)` distance matrix via broadcasting
- Convergence on max centroid shift < `tol`
- Best-of-`n_init` restarts by lowest inertia

### GMM (EM)
Full-covariance EM with:
- **E-step** — log-sum-exp trick for numerical stability
- **M-step** — weighted outer-product covariance update
- Cholesky decomposition for log-determinant and Mahalanobis distance
- Covariance regularisation (`reg_covar * I`) to prevent singularity
- K-Means warm-start for component means
- Best-of-`n_init` restarts by highest log-likelihood

---

## Dependencies

- Python ≥ 3.9
- NumPy
- Matplotlib
- Pillow

---

## License

MIT
