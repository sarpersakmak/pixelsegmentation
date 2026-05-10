# 🎨 ChromaSeg: Unsupervised Image Segmentation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ChromaSeg is a robust, unsupervised image segmentation tool utilizing **K-Means** and **Gaussian Mixture Models (GMM)**. 

What sets this project apart is its underlying architecture: both algorithms are **implemented entirely from scratch using NumPy**. It purposefully avoids high-level machine learning frameworks like `scikit-learn` to demonstrate a deep understanding of the core mathematical principles, linear algebra, and vectorised operations.

---

## 📊 Segmentation Results

*Below are examples of segmentation outputs comparing K-Means and GMM algorithms using 8 clusters.*

| Original Image | K-Means (K=8) | GMM (K=8) |
|:---:|:---:|:---:|
| ![](images/Photo1.jpg) | *[Insert Output Path]* | *[Insert Output Path]* |
| ![](images/Photo3.jpg) | *[Insert Output Path]* | *[Insert Output Path]* |

> **Note:** Replace the `*[Insert Output Path]*` placeholders with the actual generated image paths once you run the algorithm.

---

## ✨ Key Features

- **Custom K-Means Implementation:** Features Lloyd's algorithm with highly optimized, vectorised distance computations via NumPy broadcasting. Includes a multi-restart strategy and empty-cluster re-seeding mechanisms.
- **Custom GMM (Expectation-Maximization):** Full-covariance Gaussian Mixture Model built from the ground up. Implements the log-sum-exp trick for numerical stability during the E-step, Cholesky-based log-determinants, covariance regularisation to prevent singularity, and K-Means warm-starting.
- **Flexibility:** Supports any arbitrary number of clusters (`K`) and processes any standard RGB image format.
- **Visualization:** Automatically generates side-by-side comparisons of the original image and the resulting segmentation maps, saving them as high-quality PNGs.

---

## 📂 Project Architecture

```text
pixelsegmentation/
├── chromaseg/
│   ├── gmm.py               # Core GaussianMixture class
│   ├── image_utils.py       # I/O, preprocessing, and map visualization utilities
│   └── kmeans.py            # Core KMeans class
├── images/                  # Directory containing sample inputs (Photo1.jpg, etc.)
├── report/                  # Documentation and analytical reports
├── main.py                  # Main entry point and execution pipeline
├── requirements.txt         # Project dependencies
├── .gitignore               # Git exclusion rules
└── README.md                # Project documentation
⚙️ Installation
Clone the repository and install the required dependencies:

Bash
git clone [https://github.com/sarpersakmak/pixelsegmentation.git](https://github.com/sarpersakmak/pixelsegmentation.git)
cd pixelsegmentation
pip install -r requirements.txt
🚀 Usage
1. Standard Execution:
To run the segmentation pipeline on the default images located in the images/ directory:

Bash
python main.py
2. Programmatic Usage (API):
You can easily integrate the custom classes into your own scripts:

Python
import numpy as np
from chromaseg.kmeans import KMeans
from chromaseg.gmm import GaussianMixture

# Load and normalize image pixels
X = image.reshape(-1, 3) / 255.0  # Shape: (H*W, 3)

# Initialize and fit K-Means
km = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X)
print("K-Means Labels:", km.labels_)
print("K-Means Inertia:", km.inertia_)

# Initialize and fit GMM
gm = GaussianMixture(n_components=8, n_init=5, random_state=42).fit(X)
print("GMM Labels:", gm.labels_)
print("GMM Lower Bound:", gm.lower_bound_)
🧠 Algorithmic Deep-Dive
K-Means
Based on Lloyd's algorithm, focusing on computational efficiency:

Initialization: Random sampling of K distinct data points.

Distance Matrix: Leverages NumPy broadcasting instead of iterative loops for massive speedups.

Convergence Criteria: Stops when the maximum centroid shift falls below a specified tol threshold.

Optimization: Utilizes a best-of-n_init restart mechanism, selecting the model with the lowest inertia.

Gaussian Mixture Models (GMM)
Full-covariance EM algorithm implementation:

E-step: Employs the log-sum-exp trick to handle extremely small probability values without underflow.

M-step: Calculates weighted outer-product covariance updates.

Mathematics: Uses Cholesky decomposition for efficient computation of the log-determinant and Mahalanobis distance.

Stability: Adds a small constant to the diagonal of covariance matrices to ensure they remain positive-definite.

Warm-start: Initializes component means using the results from K-Means for faster convergence.

📦 Dependencies
Python >= 3.9

numpy

matplotlib

Pillow

📄 License
This project is licensed under the MIT License.
