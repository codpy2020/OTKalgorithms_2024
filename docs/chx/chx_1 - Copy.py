"""
=============================================
X.1 Clustering
=============================================
In this experiment, we explore various machine learning and interpolation techniques
to model and predict a sinusoidal function. We use different models, including CodPy,
SciPy's RBF interpolator, Scikit-learn's SVR, Decision Trees, AdaBoost, Random Forest,
TensorFlow Neural Network, and XGBoost.
"""

# Import necessary modules and setup the environment
import os
import sys

from codpy.clustering import GreedySearch, SharpDiscrepancy

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)


import warnings

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

warnings.filterwarnings("ignore")

# Ensure utils path is in sys.path
curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)

#########################################################################
# **Define Functions for Generating Blob Data and Clustering**


def gen_blobs(Nx, Nz, Ny, D):
    """
    Generate blob data for clustering.

    Parameters:
    - Nx: Number of samples for training.
    - Nz: Number of samples for testing.
    - Ny: Number of centers (clusters).
    - D: Number of dimensions (features).

    Returns:
    - X: Generated data points.
    - y: Cluster labels for each point.
    """
    X, y = make_blobs(
        n_samples=Nx + Nz,
        n_features=int(D),
        centers=int(Ny),
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )
    return X, y


#########################################################################
# Clustering Models
# ------------------------
# This section defines the K-means and CodPy clustering models.
#########################################################################


def greedy_clustering(x, z, Ny):
    """
    Apply a fast greedy search algorithms for clustering.

    Parameters:
    - x: Training data.
    - z: Test data.
    - Ny: Number of clusters.

    Returns:
    - f_z: Predicted cluster labels for test data.
    - centers: Selected cluster centers.
    """
    # Set up a kernel and select centers
    kernel = GreedySearch(x=x, N=Ny)

    # retrieve the centers
    centers = kernel.cluster_centers_
    # retrieve labels associated to a set of points
    f_z = kernel(z)
    return f_z, centers


def sharp_clustering(x, z, Ny):
    # Set up a kernel and select centers
    kernel = SharpDiscrepancy(x=x, N=Ny)
    centers = kernel.cluster_centers_
    f_z = kernel(z)
    return f_z, centers


def kmeans_clustering(x, z, K):
    kmeans = KMeans(n_clusters=K, random_state=1).fit(x)
    f_z = kmeans.predict(z)
    cluster_centers = kmeans.cluster_centers_
    return f_z, cluster_centers


#########################################################################
# Running the Experiment
# ------------------------
# This section runs the experiment to compare K-means and various CodPy clustering.
#########################################################################


import time

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from codpy.kernel import Kernel


def compute_mmd(x_test, z):
    kernel = Kernel(x=x_test, order=1)
    mmd = kernel.discrepancy(z)
    return mmd


def run_experiment(Nx_values, D, Ny_values):
    """
    Run the clustering experiment with different values of Nx, Ny and collect results,
    including inertia, Euclidean distance, and execution time.

    Parameters:
    - Nx_values: List of sample counts for training and testing (Nz = Nx).
    - D: Number of dimensions (features).
    - Ny_values: List of cluster counts to experiment with.

    Returns:
    - A compact DataFrame with metrics for each experiment configuration.
    """

    # Prepare to store the results
    compact_results = []

    for Nx in Nx_values:
        for Ny in Ny_values:
            Nz = Nx  # Set Nz equal to Nx

            # Generate the blob data
            X, _ = gen_blobs(Nx, Nz, Ny, D)

            # Split the data into training and testing sets
            x_train, x_test = X[:Nx], X[Nx:]

            # Apply K-means clustering with the current number of clusters (Ny)
            start_time = time.time()
            kmeans = KMeans(n_clusters=Ny, random_state=1).fit(x_train)
            y_pred = kmeans.predict(x_test)
            kmeans_cluster_centers = kmeans.cluster_centers_
            kmeans_inertia = kmeans.inertia_
            kmeans_execution_time = time.time() - start_time

            # Calculate MMD for K-means
            mmd_km = compute_mmd(x_test, kmeans_cluster_centers)
            # Append K-means results
            compact_results.append(
                {
                    "Nx": Nx,
                    "Ny": Ny,
                    "Method": "K-means",
                    "Inertia": kmeans_inertia,
                    "MMD": mmd_km,
                    "Execution Time (s)": kmeans_execution_time,
                }
            )

            # CodPy Greedy clustering
            start_time = time.time()
            codpy_labels, codpy_centers = greedy_clustering(x_train, x_test, Ny)
            codpy_execution_time = time.time() - start_time

            # Calculate "inertia" for CodPy (sum of squared distances to cluster centers)
            codpy_inertia = np.sum(
                (pairwise_distances(x_test, codpy_centers) ** 2).min(axis=1)
            )
            mmd_codpy = compute_mmd(x_test, codpy_centers)

            # Append CodPy Greedy results
            compact_results.append(
                {
                    "Nx": Nx,
                    "Ny": Ny,
                    "Method": "Greedy-search",
                    "Inertia": codpy_inertia,
                    "MMD": mmd_codpy,
                    "Execution Time (s)": codpy_execution_time,
                }
            )

            # CodPy Sharp Discrepancy clustering
            start_time = time.time()
            codpy_labels, codpy_centers = sharp_clustering(x_train, x_test, Ny)
            codpy_execution_time = time.time() - start_time

            # Calculate "inertia" for CodPy Sharp
            codpy_inertia = np.sum(
                (pairwise_distances(x_test, codpy_centers) ** 2).min(axis=1)
            )
            mmd_codpy = compute_mmd(x_test, codpy_centers)

            # Append CodPy Sharp results
            compact_results.append(
                {
                    "Nx": Nx,
                    "Ny": Ny,
                    "Method": "Sharp Discrepancy",
                    "Inertia": codpy_inertia,
                    "MMD": mmd_codpy,
                    "Execution Time (s)": codpy_execution_time,
                }
            )

    # Convert compact results to DataFrame
    df = pd.DataFrame(compact_results)

    # Pivot the table to make it more compact
    df_pivot = df.pivot_table(
        index=["Nx", "Ny"],
        columns="Method",
        values=["Inertia", "MMD", "Execution Time (s)"],
    )

    # Flatten the multi-level columns
    df_pivot.columns = [f"{metric}_{method}" for metric, method in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    return df_pivot


#########################################################################
# Experiment Setup and Execution
# ------------------------
# We run the experiment for D=2, Nx=100, Nz=100, and cluster counts of 3 and 4.
#########################################################################

# Run the experiment with two-dimensional data and different numbers of clusters


def run_experiment_summary(Nx_values, D, Ny_values):
    # Run the experiment as before
    df = run_experiment(Nx_values, D, Ny_values)

    # Create a more compact display by grouping metrics under each method
    df_compact = pd.DataFrame()
    for method in ["K-means", "Greedy-search", "Sharp Discrepancy"]:
        # Select only the columns for Nx, Ny, and the current method's metrics
        columns_to_select = [
            "Nx",
            "Ny",
            f"Inertia_{method}",
            f"MMD_{method}",
            f"Execution Time (s)_{method}",
        ]
        df_method = df[columns_to_select].copy()

        # Round and format numeric columns in scientific notation
        df_method[f"Inertia_{method}"] = df_method[f"Inertia_{method}"].apply(
            lambda x: f"{x:.2e}"
        )
        df_method[f"MMD_{method}"] = df_method[f"MMD_{method}"].apply(
            lambda x: f"{x:.2e}"
        )
        df_method[f"Execution Time (s)_{method}"] = (
            df_method[f"Execution Time (s)_{method}"]
            .round(4)
            .apply(lambda x: f"{x:.2e}")
        )

        df_method[f"{method}"] = (
            "Inert.: "
            + df_method[f"Inertia_{method}"]
            + ", MMD: "
            + df_method[f"MMD_{method}"]
            + ", Time: "
            + df_method[f"Execution Time (s)_{method}"]
            + "s"
        )

        # Keep only "Nx", "Ny", and the new concatenated metric column for this method
        df_method = df_method[["Nx", "Ny", f"{method}"]]

        # Merge with the compact table
        if df_compact.empty:
            df_compact = df_method
        else:
            df_compact = df_compact.merge(df_method, on=["Nx", "Ny"])

    # Set multi-index for easier reading
    df_compact.set_index(["Nx", "Ny"], inplace=True)

    return df_compact


# Run the summary function
df_summary = run_experiment_summary([256, 512, 1024], 2, [32, 64, 256])
df_summary

pd.set_option("display.max_columns", None)
print(df_summary)
pass
