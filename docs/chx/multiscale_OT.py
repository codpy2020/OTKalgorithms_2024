import os

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "1"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal as mvn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

from codpy.kernel import Kernel
from codpy.multiscale_kernel import MultiScaleT
from codpy.permutation import lsap

script_dir = os.path.dirname(os.path.abspath(__file__))


def sample_uniform(n, d, amin=-1, range=2):
    """
    sample from the uniform distribution between [amin,amin+range]
    """
    return np.random.rand(n, d) * range + amin


def sample_mvn(n, A):
    """
    sample from the normal distribution with covariance A and mean 0
    """
    dim = A.shape[0]
    return mvn(
        mean=np.zeros(
            dim,
        ),
        cov=A,
        size=(n,),
    )


# def OT_exp(x):
#     """
#     coordinate-wise exponential map
#     """
#     return np.exp(x)


def OT_exp(x):
    return x * np.sum(x * x, axis=1).reshape(-1, 1)


def OTT(source_mc, source, target, epsilon=None, max_iters=1000):
    # the point cloud geometry
    if epsilon is None:
        geom = pointcloud.PointCloud(source, target)
    else:
        geom = pointcloud.PointCloud(source, target, epsilon=epsilon)

    # solution of the OT problem
    problem = linear_problem.LinearProblem(geom)
    output = sinkhorn.Sinkhorn(max_iterations=max_iters)(problem)

    dual_potentials = output.to_dual_potentials()

    # transport_map#(out-of-sample points)
    transported_points = dual_potentials.transport(source_mc)

    return transported_points


def test1(n_samples=1024, Ns=[2, 16, 32, 128], centers=2):
    X1, _ = make_blobs(n_samples, centers=centers, random_state=42, cluster_std=1.5)
    X2, _ = make_blobs(n_samples, centers=centers, random_state=84, cluster_std=1.5)
    Z, _ = make_blobs(n_samples, centers=centers, random_state=7, cluster_std=1.5)

    n_cols = 2
    n_rows = int(np.ceil(len(Ns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, N in enumerate(Ns):
        MS = MultiScaleT(N)
        MS.set(X1, X2)
        Zmapped = MS(X1)

        k = Kernel(X1)
        C = k.kernel_distance(X2)
        perm = lsap(C)
        X1_perm = X1[perm]
        k.map(X1_perm, X2)
        X1_mapped = k(X1)

        ax = axes[i]
        ax.scatter(X1[:, 0], X1[:, 1], color="blue", alpha=0.5, label="X1")
        ax.scatter(X2[:, 0], X2[:, 1], color="red", alpha=0.5, label="X2")
        # ax.scatter(Z[:, 0], Z[:, 1], color="green", label="Z (original)")
        ax.scatter(
            Zmapped[:, 0], Zmapped[:, 1], color="purple", label="Z (mapped)", marker="x"
        )
        ax.scatter(
            X1_mapped[:, 0], X1_mapped[:, 1], color="orange", label="LSAP", marker="+"
        )

        ax.set_title(f"N={N}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.suptitle("LSAP Mapping, Multiscale experiment with varying Ns")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(script_dir, "figs", "OT_multiscale_blobs.png")
    plt.savefig(fig_path)
    plt.close(fig)


def test2(
    n_samples_list=[1024, 2048, 4096],
    Ns=[2, 64, 128],
    Ds=[2, 4, 8],
    LSAP=False,
    script_dir=None,
):
    # Initialization de DF
    data = np.zeros((len(Ns), len(Ds)))

    # Iteration for each D and N
    for k, D in enumerate(Ds):
        X1 = sample_uniform(n_samples_list[0], D)  # Use first n_samples for plot
        X2 = OT_exp(X1)

        for i, N in enumerate(Ns):
            ## Multiscale avec clustering ####
            MS = MultiScaleT(N)
            MS.set(X1, X2)
            Zmapped = MS(X1)

            ### LSAP Sans clustering ###
            if LSAP:
                k = Kernel(X1)
                C = pairwise_distances(X1, X2)
                perm = lsap(C)
                X1_perm = X1[perm]
                k.map(X1_perm, X2)
                X1_mapped = k(X1)
                mse_lsap = np.sqrt(mean_squared_error(X1_mapped, X2))
            ######################

            mse = np.sqrt(mean_squared_error(Zmapped, X2))
            data[i, k] = mse

    # generate and save figure if len(n_samples_list) == 1
    if len(n_samples_list) == 1:
        fig, axes = plt.subplots(len(Ds), len(Ns), figsize=(5 * len(Ns), 5 * len(Ds)))

        for k, D in enumerate(Ds):
            for i, N in enumerate(Ns):
                ax = axes[k, i] if len(Ds) > 1 else axes[i]  # Handle subplot dimensions
                ax.scatter(X1[:, 0], X1[:, 1], color="blue", alpha=0.5, label="X1")
                ax.scatter(X2[:, 0], X2[:, 1], color="red", alpha=0.5, label="X2")
                ax.scatter(
                    Zmapped[:, 0],
                    Zmapped[:, 1],
                    color="purple",
                    label="Z (mapped)",
                    marker="x",
                )
                ax.set_title(f"N={N}, D={D}")
                ax.set_xlabel("X-axis")
                ax.set_ylabel("Y-axis")
                ax.grid(True)
                if k == 0 and i == 0:
                    ax.legend()

        plt.suptitle(
            f"LSAP Mapping, Multiscale Experiment for n_samples={n_samples_list[0]}"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Saving the figure
        fig_path = os.path.join(script_dir, "figs", "OT_multiscale2.png")
        plt.savefig(fig_path)
        plt.close(fig)

    # Create and save DataFrame for Ns and Ds
    data_df = pd.DataFrame(data, index=Ns, columns=Ds)
    data_df.index.name = "N"
    data_df.columns.name = "D"

    csv_path = os.path.join(script_dir, "figs", "OT_multiscale2.csv")
    data_df.to_csv(csv_path)

    return data_df


def test3(
    n_samples_list=[1024, 2048, 4096],
    Ns=[2, 64, 128],
    Ds=[2, 4, 8],
    script_dir=None,
):
    data = np.zeros((len(Ns), len(Ds)))

    # Iterate over Ds and Ns
    for k, D in enumerate(Ds):
        X1 = sample_uniform(n_samples_list[0], D)
        X2 = OT_exp(X1)

        for j, N in enumerate(Ns):
            # Apply OT mapping
            X1mapped = OTT(X1, X1, X2)

            # Compute MSE
            mse = mean_squared_error(X1mapped, X2)
            data[j, k] = mse

    # Fifure if len(n_samples_list) == 1
    if len(n_samples_list) == 1:
        fig, axes = plt.subplots(len(Ds), len(Ns), figsize=(5 * len(Ns), 5 * len(Ds)))

        for k, D in enumerate(Ds):
            for j, N in enumerate(Ns):
                ax = axes[k, j] if len(Ds) > 1 else axes[j]  # Handle subplot dimensions
                ax.scatter(X1[:, 0], X1[:, 1], color="blue", alpha=0.5, label="X1")
                ax.scatter(X2[:, 0], X2[:, 1], color="red", alpha=0.5, label="X2")
                ax.scatter(
                    X1mapped[:, 0],
                    X1mapped[:, 1],
                    color="purple",
                    label="X1mapped",
                    marker="x",
                )
                ax.set_title(f"N={N}, D={D}")
                ax.set_xlabel("X-axis")
                ax.set_ylabel("Y-axis")
                ax.grid(True)
                if k == 0 and j == 0:
                    ax.legend()

        plt.suptitle(
            f"OTT Mapping, Multiscale Experiment for n_samples={n_samples_list[0]}"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = os.path.join(script_dir, "figs", "OTT.png")
        plt.savefig(fig_path)
        plt.close(fig)

    df = pd.DataFrame(data, index=Ns, columns=Ds)
    df.index.name = "N"
    df.columns.name = "D"

    csv_path = os.path.join(script_dir, "figs", "OTT.csv")
    df.to_csv(csv_path)

    return df


if __name__ == "__main__":
    # test1(512, Ns=[2])
    print(
        test2(
            n_samples_list=[1024],
            Ns=[2, 4, 16],
            Ds=[2, 4, 8],
            script_dir=script_dir,
        )
    )
    fig_path = os.path.join(script_dir, "figs", "OT_multiscale2.png")
    img = plt.imread(fig_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # print(test3(n_samples_list=[512], Ns=[2,16, 32, 128], Ds = [2, 4, 8], script_dir=script_dir))  # 2, 16, 32, 128

    test1()
    pass
