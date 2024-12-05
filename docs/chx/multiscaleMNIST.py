"""
Multiscale MNIST Examples
==========================

We illustrate the class :class:`codpy.multiscale_kernel.MultiScaleKernel`, applying it to the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ problem.
The methodology is similar to the gallery example :ref:`MNIST <MNIST>`_
"""

import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "4"

import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# We use a custom hot encoder for performances reasons.
from codpy.data_processing import hot_encoder

# Standard codpy kernel class.
# A multi scale kernel method.
from codpy.multiscale_kernel import *


# %% [markdown]
def get_MNIST_data(N=-1):
    import tensorflow as tf

    (x, fx), (z, fz) = tf.keras.datasets.mnist.load_data()
    x, z = x / 255.0, z / 255.0
    x, z, fx, fz = (
        x.reshape(len(x), -1),
        z.reshape(len(z), -1),
        fx.reshape(len(fx), -1),
        fz.reshape(len(fz), -1),
    )
    fx, fz = (
        hot_encoder(pd.DataFrame(data=fx), cat_cols_include=[0], sort_columns=True),
        hot_encoder(pd.DataFrame(data=fz), cat_cols_include=[0], sort_columns=True),
    )
    x, fx, z, fz = (x, fx.values, z, fz.values)
    if N != -1:
        indices = random.sample(range(x.shape[0]), N)
        x, fx = x[indices], fx[indices]

    return x, fx, z, fz


def one_experiment(N_partition, get_predictor, **kwargs):
    def get_score(predictor):
        f_z = predictor(z).argmax(axis=-1)
        ground_truth = fz.argmax(axis=-1)
        out = confusion_matrix(ground_truth, f_z)
        return np.trace(out) / np.sum(out)

    elapsed_time = time.time()
    predictor = get_predictor(N_partition, **kwargs)
    score = get_score(predictor)
    elapsed_time = time.time() - elapsed_time
    print("N_partitions:", N_partitions, " time:", elapsed_time)
    return score, elapsed_time


def run_experiment(N_partitions, get_predictors, labels):
    results = []
    for N_partition in N_partitions:
        for get_predictor, label in zip(get_predictors, labels):
            score, elapsed_time = one_experiment(N_partition, get_predictor, all=True)
            results.append(
                {
                    "Ny": N_partition,
                    "Method": label,
                    "Execution Time (s)": elapsed_time,
                    "score": score,
                }
            )
    out = pd.DataFrame(results)
    print(out)
    out.to_csv("results_MNISTMultiscale.csv")
    return out


class Random_clusters:
    def __init__(self, x, N, **kwargs):
        self.x = x
        self.indices = random.sample(range(self.x.shape[0]), N)
        self.cluster_centers_ = self.x[self.indices]

    def __call__(self, z, **kwargs):
        return self.distance(z, self.cluster_centers_).argmin(axis=1)

    def distance(self, x, y):
        return core.op.Dnm(x, y, distance="norm22")


# %% [markdown]
# The training set is `x,fx`, the test set is `z,fz`.
# N_partitions=[5,10,20,40,80]
N_partitions = [5, 10, 20, 40, 80]
N_MNIST_pics = 40000
x, fx, z, fz = get_MNIST_data(N_MNIST_pics)
core.kernel_interface.set_verbose(False)
labels = ["random", "Sharp Disc.", "Greedy", "K-Means"]
get_predictors = [
    lambda N_partition, **kwargs: MultiScaleKernelClassifier(
        x=x, fx=fx, N=N_partition, method=Random_clusters, **kwargs
    ),
    lambda N_partition, **kwargs: MultiScaleKernelClassifier(
        x=x, fx=fx, N=N_partition, method=SharpDiscrepancy, **kwargs
    ),
    lambda N_partition, **kwargs: MultiScaleKernelClassifier(
        x=x, fx=fx, N=N_partition, method=GreedySearch, **kwargs
    ),
    lambda N_partition, **kwargs: MultiScaleKernelClassifier(
        x=x, fx=fx, N=N_partition, method=MiniBatchkmeans, **kwargs
    ),
]

# %% [markdown]
# Select a multi scale kernel method where the centers are given by a k-mean algorithm.

run_experiment(N_partitions=N_partitions, get_predictors=get_predictors, labels=labels)
