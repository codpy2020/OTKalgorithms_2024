"""
Multiscale MNIST Examples
==========================

We illustrate the class :class:`codpy.multiscale_kernel.MultiScaleKernel`, applying it to the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ problem.
The methodology is similar to the gallery example :ref:`MNIST <MNIST>`_
"""
import os,time
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
import random
# We use a custom hot encoder for performances reasons.
from codpy.data_processing import hot_encoder
# Standard codpy kernel class.
from codpy.kernel import Kernel
# A multi scale kernel method.
from codpy.multiscale_kernel import *
from sklearn.metrics import confusion_matrix


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




def one_experiment(N_partition,get_predictor):
    def get_score(predictor):
        f_z = predictor(z)
        f_z = f_z.argmax(axis=-1)
        ground_truth = fz.argmax(axis=-1)
        out = confusion_matrix(ground_truth, f_z)
        return np.trace(out) / np.sum(out)

    elapsed_time = time.time()
    predictor = get_predictor(N_partition)
    score = get_score(predictor)
    elapsed_time = time.time()-elapsed_time
    return score, elapsed_time

def run_experiment(N_partitions,get_predictors,labels,file_name=None):

    results=[]
    for N_partition in N_partitions:
        for get_predictor,label in zip(get_predictors,labels):
            score, elapsed_time = one_experiment(N_partition,get_predictor)
            print("Method:",label,"N_partition:",N_partition," score:",score," time:",elapsed_time)
            results.append(
                {
                    "Ny": N_partition,
                    "Method": label,
                    "Execution Time (s)": elapsed_time,
                    "score": score
                }
    )
    out =   pd.DataFrame(results)
    print(out)
    if file_name is not None: out.to_csv(file_name)
    return out


class Random_clusters:
    def __init__(self,x, N, max_iter = 300, random_state = 42,batch_size = 1024,verbose = False,**kwargs):
        self.x = x
        self.indices = random.sample(range(self.x.shape[0]), N)
        self.cluster_centers_ = self.x[self.indices]
    def __call__(self,z, **kwargs):
        return self.distance(z,self.cluster_centers_).argmin(axis=1)
    def distance(self,x,y):
        return core.op.Dnm(x, y, distance="norm22")


class KernelClusteringClassifierXY(KernelClassifier):
    def __init__(self,x,N,clustering_method,fx,**kwargs):
        method = clustering_method(x,N,fx=fx,**kwargs)
        y = method.cluster_centers_
        # fy = fx[method.indices]
        super().__init__(x=x,y=y,fx=fx,**kwargs)

class KernelClusteringClassifierYY(KernelClassifier):
    def __init__(self,x,N,clustering_method,fx,**kwargs):
        method = clustering_method(x,N,fx=fx,**kwargs)
        y = method.cluster_centers_
        fy = fx[method.indices]
        super().__init__(x=y,fx=fy,**kwargs)


# %% [markdown]
# The training set is `x,fx`, the test set is `z,fz`.
N_partitions=[16,32,64,128,256,512,1024,2048,4096,8192]
# N_partitions=[16,32,64,128]
N_MNIST_pics=-1
x, fx, z, fz = get_MNIST_data(N_MNIST_pics)
core.kernel_interface.set_verbose(False)
# labels = ["random","Greedy","K-Means","Sharp Disc."]
labels = ["random","Greedy","K-Means"]
get_predictorsXY = [
    lambda N_partition: KernelClusteringClassifierXY(x=x,fx=fx,N=N_partition,clustering_method=Random_clusters),
    lambda N_partition: KernelClassifier().greedy_select(x=x,fx=fx,N=N_partition,all=True),
    lambda N_partition: KernelClusteringClassifierXY(x=x,fx=fx,N=N_partition,clustering_method=MiniBatchkmeans)
    # lambda N_partition: KernelClusteringClassifier(x=x,fx=fx,N=N_partition,clustering_method=SharpDiscrepancy),
]
get_predictorsYY = [
    lambda N_partition: KernelClusteringClassifierYY(x=x,fx=fx,N=N_partition,clustering_method=Random_clusters),
    lambda N_partition: KernelClassifier().greedy_select(x=x,fx=fx,N=N_partition,all=False),
    lambda N_partition: KernelClusteringClassifierYY(x=x,fx=fx,N=N_partition,clustering_method=MiniBatchkmeans)
    # lambda N_partition: KernelClusteringClassifier(x=x,fx=fx,N=N_partition,clustering_method=SharpDiscrepancy),
]

# %% [markdown]
# Select a multi scale kernel method where the centers are given by a k-mean algorithm.
core.kernel_interface.set_verbose()
run_experiment(N_partitions=N_partitions,get_predictors=get_predictorsXY,labels=labels,file_name = "results_MNISTXY.csv")
run_experiment(N_partitions=N_partitions,get_predictors=get_predictorsYY,labels=labels,file_name = "results_MNISTYY.csv")