import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from clustering_results import *
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.stats import norm

from codpy.kernel import Kernel
from codpy.plot_utils import multi_plot


## Base computations for Bachelier problem
def get_basket_values(x, weights, **kwargs):
    if len(x) == 0:
        return
    if type(x) == type([]):
        return [get_basket_values(s, weights, **kwargs) for s in x]
    return np.dot(x, weights)


def get_payoff_values(x, weights, K, **kwargs):
    if len(x) == 0:
        return
    if type(x) == type([]):
        return [payoff_values(s, weights, K, **kwargs) for s in x]
    bkt = get_basket_values(x, weights, **kwargs)
    pay = np.maximum(0, bkt - K)
    return np.asarray(pay).reshape(-1, 1)


def payoff_nabla_values(x, weights, K, time, **kwargs):
    if len(x) == 0:
        return
    if type(x) == type([]):
        return [payoff_nabla_values(s, weights, K, time, **kwargs) for s in x]
    bkt = get_basket_values(x, weights, **kwargs)
    Z = np.where(bkt > K, 1.0, 0.0).reshape((-1, 1)) * weights.reshape((1, -1))
    return Z


# helper analytics
def BachelierPrice(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return vol * np.sqrt(T) * norm.pdf(d) + (spot - strike) * norm.cdf(d)


def BachelierDelta(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return norm.cdf(d)


def BachelierVega(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return np.sqrt(T) * norm.pdf(d)


def Bacheliervalues(x, sigma, K, T, weights):
    return BachelierPrice(get_basket_values(x, weights), K, sigma, T).reshape((-1, 1))


def genCorrel(n, seed=None):
    if seed:
        np.random.seed(seed)
    randoms = np.random.uniform(low=-1.0, high=1.0, size=(2 * n, n))
    cov = randoms.T @ randoms
    invvols = np.diag(1.0 / np.sqrt(np.diagonal(cov)))
    return np.linalg.multi_dot([invvols, cov, invvols])


def get_weights(n, seed=None):
    if seed:
        np.random.seed(seed)
    weights = np.random.uniform(low=1.0, high=10.0, size=n)
    weights /= np.sum(weights)
    return weights


def genVols(n, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.uniform(low=5.0, high=50.0, size=n)


def get_correlation(
    d, Vol=0.2, corrs=None, vols=None, weights=None, seed=None, **kwargs
):
    # random correls, but normalization to bktvol
    if corrs is None:
        corrs = genCorrel(d, seed)
    if vols is None:
        vols = genVols(d, seed)
    diagv = np.diag(vols)
    cov = np.linalg.multi_dot([diagv, corrs, diagv])
    correlation = np.linalg.cholesky(cov)
    if weights is None:
        weights = get_weights(d, seed)
    tempvol = np.sqrt(np.linalg.multi_dot([weights.T, cov, weights]))
    cov *= Vol * Vol / (tempvol * tempvol)
    correlation *= Vol / tempvol
    testvol = np.sqrt(np.linalg.multi_dot([weights.T, cov, weights]))
    return correlation


def get_variables(time, x0, corrs, **kwargs):
    normals = np.random.normal(size=(x0.shape[0], x0.shape[1]))
    inc = np.sqrt(time) * normals @ corrs.T
    x1 = x0 + inc
    return np.asarray(x1)


def run_experiment(
    Nxs, Ds, get_predictors, labels, file_name=None, vol=0.2, K=0.0, T=1.0, **kwargs
):
    def one_experiment(X1, X2, P, ground_truth, get_predictor, **kwargs):
        def get_rmse(x, y):
            return np.square(x - y).mean()

        elapsed_time = time.time()
        values = get_predictor(X1, X2, P, **kwargs)
        elapsed_time = time.time() - elapsed_time
        rmse = get_rmse(values, ground_truth)
        return rmse, elapsed_time

    results = []
    for Nx in Nxs:
        for D in Ds:
            weights, corrs = get_weights(D), get_correlation(D)
            X0 = np.zeros([Nx, D])
            X1 = get_variables(time=T, x0=X0, corrs=corrs, vol=vol, seed=42)
            X2 = get_variables(time=T, x0=X1, corrs=corrs, vol=vol, seed=43)
            X1 -= X1.mean(axis=1)[:, None]
            X2 -= X2.mean(axis=1)[:, None]
            ground_truth = Bacheliervalues(X1, vol, K, T, weights)
            payoff_values = get_payoff_values(X2, weights, K)
            for get_predictor, label in zip(get_predictors, labels):
                score, elapsed_time = one_experiment(
                    X1, X2, payoff_values, ground_truth, get_predictor, **kwargs
                )
                print(
                    "Method:",
                    label,
                    ", Nx:",
                    Nx,
                    ", D:",
                    D,
                    "score:",
                    score,
                    " time:",
                    elapsed_time,
                )
                results.append(
                    {
                        "Method": label,
                        "Nx": Nx,
                        "D": D,
                        "time (s)": elapsed_time,
                        "score": score,
                    }
                )
    out = pd.DataFrame(results)
    print(out)
    if file_name is not None:
        out.to_csv(file_name, index=False)
    return out


def get_results():
    # path = os.path.join(fig_path,"results_MNISTYY.csv")
    path = "Bachelier.csv"
    return pd.read_csv(path)


def plot_results(file_name, **kwargs):
    def plot_helper(data, ax, legend="", **kwargs):
        results, col = data
        D, results = results[0], results[1]
        methods = results["Method"].unique()
        Nys = [int(n) for n in results["Nx"].unique()]
        vals, methods = [], []
        for group in results.groupby("Method")[col]:
            vals.append([float(v) for v in group[1]])
            methods.append(group[0])
        xlims = kwargs.get("xlims", None)
        ylims = kwargs.get("ylims", None)
        if xlims is not None and col in xlims.keys():
            ax.set_xlim(xlims[col])
        if ylims is not None and col in ylims.keys():
            ax.set_ylim(ylims[col])
        vals = np.array(vals).T
        # ax.loglog(Nys,vals,label=methods)
        ax.plot(Nys, vals, label=methods)
        plt.xlabel("Nx")
        plt.ylabel(col)
        ax.legend()

    results = pd.read_csv(file_name)
    datas = [(x, "score") for x in results.groupby("D")]
    multi_plot(
        datas,
        plot_helper,
        mp_nrows=1,
        mp_ncols=len(datas),
        mp_figsize=(14, 4),
        **kwargs,
    )
    datas = [(x, "time (s)") for x in results.groupby("D")]
    multi_plot(
        datas,
        plot_helper,
        mp_nrows=1,
        mp_ncols=len(datas),
        mp_figsize=(14, 4),
        **kwargs,
    )


def POT(X, Y, Z, P, **kwargs):
    def OT_0(X, Y, epsilon=None, **kwargs):
        n = X.shape[0]
        a = np.ones(n)
        b = np.ones(n)
        M = ot.dist(X, Y)
        if epsilon == None:
            epsilon = 1.0 / M.max()
        G0 = ot.sinkhorn(a, b, M, epsilon)
        return G0

    print("error OTT before: ", np.square(X - Y).sum() * 0.5)
    out = OT_0(X, Y, **kwargs)
    print("error POT after: ", np.square(X - lalg.prod(out, Y)).sum() * 0.5)
    out = out @ P
    k = KernelClassifier(x=X, fx=out, **kwargs)
    out = lalg.prod(k(Z), P)
    return out


def OTT(X, Y, Z, P, max_iters=1000, **kwargs):
    # the point cloud geometry
    N = X.shape[0]
    geom = pointcloud.PointCloud(X, Y)
    print("error OTT before: ", np.square(X - Y).sum() * 0.5)
    # solution of the OT problem
    problem = linear_problem.LinearProblem(geom)
    out = (
        jax.numpy.asarray(sinkhorn.Sinkhorn(max_iterations=max_iters)(problem).matrix)
        * N
    )
    print("error OTT after: ", np.square(X - lalg.prod(out, Y)).sum() * 0.5)
    k = KernelClassifier(x=X, fx=out, **kwargs)
    out = lalg.prod(k(Z), P)
    return out


def dumb(X, Y, Z, P, **k):
    # k = KernelClassifier(x=X,fx=np.identity(),**k)
    # out = lalg.prod(k(Z),P)
    return P


def KernelOT(X, Y, Z, P, **k):
    Kernel(**k).set(x=X, y=Y)
    print("error KOT before: ", np.square(X - Y).sum() * 0.5)
    out = alg.Pi(x=X, y=Y)
    print("error KOT after: ", np.square(X - lalg.prod(out, Y)).sum() * 0.5)
    k = KernelClassifier(x=X, fx=out, **k)
    out = lalg.prod(k(Z), P)
    return out


get_predictors = [
    lambda X, Y, Z, P, **k: dumb(X, Y, Z, P, **k),
    lambda X, Y, Z, P, **k: POT(X, Y, Z, P, **k),
    lambda X, Y, Z, P, **k: OTT(X, Y, Z, P, **k),
    lambda X, Y, Z, P, **k: KernelOT(X, Y, Z, P, **k),
]
labels = ["dumb", "POT", "OTT", "KKmeans"]
if __name__ == "__main__":
    # use of scenario list instead
    core.kernel_interface.set_verbose()
    Nxs = [2**i for i in np.arange(5, 13, 1)]
    Ds = [2, 10, 100]
    # run_experiment(Nxs, Ds,get_predictors,labels,file_name="Bachelier.csv")
    plot_results(file_name="Bachelier.csv")
    plt.show()
    pass
