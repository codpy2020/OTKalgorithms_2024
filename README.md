# A class of kernel-based scalable algorithms for data science, 2024

This repository contains the experiments conducted for the paper:  
[A class of kernel-based scalable algorithms for data science](https://arxiv.org/abs/2410.14323).

The experiments are presented in the form of Jupyter notebooks (`.ipynb`), demonstrating the concepts and algorithms described in the paper.

---

## Contents

### Jupyter Notebooks

Click the links below to open the corresponding Jupyter notebooks directly in GitHub:
 
1. [Clustering Example](./clustering_example.ipynb): Illustrates the experiments on clustering tasks.
2. [MNIST Example](./MNIST_example.ipynb): Showcases experiments with the MNIST dataset.
3. [Multiscale MNIST Example](./multiscaleMNIST_example.ipynb): Explores multiscale methods on the MNIST dataset.
4. [Multiscale OT Example](./multiscaleOT_example.ipynb): Provides a detailed study of multiscale optimal transport.
5. [Bachelier Example](./bachelier_example.ipynb): Demonstrates the application of optimal transport kernels in the Bachelier problem.

---

### Data Files

- `Bachelier.csv`: Dataset used in the Bachelier example.
- `clustering.csv`: Dataset used in the clustering example.


## Dependencies

This repository relies on several Python packages for implementing optimal transport kernels and related experiments:

- [**CodPy**](https://codpy.readthedocs.io/en/latest/): Codpy is a kernel based, open source software library for high performance numerical computation, relying on the RKHS theory.
- [**OTT**](https://ott-jax.readthedocs.io): A library for scalable optimal transport in machine learning.
- [**POT**](https://pythonot.github.io): Python Optimal Transport library.

```
@article{OTK2024,
  title={A class of kernel-based scalable algorithms for data science},
  author={Philippe G. LeFloch, Jean-Marc Mercier, Shohruh Miryusupov},
  journal={arXiv preprint arXiv:2410.14323},
  year={2024}
}
```