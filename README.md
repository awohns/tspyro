# tspyro
`tspyro` is a [`Pyro`](https://pyro.ai)-based method for inferring posterior distributions on the geographic location and age of ancestral nodes in a tree sequence. `tspyro` can operate on any valid tree sequence in the [`tskit`](https://tskit.dev/tskit/docs/stable/introduction.html) format, including those simulated with [`SLiM`](https://github.com/MesserLab/SLiM) (possibly using [`Slendr`](https://github.com/bodkan/slendr)) or [`msprime`](https://tskit.dev/msprime/docs/stable/intro.html), or inferred by [`tsinfer`](https://tsinfer.readthedocs.io/en/latest/).

## Installation
`tspyro` is not yet available via PyPI or conda, so to install `tspyro` please clone this repo.

The required Python packages can be installed with:
```
pip3 install -r requirements.txt
```
