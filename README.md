# billiard simulations

A collection of various Python scripts for simulating random flights of gas particles in a channel with boundary micro-structure. Please see the references for more details. 

The main funtions are located in `src/billiard_library.py`. Documentation of the main functions is found in `Documentation.ipynb`. The remaining scripts in the `src/` directory generate transition matrices for various familes of billiard cells. 

The `notebooks/` directory contains Jupyter notebooks that analyze the data from the transition matrices. For example, computation of diffusivity of the random flight and spectral gap of the transition operator.

## Dependencies
These scripts are written in Python and use the following dependencies - all of which are included in an Anaconda distribution of Python 3. 
- `numpy` and `pandas`: for linear algebra functionality
- `scipy`: for random variables and linear algebra
- `matplotlib` and `seaborn`: for data visualization

## References

1. Timothy Chumley, Renato Feres, and Luis Alberto Garcia German. "Computation of Knudsen diffusivity in domains with boundary microstructure." arXiv preprint arXiv:2005.14318 (2020).