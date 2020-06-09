# billiard simulations

A collection of various Python scripts for simulating random flights of gas particles in a channel with boundary micro-structure. Please see the references for more details. 

The main funtions are located in `src/billiard_library.py`. Documentation of the main functions is found in `Documentation.ipynb`; the documentation Jupyter notebook has many of the visualizations commented out and toned down (e.g. `mpl.rcParams['figure.dpi']` is set very low) in order for it to be displayed by github. If your browser is still having issues displaying `Documentation.ipynb` add `?flush_cache=true` to the end of the URL, i.e. click [here](https://github.com/garciagerman/billiard_simulations/blob/master/Documentation.ipynb?flush_cache=true). 

The remaining scripts in the `src/` directory generate transition matrices for various familes of billiard cells. The `notebooks/` directory contains Jupyter notebooks that analyze the data from the transition matrices. For example, computation of diffusivity of the random flight and spectral gap of the transition operator.

## Dependencies
These scripts are written in Python and use the following dependencies - all of which are included in an Anaconda distribution of Python 3. 
- `numpy` and `pandas`: for linear algebra functionality
- `scipy`: for random variables and linear algebra
- `matplotlib` and `seaborn`: for data visualization

## References

1. Timothy Chumley, Renato Feres, and Luis Alberto Garcia German. "Computation of Knudsen diffusivity in domains with boundary microstructure." arXiv preprint arXiv:2005.14318 (2020).