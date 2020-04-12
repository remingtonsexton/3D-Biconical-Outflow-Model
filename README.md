# 3D Biconical Outflow Model
Reconstruction of 3D Biconical Outflow Models from [Bae & Woo (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...828...97B/abstract)
The Jupyter Notebook and accompanying Python script reproduces the 3D model, 2D flux, velocity, and dispersion maps, and 1D emission line profiles with the same paramters as found in the original manuscript.

**Note**: due to incompatibility between Python 2 and Python 3 for `scipy.interpolate.griddata` function, this notebook only works with Python 2. 

It outputs the following figures:

![](https://github.com/remingtonsexton/3D-Biconical-Outflow-Model/blob/master/figures/model_3d.png)

![](https://github.com/remingtonsexton/3D-Biconical-Outflow-Model/blob/master/figures/maps_2d.png)

![](https://github.com/remingtonsexton/3D-Biconical-Outflow-Model/blob/master/figures/emission_model.png)
