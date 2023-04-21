# HI-infrastructure

This repository contains the python modules and iPython notebooks used to produce the results and figures in Mason, Vaughn & Asner (2023). As Global Airborne Observatory data proprietary, users will not be able to fully reproduce this work and the code has not been written with this in mind. Nonetheless, the workflow, comments, and environment files may be useful for those who are applying similar methods to their own data.

Each notebook calls an associated python module that contains the bulk of the code needed to do the work:

- RunCNN.ipynb, run_cnn.py: train CNNs on LiDAR data and produce initial maps
- RunXGB.ipynb, run_xgb.py: train XGBOOST on the initial CNN maps, a canopy height map, and 147 bands of imaging spectroscopy data; produce maps; then use apply vector operations to refine the maps.
  - Also calls apply.py to apply models to entire study regions (as opposed to training and test regions)
- MakeFigures.ipynb, make_figures.py: create the figures in the paper

Three .yaml files record the package versions used, and the BFGN configuration file.
