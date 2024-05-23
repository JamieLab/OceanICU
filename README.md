# OceanICU Neural Network and air-sea CO2 flux uncertainty framework

This repository holds the neural network framework developed to extrapolate/interpolate SOCAT fCO2sw observations into globally complete fields, based
on a feed forward neural network approach.
The framework has been developed to allow input parameter combinations to be modified whilst keeping the neural network and CO2 flux calculations
consistent. The framework also expands on the uncertainty determination approaches, allowing time varying uncertainties to be calculated and then
propagated through the flux calculations.

# Instructions

# Supporting Manuscript

Ford, D. J., Blannin, J., Watts, J., Watson, A., Landschützer, P., Jersild, A., & Shutler, J. (in review). A comprehensive analysis of air-sea CO2 flux uncertainties constructed from surface ocean data products. https://doi.org/10.22541/essoar.171199280.05732707/v1

# Developed by
Daniel J. Ford (d.ford@exeter.ac.uk) - Main contact  
Jamie D. Shutler  
Josh Blannin  
Andy Watson  

# Funding
This work was funded by the European Union under grant agreement no. 101083922 (OceanICU; https://ocean-icu.eu/) and UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10054454, 10063673, 10064020, 10059241, 10079684, 10059012, 10048179]. The views, opinions and practices used to produce this dataset/software are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
Further updates to the framework to allow spatial resolution variations was funded by the Convex Seascape Survey (https://convexseascapesurvey.com/).
Further updates to the framework for the generation of Total Alkalinity data, and the extension to full surface ocean carbonate system funded by the ESA Scope project ()

# References
Implementation of the NOAA COARE for cool skin estimation
Bariteau Ludovic, Blomquist Byron, Fairall Christopher, Thompson Elizabeth, Edson Jim, & Pincus Robert. (2021). Python implementation of the COARE 3.5 Bulk Air-Sea Flux algorithm (v1.1). Zenodo. https://doi.org/10.5281/zenodo.5110991

Implementation of semi-variogram analysis in Python
Mälicke, M. (2022). SciKit-GStat 1.0: a SciPy-flavored geostatistical variogram estimation toolbox written in Python. Geoscientific Model Development, 15(6), 2505–2532. https://doi.org/10.5194/gmd-15-2505-2022

FluxEngine: air-sea gas exchange calculation Python toolbox
Shutler, J. D., Land, P. E., Piolle, J. F., Woolf, D. K., Goddijn-Murphy, L., Paul, F., et al. (2016). FluxEngine: A flexible processing system for calculating atmosphere-ocean carbon dioxide gas fluxes and climatologies. Journal of Atmospheric and Oceanic Technology, 33(4), 741–756. https://doi.org/10.1175/JTECH-D-14-00204.1
Holding, T., Ashton, I. G., Shutler, J. D., Land, P. E., Nightingale, P. D., Rees, A. P., et al. (2019). The FluxEngine air–sea gas flux toolbox: simplified interface and extensions for in situ analyses and multiple sparingly soluble gases. Ocean Science, 15(6), 1707–1728. https://doi.org/10.5194/os-15-1707-2019
