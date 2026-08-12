# Source Code for: 

Luke Colosi, Matthew Mazloff, and Sarah T. Gille. Decorrelation Time Scales Variations Across the Coastal-Open Ocean Transition Zone. Journal of Geophysical Research: Oceans, in preparation. 

# Abstract 

The coastal-open ocean transition zone (COTZ) encompasses the continental shelf, slope, and adjacent offshore waters where complex physical processes, including mesoscale eddies, internal waves, and coastal upwelling, regulate the redistribution of heat, freshwater, and nutrients that sustain productive marine ecosystems. Understanding the processes that link the coastal and open ocean across the COTZ requires knowledge of the intrinsic scales of ocean variability to ensure that the ocean is accurately measured, mapped, and modeled. Characteristic time scales of potential density are expected to increase with distance offshore, mirroring the lengthening of spatial scales, including the baroclinic Rossby deformation radius, across the COTZ. However, the spatial structure and abruptness of this transition remain incompletely understood. Here, we characterize the decorrelation time scales of potential density across the COTZ using a high-resolution ocean model of the California Current System (CCS) and long-term mooring observations, assess the model's ability to reproduce observed scales, and identify the physical drivers. We find that decorrelation time scales do not increase monotonically with distance offshore, but instead exhibit significant spatial variability driven by dynamical transitions: the continental shelf break, the base of the mixed layer, and regions of reversing alongshore flow in the open ocean. The model broadly reproduces the observed spatial patterns but systematically overestimates decorrelation time scales, suggesting that high-frequency variability present in observations is not fully captured by the model. These results highlight the importance of accounting for spatially variable decorrelation time scales in the design of observation systems, data assimilation schemes, and objective analyses spanning the COTZ, where a single characteristic scale is insufficient to capture the full range of ocean variability.

# Plain Language Summary

The shallow coastal seas and the deep open ocean continuously exchange energy, heat, freshwater, and nutrients through ocean currents, waves, and eddies. Understanding how coastal and open-ocean waters are connected is important because coastal oceans help regulate Earth's climate, support productive marine ecosystems, and sustain ocean fisheries. Understanding these connections requires knowledge of how the ocean evolves over time, so we can accurately measure, map, and model ocean variables such as temperature, salinity, and density. However, we do not fully understand how long ocean features and conditions tend to persist before changing, how this persistence varies from the coast to the open ocean, whether these spatial changes in persistence occur gradually or abruptly, or what drives this variability. This study uses data from a high-resolution ocean model and mooring observations to determine the persistence of ocean density off the California coast and how this persistence varies throughout the water column and across the coastal ocean. We find that the persistence of ocean density varies significantly throughout the ocean, with the most pronounced changes at transitions in ocean currents, water depth, and how density changes with depth. These results show that ocean density does not vary in a simple, uniform way across the coastal ocean, and that assuming a single typical timescale is not sufficient to describe or predict variability. This has important implications for how we design observing systems and build models that aim to represent the full range of ocean variability.

# Authors 
* [Luke V. Colosi](https://lcolosi.github.io/) <<lcolosi@ucsd.edu>>
* [Matthew R. Mazloff](https://mmazloff.scrippsprofiles.ucsd.edu/) <<mmazloff@ucsd.edu>>
* [Sarah T. Gille](https://sgille.scrippsprofiles.ucsd.edu/) <<sgille@ucsd.edu>>

# Data
All data needed to reproduce the analysis in this paper will be available for download through the University of California Digital Collections.

# Funding
This work was supported by the Office of Naval Research (Grant #), by the National Defense Science and Engineering Graduate Fellowship.

# How to use this repository

All figures in Colosi et al. (2026) can be reproduced using the Python scripts from this repository (or from [Zenodo](add_link_here)) and processed [data](add_link_here) published to the UCSD library digital collections. To do so, follow these steps:

1. Make a local copy of this repository by either cloning or downloading it.

2. Download the processed [data](add_link_here), unzip the files, and move all directories to `data` in the project root. After doing so, your directory tree should look like this:

```
WaveSpectrum/
├── data
│   ├── cce
│   ├── mitgcm
│   ├── bathymetry
│   └── calcofi
├── figs
├── src
└── tools
```

3. Make sure that you create an environment with the package versions specified in `environment.yml`. If you are using [Conda](https://docs.conda.io/en/latest/) you can run 

`conda env create -f environment.yml`

from the project root to create the environment from the .yml file and run `conda activate ocean_scales` to activate the environment.

4. If you follow the steps above you should be able to reproduce all figures, by running `python figXX.py` from the `src` directory without having to adjust any paths.

# How to cite this code

If you wish to use the code from this repository, you may cite it as: 

Colosi, Luke V. (2026, October 2026). Source code for: 'Decorrelation Time Scales Variations Across the Coastal-Open Ocean Transition Zone'. Zenodo. (add_link_here) 

