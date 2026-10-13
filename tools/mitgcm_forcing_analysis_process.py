# =============================================================================
# Processing MITgcm data for the Forcing Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the regional 
#   decorrelation time scale analysis. These include: 
# 
#       (1) Conservative Temperature
#       (2) Absolute Salinity
#       (3) Potential Density (referenced to the surface)
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-11
# =============================================================================

# Import python libraries 
import os
from pathlib import Path
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import gsw


