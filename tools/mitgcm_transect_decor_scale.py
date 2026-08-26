# =============================================================================
# Compute CalCOFI line 80.0 Transect decorrelation time scales from MITgcm data 
# =============================================================================
#
# Description:
#   Computes CalCOFI line 80.0 transect decorrelation time scales and their
#   uncertainty at mooring locations from MITgcm data and saves the results
#   to a NetCDF file.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-25
# =============================================================================

# Import libraries 
import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
from netCDF4 import Dataset, num2date
from datetime import datetime
from tqdm import tqdm

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import analysis functions 
from autocorr import compute_autocorr_biased, compute_decor_scale, compute_decor_scale_unc, segment_time_series
from lsf import unweighted_lsf, detrend, compute_fve
from filter import gaussian_low_pass_filter  

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density", "uvel", or "vvel".
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_harmonics : Specify the number of seasonal cycle harmonics to fit.
# - option_detrend_Sseg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - dt: Sampling interval of the model data (units: seconds). 
# - T_annual: Specifies the annual cycle period (one Julian year) in units of seconds. 
# - segment_overlap: Specifies the fractional overlap between segments 
#                    (e.g., 0.75 for 75% overlap).
# - segment_duration: Specifies the length of each segment in years.
# - depth_lim : Specifies the deepest depth to preform analysis. 
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_harmonics   = 2      
option_detrend_seg = False

# Set time and space parameters
dt               = 3600    
T_annual         = 365.25*(24)*(60)*(60)    
segment_overlap  = 0.5                                        
segment_duration = 1   
depth_lim        = -220 

# Parameter verification
if option_data not in ("temp", "salt", "density", "uvel", "vvel"):
    raise ValueError(
        f"Invalid option_data: {option_data}. "
        "Choose 'temp', 'salt', 'density', 'uvel', or 'vvel'."
    )
if option_interannual not in ("linear", "gaussian"):
    raise ValueError(
        f"Invalid option_interannual: {option_interannual}. "
        "Choose 'linear' or 'gaussian'."
    )

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"
