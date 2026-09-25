# =============================================================================
# Analytic Decorrelation Scales for Power-Law Red Spectra  
# =============================================================================
#
# Description:
#   Compute the autocorrelation and decorrelation scales of Power-Law red spectra
#   using our  analytic solution. We will compute an autocorrelation
#   for a range of:   
# 
#       (1) Spectral slopes, 
#       (2) Record durations.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-22
# =============================================================================

# Import libraries 
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import xarray as xr

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox 
from autocorr import autocorrelation_analytic, decorrelation_scale_analytic

# -----------------------------------------------------------------------------
# Set parameters for Autocorrelation calculation
# -----------------------------------------------------------------------------

# Set sampling interval and record duration parameters (units: days)
dt = 1/24                                         
T_ac  = np.array([3/12, 6/12, 8/12, 1]) * (365)  

# Set maximum and minimum frequency in units of cpd
fmax = 1/dt                                         
fmin_values_ac = 1/T_ac               

# Set spectral slope values 
alpha_values_ac = [1, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0]

# Convert time to units of months 
days_per_month = 365 / 12 
T_months_ac = T_ac / days_per_month

# Set time lag (units: days)
tau = np.linspace(dt, max(T_ac), 200) 

# Set precision for complex special functions
mp.dps = 25

# -----------------------------------------------------------------------------
# Compute autocorrelation 
# -----------------------------------------------------------------------------

# Set dimension lengths
nfmin, nalpha, ntau = len(fmin_values_ac), len(alpha_values_ac), len(tau)

# Initialize array
rho = np.zeros((ntau, nfmin, nalpha))

# Loop through f_min 
for i, fmin in enumerate(fmin_values_ac):

    # Loop through spectral slope
    for j, alpha in enumerate(alpha_values_ac):

        # Compute the autocorrelation function 
        _, rho[:,i,j], _ = autocorrelation_analytic(tau, fmin, fmax, alpha)

# -----------------------------------------------------------------------------
# Set parameters for decorrelation scale calculation
# -----------------------------------------------------------------------------

# Set sampling interval and record duration parameters (units: days)
dt = 1/24                                                     
T_ds = np.flipud(np.arange(0.025, 1 + 0.025, 0.025) * (365))  

# Set maximum and minimum frequency in units of cpd
fmax = 1/dt                                         
fmin_values_ds = 1/T_ds               

# Set spectral slope values 
alpha_values_ds = np.arange(0.05,5+0.05,0.05)

# Convert time to units of months 
days_per_month = 365 / 12 
T_months_ds = T_ds / days_per_month

# -----------------------------------------------------------------------------
# Compute decorrelation scale 
# -----------------------------------------------------------------------------

# Set parameters
nfmin, nalpha = len(fmin_values_ds), len(alpha_values_ds)

# Initialize array
T_tilde = np.zeros((nfmin, nalpha))

# Loop through f_min
for i, fmin in enumerate(fmin_values_ds):

    # Loop through spectral slope
    for j, alpha in enumerate(alpha_values_ds):

        # Compute the decorrelation scale (units: days)
        T_tilde[i,j], _ = decorrelation_scale_analytic(fmin, fmax, alpha) 

# -----------------------------------------------------------------------------
# Save data in a netcdf file 
# -----------------------------------------------------------------------------

# --- Autocorrelation --- # 
autocorr = xr.DataArray(data=rho,
                        dims=['lag','duration_ac','slope_ac'],
                        coords=dict(lag=tau,duration_ac=T_ac,slope_ac=alpha_values_ac),
                        attrs=dict(
                            description=(f'Analytic autocorrelation solutions for ' + 
                                        'a range of window durations and spectral slopes.'),
                            units='unitless'
                        )
)

# --- Decorrelation Scales --- #
decor_scale = xr.DataArray(data=T_tilde,
                            dims=['duration_ds','slope_ds'],
                            coords=dict(duration_ds=T_ds,slope_ds=alpha_values_ds),
                            attrs=dict(
                                description=(f'Analytic decorrelation scale solutions for ' + 
                                            'a range of window durations and spectral slopes.'),
                                units='days'
                            )
)

# Create data set from data arrays 
data = xr.Dataset({'autocorr':autocorr,'decor_scale':decor_scale})

# Set path to processed data 
PATH_processed = PATH_data / 'analytic' 

# Set file path for saving the netcdf file
file_path = PATH_processed / f"analytic_decor_scale.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
