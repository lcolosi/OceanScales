# =============================================================================
# Analytic Decorrelation Scales for Power-Law Red Spectra  
# =============================================================================
#
# Description:
#   Compute the autocorrelation and decorrelation scales of Power-Law red spectra
#   using our  analytic solution. We will compute both for a range of:   
# 
#       (1) Spectral slopes, 
#       (2) Record durations.
# 
#   Additionally, we compute the decorrelation scale from the analytic 
#   autocorrelation solution using the "maximizing I(\tau)" method. 
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
import mpmath as mp
import xarray as xr
from tqdm import tqdm

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox 
from autocorr import autocorrelation_analytic, decorrelation_scale_analytic, compute_decor_scale

# -----------------------------------------------------------------------------
# Set parameters for Autocorrelation calculation
# -----------------------------------------------------------------------------

# Set whether to compute the biased or unbiased estimate 
option_biased = True

# Set sampling interval and record duration parameters (units: days)
dt = 1 / 24                                         
T_ac  = np.array([1/12, 2/12, 3/12, 4/12, 6/12, 8/12, 1]) * (365)  

# Set maximum and minimum frequency in units of cpd
fmax = 1 / (2*dt)                                         
fmin_values_ac = 1 / T_ac               

# Set spectral slope values 
alpha_values_ac = np.arange(0, 3 + 0.05, 0.05)

# Convert time to units of months 
days_per_month = 365 / 12 
T_months_ac = T_ac / days_per_month

# Set time lag (units: days)
tau = np.arange(0, max(T_ac) + dt/2, dt)

# Set precision for complex special functions
mp.dps = 25

# -----------------------------------------------------------------------------
# Compute autocorrelation 
# -----------------------------------------------------------------------------

# Set dimension lengths
nfmin, nalpha, ntau = len(fmin_values_ac), len(alpha_values_ac), len(tau)

# Initialize arrays
rho = np.ma.masked_all((ntau, nfmin, nalpha))
T_tilde_maxI = np.full((nfmin, nalpha), np.nan)
M_lag_maxI = np.full((nfmin, nalpha), np.nan)

# Loop through f_min (record durations)
for i, fmin in enumerate(tqdm(fmin_values_ac,desc="Computing Analytic Autocorrelation",unit="duration")):

    # Only use lags that fall within this record duration
    valid_lags = tau <= T_ac[i]

    # Set nonnegative lag vector 
    tau_valid = tau[valid_lags]

    # Loop through spectral slope
    for j, alpha in enumerate(alpha_values_ac):

        # Compute the autocorrelation function (evaulating only at positive lags)
        _, autocorr_pos, _ = autocorrelation_analytic(tau_valid[1:], 
                                                      fmin, 
                                                      fmax, 
                                                      alpha, 
                                                      biased=option_biased, 
                                                      record_duration=T_ac[i]
                                                     )

        # Add exact zero-lag autocorrelation explicitly
        autocorr = np.concatenate(([1.0],autocorr_pos))

        # Construct the full autocorrelation with negative and positive lags
        autocorr_full = np.concatenate((autocorr[:0:-1], autocorr))
        lag_full      = np.concatenate((-tau[valid_lags][:0:-1], tau_valid))

        # Compute the decorrelation scale from the analytic autocorrelation 
        # using the "maximizing I(tau)" method.
        T_tilde_maxI[i,j], M_lag_maxI[i,j] = compute_decor_scale(autocorr_full,lag_full)

        # Save nonnegative autocorrelation in masked array
        rho[valid_lags, i, j] = autocorr

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
for i, fmin in enumerate(tqdm(fmin_values_ds,desc="Computing Analytic decorrelation scale",unit="duration")):

    # Loop through spectral slope
    for j, alpha in enumerate(alpha_values_ds):

        # Compute the decorrelation scale (units: days)
        T_tilde[i,j], _ = decorrelation_scale_analytic(fmin, fmax, alpha) 

# -----------------------------------------------------------------------------
# Save data in a netcdf file 
# -----------------------------------------------------------------------------

# Set label for autocorrelation estimate  
autocorr_estimate = "biased" if option_biased else "unbiased"

# --- Autocorrelation --- # 
autocorr = xr.DataArray(data=rho,
                        dims=['lag','duration_ac','slope_ac'],
                        coords=dict(lag=tau,duration_ac=T_ac,slope_ac=alpha_values_ac),
                        attrs=dict(
                            description=(f'Analytic ({autocorr_estimate}) autocorrelation solutions for ' + 
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

decor_scale_maxI = xr.DataArray(data=T_tilde_maxI,
                                dims=['duration_ac','slope_ac'],
                                coords=dict(duration_ac=T_ac,slope_ac=alpha_values_ac),
                                attrs=dict(
                                    description=(f'Decorrelation scale computed from ' + 
                                                 'the analytic autocorrelation solution ' + 
                                                 'using the maximum I(tau) method for ' + 
                                                 'a range of window durations and spectral slopes.'),
                                    units='days'
                                )
)

M_lag = xr.DataArray(data=M_lag_maxI,
                    dims=['duration_ac','slope_ac'],
                    coords=dict(duration_ac=T_ac,slope_ac=alpha_values_ac),
                    attrs=dict(
                        description=(f'Lag number corresponding to the maximum symmetric integral. Computed from ' + 
                                      'the analytic autocorrelation solution ' + 
                                      'using the maximum I(tau) method for ' + 
                                      'a range of window durations and spectral slopes.'),
                        units='days'
                    )
)

# Create data set from data arrays 
data = xr.Dataset({'autocorr':autocorr,'decor_scale':decor_scale,'decor_scale_maxI':decor_scale_maxI,'M_lag':M_lag})

# Set path to processed data 
PATH_processed = PATH_data / 'analytic' 

# Set file path for saving the netcdf file
file_path = PATH_processed / f"analytic_decor_scale_{autocorr_estimate}.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
