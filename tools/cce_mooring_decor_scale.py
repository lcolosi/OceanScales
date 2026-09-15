# =============================================================================
# Compute decorrelation time scales from CCE mooring observation 
# =============================================================================
#
# Description:
#   Computes decorrelation time scales and their uncertainty from CCE mooring 
# observations and saves the results to a NetCDF file.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-10
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
from autocorr import compute_autocorr_biased_masked, compute_decor_scale_masked, compute_decor_scale_unc_masked, segment_time_series
from lsf import unweighted_lsf, detrend, compute_fve
from filter import gaussian_low_pass_filter

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_mooring: Specifies which cce mooring will be processed. 
#                   Options include: "cce1" or "cce2"
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density".
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_harmonics : Specify the number of seasonal cycle harmonics to fit.
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - dt: Sampling interval of the model data (units: seconds). 
# - T_annual: Specifies the annual cycle period (one Julian year) in units of seconds. 
# - segment_overlap: Specifies the fractional overlap between segments 
#                    (e.g., 0.75 for 75% overlap).
# - segment_duration: Specifies the length of each segment in years.
# - depth_lim: Specifies the deepest depth to preform analysis. 
# - norm: Specifies the method used to normalize the autocovariance when 
#         observations are missing.
#
# ------------#

# Set processing parameters
option_mooring     = 'cce2'
option_data        = 'density'    
option_interannual = 'gaussian' 
option_harmonics   = 2      
option_detrend_seg = True

# Set time and space parameters
dt               = 3600    
T_annual         = 365.25*(24)*(60)*(60)    
segment_overlap  = 0.5                                        
segment_duration = 0.5   
depth_lim        = -220 
norm             = "corrected"

# Parameter verification
if option_data not in ("temp", "salt", "density"):
    raise ValueError(
        f"Invalid option_data: {option_data}. "
        "Choose 'temp', 'salt', or 'density'."
    )
if option_interannual not in ("linear", "gaussian"):
    raise ValueError(
        f"Invalid option_interannual: {option_interannual}. "
        "Choose 'linear' or 'gaussian'."
    )

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# -----------------------------------------------------------------------------
# Load cce data
# -----------------------------------------------------------------------------

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "cce" / option_mooring / "processed"

# Set NetCDF variable names
variable_names = {
    "temp": "CTemp",
    "sal": "ASal",
    "density": "SIG",
}

# Set filename based on selected data type
if option_data in ("temp", "sal", "density"):
    filename = (
        PATH_processed
        / f"{option_mooring}_proc_density_hrly_mooring.nc"
    )
else:
    raise ValueError(f"Invalid option_data: {option_data}")

# Load NetCDF data
with Dataset(filename, "r") as nc:
    depth = nc.variables["depth"][:]

    time = num2date(
        nc.variables["time"][:],
        units=nc.variables["time"].units,
    )

    data = nc.variables[variable_names[option_data]][:]

# Convert cftime.DatetimeGregorian to Python datetime objects
time_dt = np.array(
    [
        datetime(
            d.year,
            d.month,
            d.day,
            d.hour,
            d.minute,
            d.second,
        )
        for d in time
    ]
)

# Mask data points previously set to NaN during processing
data = np.ma.masked_invalid(data)

# Select depth levels shallower than the depth limit 
idx_depth = depth >= depth_lim

# Extract depth and data from the specified depth levels
depth = depth[idx_depth]
data = data[:, idx_depth]

# -----------------------------------------------------------------------------
# Remove seasonal and interannual variability from time series
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
# The interannual variability is estimated by applying the Gaussian low-pass filter
# to the original time series, rather than to the residual after removing the 
# annual and semi-annual cycles. This prevents year-to-year variations in the 
# amplitude or phase of the seasonal cycle from being interpreted as interannual 
# variability. The seasonal and interannual components are therefore estimated 
# independently from the original time series before both are removed.
# ------------ # 

# Set the radian frequencies for the seasonal cycle LSF (units: rad/sec)
w = 2 * np.pi * np.arange(1, option_harmonics + 1) / T_annual

# Set option for linear trend
linear_trend = option_interannual == "linear"

# Compute the elapsed time from beginning of time series (units: seconds)
t0 = time[0]
time_elapsed = np.array([(t - t0).total_seconds() for t in time])

# Obtain the dimensions of the longitude and latitude 
ntime,ndepth = np.shape(data)

# Initialize arrays 
fit      = np.ma.masked_all((ntime,ndepth))
data_res = np.ma.masked_all((ntime,ndepth))

# Loop over each depth
for idepth in tqdm(range(ndepth), desc="Computing Least-Squares Fit", unit="depth"):

    # Set the time series 
    data_ts = data[:,idepth]

    # Skip time series containing only masked data
    if np.ma.getmaskarray(data_ts).all():
        continue

    # Compute seasonal harmonic fit
    fit[:,idepth], _, _, _ = unweighted_lsf(data_ts, 
                                            time_elapsed, 
                                            parameters=option_harmonics, 
                                            freqs=w, 
                                            sigma=None, 
                                            linear_trend=linear_trend,
                                           )

    # Compute the residual time series 
    data_res[:,idepth] = data_ts - fit[:,idepth]

# Apply Gaussian low-pass filtering when selected
if option_interannual == 'gaussian': 

    # Initialize interannual variability array
    data_interannual = np.ma.masked_all((ntime,ndepth))

    # Loop over each depth
    for idepth in tqdm(range(ndepth), desc="Low-pass Filtering Time Series", unit="depth"):

        # Set the ith time series 
        data_ts = np.ma.masked_invalid(data[:,idepth])

        # Remove the time mean
        data_anomaly = data_ts - np.ma.mean(data_ts)

        # Skip grid points containing only masked data
        if np.ma.getmaskarray(data_anomaly).all():
            continue

        # Estimate interannual variability using 365-day FWHM Gaussian low-pass
        data_interannual[:,idepth] = gaussian_low_pass_filter(data_anomaly,
                                                              fwhm_days=365,
                                                              dt_hours=1,
                                                              mode='constant',
                                                              truncate=4,
                                                             )

    # Remove seasonal and interannual variability
    data_res = data - fit - data_interannual

# Set the model for the interannual and seasonal cycles 
if option_interannual == 'gaussian': 
    model = fit + data_interannual 
else: 
    model = fit 

# Initialize arrays 
fve = np.ma.masked_all(ndepth)

# Loop over each depth
for idepth in tqdm(range(ndepth), desc="Computing Fraction of Variance Explained", unit="depth"):

    # Set the data and model time series 
    data_ts  = data[:,idepth]
    model_ts = model[:,idepth]

    # Skip grid points containing only masked data
    if np.ma.getmaskarray(data_ts).all():
        continue

    # Compute the fraction of variance explained by the interannual and season model
    fve[idepth] = compute_fve(data_ts, model_ts)

# -----------------------------------------------------------------------------
# Compute decorrelation time scales and their uncertainty
# -----------------------------------------------------------------------------

# Segment a single time series 
segments = segment_time_series(time_dt, 
                               data_res[:,0], 
                               duration=segment_duration, 
                               overlap=segment_overlap,
                               )

# Obtain the dimensions of the segmented time series
nseg = len(segments)
ntime_seg = len(segments[0][0])

# Initialize arrays 
Lt      = np.ma.masked_all(ndepth)
Lt_stdm = np.ma.masked_all(ndepth)
Lt_std  = np.ma.masked_all(ndepth)
Lt_stds = np.ma.masked_all(ndepth)

# Loop over each depth
for idepth in tqdm(range(ndepth), desc="Computing Decorrelation Scales", unit="depth"):

    # Set the time series 
    data_ts = data_res[:,idepth]

    # Skip grid points containing only masked data
    if np.ma.getmaskarray(data_ts).all():
        continue

    # Segment the time series 
    segments = segment_time_series(time_dt, 
                                    data_ts, 
                                    duration=segment_duration, 
                                    overlap=segment_overlap,
                                    )

    # Initialize arrays
    autocorr_seg = np.ma.masked_all((nseg,2*ntime_seg-1))

    # Loop through segments
    for iseg, (tseg, dseg) in enumerate(segments):

        # Compute the elapsed time from beginning of segmented time series
        t0 = tseg[0]
        time_elapsed_seg = np.array([(t - t0).total_seconds() for t in tseg])
        
        # Remove segment-wise mean or linear trend
        if option_detrend_seg: 
            data_dt = detrend(dseg, time_elapsed_seg, mean = 0)
        else: 
            data_dt = dseg - np.ma.mean(dseg)

        # Compute autocorrelation function
        autocorr_seg[iseg,:], time_lag = compute_autocorr_biased_masked(data_dt, time_elapsed_seg, normalization=norm)

    # Compute the mean autocorrelation function 
    autocorr_mean = np.ma.mean(autocorr_seg, axis=0)

    # Compute the decorrelation scale of the mean autocorrelation 
    Lt[idepth], M_lag = compute_decor_scale_masked(autocorr_mean,time_lag) 

    # Compute the standard error of the decorrelation scale
    Lt_stdm[idepth], Lt_std[idepth], Lt_stds[idepth]  = compute_decor_scale_unc_masked(autocorr_mean, 
                                                                                        autocorr_seg, 
                                                                                        M_lag, 
                                                                                        dt, 
                                                                                        segment_overlap,
                                                                                        )

# Convert time scale to units of days
Lt_days      = Lt/(24*60*60) 
Lt_stdm_days = Lt_stdm/(24*60*60) 
Lt_std_days  = Lt_std/(24*60*60) 
Lt_stds_days = Lt_stds/(24*60*60)   


# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Decorrelation Time Scales --- # 
decor_scale = xr.DataArray(data=Lt_days,
                           dims=['depth'],
                           coords=dict(depth=depth),
                           attrs=dict(
                               description=(f'Decorrelation time scale at the {option_mooring.upper()} ' +
                                            'mooring location.'),
                               units='days'
                           )
)

decor_scale_stdm = xr.DataArray(data=Lt_stdm_days,
                           dims=['depth'],
                            coords=dict(depth=depth),
                           attrs=dict(
                               description=('Standard error of the decorrelation time ' +
                                            'scale computed from the mean ' + 
                                            f'autocorrelation at the {option_mooring.upper()}  mooring location, ' +
                                            'accounting approximately ' +
                                            'for dependence between overlapping segments.'),
                               units='days'
                           )
)

decor_scale_std = xr.DataArray(data=Lt_std_days,
                           dims=['depth'],
                           coords=dict(depth=depth),
                           attrs=dict(
                               description=('Standard deviation of the decorrelation time ' +
                                            'scale for individual realizations, ' + 
                                            f'at the {option_mooring.upper()}  mooring location, ' +
                                            'accounting approximately ' +
                                            'for dependence between overlapping segments.'),
                               units='days'
                           )
)

decor_scale_stds = xr.DataArray(data=Lt_stds_days,
                        dims=['depth'],
                        coords=dict(depth=depth),
                        attrs=dict(
                            description=('Standard error of the standard deviatio of the decorrelation time ' +
                                         'scale computed from the mean ' + 
                                         f'autocorrelation at the {option_mooring.upper()}  mooring location, ' + 
                                         'accounting approximately ' +
                                         'for dependence between overlapping segments.'),
                            units='days'
                        )
)

# --- Model Diagnostics --- # 
FVE = xr.DataArray(data=fve,
                   dims=['depth'],
                   coords=dict(depth=depth),
                   attrs=dict(
                       description=('Fraction of variance explained by the ' +
                                    'interannual and seasonal variability.'),
                       units='fractional'
                    )
)

# Create data set from data arrays 
data = xr.Dataset({'decor_scale':decor_scale,'decor_scale_stdm':decor_scale_stdm, 'decor_scale_std':decor_scale_std, 'decor_scale_stds':decor_scale_stds, 'FVE':FVE})

# Set global variables to document the processing parameters used 
data.attrs.update({
    "mooring": option_mooring,
    "variable": option_data,
    "interannual_method": option_interannual,
    "seasonal_harmonics": option_harmonics,
    "segment_duration_years": segment_duration,
    "segment_overlap": segment_overlap,
    "segment_processing": seg_proc,
    "sampling_interval_seconds": dt,
})

# Set segment duration in months
segment_months = int(round(segment_duration * 12))

# Set file path for saving the netcdf file
file_path = PATH_processed / f"{option_mooring}_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')





