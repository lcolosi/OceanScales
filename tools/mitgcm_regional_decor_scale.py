# =============================================================================
# Compute regional decorrelation time scales from MITgcm data 
# =============================================================================
#
# Description:
#   Computes regional decorrelation time scales and their uncertainty from
#   MITgcm data and saves the results to a NetCDF file.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-18
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

# Import plotting toolbox 
from autocorr import compute_autocorr_biased, compute_decor_scale, compute_decor_scale_unc, segment_time_series, compute_decor_scale_weighted
from lsf import unweighted_lsf, detrend
from filter import gaussian_low_pass_filter

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density", "uvel", "vvel", or "ssh".
#
# - option_depth: Depth at which the decorrelation time scale is computed
#                 (units: meters). Not used for SSH.
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_harmonics : Specify the number of seasonal cycle harmonics to fit.
# - dt: Sampling interval of the model data (units: seconds). 
# - T_annual: Specifies the annual cycle period (one Julian year) in units of seconds. 
# - segment_overlap: Specifies the fractional overlap between segments 
#                    (e.g., 0.75 for 75% overlap).
# - segment_duration: Specifies the length of each segment in years.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_depth       = 9   
option_interannual = 'linear' 
option_harmonics   = 2      

# Set time and space parameters
dt               = 3600    
T_annual         = 365.25*(24)*(60)*(60)    
segment_overlap  = 0.5                                        
segment_duration = 1    

# -----------------------------------------------------------------------------
# Load MITgcm data
# -----------------------------------------------------------------------------

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Set NetCDF variable names
variable_names = {
    "temp": "CTemp",
    "sal": "ASal",
    "density": "SIG",
    "uvel": "u",
    "vvel": "v",
    "ssh": "ssh",
}

# Set filename based on selected data type
if option_data in ("temp", "sal", "density"):
    filename = (
        PATH_processed
        / f"mitgcm_proc_density_hrly_reg_depth_{option_depth}m.nc"
    )
elif option_data in ("uvel", "vvel"):
    filename = (
        PATH_processed
        / f"mitgcm_proc_vel_hrly_reg_depth_{option_depth}m.nc"
    )
elif option_data == "ssh":
    filename = PATH_processed / "mitgcm_proc_ssh_hrly_reg.nc"
else:
    raise ValueError(f"Invalid option_data: {option_data}")

# Load NetCDF data
with Dataset(filename, "r") as nc:
    lon = nc.variables["lon"][:]
    lat = nc.variables["lat"][:]

    if option_data != "ssh":
        depth = nc.variables["Depth"][:]

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
ntime,nlat,nlon = np.shape(data)

# Initialize arrays 
fit      = np.ma.masked_all((ntime,nlat,nlon))
data_res = np.ma.masked_all((ntime,nlat,nlon))

# Loop through longitude 
for ilon in tqdm(range(nlon), desc="Computing Least-Squares Fit", unit="lon"):

    # Loop through latitude
    for ilat in range(nlat):

        # Set the time series 
        data_ts = data[:,ilat,ilon]

        # Skip grid points containing only masked data
        if np.ma.getmaskarray(data_ts).all():
            continue

        # Compute seasonal harmonic fit
        fit[:,ilat,ilon], _, _, _ = unweighted_lsf(data_ts, 
                                                   time_elapsed, 
                                                   parameters=option_harmonics, 
                                                   freqs=w, 
                                                   sigma=None, 
                                                   linear_trend=linear_trend,
                                                   )
    
        # Compute the residual time series 
        data_res[:,ilat,ilon] = data_ts - fit[:,ilat,ilon]

# Apply Gaussian low-pass filtering when selected
if option_interannual == 'gaussian': 

    # Initialize interannual variability array
    data_interannual = np.ma.masked_all((ntime, nlat, nlon))

    # Loop through longitude 
    for ilon in tqdm(range(nlon), desc="Low-pass Filtering Time Series", unit="lon"):

        # Loop through latitude
        for ilat in range(nlat):

            # Set the time series 
            data_ts = np.ma.masked_invalid(data[:, ilat, ilon])

            # Skip grid points containing only masked data
            if np.ma.getmaskarray(data_ts).all():
                continue

            # Estimate interannual variability using 365-day FWHM Gaussian low-pass
            data_interannual[:, ilat, ilon] = gaussian_low_pass_filter(data_ts,
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

# Compute the fraction of variance explained by the interannual and season model


# -----------------------------------------------------------------------------
# Compute decorrelation time scales and their uncertainty
# -----------------------------------------------------------------------------

# Find a valid grid point
valid_grid = ~np.ma.getmaskarray(data_res).all(axis=0)
ilat_valid, ilon_valid = np.argwhere(valid_grid)[0]

# Segment a single time series 
segments = segment_time_series(time, 
                               data_res[:,ilat_valid, ilon_valid], 
                               duration=segment_duration, 
                               overlap=segment_overlap,
                               )

# Obtain the dimensions of the segmented time series
nseg = len(segments)
ntime_seg = len(segments[0][0])

# Initialize arrays 
Lt             = np.ma.masked_all((nlat,nlon))
Lt_stdm        = np.ma.masked_all((nlat,nlon))
Lt_std         = np.ma.masked_all((nlat,nlon))

# Loop through longitude 
for ilon in tqdm(range(nlon), desc="Computing Decorrelation Scales", unit="lon"):

    # Loop through latitude
    for ilat in range(nlat):
        
        # Set the time series 
        data_ts = data_res[:,ilat,ilon]

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
            
            # Detrend data record 
            data_dt = detrend(dseg, time_elapsed_seg, mean = 0)

            # Compute autocorrelation function
            autocorr_seg[iseg,:], time_lag = compute_autocorr_biased(data_dt, time_elapsed_seg)

        # Compute the mean autocorrelation function 
        autocorr_mean = np.ma.mean(autocorr_seg, axis=0)

        # Compute the decorrelation scale of the mean autocorrelation 
        Lt[ilat,ilon], M_lag = compute_decor_scale(autocorr_mean,time_lag) 
    
        # Compute the standard error of the decorrelation scale
        Lt_stdm[ilat,ilon], Lt_std[ilat,ilon] = compute_decor_scale_unc(autocorr_mean, 
                                                                        autocorr_seg, 
                                                                        M_lag, 
                                                                        dt, 
                                                                        segment_overlap,
                                                                        )

# Convert time scale to units of days
Lt_days      = Lt/(24*60*60) 
Lt_stdm_days = Lt_stdm/(24*60*60) 
Lt_std_days  = Lt_std/(24*60*60) 

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Decorrelation Time Scales --- # 
decor_scale = xr.DataArray(data=Lt_days,
                           dims=['lat','lon'],
                           coords=dict(lat=lat,lon=lon),
                           attrs=dict(
                               description=('Decorrelation time scale across the Point ' +
                                           'Conception Study region at ' +
                                           str(option_depth) + ' meter depth.'),
                               units='days'
                           )
)

decor_scale_stdm = xr.DataArray(data=Lt_stdm_days,
                           dims=['lat','lon'],
                           coords=dict(lat=lat,lon=lon),
                           attrs=dict(
                               description=('Standard error of the decorrelation time ' +
                                            'scale computed from the mean ' + 
                                            'autocorrelation across the Point ' +
                                            'Conception Study region at ' +
                                            str(option_depth) + 
                                            ' meter depth, accounting approximately ' +
                                            'for dependence between overlapping segments.'),
                               units='days'
                           )
)

decor_scale_std = xr.DataArray(data=Lt_std_days,
                           dims=['lat','lon'],
                           coords=dict(lat=lat,lon=lon),
                           attrs=dict(
                               description=('Standard deviation of the decorrelation time ' +
                                            'scale for individual realizations, ' + 
                                            ' across the Point ' +
                                            'Conception Study region at ' +
                                            str(option_depth) + 
                                            ' meter depth, accounting approximately ' +
                                            'for dependence between overlapping segments.'),
                               units='days'
                           )
)

# Create data set from data arrays 
data = xr.Dataset({'decor_scale':decor_scale,'decor_scale_stdm':decor_scale_stdm, 'decor_scale_std':decor_scale_std})

# Set file path for saving the netcdf file
file_path = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_reg_depth_{option_depth}m_{option_interannual}.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
