# =============================================================================
# Compute spectrograms across the study domain from MITgcm data 
# =============================================================================
#
# Description:
#   Computes spectrograms at mooring locations from
#   MITgcm data and saves the results to a NetCDF file.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-15
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
from spectra import compute_spectrum1D, spectral_slope, spectral_diags, spectral_uncertainty
from autocorr import segment_time_series
from lsf import unweighted_lsf
from filter import gaussian_low_pass_filter

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "salt", "density", "uvel", "vvel", or "ssh".
# - option_depth: Depth at which the decorrelation time scale is computed
#                 (units: meters). Not used for SSH.
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_harmonics : Specify the number of seasonal cycle harmonics to fit.
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: "detrend", "demean", or None
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
option_detrend_seg = "detrend"

# Set time and space parameters
seconds_per_day  = 24 * 60 * 60
dt               = 3600    
T_annual         = 365.25 * seconds_per_day  
segment_overlap  = 0.5                                        
segment_duration = 0.5   
depth_lim        = -220 
f_cut            = 1 / (30 * seconds_per_day)
fmin_slope       = (1/182.5) / seconds_per_day
fmax_slope       = (1/4) / seconds_per_day 

# Parameter verification
if option_data not in ("temp", "salt", "density", "uvel", "vvel", "ssh"):
    raise ValueError(
        f"Invalid option_data: {option_data}. "
        "Choose 'temp', 'salt', 'density', 'uvel', 'vvel', or 'ssh'."
    )
if option_interannual not in ("linear", "gaussian"):
    raise ValueError(
        f"Invalid option_interannual: {option_interannual}. "
        "Choose 'linear' or 'gaussian'."
    )

if option_detrend_seg not in ("detrend", "demean", None):
    raise ValueError(
        "option_detrend_seg must be 'detrend', 'demean', or None."
    )

# Label segment processing method 
seg_proc = "none" if option_detrend_seg is None else option_detrend_seg

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

# -----------------------------------------------------------------------------
# Compute depth spectrogram and their diagnostics
# -----------------------------------------------------------------------------

# Create reference segments to determine the segment length
segments_ref = segment_time_series(time_dt, 
                                   np.zeros(ntime), 
                                   duration=segment_duration, 
                                   overlap=segment_overlap,
                                   )

# Check if no segments were generated
if len(segments_ref) == 0:
    raise ValueError("No segments were generated.")

# Check that all segments have the same number of samples
segment_lengths = {len(dseg) for _, dseg in segments_ref}

if len(segment_lengths) != 1:
    raise ValueError(
        "All segments must contain the same number of samples."
    )

# Obtain the dimensions of the segmented time series
nseg = len(segments_ref)
ntime_seg = segment_lengths.pop()

# Compute expected cyclic frequency vector for each segment
f = np.fft.rfftfreq(ntime_seg, d=dt)

# Set number of non-negative frequency bins
nfreq = len(f)

# Initialize arrays 
psd_mean        = np.ma.masked_all((nlat,nlon,nfreq))
psd_CI          = np.ma.masked_all((nlat,nlon,nfreq,2))
spec_slope      = np.ma.masked_all((nlat,nlon))
spec_slope_stde = np.ma.masked_all((nlat,nlon))
moments         = np.ma.masked_all((nlat,nlon,4))
fve             = np.ma.masked_all((nlat,nlon,2))
mean_period     = np.ma.masked_all((nlat,nlon))

# Loop through longitude 
for ilon in tqdm(range(nlon), desc="Computing power spectra", unit="lon"):

    # Loop through latitude
    for ilat in range(nlat):
        
        # Set the time series 
        data_ts = data_res[:,ilat,ilon]

        # Skip grid points containing only masked data
        if np.ma.getmaskarray(data_ts).all():
            continue

        # Error program if a partially masked time series is present. 
        if np.ma.getmaskarray(data_ts).any():
            raise ValueError(
                f"Partially masked time series at lat index {ilat}, "
                f"lon index {ilon}."
            )

        # Segment the time series 
        segments = segment_time_series(time_dt, 
                                       data_ts, 
                                       duration=segment_duration, 
                                       overlap=segment_overlap,
                                       )

        # Initialize arrays
        psd_seg = np.ma.masked_all((nseg,nfreq),dtype=float)

        # Loop through segments
        for iseg, (tseg, dseg) in enumerate(segments):

            # Check segment length
            if len(dseg) != ntime_seg:
                raise ValueError(
                    "Segment length differs from the reference segment length."
                )

            # Compute the hanning-windowed power spectrum 
            psd_seg[iseg,:], f_i, *_  = compute_spectrum1D(dseg, 
                                                            dt, 
                                                            1, 
                                                            'cyclic', 
                                                            segment_preprocess = seg_proc
                                                            ) 

            # Check frequency grid
            if not np.array_equal(f_i, f):
                raise ValueError(
                    "Frequency grid differs between segments."
                )

        # Compute the mean power spectral density function 
        psd_mean[ilat,ilon,:] = np.ma.mean(psd_seg, axis=0)

        # Compute the 95% confidence interval 
        psd_CI[ilat,ilon,:,:] = spectral_uncertainty(alpha=0.05,
                                                        psd=psd_mean[ilat,ilon,:],
                                                        estimator="fft",
                                                        nseg=nseg,
                                                        )

        # Compute the spectral slope 
        spec_slope[ilat,ilon], spec_slope_stde[ilat,ilon], *_ = spectral_slope(f,
                                                                               psd_mean[ilat,ilon,:], 
                                                                               fmin_slope, 
                                                                               fmax_slope,
                                                                              )

        # Compute spectral moments, FVE and mean period
        moments[ilat,ilon,:], fve[ilat,ilon,:], mean_period[ilat,ilon] = spectral_diags(psd_mean[ilat,ilon,:], 
                                                                                        f, 
                                                                                        f_cutoff=f_cut
                                                                                        )

# Convert mean period to units of days
mean_period_days = mean_period/(24*60*60) 

# Convert frequency and psd to units of cycles per day 
f_cpd      = f * seconds_per_day
psd_cpd    = psd_mean / seconds_per_day 
psd_CI_cpd = psd_CI / seconds_per_day 

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# Set coordinates
CI_coord      = ['lower', 'upper']
moments_coord = ["m0", "m1", "m2", "m3"]
fve_coord     = ['low_FVE', 'high_FVE']

# --- Spectral analysis diagnostics --- # 
PSD = xr.DataArray(data=psd_cpd,
                           dims=['lat','lon','freq'],
                           coords=dict(lat=lat,lon=lon,freq=f_cpd),
                           attrs=dict(
                               description=('Power Spectral Density depth spectrogram at the CCE ' +
                                            'mooring locations.'),
                               units='variance/cycles/day'
                           )
)

PSD_CI = xr.DataArray(data=psd_CI_cpd,
                           dims=['lat','lon','freq','CI_coord'],
                           coords=dict(lat=lat,lon=lon,freq=f_cpd,CI_coord=CI_coord),
                           attrs=dict(
                               description=('95% confidence interval for the power spectral ' +
                                            'density depth spectrogram at CCE mooring ' + 
                                            'locations.'),
                               units='variance/cycles/day'
                           )
)

SPEC_slope = xr.DataArray(data=spec_slope,
                           dims=['lat','lon'],
                           coords=dict(lat=lat,lon=lon),
                           attrs=dict(
                               description=('Spectral Slope for the power spectral ' +
                                            'density depth spectrogram at CCE mooring ' + 
                                            'locations.'),
                               units='unitless'
                           )
)

SPEC_slope_stde = xr.DataArray(data=spec_slope_stde,
                        dims=['lat','lon'],
                        coords=dict(lat=lat,lon=lon),
                        attrs=dict(
                            description=('Standard error of the Spectral Slope ' +
                                         'for the power spectral density depth spectrogram' + 
                                         'at the CCE mooring locations.'),
                            units='unitless'
                        )
)

MOMENTS = xr.DataArray(data=moments,
                   dims=['lat','lon','moments_coord'],
                   coords=dict(lat=lat,lon=lon,moments_coord=moments_coord),
                   attrs=dict(
                       description=('First 4 moments (zeroth to third) of the ' +
                                    'power spectral density depth spectrogram.')
                    )
)

FVE = xr.DataArray(data=fve,
                   dims=['lat','lon','fve_coord'],
                   coords=dict(lat=lat,lon=lon,fve_coord=fve_coord),
                   attrs=dict(
                       description=('Fraction of variance explained by the ' +
                                    'low and high frequency bands.'),
                       units='precent'
                    )
)

MEAN_PERIOD_days = xr.DataArray(data=mean_period_days,
                        dims=['lat','lon'],
                        coords=dict(lat=lat,lon=lon),
                        attrs=dict(
                            description=('Mean Period in units days ' +
                                         'for the power spectral density depth spectrogram' + 
                                         'at the CCE mooring locations.'),
                            units='days'
                        )
)

# Create data set from data arrays 
data = xr.Dataset({'PSD':PSD,'PSD_CI':PSD_CI,'SPEC_slope':SPEC_slope,'SPEC_slope_stde':SPEC_slope_stde,'MOMENTS':MOMENTS,'FVE':FVE,'MEAN_PERIOD_days':MEAN_PERIOD_days})

# Set global variables to document the processing parameters used 
data.attrs.update({
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

# Set the range of periods for the spectral slope calculation 
Tmin_slope = 1 / (fmax_slope * seconds_per_day)
Tmax_slope = 1 / (fmin_slope * seconds_per_day)

# Set file path for saving the netcdf file
file_path = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_reg_{option_interannual}_slope_{Tmax_slope}_{Tmin_slope}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')



