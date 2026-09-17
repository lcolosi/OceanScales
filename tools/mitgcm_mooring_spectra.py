# =============================================================================
# Compute spectrograms at mooring locations from MITgcm data 
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
#                Options: "temp", "salt", "density", "uvel", or "vvel".
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_harmonics : Specify the number of seasonal cycle harmonics to fit.
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - seconds_per_day: Number of seconds in a day. 
# - dt: Sampling interval of the model data (units: seconds). 
# - T_annual: Specifies the annual cycle period (one Julian year) in units of seconds. 
# - segment_overlap: Specifies the fractional overlap between segments 
#                    (e.g., 0.75 for 75% overlap). NOTE: The confidence interval 
#                    calculation assumes 50% overlap, so if this is changed from 0.5, 
#                    the CI calculation must be adjusted accordingly.
# - segment_duration: Specifies the length of each segment in years.
# - depth_lim : Specifies the deepest depth to preform analysis. 
# - f_cut: Specifies the cutoff frequency in units of cpd for the Fraction of
#          variance explained.
# - fmin_slope: Lower frequency limit for the spectral slope calculation.  
# - fmax_slope: Upper frequency limit for the spectral slope calculation. 
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_harmonics   = 2      
option_detrend_seg = True

# Set time and space parameters
seconds_per_day  = 24 * 60 * 60
dt               = 3600    
T_annual         = 365.25 * seconds_per_day  
segment_overlap  = 0.5                                        
segment_duration = 0.5   
depth_lim        = -220 
f_cut            = 1 / (14 * seconds_per_day)
fmin_slope       = (1/30) / seconds_per_day
fmax_slope       = (1/14) / seconds_per_day 

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

# -----------------------------------------------------------------------------
# Load MITgcm data
# -----------------------------------------------------------------------------

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"

# Set NetCDF variable names
variable_names = {
    "temp": "CTemp",
    "sal": "ASal",
    "density": "SIG",
    "uvel": "u",
    "vvel": "v",
}

# Set filename based on selected data type
if option_data in ("temp", "sal", "density"):
    filename = (
        PATH_processed
        / f"mitgcm_proc_density_hrly_mooring.nc"
    )
elif option_data in ("uvel", "vvel"):
    filename = (
        PATH_processed
        / f"mitgcm_proc_vel_hrly_mooring.nc"
    )
else:
    raise ValueError(f"Invalid option_data: {option_data}")

# Load NetCDF data
with Dataset(filename, "r") as nc:
    site  = nc.variables["site"][:]
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

# Select depth levels shallower than the depth limit 
idx_depth = depth >= depth_lim

# Extract depth and data from the specified depth levels
depth = depth[idx_depth]
data = data[:, :, idx_depth]

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
nsite,ntime,ndepth = np.shape(data)

# Initialize arrays 
fit      = np.ma.masked_all((nsite,ntime,ndepth))
data_res = np.ma.masked_all((nsite,ntime,ndepth))

# Loop over each site
for isite in tqdm(range(nsite), desc="Computing Least-Squares Fit", unit="mooring site"):

    # Loop over each depth
    for idepth in range(ndepth):

        # Set the time series 
        data_ts = data[isite,:,idepth]

        # Skip time series containing only masked data
        if np.ma.getmaskarray(data_ts).all():
            continue

        # Compute seasonal harmonic fit
        fit[isite,:,idepth], _, _, _ = unweighted_lsf(data_ts, 
                                                    time_elapsed, 
                                                    parameters=option_harmonics, 
                                                    freqs=w, 
                                                    sigma=None, 
                                                    linear_trend=linear_trend,
                                                    )
    
        # Compute the residual time series 
        data_res[isite,:,idepth] = data_ts - fit[isite,:,idepth]

# Apply Gaussian low-pass filtering when selected
if option_interannual == 'gaussian': 

    # Initialize interannual variability array
    data_interannual = np.ma.masked_all((nsite,ntime,ndepth))

    # Loop over each site
    for isite in tqdm(range(nsite), desc="Low-pass Filtering Time Series", unit="mooring site"):

        # Loop over each depth
        for idepth in range(ndepth):

            # Set the time series 
            data_ts = np.ma.masked_invalid(data[isite,:,idepth])

            # Remove the time mean
            data_anomaly = data_ts - np.ma.mean(data_ts)

            # Skip grid points containing only masked data
            if np.ma.getmaskarray(data_anomaly).all():
                continue

            # Estimate interannual variability using 365-day FWHM Gaussian low-pass
            data_interannual[isite,:,idepth] = gaussian_low_pass_filter(data_anomaly,
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
psd_mean        = np.ma.masked_all((nsite,ndepth,nfreq))
psd_CI          = np.ma.masked_all((nsite,ndepth,nfreq,2))
spec_slope      = np.ma.masked_all((nsite,ndepth))
spec_slope_stde = np.ma.masked_all((nsite,ndepth))
moments         = np.ma.masked_all((nsite,ndepth,4))
fve             = np.ma.masked_all((nsite,ndepth,2))
mean_period     = np.ma.masked_all((nsite,ndepth))

# Loop over each site
for isite in tqdm(range(nsite), desc="Computing power spectra", unit="mooring site"):

    # Loop over each depth
    for idepth in range(ndepth):

        # Set the time series 
        data_ts = data_res[isite,:,idepth]

        # Skip grid points containing only masked data
        if np.ma.getmaskarray(data_ts).all():
            continue

        # Error program if a partially masked time series is present. 
        if np.ma.getmaskarray(data_ts).any():
            raise ValueError(
                f"Partially masked time series at depth index {idepth}, "
                f"site index {isite}."
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
        psd_mean[isite,idepth,:] = np.ma.mean(psd_seg, axis=0)

        # Compute the 95% confidence interval 
        psd_CI[isite,idepth,:,:] = spectral_uncertainty(alpha=0.05,
                                                        psd=psd_mean[isite,idepth,:],
                                                        estimator="fft",
                                                        nseg=nseg,
                                                        )

        # Compute the spectral slope 
        spec_slope[isite,idepth], spec_slope_stde[isite,idepth], *_ = spectral_slope(f,
                                                                                     psd_mean[isite,idepth,:], 
                                                                                     fmin_slope, 
                                                                                     fmax_slope,
                                                                                     )

        # Compute spectral moments, FVE and mean period
        moments[isite,idepth,:], fve[isite,idepth,:], mean_period[isite,idepth] = spectral_diags(psd_mean[isite,idepth,:], 
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
                           dims=['site','depth','freq'],
                           coords=dict(site=site,depth=depth,freq=f_cpd),
                           attrs=dict(
                               description=('Power Spectral Density depth spectrogram at the CCE ' +
                                            'mooring locations.'),
                               units='variance/cycles/day'
                           )
)

PSD_CI = xr.DataArray(data=psd_CI_cpd,
                           dims=['site','depth','freq','CI_coord'],
                           coords=dict(site=site,depth=depth,freq=f_cpd,CI_coord=CI_coord),
                           attrs=dict(
                               description=('95% confidence interval for the power spectral ' +
                                            'density depth spectrogram at CCE mooring ' + 
                                            'locations.'),
                               units='variance/cycles/day'
                           )
)

SPEC_slope = xr.DataArray(data=spec_slope,
                           dims=['site','depth'],
                           coords=dict(site=site,depth=depth),
                           attrs=dict(
                               description=('Spectral Slope for the power spectral ' +
                                            'density depth spectrogram at CCE mooring ' + 
                                            'locations.'),
                               units='unitless'
                           )
)

SPEC_slope_stde = xr.DataArray(data=spec_slope_stde,
                        dims=['site','depth'],
                        coords=dict(site=site,depth=depth),
                        attrs=dict(
                            description=('Standard error of the Spectral Slope ' +
                                         'for the power spectral density depth spectrogram' + 
                                         'at the CCE mooring locations.'),
                            units='unitless'
                        )
)

MOMENTS = xr.DataArray(data=moments,
                   dims=['site','depth','moments_coord'],
                   coords=dict(site=site,depth=depth,moments_coord=moments_coord),
                   attrs=dict(
                       description=('First 4 moments (zeroth to third) of the ' +
                                    'power spectral density depth spectrogram.')
                    )
)

FVE = xr.DataArray(data=fve,
                   dims=['site','depth','fve_coord'],
                   coords=dict(site=site,depth=depth,fve_coord=fve_coord),
                   attrs=dict(
                       description=('Fraction of variance explained by the ' +
                                    'low and high frequency bands.'),
                       units='precent'
                    )
)

MEAN_PERIOD_days = xr.DataArray(data=mean_period_days,
                        dims=['site','depth'],
                        coords=dict(site=site,depth=depth),
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

# Set file path for saving the netcdf file
file_path = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')



