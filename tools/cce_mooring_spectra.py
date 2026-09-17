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
from spectra import compute_spectrum1D, spectral_slope, spectral_diags, spectral_uncertainty
from autocorr import segment_time_series
from lsf import unweighted_lsf
from filter import gaussian_low_pass_filter
from processing_utils import longest_masked_gap

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
# - seconds_per_day: Number of seconds in a day. 
# - dt: Sampling interval of the model data (units: seconds). 
# - T_annual: Specifies the annual cycle period (one Julian year) in units of seconds. 
# - segment_overlap: Specifies the fractional overlap between segments 
#                    (e.g., 0.75 for 75% overlap).
# - segment_duration: Specifies the length of each segment in years.
# - depth_lim: Specifies the deepest depth to preform analysis. 
# - max_gap_duration: Specifies the maximum missing data gap within a segment in 
#                     units of seconds. 
# - f_cut: Specifies the cutoff frequency in units of cpd for the Fraction of
#          variance explained.
# - fmin_slope: Lower frequency limit for the spectral slope calculation.  
# - fmax_slope: Upper frequency limit for the spectral slope calculation. 
#
# ------------#

# Set processing parameters
option_mooring     = 'cce2'
option_data        = 'density'    
option_interannual = 'gaussian' 
option_harmonics   = 2      
option_detrend_seg = True

# Set time and space parameters
seconds_per_day  = 24 * 60 * 60
dt               = 3600    
T_annual         = 365.25 * seconds_per_day    
segment_overlap  = 0.5                                        
segment_duration = 0.5   
depth_lim        = -220 
max_gap_duration = (24) * 60 * 60
f_cut            = 1 / (14 * seconds_per_day)
fmin_slope       = (1/30) / seconds_per_day
fmax_slope       = (1/14) / seconds_per_day 

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

# -----------------------------------------------------------------------------
# Compute decorrelation time scales and their uncertainty
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
psd_mean        = np.ma.masked_all((ndepth,nfreq))
psd_CI          = np.ma.masked_all((ndepth,nfreq,2))
spec_slope      = np.ma.masked_all((ndepth))
spec_slope_stde = np.ma.masked_all((ndepth))
moments         = np.ma.masked_all((ndepth,4))
fve             = np.ma.masked_all((ndepth,2))
mean_period     = np.ma.masked_all((ndepth))
nseg_used       = np.zeros(ndepth, dtype=int)
longest_gap     = np.zeros(ndepth, dtype=float)

# Loop over each depth
for idepth in tqdm(range(ndepth), desc="Computing power spectra", unit="depth"):

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
    psd_seg = np.ma.masked_all((nseg,nfreq),dtype=float)

    # Initialize number of usable segments
    nseg_valid = 0

    # Loop through segments
    for iseg, (tseg, dseg) in enumerate(segments):

        # Check segment length
        if len(dseg) != ntime_seg:
            raise ValueError(
                "Segment length differs from the reference segment length."
            )

        # Obtain mask array for the time series 
        mask = np.ma.getmaskarray(dseg)

        # Skip segment if all data are missing
        if mask.all():
            continue

        # Determine longest missing-data gap
        gap_duration, _ = longest_masked_gap(mask, dt)

        # Save longest gap encountered at this depth
        longest_gap[idepth] = max(longest_gap[idepth], gap_duration)

        # Skip segment if gap exceeds threshold
        if gap_duration > max_gap_duration:
            continue

        # Compute the elapsed time from beginning of segmented time series
        t0 = tseg[0]
        time_elapsed_seg = np.array([(t - t0).total_seconds() for t in tseg])

        # Interpolate only if there are masked data points 
        if np.any(mask): 
            dseg_interp = np.interp(time_elapsed_seg, time_elapsed_seg[~mask], dseg[~mask])

        # Do not interpolate if there no masked data points
        else: 
            dseg_interp = np.asarray(dseg)


        # Interpolate masked data points 
        dseg_interp = np.interp(time_elapsed_seg, time_elapsed_seg[~mask], dseg[~mask])
        
        # Compute the hanning-windowed power spectrum 
        psd_seg[iseg,:], f_i, *_  = compute_spectrum1D(dseg_interp, 
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

        # Count accepted segment
        nseg_valid += 1

    # Skip depth if no usable segments remain
    if nseg_valid == 0:
        continue

    # Compute the mean power spectral density function 
    psd_mean[idepth,:] = np.ma.mean(psd_seg, axis=0)

    # Compute the 95% confidence interval 
    psd_CI[idepth,:,:] = spectral_uncertainty(alpha=0.05,
                                              psd=psd_mean[idepth,:],
                                              estimator="fft",
                                              nseg=nseg_valid,
                                              )

    # Compute the spectral slope 
    spec_slope[idepth], spec_slope_stde[idepth], *_ = spectral_slope(f,
                                                                     psd_mean[idepth,:], 
                                                                     fmin_slope, 
                                                                     fmax_slope,
                                                                     )

    # Compute spectral moments, FVE and mean period
    moments[idepth,:], fve[idepth,:], mean_period[idepth] = spectral_diags(psd_mean[idepth,:], 
                                                                           f, 
                                                                           f_cutoff=f_cut
                                                                           )

    # Save the number of used segments
    nseg_used[idepth] = nseg_valid

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
                           dims=['depth','freq'],
                           coords=dict(depth=depth,freq=f_cpd),
                           attrs=dict(
                               description=(f'Power Spectral Density depth spectrogram at the {option_mooring} ' +
                                            'mooring location.'),
                               units='variance/cycles/day'
                           )
)

PSD_CI = xr.DataArray(data=psd_CI_cpd,
                           dims=['depth','freq','CI_coord'],
                           coords=dict(depth=depth,freq=f_cpd,CI_coord=CI_coord),
                           attrs=dict(
                               description=('95% confidence interval for the power spectral ' +
                                            f'density depth spectrogram at the {option_mooring} mooring ' + 
                                            'location.'),
                               units='variance/cycles/day'
                           )
)

SPEC_slope = xr.DataArray(data=spec_slope,
                           dims=['depth'],
                           coords=dict(depth=depth),
                           attrs=dict(
                               description=('Spectral Slope for the power spectral ' +
                                            f'density depth spectrogram at the {option_mooring} mooring ' + 
                                            'location.'),
                               units='unitless'
                           )
)

SPEC_slope_stde = xr.DataArray(data=spec_slope_stde,
                        dims=['depth'],
                        coords=dict(depth=depth),
                        attrs=dict(
                            description=('Standard error of the Spectral Slope ' +
                                         'for the power spectral density depth spectrogram' + 
                                         f'at the {option_mooring} mooring location.'),
                            units='unitless'
                        )
)

MOMENTS = xr.DataArray(data=moments,
                   dims=['depth','moments_coord'],
                   coords=dict(depth=depth,moments_coord=moments_coord),
                   attrs=dict(
                       description=('First 4 moments (zeroth to third) of the ' +
                                    'power spectral density depth spectrogram.')
                    )
)

FVE = xr.DataArray(data=fve,
                   dims=['depth','fve_coord'],
                   coords=dict(depth=depth,fve_coord=fve_coord),
                   attrs=dict(
                       description=('Fraction of variance explained by the ' +
                                    'low and high frequency bands.'),
                       units='precent'
                    )
)

MEAN_PERIOD_days = xr.DataArray(data=mean_period_days,
                        dims=['depth'],
                        coords=dict(depth=depth),
                        attrs=dict(
                            description=('Mean Period in units days ' +
                                         'for the power spectral density depth spectrogram' + 
                                         f'at the {option_mooring} mooring location.'),
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
file_path = PATH_processed / f"{option_mooring}_spectra_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')

# Print the number of segements used at each depth
print("\nSpectral segment summary:")
print("-" * 60)
print(
    f"{'Depth (m)':>12} "
    f"{'Used':>8} "
    f"{'Total':>8} "
    f"{'Percent':>10} "
    f"{'Max gap (hr)':>14}"
)

for idepth in range(ndepth):

    percent_used = 100 * nseg_used[idepth] / nseg
    longest_gap_hours = longest_gap[idepth] / 3600

    print(
        f"{depth[idepth]:12.1f} "
        f"{nseg_used[idepth]:8d} "
        f"{nseg:8d} "
        f"{percent_used:9.1f}% "
        f"{longest_gap_hours:14.1f}"
    )




