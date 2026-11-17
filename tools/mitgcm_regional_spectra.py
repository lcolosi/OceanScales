# =============================================================================
# Compute power spectra and their diagnotics across the study domain from MITgcm data 
# =============================================================================
#
# Description:
#   Computes power spectra and their diagnostics at each regional MITgcm
#   grid point, including spectral variance fractions and uncertainties.
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
from spectra import compute_spectrum1D, spectral_slope, spectral_diags, spectral_uncertainty, compute_spectral_var_frac
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
fmin_slope       = (1/182.5) / seconds_per_day
fmax_slope       = (1/4) / seconds_per_day 
period_bands     = {
                    "low": (30.0, 182.5),
                    "mid": (4.0, 30.0),
                    "high": (2 / 24, 4.0),
                   }
band_labels      = list(period_bands)

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
    
        # Remove seasonal cycles (and interannual variability if a global linear trend is fit)
        data_res[:,ilat,ilon] = data_ts - fit[:,ilat,ilon]

# Apply Gaussian low-pass filtering when selected
if option_interannual == 'gaussian': 

    # Initialize interannual variability array
    data_interannual = np.ma.masked_all((ntime, nlat, nlon))

    # Loop through longitude 
    for ilon in tqdm(range(nlon), desc="Low-pass Filtering Time Series", unit="lon"):

        # Loop through latitude
        for ilat in range(nlat):

            # Set the ith residual time series 
            data_ts = np.ma.masked_invalid(data_res[:, ilat, ilon])

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

    # Remove interannual variability from the seasonal residual
    data_res = data_res - data_interannual

# -----------------------------------------------------------------------------
# Compute power spectra and their diagnostics
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

# Compute expected cyclic frequency vector for each segment and convert to cycles per day
f     = np.fft.rfftfreq(ntime_seg, d=dt)
f_cpd = f * seconds_per_day

# Set number of non-negative frequency bins
nfreq = len(f)

# --- Create non-overlapping band masks --- #

# Initialize mask list 
band_masks = []

# Loop through frequency bands 
for iband, (band_name, (period_min, period_max)) in enumerate(period_bands.items()):

    # Set low and high frequency bounds of band
    f_lo = 1.0 / period_max
    f_hi = 1.0 / period_min

    # Define the band mask for the ith frequency band (shared boundary belongs to the higher-frequency band)
    band_mask = (f_cpd >= f_lo) & (f_cpd < f_hi)

    # For the high-frequency band, let the nyquist frequency be included in the frequency band
    if iband == len(period_bands) - 1:
        band_mask = (f_cpd >= f_lo) & (f_cpd <= f_hi)

    # Error if no frequencies are contained in the frequency band
    if not np.any(band_mask):
        raise ValueError(f"The {band_name} band contains no FFT frequencies.")

    # Add band mask to list
    band_masks.append(band_mask)

# Define the mask for the union of all frequency bands
total_mask = np.logical_or.reduce(band_masks)

# Check if any frequency bands overlap 
if np.any(np.sum(band_masks, axis=0) > 1):
    raise ValueError("The variance frequency bands overlap.")

# Check if we have enough windows to estimate uncertainties
if nseg < 2:
    raise ValueError("At least two time segments are needed for uncertainty estimates.")

# Initialize arrays 
psd_mean        = np.ma.masked_all((nlat,nlon,nfreq))
psd_CI          = np.ma.masked_all((nlat,nlon,nfreq,2))
spec_slope      = np.ma.masked_all((nlat,nlon))
spec_slope_stdm = np.ma.masked_all((nlat,nlon))
moments         = np.ma.masked_all((nlat,nlon,4))
var_band        = np.ma.masked_all((nlat,nlon,3))
phi_mean        = np.ma.masked_all((nlat,nlon,3))
phi_stdm        = np.ma.masked_all((nlat,nlon,3))
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
        spec_slope[ilat,ilon], spec_slope_stdm[ilat,ilon], *_ = spectral_slope(f,
                                                                               psd_mean[ilat,ilon,:], 
                                                                               fmin_slope, 
                                                                               fmax_slope,
                                                                              )

        # Compute spectral moments and mean period
        moments[ilat,ilon,:], _, _, mean_period[ilat,ilon] = spectral_diags(psd_mean[ilat,ilon,:], 
                                                                            f, 
                                                                            f_cutoff= 1 / (30 * seconds_per_day)
                                                                            )

        # Loop through frequency bands 
        for iband, band_mask in enumerate(band_masks):

            # Compute the spectral variance fraction and its uncertainty
            var_band[ilat,ilon,iband], phi_mean[ilat,ilon,iband], phi_stdm[ilat,ilon,iband], _, _ = compute_spectral_var_frac(psd_mean[ilat,ilon,:],
                                                                                                                              psd_seg,
                                                                                                                              f,
                                                                                                                              band_mask,
                                                                                                                              total_mask,
                                                                                                                              overlap=segment_overlap,
                                                                                                                              )

        # Check if the svf in the three bands sum to 1 
        phi_sum = np.sum(phi_mean[ilat, ilon, :])

        if not np.isclose(phi_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Spectral variance fractions do not sum to 1 at "
                f"lat={ilat}, lon={ilon}."
            )

# Convert mean period to units of days
mean_period_days = mean_period/(24*60*60) 

# Convert spectral variance fraction to percentage
phi_mean *= 100
phi_stdm *= 100

# Convert psd to units of cycles per day 
psd_cpd    = psd_mean / seconds_per_day 
psd_CI_cpd = psd_CI / seconds_per_day 

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# Set coordinates
CI_coord      = ['lower', 'upper']
moments_coord = ["m0", "m1", "m2", "m3"]
band_coord    = band_labels

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

SPEC_slope_stdm = xr.DataArray(data=spec_slope_stdm,
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

VAR = xr.DataArray(data=var_band,
                   dims=['lat','lon','band_coord'],
                   coords=dict(lat=lat,lon=lon,band_coord=band_coord),
                   attrs=dict(
                       description=('Absolute variance contained in the ' +
                                    'low, mid, and high frequency bands.'),
                       units='variance'
                    )
)

SVF = xr.DataArray(data=phi_mean,
                   dims=['lat','lon','band_coord'],
                   coords=dict(lat=lat,lon=lon,band_coord=band_coord),
                   attrs=dict(
                       description=('Spectral variance fraction of the mean PSD in the ' +
                                    'low, mid, and high frequency bands.'),
                       units='percent'
                    )
)

SVF_stdm = xr.DataArray(data=phi_stdm,
                        dims=['lat', 'lon', 'band'],
                        coords=dict(lat=lat, lon=lon, band=band_coord),
                        attrs=dict(
                            description=('Standard error of spectral variance fraction using ' 
                                        'segment-wise delta-method residuals with overlap correction.'),
                            units='percent',
                        ),
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
data = xr.Dataset({'PSD':PSD,'PSD_CI':PSD_CI,'SPEC_slope':SPEC_slope,'SPEC_slope_stdm':SPEC_slope_stdm,'MOMENTS':MOMENTS,'VAR':VAR,'SVF':SVF,'SVF_stdm':SVF_stdm,'MEAN_PERIOD_days':MEAN_PERIOD_days})

# Set global variables to document the processing parameters used 
data.attrs.update({
    "variable": option_data,
    "interannual_method": option_interannual,
    "seasonal_harmonics": option_harmonics,
    "segment_duration_years": segment_duration,
    "segment_overlap": segment_overlap,
    "segment_processing": seg_proc,
    "sampling_interval_seconds": dt,
    "band_period_bounds_days": str(period_bands),
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



