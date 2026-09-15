# =============================================================================
# Figure S09
# =============================================================================
#
# Caption:
#   Time series of potential density at (a) CCE1 and (b) CCE2 at 9.6 meter
#   water depth. Dashed black line represents the annual cycle, semi-annual cycle,
#   and interannual variability. Residual time series of potential density at
#   (c) CCE1 and (d) CCE2 calculated by subtracting seasonal cycles and
#   interannual variability from the original time series. Autocorrelation estimated
#   from (e) CCE1 and (f) CCE2 residual potential density time series.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-11
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_figs = ROOT / "figs"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import analysis functions 
from autocorr import compute_autocorr_biased_masked, compute_decor_scale_masked, compute_decor_scale_unc_masked, segment_time_series
from lsf import unweighted_lsf, detrend, compute_fve
from filter import gaussian_low_pass_filter
from plotting import add_corner_label

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_depth: Specifies the depth to analyze. 
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
# - norm: Specifies the method used to normalize the autocovariance when 
#         observations are missing.
#
# ------------#

# Set processing parameters
option_depth       = 9
option_data        = 'density'    
option_interannual = 'gaussian' 
option_harmonics   = 2      
option_detrend_seg = True

# Set time and space parameters
dt               = 3600    
T_annual         = 365.25*(24)*(60)*(60)    
segment_overlap  = 0.5                                        
segment_duration = 0.5   
norm             = "standard"

# Set font and fontsize
fontsize=16
fontsize_ins=9
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",        
    "text.latex.preamble": r"\usepackage{amsmath}"  
}) 

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
if norm not in ("standard", "corrected"):
    raise ValueError(
        "norm must be 'standard' or 'corrected'."
    )

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# -----------------------------------------------------------------------------
# Load cce data
# -----------------------------------------------------------------------------

# Set path to processed regional MITgcm data
PATH_processed_cce1 = PATH_data / "cce" / "cce1" / "processed"
PATH_processed_cce2 = PATH_data / "cce" / "cce2" / "processed"

# Set NetCDF variable names
variable_names = {
    "temp": "CTemp",
    "sal": "ASal",
    "density": "SIG",
}

# Set filename based on selected data type
if option_data in ("temp", "sal", "density"):
    filename_cce1 = (
        PATH_processed_cce1
        / f"cce1_proc_density_hrly_mooring.nc"
    )
    filename_cce2 = (
            PATH_processed_cce2
            / f"cce2_proc_density_hrly_mooring.nc"
        )
else:
    raise ValueError(f"Invalid option_data: {option_data}")

# --- CCE1 --- # 

# Load NetCDF data
with Dataset(filename_cce1, "r") as nc:
    depth1 = nc.variables["depth"][:]

    time1 = num2date(
        nc.variables["time"][:],
        units=nc.variables["time"].units,
    )

    data1 = nc.variables[variable_names[option_data]][:]

# Convert cftime.DatetimeGregorian to Python datetime objects
time1_dt = np.array(
    [
        datetime(
            d.year,
            d.month,
            d.day,
            d.hour,
            d.minute,
            d.second,
        )
        for d in time1
    ]
)

# --- CCE2 --- # 

# Load NetCDF data
with Dataset(filename_cce2, "r") as nc:
    depth2 = nc.variables["depth"][:]

    time2 = num2date(
        nc.variables["time"][:],
        units=nc.variables["time"].units,
    )

    data2 = nc.variables[variable_names[option_data]][:]

# Convert cftime.DatetimeGregorian to Python datetime objects
time2_dt = np.array(
    [
        datetime(
            d.year,
            d.month,
            d.day,
            d.hour,
            d.minute,
            d.second,
        )
        for d in time2
    ]
)

# Mask data points previously set to NaN during processing
data1 = np.ma.masked_invalid(data1)
data2 = np.ma.masked_invalid(data2)

# -----------------------------------------------------------------------------
# Obtain the time series at the specified depth for each mooring
# -----------------------------------------------------------------------------

# Set index for the depth of interest
depth1_index =  np.abs(np.abs(depth1) - option_depth).argmin()
depth2_index =  np.abs(np.abs(depth2) - option_depth).argmin()

# Extract the data at the specified depth
data1_depth = data1[:, depth1_index]
data2_depth = data2[:, depth2_index]

# Print actual depths
print("CCE1 depth:", depth1[depth1_index])
print("CCE2 depth:", depth2[depth2_index])

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
time1_elapsed = np.array([(t - time1[0]).total_seconds() for t in time1])
time2_elapsed = np.array([(t - time2[0]).total_seconds() for t in time2])

# Obtain the dimensions of the longitude and latitude 
ntime1 = len(data1_depth)
ntime2 = len(data2_depth)

# Compute seasonal harmonic fit
fit1, *_ = unweighted_lsf(data1_depth, 
                          time1_elapsed, 
                          parameters=option_harmonics, 
                          freqs=w, 
                          sigma=None, 
                          linear_trend=linear_trend,
                          )
fit2, *_ = unweighted_lsf(data2_depth, 
                          time2_elapsed, 
                          parameters=option_harmonics, 
                          freqs=w, 
                          sigma=None, 
                          linear_trend=linear_trend,
                          )

# Compute the residual time series 
data1_res = data1_depth - fit1
data2_res = data2_depth - fit2

# Compute the time mean 
data1_mean = np.ma.mean(data1_depth)
data2_mean = np.ma.mean(data2_depth)

# Apply Gaussian low-pass filtering when selected
if option_interannual == 'gaussian': 

    # Estimate interannual variability using 365-day FWHM Gaussian low-pass
    data1_interannual = gaussian_low_pass_filter(data1_depth - data1_mean,
                                                 fwhm_days=365,
                                                 dt_hours=1,
                                                 mode='constant',
                                                 truncate=4,
                                                )
    data2_interannual = gaussian_low_pass_filter(data2_depth - data2_mean,
                                                 fwhm_days=365,
                                                 dt_hours=1,
                                                 mode='constant',
                                                 truncate=4,
                                                )

    # Remove seasonal and interannual variability
    data1_res = data1_depth - fit1 - data1_interannual
    data2_res = data2_depth - fit2 - data2_interannual

# Set the model for the interannual and seasonal cycles 
if option_interannual == 'gaussian': 
    model1 = fit1 + data1_interannual 
    model2 = fit2 + data2_interannual 
else: 
    model1 = fit1 
    model2 = fit2 

# Compute the fraction of variance explained by the interannual and season model
fve1 = compute_fve(data1_depth, model1)
fve2 = compute_fve(data2_depth, model2)

# -----------------------------------------------------------------------------
# Compute autocorrelation and the decorrelation time scales 
# -----------------------------------------------------------------------------

# Segment the time series 
segments1 = segment_time_series(time1_dt, 
                                data1_res, 
                                duration=segment_duration, 
                                overlap=segment_overlap,
                               )
segments2 = segment_time_series(time2_dt, 
                                data2_res, 
                                duration=segment_duration, 
                                overlap=segment_overlap,
                               )

# Obtain the dimensions of the segmented time series
nseg1 = len(segments1)
ntime_seg1 = len(segments1[0][0])
nseg2 = len(segments2)
ntime_seg2 = len(segments2[0][0])

# Initialize arrays 
autocorr1_seg   = np.ma.masked_all((nseg1,2*ntime_seg1-1))
autocorr2_seg   = np.ma.masked_all((nseg2,2*ntime_seg2-1))

# --- CCE1 --- # 

# Loop through segments
for iseg, (tseg, dseg) in enumerate(segments1):

    # Compute the elapsed time from beginning of segmented time series
    t0 = tseg[0]
    time_elapsed_seg = np.array([(t - t0).total_seconds() for t in tseg])
    
    # Remove segment-wise mean or linear trend
    if option_detrend_seg: 
        data_dt = detrend(dseg, time_elapsed_seg, mean = 0)
    else: 
        data_dt = dseg - np.ma.mean(dseg)

    # Compute autocorrelation function
    autocorr1_seg[iseg,:], time1_lag = compute_autocorr_biased_masked(data_dt, 
                                                                      time_elapsed_seg, 
                                                                      normalization=norm
                                                                     )

# Compute the mean autocorrelation function 
autocorr1_mean = np.ma.mean(autocorr1_seg, axis=0)

# Compute the decorrelation scale of the mean autocorrelation 
Lt1, M_lag1 = compute_decor_scale_masked(autocorr1_mean,time1_lag) 

# Compute the standard error of the decorrelation scale
Lt1_stdm, Lt1_std, Lt1_stds  = compute_decor_scale_unc_masked(autocorr1_mean, 
                                                              autocorr1_seg, 
                                                              M_lag1, 
                                                              dt, 
                                                              segment_overlap,
                                                             )

# Convert time scale to units of days
time1_lag_days = time1_lag/(24*60*60) 
Lt1_days       = Lt1/(24*60*60) 
Lt1_stdm_days  = Lt1_stdm/(24*60*60) 
Lt1_std_days   = Lt1_std/(24*60*60) 
Lt1_stds_days  = Lt1_stds/(24*60*60)   

# --- CCE2 --- # 

# Loop through segments
for iseg, (tseg, dseg) in enumerate(segments2):

    # Compute the elapsed time from beginning of segmented time series
    t0 = tseg[0]
    time_elapsed_seg = np.array([(t - t0).total_seconds() for t in tseg])
    
    # Remove segment-wise mean or linear trend
    if option_detrend_seg: 
        data_dt = detrend(dseg, time_elapsed_seg, mean = 0)
    else: 
        data_dt = dseg - np.ma.mean(dseg)

    # Compute autocorrelation function
    autocorr2_seg[iseg,:], time2_lag = compute_autocorr_biased_masked(data_dt, 
                                                                      time_elapsed_seg, 
                                                                      normalization=norm
                                                                     )

# Compute the mean autocorrelation function 
autocorr2_mean = np.ma.mean(autocorr2_seg, axis=0)

# Compute the decorrelation scale of the mean autocorrelation 
Lt2, M_lag2 = compute_decor_scale_masked(autocorr2_mean,time2_lag) 

# Compute the standard error of the decorrelation scale
Lt2_stdm, Lt2_std, Lt2_stds  = compute_decor_scale_unc_masked(autocorr2_mean, 
                                                              autocorr2_seg, 
                                                              M_lag2, 
                                                              dt, 
                                                              segment_overlap,
                                                             )

# Convert time scale to units of days
time2_lag_days = time2_lag/(24*60*60) 
Lt2_days       = Lt2/(24*60*60) 
Lt2_stdm_days  = Lt2_stdm/(24*60*60) 
Lt2_std_days   = Lt2_std/(24*60*60) 
Lt2_stds_days  = Lt2_stds/(24*60*60)   

# -----------------------------------------------------------------------------
# Plot time series, least-squares fit, residual, and autocorrelation
# -----------------------------------------------------------------------------

# Find the zero lag index 
zero_lag_index1 = (2 * ntime_seg1 - 1) // 2
zero_lag_index2 = (2 * ntime_seg2 - 1) // 2

# Obtain the positive lags from the autocorrelation for plotting 
time1_lag_pos      = time1_lag_days[zero_lag_index1:]
autocorr1_pos      = autocorr1_seg[:,zero_lag_index1:]
autocorr1_mean_pos = autocorr1_mean[zero_lag_index1:]
time2_lag_pos      = time2_lag_days[zero_lag_index2:]
autocorr2_pos      = autocorr2_seg[:,zero_lag_index2:]
autocorr2_mean_pos = autocorr2_mean[zero_lag_index2:]

# Set plotting parameters 
x_max = 182.5
dx = 20

# Create figure and axis objects 
fig, axes = plt.subplots(3,2,figsize=(18, 12))

#--- Subplot 1 ---# 
ax = axes[0,0]

# Plot the time series of potential density at CCE1
ax.plot(time1_dt, data1_depth, color='tab:green', linewidth=1.5)

# Plot the seasonal cycles fit and interannual variability
ax.plot(time1_dt, model1, color='k', ls='--', linewidth=1.5, label='Least-Squares Fit')

# Set axis attributes
ax.set_title('CCE1')
ax.set_ylim(22.75, 26.0)
ax.set_ylabel('Potential Density (kg/m$^3$)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)
ax.legend(loc='upper left', fontsize=fontsize)

#--- Subplot 2 ---# 
ax = axes[0,1]

# Plot the time series of potential density at CCE2
ax.plot(time2_dt, data2_depth, color='tab:red', linewidth=1.5)

# Plot the seasonal cycles fit plus interannual variability
ax.plot(time2_dt, model2, color='k', ls='--', linewidth=1.5, label='Least-Squares Fit')

# Set axis attributes
ax.set_title('CCE2')
ax.set_ylim(22.75, 26.0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

#--- Subplot 3 ---# 
ax = axes[1,0]

# Plot the residual time series of potential density at CCE1
ax.plot(time1_dt, data1_res, color='tab:green', label='CCE1', linewidth=1.5)

# Set axis attributes
ax.set_ylim(-1.25, 1.25)
ax.set_ylabel('Potential Density (kg/m$^3$)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=True, right=True, left=True, bottom=True, length=5)

#--- Subplot 4 ---# 
ax = axes[1,1]

# Plot the residual time series of potential density at CCE2
ax.plot(time2_dt, data2_res, color='tab:red', linewidth=1.5)

# Set axis attributes
ax.set_ylim(-1.25, 1.25)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=True, right=True, left=True, bottom=True, length=5)

#--- Subplot 5 ---# 
ax = axes[2,0]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the autocorrelation of each of the windows 
for iseg in range(0,nseg1): 

    # Plot the ith window autocorrelation at CCE1
    ax.plot(time1_lag_pos, autocorr1_seg[iseg,zero_lag_index1:], color='tab:green', alpha=0.4, linewidth=1)

# Plot the mean autocorrelation at CCE1
ax.plot(time1_lag_pos, autocorr1_mean[zero_lag_index1:], color='tab:green', linewidth=3)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.45, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.38, 0.59, 0.3, 0.38])

# Plot the zero line 
axins.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=0.75)

# Plot the autocorrelation of each window
for iseg in range(nseg1):

    axins.plot(
        time1_lag_pos,
        autocorr1_pos[iseg, :],
        color='tab:green',
        alpha=0.4,
        linewidth=0.8
    )

# Plot the mean autocorrelation
axins.plot(
    time1_lag_pos,
    autocorr1_mean_pos,
    color='tab:green',
    linewidth=1.5
)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 15)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 16+2, 2))
axins.set_yticks(np.arange(0,1+0.25,0.25))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(
    which='both',
    direction='out',
    top=False,
    right=False,
    left=True,
    bottom=True,
    labelsize=fontsize_ins,
    length=3
)

#--- Subplot 6 ---# 
ax = axes[2,1]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the autocorrelation of each of the windows 
for iseg in range(0,nseg2): 

    # Plot the ith window autocorrelation at CCE2
    ax.plot(time2_lag_pos, autocorr2_seg[iseg,zero_lag_index2:], color='tab:red', alpha=0.4, linewidth=1)

# Plot the mean autocorrelation at CCE2
ax.plot(time2_lag_pos, autocorr2_mean[zero_lag_index2:], color='tab:red', linewidth=3)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.45, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.38, 0.59, 0.3, 0.38])

# Plot the zero line 
axins.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=0.75)

# Plot the autocorrelation of each window
for iseg in range(nseg2):

    axins.plot(
        time2_lag_pos,
        autocorr2_pos[iseg, :],
        color='tab:red',
        alpha=0.4,
        linewidth=0.8
    )

# Plot the mean autocorrelation
axins.plot(
    time2_lag_pos,
    autocorr2_mean_pos,
    color='tab:red',
    linewidth=1.5
)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 15)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 16+2, 2))
axins.set_yticks(np.arange(0,1+0.25,0.25))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(
    which='both',
    direction='out',
    top=False,
    right=False,
    left=True,
    bottom=True,
    labelsize=fontsize_ins,
    length=3
)

# Label each subplot
pos = [0.95, 0.91] 
add_corner_label(axes[0,0], pos, 'A', fontsize=fontsize)
add_corner_label(axes[0,1], pos, 'B', fontsize=fontsize)
add_corner_label(axes[1,0], pos, 'C', fontsize=fontsize)
add_corner_label(axes[1,1], pos, 'D', fontsize=fontsize)
add_corner_label(axes[2,0], pos, 'E', fontsize=fontsize)
add_corner_label(axes[2,1], pos, 'F', fontsize=fontsize)

# Adjust spacing 
plt.subplots_adjust(hspace=0.21, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'figS09.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)

# -----------------------------------------------------------------------------
# Print decorrelation scale statistics
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Decorrelation Scale Statistics")
print("=" * 60)

print(
    f"CCE1:\n"
    f"  Decorrelation scale       : {Lt1_days:.2f} days\n"
    f"  Standard error            : {Lt1_stdm_days:.2f} days\n"
    f"  Standard deviation        : {Lt1_std_days:.2f} days\n"
    f"  Std. error of std. dev.   : {Lt1_stds_days:.2f} days\n"
    f"  Number of segments        : {nseg1}"
)

print()

print(
    f"CCE2:\n"
    f"  Decorrelation scale       : {Lt2_days:.2f} days\n"
    f"  Standard error            : {Lt2_stdm_days:.2f} days\n"
    f"  Standard deviation        : {Lt2_std_days:.2f} days\n"
    f"  Std. error of std. dev.   : {Lt2_stds_days:.2f} days\n"
    f"  Number of segments        : {nseg2}"
)

print("=" * 60)



