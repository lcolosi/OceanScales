# =============================================================================
# Figure 02
# =============================================================================
#
# Caption:
#   Time series of potential density at (a) CCE1, (b) CCE2, and (c) CCE3 at 9.6 meter
#   water depth. Dashed black line represents the annual cycle, semi-annual cycle,
#   and interannual variability. Residual time series of potential density at
#   (d) CCE1, (e) CCE2, and (f) CCE3 calculated by subtracting seasonal cycles and
#   interannual variability from the original time series. Autocorrelation estimated
#   from (g) CCE1, (h) CCE2, and (i) CCE3 residual potential density time series.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-13
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset, num2date
from datetime import datetime
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

# Import plotting toolbox 
from plotting import month_fmt, add_corner_label
from autocorr import compute_autocorr_biased, compute_decor_scale, compute_decor_scale_unc, segment_time_series
from lsf import unweighted_lsf, detrend
from filter import gaussian_low_pass_filter

# -----------------------------------------------------------------------------
# Set data analysis and plotting parameters
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_depth: Specifies the water depth which the time series is extracted from 
#                 at each mooring.
# - option_harmonics : Specify the number of seasonal cycle harmonics to fit. 
# - dt: Specify the model time resolution (units: seconds)
# - T_annual: Specifies the annual cycle period (one Julian year) in units of seconds. 
# - segment_overlap: Specifies the fractional overlap between segments 
#                    (e.g., 0.75 for 75% overlap).
# - segment_duration: Specifies the length of each segment in years.
# 
# ------------ # 

# Set processing parameters
option_interannual = 'linear'    
option_depth       = 9 
option_harmonics   = 2
 
# Set time and space parameters
dt               = 1*(60)*(60)     
T_annual         = 365.25*(24)*(60)*(60)    
segment_overlap  = 0.5                                        
segment_duration = 1                                           

# Set font and fontsize
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",        
    "text.latex.preamble": r"\usepackage{amsmath}"  
}) 

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------

# Obtain filename paths
filename = PATH_data / "mitgcm" / "mooring" / "processed" / "mitgcm_proc_density_hrly_mooring.nc"

# Generate the nc data structure
nc = Dataset(filename, 'r')

# Extract data variables
depth = nc.variables['depth'][:]
lon   = nc.variables['LON'][:]
lat   = nc.variables['LAT'][:]
time  =  num2date(nc.variables['time'][:], nc.variables['time'].units)
sig = nc.variables['SIG'][:]

# Mask data at fill values (zero for the MITgcm output)
sig_m = np.ma.masked_where(sig == 0, sig)

# Convert cftime.DatetimeGregorian to Python datetime objects
time_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time])

# -----------------------------------------------------------------------------
# Obtain the time series at the specified depth for each mooring
# -----------------------------------------------------------------------------

# Set index for the depth of interest
depth_index =  np.abs(np.abs(depth) - option_depth).argmin()

# Extract the data at the specified depth
sig_depth = sig_m[:, :, depth_index]

# -----------------------------------------------------------------------------
# Remove seasonal and interannual variability from time series
# -----------------------------------------------------------------------------

# Set the dimensions of the array
nsite,ntime = np.shape(sig_depth)

# Set the radian frequencies for the annual, semi-annual, and 
# tri-annual cycles (units: rad/sec)
w = 2 * np.pi * np.arange(1, option_harmonics + 1) / T_annual

# Set option for linear trend
linear_trend = option_interannual == "linear"

# Compute the elapsed time from beginning of time series (units: seconds)
t0 = time[0]
time_elapsed = np.array([(t - t0).total_seconds() for t in time])

# Initialize arrays 
fit = np.ma.zeros((nsite,ntime))
sig_res = np.ma.zeros((nsite,ntime))

# Loop through mooring sites 
for isite in range(nsite): 

    # Set the ith site time series 
    data_ts = sig_depth[isite,:]

    # Compute seasonal harmonic fit
    fit[isite,:], _, _, _ = unweighted_lsf(data_ts, 
                                           time_elapsed, 
                                           parameters=option_harmonics, 
                                           freqs=w, 
                                           sigma=None, 
                                           linear_trend=linear_trend)

    # Compute the residual time series 
    sig_res[isite,:] = data_ts - fit[isite,:]

# Apply Gaussian low-pass filtering when selected
if option_interannual == 'gaussian': 

    # Initialize arrays for the low-pass interannual signal and final residuals
    sig_interannual = np.ma.zeros((nsite, ntime))
    sig_res_filtered = np.ma.zeros((nsite, ntime))

    # Loop through mooring sites
    for isite in range(nsite):

        # Residual after removing seasonal cycle
        data_ts = np.ma.masked_invalid(sig_res[isite, :])

        # Estimate interannual variability using 365-day FWHM Gaussian low-pass
        sig_interannual[isite, :] = gaussian_low_pass_filter(
            data_ts,
            fwhm_days=365,
            dt_hours=1,
            mode='constant',
            truncate=4,
        )

        # Remove interannual variability
        sig_res_filtered[isite, :] = data_ts - sig_interannual[isite, :]

    # Replace residual with the filtered residual
    sig_res = sig_res_filtered

# Set the model for the interannual and seasonal cycles 
if option_interannual == 'gaussian': 
    model = fit + sig_interannual 
else: 
    model = fit 

# -----------------------------------------------------------------------------
# Compute autocorrelation and the decorrelation time scales 
# -----------------------------------------------------------------------------

# Segment a single time series 
segments = segment_time_series(time, 
                               sig_res[0,:], 
                               duration=segment_duration, 
                               overlap=segment_overlap)

# Obtain the dimensions of the segmented time series
nseg,ntime_seg = np.shape(segments)[0], np.shape(segments)[2]

# Intialize arrays 
autocorr_seg   = np.zeros((nsite,nseg,2*ntime_seg-1))
autocorr_mean  = np.zeros((nsite,2*ntime_seg-1))
Lt             = np.zeros(nsite)
Lt_stdm        = np.zeros(nsite)
Lt_std         = np.zeros(nsite)

# Loop through mooring sites 
for isite in range(nsite): 

    # Set the ith site residual time series 
    data_ts = sig_res[isite,:]

    # Segment time series 
    segments = segment_time_series(time_dt, 
                                   data_ts, 
                                   duration=segment_duration, 
                                   overlap=segment_overlap)

    # Loop through segments
    for iseg, (tseg, dseg) in enumerate(segments):

        # Compute the elapsed time from beginning of segmented time series
        t0 = tseg[0]
        time_elapsed_seg = np.array([(t - t0).total_seconds() for t in tseg])
        
        # Detrend data record 
        data_dt = detrend(dseg, time_elapsed_seg, mean = 0)
        
        # Compute autocorrelation function
        autocorr_seg[isite,iseg,:], time_lag = compute_autocorr_biased(data_dt, time_elapsed_seg)

    # Compute the mean autocorrelation  
    autocorr_mean[isite,:] = np.mean(autocorr_seg[isite,:,:], axis=0)

    # Compute the decorrelation scale of the mean autocorrelation 
    Lt[isite], M_lag = compute_decor_scale(autocorr_mean[isite,:],time_lag) 

    # Compute the standard error of the decorrelation scale
    Lt_stdm[isite], Lt_std[isite] = compute_decor_scale_unc(autocorr_mean[isite,:], 
                                                            autocorr_seg[isite,:,:], 
                                                            M_lag, 
                                                            dt, 
                                                            segment_overlap)

    # Convert from seconds to days
    time_lag_days = time_lag/(24*60*60) 
    t_days = Lt/(24*60*60) 
    Lt_stdm_days = Lt_stdm/(24*60*60)

# -----------------------------------------------------------------------------
# Plot time series, least-squares fit, residual, and autocorrelation
# -----------------------------------------------------------------------------

# Find the zero lag index 
zero_lag_index = (2 * ntime_seg - 1) // 2

# Obtain the positive lags from the autocorrelation for plotting 
time_lag_pos      = time_lag_days[zero_lag_index:]
autocorr_pos      = autocorr_seg[:,:,zero_lag_index:]
autocorr_mean_pos = autocorr_mean[:,zero_lag_index:]

# Set plotting parameters 
x_max = 200
dx = 25

# Create figure and axis objects 
fig, axes = plt.subplots(3,3,figsize=(20, 12))

#--- Subplot 1 ---# 
ax = axes[0,0]

# Plot the time series of potential density at CCE1
ax.plot(time_dt, sig_depth[0,:], color='tab:green', linewidth=1.5)

# Plot the seasonal cycles fit and interannual variability
ax.plot(time_dt, model[0,:], color='k', ls='--', linewidth=1.5, label='Least-Squares Fit')

# Set axis attributes
ax.set_title('CCE1')
ax.set_ylim(23.4, 26.0)
ax.set_ylabel('Potential Density (kg/m$^3$)')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(month_fmt))
ax.set_xticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)
ax.legend(loc='upper left', fontsize=fontsize)

#--- Subplot 2 ---# 
ax = axes[0,1]

# Plot the time series of potential density at CCE2
ax.plot(time_dt, sig_depth[1,:], color='tab:red', linewidth=1.5)

# Plot the seasonal cycles fit plus interannual variability
ax.plot(time_dt, model[1,:], color='k', ls='--', linewidth=1.5, label='Least-Squares Fit')

# Set axis attributes
ax.set_title('CCE2')
ax.set_ylim(23.4, 26.0)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(month_fmt))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

#--- Subplot 3 ---# 
ax = axes[0,2]

# Plot the time series of potential density at CCE3
ax.plot(time_dt, sig_depth[2,:], color='tab:blue', linewidth=1.5)

# Plot the seasonal cycles fit plus interannual variability
ax.plot(time_dt, model[2,:], color='k', ls='--', linewidth=1.5, label='Least-Squares Fit')

# Set axis attributes
ax.set_title('CCE3')
ax.set_ylim(23.4, 26.0)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(month_fmt))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)

#--- Subplot 4 ---# 
ax = axes[1,0]

# Plot the residual time series of potential density at CCE1
ax.plot(time_dt, sig_res[0,:], color='tab:green', label='CCE1', linewidth=1.5)

# Set axis attributes
ax.set_ylim(-1.0, 1.0)
ax.set_ylabel('Potential Density (kg/m$^3$)')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(month_fmt))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=True, right=True, left=True, bottom=True, length=5)

#--- Subplot 5 ---# 
ax = axes[1,1]

# Plot the residual time series of potential density at CCE2
ax.plot(time_dt, sig_res[1,:], color='tab:red', linewidth=1.5)

# Set axis attributes
ax.set_ylim(-1.0, 1.0)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(month_fmt))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=True, right=True, left=True, bottom=True, length=5)

#--- Subplot 6 ---# 
ax = axes[1,2]

# Plot the residual time series of potential density at CCE3
ax.plot(time_dt, sig_res[2,:], color='tab:blue', linewidth=1.5)

# Set axis attributes
ax.set_ylim(-1.0, 1.0)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(month_fmt))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=True, right=False, left=True, bottom=True, length=5)

#--- Subplot 7 ---# 
ax = axes[2,0]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the autocorrelation of each of the windows 
for iseg in range(0,nseg): 

    # Plot the ith window autocorrelation at CCE1
    ax.plot(time_lag_pos, autocorr_seg[0,iseg,zero_lag_index:], color='tab:green', alpha=0.4, linewidth=1)

# Plot the mean autocorrelation at CCE1
ax.plot(time_lag_pos, autocorr_mean[0,zero_lag_index:], color='tab:green', linewidth=3)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.4, 1.05)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

#--- Subplot 8 ---# 
ax = axes[2,1]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the autocorrelation of each of the windows 
for iseg in range(0,nseg): 

    # Plot the ith window autocorrelation at CCE2
    ax.plot(time_lag_pos, autocorr_seg[1,iseg,zero_lag_index:], color='tab:red', alpha=0.4, linewidth=1)

# Plot the mean autocorrelation at CCE2
ax.plot(time_lag_pos, autocorr_mean[1,zero_lag_index:], color='tab:red', linewidth=3)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.4, 1.05)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

#--- Subplot 9 ---# 
ax = axes[2,2]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the autocorrelation of each of the windows 
for iseg in range(0,nseg): 

    # Plot the ith window autocorrelation at CCE3
    ax.plot(time_lag_pos, autocorr_seg[2,iseg,zero_lag_index:], color='tab:blue', alpha=0.4, linewidth=1)

# Plot the mean autocorrelation at CCE3
ax.plot(time_lag_pos, autocorr_mean[2,zero_lag_index:], color='tab:blue', linewidth=3)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.4, 1.05)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)

# Label each subplot
pos = [0.95, 0.91] 
add_corner_label(axes[0,0], pos, 'A', fontsize = fontsize)
add_corner_label(axes[0,1], pos, 'B', fontsize = fontsize)
add_corner_label(axes[0,2], pos, 'C', fontsize = fontsize)
add_corner_label(axes[1,0], pos, 'D', fontsize = fontsize)
add_corner_label(axes[1,1], pos, 'E', fontsize = fontsize)
add_corner_label(axes[1,2], pos, 'F', fontsize = fontsize)
add_corner_label(axes[2,0], pos, 'G', fontsize = fontsize)
add_corner_label(axes[2,1], pos, 'H', fontsize = fontsize)  
add_corner_label(axes[2,2], pos, 'I', fontsize = fontsize)

# Adjust spacing 
plt.subplots_adjust(hspace=0.21, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'fig02.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
