# =============================================================================
# Figure 09
# =============================================================================
#
# Caption:
#   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-17
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_figs = ROOT / "figs"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox 
from plotting import add_corner_label

# -----------------------------------------------------------------------------
# Set processing and plotting parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density", "uvel", or "vvel".
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - segment_months : Specifies the window duration. 
# - option_depth: Specifies the water depth which the data is extracted from 
#                 at each mooring.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'gaussian' 
option_detrend_seg = True
segment_months     = 6
option_depth       = 10 
option_slope       = 1

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set plotting parameters
seg_durations = [1, 2, 3, 4, 6, 8, 12]

# Path to processed data 
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"
PATH_analytic_processed = PATH_data / "analytic" 

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm and CCE decorrelation-scale estimates
# -----------------------------------------------------------------------------

# --- Analytic Data --- # 

# Set filename to saved the netcdf file
filename = PATH_analytic_processed / f"analytic_autocor_decor_scale.nc"

# Load in data
with Dataset(filename, "r") as nc:

    # Load analytic autocorrelation and decorrelation scale
    autocorr_analytic    = nc.variables["autocorr"][:]
    decor_scale_analytic = nc.variables["decor_scale"][:]

    # Load coordinates
    lag_analytic = nc.variables["lag"][:]
    duration_ac  = nc.variables["duration_ac"][:]
    duration_ds  = nc.variables["duration_ds"][:]
    slope_ac     = nc.variables["slope_ac"][:]
    slope_ds     = nc.variables["slope_ds"][:]

# --- MITgcm --- # 

# Initialize array
autocorr_m         = []
lag_m              = []
decor_scale_m      = []
decor_scale_stdm_m = []

# Loop through window durations
for seg_duration in seg_durations:

    filename = (
        PATH_processed
        / f"mitgcm_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_"
          f"seg_duration_{seg_duration}mo.nc"
    )

    with Dataset(filename, "r") as nc:

        # Load autocorrelation and lag
        autocorr = nc.variables["autocorr"][:]
        lag = nc.variables["lag"][:]

        # Find zero-lag index
        zero_lag = np.abs(lag).argmin()

        # Retain only non-negative lags
        autocorr = autocorr[..., zero_lag:]
        lag = lag[zero_lag:]

        # Append to lists
        autocorr_m.append(autocorr)
        lag_m.append(lag)

        # Append decorrelation scale to list
        decor_scale_m.append(nc.variables["decor_scale"][:])
        decor_scale_stdm_m.append(nc.variables["decor_scale_stdm"][:])

        # Save depth coordinate once
        if seg_duration == seg_durations[0]:
            depth_m = nc.variables["depth"][:]


# Stack window durations along a new first dimension for decorrelation scales
decor_scale_m = np.ma.stack(decor_scale_m, axis=0)
decor_scale_stdm_m = np.ma.stack(decor_scale_stdm_m, axis=0)

#------------------------------------------# 
# Padd and stack autocorrelation  
#------------------------------------------# 

# Find maximum number of lags across all window durations
nlag_max = max(ac.shape[-1] for ac in autocorr_m)

# Determine output shape
shape_out = (len(autocorr_m), *autocorr_m[0].shape[:-1], nlag_max)

# Initialize masked array
autocorr_m_stack = np.ma.masked_all(shape_out)

# Insert each autocorrelation array
for j, ac in enumerate(autocorr_m):
    nlag = ac.shape[-1]
    autocorr_m_stack[j, ..., :nlag] = ac

# Replace list with stacked array
autocorr_m = autocorr_m_stack

#------------------------------------------# 
# Padd and stack the time lag  
#------------------------------------------# 

# Find the maximum number of lags
nlag_max = max(len(lag) for lag in lag_m)

# Initialize masked lag array
lag_m_stack = np.ma.masked_all((len(seg_durations), nlag_max))

# Fill available lags
for j, lag in enumerate(lag_m):
    lag_m_stack[j, :len(lag)] = lag

# Replace list with stacked array
lag_m = lag_m_stack

# -----------------------------------------------------------------------------
# Obtain the autocorrelation at the specified depth for each mooring
# -----------------------------------------------------------------------------

# Set index for the depth of interest
depth_m_index = np.abs(np.abs(depth_m) - option_depth).argmin()

# Extract the data at the specified depth
autocorr_m_depth         = autocorr_m[:,:,depth_m_index,:]
decor_scale_m_depth      = decor_scale_m[:,:,depth_m_index]
decor_scale_stdm_m_depth = decor_scale_stdm_m[:,:,depth_m_index]

# Find the zero lag index 
zero_lag_m = np.abs(lag_m[:,0]).argmin()

# Obtain non-negative lags
lag_m_pos = lag_m[:,zero_lag_m:]

# Obtain corresponding autocorrelations
autocorr_m_depth_pos = autocorr_m_depth[:,:,zero_lag_m:]

# -----------------------------------------------------------------------------
# Obtain the analytic autocorrelation at the specified slope
# -----------------------------------------------------------------------------

# Set index for the spectral slope of interest
slope_m_index = np.abs(slope_ac - option_slope).argmin()

# Extract the data at the specified slope
autocorr_analytic_slope = autocorr_analytic[:,:,slope_m_index]

# -----------------------------------------------------------------------------
# Plot autocorrelation and decorrelation time scales at mooring locations 
# -----------------------------------------------------------------------------

# Set plotting parameters 
depth_m_pos = abs(depth_m)
duration_m  = np.asarray(seg_durations)
duration_ds = duration_ds * (12 / 365.25) 
slope_p     = [1, 1.5, 2]
x_min       = -3
x_max       = 60
dx          = 10

# Use a discrete colormap for the seven window durations
colors = cmo.haline_r(np.linspace(0.05, 0.95, len(seg_durations)))

# Create figure
fig, axes = plt.subplots(2,3,figsize=(18, 10))
ax_flat = axes.flatten()

# --- Subplot 1 --- # 
ax = ax_flat[0]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract autocorrelation and lag
    autocorr = autocorr_m_depth_pos[j,0,:]
    lag      = lag_m_pos[j,:]

    # Plot mean autocorrelation 
    ax.plot(
        lag,
        autocorr,
        '-',
        color=colors[j],
        linewidth=2,
        markersize=5,
        label=f"{seg_duration} month"
    )

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_xlim(x_min,x_max)
ax.set_ylim(-0.3, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)
ax.legend(loc='upper right', fontsize=13)

# --- Subplot 2 --- # 
ax = ax_flat[1]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract autocorrelation and lag
    autocorr = autocorr_m_depth_pos[j,1,:]
    lag      = lag_m_pos[j,:]

    # Plot mean autocorrelation 
    ax.plot(
        lag,
        autocorr,
        '-',
        color=colors[j],
        linewidth=2,
        markersize=5,
        label=f"{seg_duration} month"
    )

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_xlim(x_min,x_max)
ax.set_ylim(-0.3, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

# --- Subplot 3 --- # 
ax = ax_flat[2]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract autocorrelation and lag
    autocorr = autocorr_m_depth_pos[j,2,:]
    lag      = lag_m_pos[j,:]

    # Plot mean autocorrelation 
    ax.plot(
        lag,
        autocorr,
        '-',
        color=colors[j],
        linewidth=2,
        markersize=5,
        label=f"{seg_duration} month"
    )

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_xlim(x_min,x_max)
ax.set_ylim(-0.3, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

# --- Subplot 4 --- # 
ax = ax_flat[3]

# Loop through a subset of alpha values 
for k, ialpha in enumerate(slope_p): 

    # Find index value
    idx_alpha = np.argmin(np.abs(np.array(slope_ds) - ialpha))

    # Plot the decorrelation scale as a function of window duration
    ax.plot(duration_ds, decor_scale_analytic[:,idx_alpha], '--', color='k', lw = 1.5) 

# Label cruves
ax.annotate(
    r'$\alpha = 2$',
    xy=(0.83, 0.95),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$\alpha = 1.5$',
    xy=(0.85, 0.69),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$\alpha = 1$',
    xy=(0.88, 0.25),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

# Plot the mean decorrelation scale as a function of segment duration 
ax.plot(
    duration_m, 
    decor_scale_m_depth[:,0], 
    '.-', 
    color='tab:green', 
    lw=2, 
    markersize=8
)

# Plot the standard error of the mean decorrelation scale as a function of segment duration 
ax.fill_between(
    duration_m, 
    decor_scale_m_depth[:,0] + decor_scale_stdm_m_depth[:,0], 
    decor_scale_m_depth[:,0] - decor_scale_stdm_m_depth[:,0], 
    color='tab:green', 
    alpha=0.25
)

# Set axis attributes 
ax.set_xlabel(r'Duration $T$ (months)')
ax.set_ylabel('Decorrelation Scale (days)')
ax.set_xticks(duration_m)
ax.set_yticks(np.arange(0,60+10,10))
ax.set_xlim(0,12.1)
ax.set_ylim(0,60)
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

# --- Subplot 5 --- # 
ax = ax_flat[4]

# Loop through a subset of alpha values 
for k, ialpha in enumerate(slope_p): 

    # Find index value
    idx_alpha = np.argmin(np.abs(np.array(slope_ds) - ialpha))

    # Plot the decorrelation scale as a function of window duration
    ax.plot(duration_ds, decor_scale_analytic[:,idx_alpha], '--', color='k', lw = 1.5) 

# Label cruves
ax.annotate(
    r'$\alpha = 2$',
    xy=(0.83, 0.95),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$\alpha = 1.5$',
    xy=(0.85, 0.69),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$\alpha = 1$',
    xy=(0.88, 0.25),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

# Plot the mean decorrelation scale as a function of segment duration 
ax.plot(
    duration_m, 
    decor_scale_m_depth[:,1], 
    '.-', 
    color='tab:red', 
    lw=2, 
    markersize=8
)

# Plot the standard error of the mean decorrelation scale as a function of segment duration 
ax.fill_between(
    duration_m, 
    decor_scale_m_depth[:,1] + decor_scale_stdm_m_depth[:,1], 
    decor_scale_m_depth[:,1] - decor_scale_stdm_m_depth[:,1], 
    color='tab:red', 
    alpha=0.25
)

# Set axis attributes 
ax.set_xlabel(r'Duration $T$ (months)')
ax.set_xticks(duration_m)
ax.set_yticks(np.arange(0,60+10,10))
ax.set_xlim(0,12.1)
ax.set_ylim(0,60)
ax.set_yticklabels([])
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

# --- Subplot 6 --- # 
ax = ax_flat[5]

# Loop through a subset of alpha values 
for k, ialpha in enumerate(slope_p): 

    # Find index value
    idx_alpha = np.argmin(np.abs(np.array(slope_ds) - ialpha))

    # Plot the decorrelation scale as a function of window duration
    ax.plot(duration_ds, decor_scale_analytic[:,idx_alpha], '--', color='k', lw = 1.5) 

# Label cruves
ax.annotate(
    r'$\alpha = 2$',
    xy=(0.83, 0.95),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$\alpha = 1.5$',
    xy=(0.85, 0.69),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$\alpha = 1$',
    xy=(0.88, 0.25),
    xycoords='axes fraction',
    fontsize=13,
    color='k',
    ha='left',
    va='center',
)

# Plot the mean decorrelation scale as a function of segment duration 
ax.plot(
    duration_m, 
    decor_scale_m_depth[:,2], 
    '.-', 
    color='tab:blue', 
    lw=2, 
    markersize=8
)

# Plot the standard error of the mean decorrelation scale as a function of segment duration 
ax.fill_between(
    duration_m, 
    decor_scale_m_depth[:,2] + decor_scale_stdm_m_depth[:,2], 
    decor_scale_m_depth[:,2] - decor_scale_stdm_m_depth[:,2], 
    color='tab:blue', 
    alpha=0.25
)

# Set axis attributes 
ax.set_xlabel(r'Duration $T$ (months)')
ax.set_xticks(duration_m)
ax.set_yticks(np.arange(0,60+10,10))
ax.set_xlim(0,12.1)
ax.set_ylim(0,60)
ax.set_yticklabels([])
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

# Label each subplot
pos = [0.94, 0.07]
add_corner_label(ax_flat[0], pos, 'A', fontsize = 16)
add_corner_label(ax_flat[1], pos, 'B', fontsize = 16)
add_corner_label(ax_flat[2], pos, 'C', fontsize = 16)
add_corner_label(ax_flat[3], pos, 'D', fontsize = 16)
add_corner_label(ax_flat[4], pos, 'E', fontsize = 16)
add_corner_label(ax_flat[5], pos, 'F', fontsize = 16)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.2, wspace=0.1)

# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------

fig.savefig(
    PATH_figs / "fig09.png",
    dpi=300,
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
)

