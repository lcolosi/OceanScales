# =============================================================================
# Figure S11
# =============================================================================
#
# Caption:
#   (a) Decorrelation time scale depth-averaged from the surface to 30 meters
#   water depth along the CalCOFI line 80.0 transect. Gray shading represents
#   $\pm1$ standard error of the mean depth averaged over the same depth range.
#   Vertical dashed lines show the position of CCE1 (green), CCE2 (red), and 
#   CCE3 (blue) moorings. (b) 
#   
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-13
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
from plotting import add_x_axis_marker, add_corner_label

# -----------------------------------------------------------------------------
# Set processing and plotting parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density", "u_along", or "v_cross".
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: "detrend", "demean", or None
# - option_averaging: Specifies whether to compute the variance-weighted or unweighted
#                     ensemble-average of the autocorrelation.
#                      Options: "weighted" for variance-weighted average.
#                               "unweighted" for unweighted average.
# - segment_months : Specifies the window duration. 
# - sn_threshold : Signal-to-noise ratio threshold for the statistical significance 
#                  criteria. Represents the number of standard deviation a
#                  decorrelation scale estimate is away from the regional spatial
#                  median.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_detrend_seg = "detrend"
option_averaging   = "weighted" 
option_slope_range = [182.5, 4.0] 
segment_months     = 6
depth_lim          = -220 
lat_bnds           = [33.0, 35.0]                                          
lon_bnds           = [237.0, 240.0]
depth_avg_max      = 30

# Label segment processing method 
seg_proc = "none" if option_detrend_seg is None else option_detrend_seg

# Set uncertainty estimate parameters
sn_threshold = 1

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm decorrelation scales, bathymetry, and velocity data
# -----------------------------------------------------------------------------

# --- Decorrelation Time Scales --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "transect" / "processed"

# Obtain filename path
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_trans_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
dist    = nc.variables['dist'][:]
depth   = nc.variables['depth'][:]
lon     = nc.variables['LON'][:]
lat     = nc.variables['LAT'][:]
Lt      = nc.variables['decor_scale'][:]
Lt_stdm = nc.variables['decor_scale_stdm'][:]

# --- Spectral data --- # 

# Obtain filename path
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_trans_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
var_band = nc.variables['VAR'][:]
fve      = nc.variables['FVE'][:]

# Set low and high frequency components of variance and fve
var_low  = var_band[:,:,0]
var_high = var_band[:,:,1]
fve_low  = fve[:,:,0]

# --- CCE Mooring Locations --- # 
lat1, lat2, lat3  = 33.457, 34.3075, 34.44825228022894           
lon1, lon2, lon3  = -122.52233, -120.8042, -120.53825701527784 

# -----------------------------------------------------------------------------
# Compute the depth average of the decorrelation scale and variance
# -----------------------------------------------------------------------------

# Convert depth to positive-downward coordinates
z = np.abs(depth)

# Select depth levels within the upper 30 m
idx_depth = z <= depth_avg_max

# Extract depths and data
z_sel        = z[idx_depth]
Lt_sel       = np.ma.masked_invalid(Lt[:, idx_depth])
Lt_stdm_sel  = np.ma.masked_invalid(Lt_stdm[:, idx_depth])
var_low_sel  = np.ma.masked_invalid(var_low[:, idx_depth])
var_high_sel = np.ma.masked_invalid(var_high[:, idx_depth])
fve_low_sel  = np.ma.masked_invalid(fve_low[:, idx_depth])

# Initialize depth-averaged decorrelation scale
Lt_depth_avg = np.ma.masked_all(len(dist))
Lt_stdm_depth_avg = np.ma.masked_all(len(dist))
var_low_depth_avg  = np.ma.masked_all(len(dist))
var_high_depth_avg = np.ma.masked_all(len(dist))
fve_low_depth_avg  = np.ma.masked_all(len(dist))

# Loop through distance offshore
for idist in range(len(dist)):

    # Extract vertical profile
    Lt_prof       = Lt_sel[idist, :]
    Lt_stdm_prof  = Lt_stdm_sel[idist, :]
    var_low_prof  = var_low_sel[idist, :]
    var_high_prof = var_high_sel[idist, :]
    fve_low_prof  = fve_low_sel[idist, :]

    # Identify valid depth levels
    valid = (~np.ma.getmaskarray(Lt_prof) 
             & ~np.ma.getmaskarray(Lt_stdm_prof) 
             & ~np.ma.getmaskarray(var_low_prof) 
             & ~np.ma.getmaskarray(var_high_prof)
             & ~np.ma.getmaskarray(fve_low_prof)
    )

    # Require at least two valid depth levels
    if np.sum(valid) < 2:
        continue

    # Obtain valid depths and decorrelation scales
    z_valid        = z_sel[valid]
    Lt_valid       = Lt_prof[valid]
    Lt_stdm_valid  = Lt_stdm_prof[valid]
    var_low_valid  = var_low_prof[valid]    
    var_high_valid = var_high_prof[valid]
    fve_low_valid  = fve_low_prof[valid]

    # Compute depth range
    H = z_valid[-1] - z_valid[0]

    # Compute depth-averaged decorrelation scale
    if H > 0:
        Lt_depth_avg[idist] = (
            np.trapezoid(Lt_valid, z_valid) / H
        )
        Lt_stdm_depth_avg[idist] = (
            np.trapezoid(Lt_stdm_valid, z_valid) / H
        )
        var_low_depth_avg[idist] = (
            np.trapezoid(var_low_valid, z_valid) / H
        )
        var_high_depth_avg[idist] = (
            np.trapezoid(var_high_valid, z_valid) / H
        )
        fve_low_depth_avg[idist] = (
            np.trapezoid(fve_low_valid, z_valid) / H
        )

# -----------------------------------------------------------------------------
# Plot depth average decorrelation time scale, velocity, and FVE  
# -----------------------------------------------------------------------------

# Set plotting parameters
levels_vel = np.arange(-0.15, 0.15 + 0.005, 0.005)
ticks_vel = np.arange(-0.15, 0.15 + 0.05, 0.05)
levels_fve = np.arange(60, 92.5+0.5, 0.5)
ticks_fve  = np.arange(60, 90+5, 5)
cmap_vel = cmo.balance
cmap_fve = cmo.rain
arrow_length_x,arrow_length_y = 0.015, 0.08  
x0, y0 = 0.075, 0.78 

# Create figure
fig, axes = plt.subplots(2,1,figsize=(10,8))

#--- Subplot 1 ---# 
ax = axes[0]

# Plot uncertainty envelope
ax.fill_between(dist, Lt_depth_avg - Lt_stdm_depth_avg, Lt_depth_avg + Lt_stdm_depth_avg, color='k', alpha=0.15, linewidth=0, zorder=2)

# Plot depth-averaged decorrelation scale
ax.plot(dist, Lt_depth_avg, color='k', linewidth=2, zorder=3)

# Set axis attributes
ax.set_ylabel(r'$\langle\overline{\mathcal{T}}\rangle_z$ (days)')
ax.set_xlim(0, dist[-1])
ax.set_ylim(5,25) 
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(5,25+5,5)) 
ax.invert_xaxis()
ax.tick_params(top=True, bottom=True, left=True, right=False, direction='out', length=3.5)
ax.grid(axis='x', linestyle='--', alpha=0.1, color='k')

# --- Create a left axis for FVE --- # 
ax_lf = ax.twinx()

# Plot the depth average fve 
ax_lf.plot(dist, fve_low_depth_avg, color='tab:purple', linewidth=2, label='FVE')

# Set axis attributes 
ax_lf.set_ylabel(r'$\langle\mathrm{FVE}\rangle_z$ ($\%$)', color='tab:purple')
ax_lf.set_ylim(60, 92.5)
ax_lf.set_yticks(np.arange(60, 92.5 + 5, 5))
ax_lf.tick_params(axis='y', colors='tab:purple', direction='out', length=3.5)

# --- Create top axis for longitude --- #
ax_top = ax.twiny()

# Make sure limits match
ax_top.set_xlim(ax.get_xlim())

# Choose where you want longitude ticks (same positions as distance ticks)
dist_ticks = ax.get_xticks()

# Interpolate longitude at those distance values
lon_180 = ((lon + 180) % 360) - 180
lon_ticks = np.interp(dist_ticks, dist, lon_180)

# Create labels but only keep every other one
labels = [
    f"{abs(x):.1f}°W" if i % 2 == 0 else ""
    for i, x in enumerate(lon_ticks)
]

# Set ticks and labels
ax_top.set_xticks(dist_ticks)
ax_top.set_xticklabels(labels) 

sort_idx = np.argsort(lon_180)
lon_sorted = lon_180[sort_idx]
dist_sorted = dist[sort_idx]

# Interpolate longtiude onto distance coordinates 
dist1 = np.interp(lon1, lon_sorted, dist_sorted)
dist2 = np.interp(lon2, lon_sorted, dist_sorted)
dist3 = np.interp(lon3, lon_sorted, dist_sorted)

# Add CCE1, CCE2, and CCE3 locations 
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.02, y_text=1.035, ms= 6, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.02, y_text=1.035, ms =6, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.02, y_text=1.035, ms =6, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Remove bottom x-axis ticklabels
ax.set_xticklabels([])

#--- Subplot 2 ---# 
ax = axes[1]

# Plot depth-averaged variances for low and high frequency bands 
ax.semilogy(dist, var_low_depth_avg, '-', color='tab:red', linewidth=2, label='Low Frequency')
ax.semilogy(dist, var_high_depth_avg, '-', color='tab:blue', linewidth=2, label='High Frequency')

# Set axis attributes
ax.set_xlabel('Distance from shore (km)')
ax.set_ylabel(r'$\langle\sigma^2\rangle_z$ ($\mathrm{kg}^2/\mathrm{m}^6$)')
ax.set_xlim(0, dist[-1])
ax.set_xticks(np.arange(0,250+25,25))
ax.invert_xaxis()
ax.tick_params(top=True, bottom=True, left=True, right=False, direction='out', length=3.5)
ax.grid(linestyle='--', alpha=0.1,color='k') 
ax.legend(loc='lower right', fontsize=14)

# --- Create top axis for longitude --- #
ax_top = ax.twiny()

# Make sure limits match
ax_top.set_xlim(ax.get_xlim())

# Choose where you want longitude ticks (same positions as distance ticks)
dist_ticks = ax.get_xticks()

# Interpolate longitude at those distance values
lon_180 = ((lon + 180) % 360) - 180
lon_ticks = np.interp(dist_ticks, dist, lon_180)

# Create labels but only keep every other one
labels = [
    f"{abs(x):.1f}°W" if i % 2 == 0 else ""
    for i, x in enumerate(lon_ticks)
]

# Set ticks and labels
ax_top.set_xticks(dist_ticks)
ax_top.set_xticklabels(labels) 

sort_idx = np.argsort(lon_180)
lon_sorted = lon_180[sort_idx]
dist_sorted = dist[sort_idx]

# Interpolate longtiude onto distance coordinates 
dist1 = np.interp(lon1, lon_sorted, dist_sorted)
dist2 = np.interp(lon2, lon_sorted, dist_sorted)
dist3 = np.interp(lon3, lon_sorted, dist_sorted)

# Add CCE1, CCE2, and CCE3 locations 
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.02, y_text=1.035, ms= 6, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.02, y_text=1.035, ms =6, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.02, y_text=1.035, ms =6, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Remove top x-axis ticklabels
ax_top.set_xticklabels([])

# Label each subplot
ax1, ax2 = axes.flatten()
add_corner_label(ax1, [0.035,0.91], 'A', fontsize = fontsize)
add_corner_label(ax2, [0.035,0.91], 'B', fontsize = fontsize)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "figS11.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)


