# =============================================================================
# Figure S16
# =============================================================================
#
# Caption:
#   
#   
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-12
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo
import matplotlib as mpl

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
lat_bnds           = [33.0, 35.0]                                          
lon_bnds           = [237.0, 240.0]

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
# Load MITgcm spectral, velocity, mixed layer, and bathymetry data
# -----------------------------------------------------------------------------

# --- Spectral data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "transect" / "processed"

# Obtain filename path
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_trans_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
dist     = nc.variables['dist'][:]
depth    = nc.variables['depth'][:]
svf      = nc.variables['SVF'][:]
svf_stdm = nc.variables['SVF_stdm'][:]

# Obtain the low, mid and high frequency band spectral variance fraction and its uncertainty
svf_low      = svf[:,:,0]
svf_mid      = svf[:,:,1]
svf_high     = svf[:,:,2]

svf_stdm_low  = svf_stdm[:,:,0]
svf_stdm_mid  = svf_stdm[:,:,1]
svf_stdm_high = svf_stdm[:,:,2]

# --- Mixed Layer Depth --- # 

# Obtain filename path
filename_mld = PATH_processed / f"mitgcm_proc_density_hrly_trans.nc"

# Generate the nc data structure
nc = Dataset(filename_mld, 'r')

# Extract data variables
mld = nc.variables['MLD'][:]

# --- Velocity --- # 

# Obtain filename path
filename_mitgcm = PATH_processed / "mitgcm_proc_vel_hrly_trans.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
dist_vel   = nc.variables['dist'][:]
depth_vel  = nc.variables['depth'][:]
lon_vel    = nc.variables['LON'][:]
lat_vel    = nc.variables['LAT'][:]

u_along_trans_bar = nc.variables['u_along_bar'][:]
v_cross_trans_bar = nc.variables['v_cross_bar'][:]

# --- Bathymetry --- # 

# Obtain filename path
filename_bathy = PATH_data / "mitgcm" / "transect" / "DEPTH_CCS_trans.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
dist_wd     = nc.variables['dist'][:]
water_depth = nc_bathy.variables['water_depth'][:]

# Set the depth at the coast to zero 
water_depth[0] = 0

# --- CCE Mooring Locations --- # 
lat1, lat2, lat3  = 33.457, 34.3075, 34.44825228022894           
lon1, lon2, lon3  = -122.52233, -120.8042, -120.53825701527784 

# -----------------------------------------------------------------------------
# Compute the time mean and standard deviation mixed layer depth 
# -----------------------------------------------------------------------------

mld_mean = np.ma.mean(mld,axis=1)
mld_std = np.ma.std(mld,axis=1,ddof=1)

# -----------------------------------------------------------------------------
# Change the velocity naming convention from transect relative to shore relative
# -----------------------------------------------------------------------------

# ------------ #
# --- Note --- #
# ------------ #
#
# Original directional convention is relative to the transect line: 
# 
# Along-transect (cross-shelf direction)
# - Positive: onshore
# - Negative: offshore
#
# Cross-transect (along-shelf direction)
# - Positive: upcoast
# - Negative: downcoast
#
# The transect coordinates are ordered offshore -> onshore.
#
# ------------ #

# Along-shore velocity: positive upcoast
u_along_bar = v_cross_trans_bar 

# Cross-shore velocity: positive onshore
v_cross_bar = u_along_trans_bar 

# -----------------------------------------------------------------------------
# Extract velocity variables from the study domain and upper ocean
# -----------------------------------------------------------------------------

# Select transect locations within the study region
dist_mask = (
    (lat_vel >= lat_bnds[0])
    & (lat_vel <= lat_bnds[1])
    & (lon_vel >= lon_bnds[0])
    & (lon_vel <= lon_bnds[1])
)
idx_dist = np.where(dist_mask)[0]

# Extract velocity data within the specified region
lon         = lon_vel[idx_dist]
lat         = lat_vel[idx_dist]
u_along_bar = u_along_bar[idx_dist, :]
v_cross_bar = v_cross_bar[idx_dist, :]

# -----------------------------------------------------------------------------
# Compute the time-mean velocity fields
# -----------------------------------------------------------------------------

u_along_bar_mean = np.mean(u_along_bar, axis=1) 
v_cross_bar_mean = np.mean(v_cross_bar, axis=1) 

# -----------------------------------------------------------------------------
# Obtain the CU-CC boundary location 
# -----------------------------------------------------------------------------

# Convert masked array to NumPy array
u = np.ma.filled(u_along_bar_mean, np.nan)

# Identify valid points
valid = np.isfinite(u) & np.isfinite(dist)

# Find indices where alongshore velocity changes sign
idx_cross = np.where(
    valid[:-1] & valid[1:] & (u[:-1] * u[1:] < 0)
)[0]

# Define list to store the distances of zero-crossings
dist_cross = []

# Loop through indices of zero-crossings
for i in idx_cross:

    # Obtain the distances and velocities at the zero-crossing points
    d0, d1 = dist[i], dist[i+1]
    u0, u1 = u[i], u[i+1]

    # Linearly Interpolate to find the distance where the alongshore velocity is zero
    d_cross = d0 - u0 * (d1 - d0) / (u1 - u0)
    dist_cross.append(d_cross)

# Include any points where velocity is exactly zero
idx_zero = np.where(valid & (u == 0))[0]
dist_cross.extend(dist[idx_zero])

# -----------------------------------------------------------------------------
# Compute the relative uncertainty of the spectral variance fraction
# -----------------------------------------------------------------------------

# Compute spatial median
svf_reg_median_l = np.ma.median(svf_low)
svf_reg_median_m = np.ma.median(svf_mid)
svf_reg_median_h = np.ma.median(svf_high)
    
# Compute the signal-to-noise ratio (with respect to the regional mean)
svf_sn_ratio_l = np.abs(svf_low - svf_reg_median_l) / svf_stdm_low
svf_sn_ratio_m = np.abs(svf_mid - svf_reg_median_m) / svf_stdm_mid
svf_sn_ratio_h = np.abs(svf_high - svf_reg_median_h) / svf_stdm_high

# Mask not statistically significant grid points
nonsig_mask_l = np.ma.getmask(np.ma.masked_less_equal(svf_sn_ratio_l, sn_threshold))
nonsig_mask_m = np.ma.getmask(np.ma.masked_less_equal(svf_sn_ratio_m, sn_threshold))
nonsig_mask_h = np.ma.getmask(np.ma.masked_less_equal(svf_sn_ratio_h, sn_threshold))

# Get land mask from spectral slope
land_mask_l = np.ma.getmaskarray(svf_low)
land_mask_m = np.ma.getmaskarray(svf_mid)
land_mask_h = np.ma.getmaskarray(svf_high)

# Combine statistical significance and land masks
combine_mask_l = nonsig_mask_l & ~land_mask_l
combine_mask_m = nonsig_mask_m & ~land_mask_m
combine_mask_h = nonsig_mask_h & ~land_mask_h

# Create a mask array where non-significant ocean points = 1, others = NaN
svf_low_mask  = np.where(combine_mask_l, 1, np.nan)
svf_mid_mask  = np.where(combine_mask_m, 1, np.nan)
svf_high_mask = np.where(combine_mask_h, 1, np.nan)

# -----------------------------------------------------------------------------
# Plot spectral variance fraction   
# -----------------------------------------------------------------------------

# Set plotting parameters
cmap_svf   = cmo.rain
mpl.rcParams["hatch.linewidth"] = 0.2 

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# --- Subplot 1 --- # 
ax = axes[0]

# Plot the time-mean along-shore velocity 
cf = ax.contourf(dist,abs(depth),svf_low.T, levels=np.arange(40, 85+0.5, 0.5), cmap=cmap_svf, extend='both')

# Overlay a contourf with hatching for the non-significant regions
ax.contourf(
    dist,
    abs(depth),
    svf_low_mask.T,
    levels=[0.5, 1.5],      
    hatches=['....'],       
    colors='none',          
    zorder=10,              
)

# Plot the seasonal mean mixed layer depth
ax.plot(dist_wd, mld_mean, '--', color='k', linewidth=1.2, label=r"$\overline{z}_{mld}$")
ax.fill_between(dist_wd, mld_mean - mld_std, mld_mean + mld_std, color='k', alpha=0.1, label=r"$\sigma_{z_{mld}}$")

# Plot the ocean bottom depth 
ax.fill_between(dist_wd, abs(water_depth), abs(depth[-1]), color='0.4') 

# Loop through zero-crossings
for d_cross in dist_cross[:1]:

    # Plot the zero-crossing points of the alongshore velocity
    ax.axvline(d_cross, ls='--', color='k', alpha=0.7, lw=1.5)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(0,dist[-1])
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(linestyle='--',alpha=0.1,color='k')

# Set colorbar
cax = fig.add_axes([0.915, 0.645, 0.02, 0.23])
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
cbar.set_label(r'$\phi_\mathrm{low}$ ($\%$)')
cbar.set_ticks( np.arange(40, 80+10, 10))

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

# Remove bottom x-axis ticklabels
ax.set_xticklabels([])

# --- Subplot 2 --- # 
ax = axes[1]

# Plot the time-mean along-shore velocity 
cf = ax.contourf(dist,abs(depth),svf_mid.T, levels=np.arange(10, 50+0.5, 0.5), cmap=cmap_svf, extend='both')

# Overlay a contourf with hatching for the non-significant regions
ax.contourf(
    dist,
    abs(depth),
    svf_mid_mask.T,
    levels=[0.5, 1.5],      
    hatches=['....'],       
    colors='none',          
    zorder=10,              
)

# Plot the seasonal mean mixed layer depth
ax.plot(dist_wd, mld_mean, '--', color='k', linewidth=1.2, label=r"$\overline{z}_{mld}$")
ax.fill_between(dist_wd, mld_mean - mld_std, mld_mean + mld_std, color='k', alpha=0.1, label=r"$\sigma_{z_{mld}}$")

# Plot the ocean bottom depth 
ax.fill_between(dist_wd, abs(water_depth), abs(depth[-1]), color='0.4') 

# Loop through zero-crossings
for d_cross in dist_cross[:1]:

    # Plot the zero-crossing points of the alongshore velocity
    ax.axvline(d_cross, ls='--', color='k', alpha=0.7, lw=1.5)

    # Plot the text label
    ax.annotate(
        f'{d_cross:.1f} km',
        xy=(d_cross, 0.11),
        xycoords=ax.get_xaxis_transform(),
        xytext=(5, 0),
        textcoords='offset points',
        rotation=90,
        va='center',
        ha='left',
        fontsize=12,
        color='k',
    )

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(0,dist[-1])
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(linestyle='--',alpha=0.1,color='k')

# Set colorbar
cax = fig.add_axes([0.915, 0.38, 0.02, 0.23])
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
cbar.set_label(r'$\phi_\mathrm{mid}$ ($\%$)')
cbar.set_ticks( np.arange(10, 50+10, 10))

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

# Remove top and bottom x-axis ticklabels
ax.set_xticklabels([])
ax_top.set_xticklabels([])

# --- Subplot 3 --- # 
ax = axes[2]

# Plot the time-mean along-shore velocity 
cf = ax.contourf(dist,abs(depth),svf_high.T, levels=np.arange(0, 20+0.25, 0.25), cmap=cmap_svf, extend='both')

# Overlay a contourf with hatching for the non-significant regions
ax.contourf(
    dist,
    abs(depth),
    svf_high_mask.T,
    levels=[0.5, 1.5],      
    hatches=['....'],       
    colors='none',          
    zorder=10,              
)

# Plot the seasonal mean mixed layer depth
ax.plot(dist_wd, mld_mean, '--', color='k', linewidth=1.2, label=r"$\overline{z}_{mld}$")
ax.fill_between(dist_wd, mld_mean - mld_std, mld_mean + mld_std, color='k', alpha=0.1, label=r"$\sigma_{z_{mld}}$")

# Plot the ocean bottom depth 
ax.fill_between(dist_wd, abs(water_depth), abs(depth[-1]), color='0.4') 

# Loop through zero-crossings
for d_cross in dist_cross[:1]:

    # Plot the zero-crossing points of the alongshore velocity
    ax.axvline(d_cross, ls='--', color='k', alpha=0.7, lw=1.5)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(0,dist[-1])
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(linestyle='--',alpha=0.1,color='k')

# Set colorbar
cax = fig.add_axes([0.915, 0.115, 0.02, 0.23])
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
cbar.set_label(r'$\phi_\mathrm{high}$ ($\%$)')
cbar.set_ticks(np.arange(0, 20+5, 5))

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

# Remove top x-axis ticklabels
ax_top.set_xticklabels([])

# Label each subplot
ax1, ax2, ax3 = axes.flatten()
add_corner_label(ax1, [0.025,0.075], 'A', fontsize = 16)
add_corner_label(ax2, [0.025,0.075], 'B', fontsize = 16)
add_corner_label(ax3, [0.025,0.075], 'C', fontsize = 16)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "figS16.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)



