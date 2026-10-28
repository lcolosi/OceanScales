# =============================================================================
# Figure 13
# =============================================================================
#
# Caption:
#   (a) Decorrelation time scale depth-averaged from the surface to 30 meters
#   water depth along the CalCOFI line 80.0 transect. Gray shading represents
#   $\pm1$ standard error of the mean depth averaged over the same depth range.
#   Vertical dashed lines show the position of CCE1 (green), CCE2 (red), and 
#   CCE3 (blue) moorings. (b) Time-mean ocean velocity vectors in the cross-shore
#   and along-shore coordinate system and depth averaged from the surface to
#   400 meters water depth. (c) Time-mean along-shore velocity component. Gray
#   shading is the ocean bottom. (d) Fraction of the total potential density 
#   variance explained (FVE) by low-frequency variability with frequencies 
#   $f < 1/30 cpd$
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
import matplotlib.gridspec as gridspec

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
Lt_std  = nc.variables['decor_scale_std'][:]

# --- Spectral data --- # 

# Obtain filename path
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_trans_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
fve      = nc.variables['FVE'][:]

# --- Velocity --- # 

# Obtain filename path
filename_mitgcm = PATH_processed / "mitgcm_proc_vel_hrly_trans.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
dist_vel    = nc.variables['dist'][:]
depth_vel   = nc.variables['depth'][:]
lon_vel     = nc.variables['LON'][:]
lat_vel     = nc.variables['LAT'][:]

u_along_trans     = nc.variables['u_along'][:]
v_cross_trans     = nc.variables['v_cross'][:]
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
u_along     = v_cross_trans 
u_along_bar = v_cross_trans_bar 

# Cross-shore velocity: positive onshore
v_cross     = u_along_trans 
v_cross_bar = u_along_trans_bar 

# -----------------------------------------------------------------------------
# Extract velocity variables from the study domain and upper ocean
# -----------------------------------------------------------------------------

# Select depth levels shallower than the depth limit
idx_depth = depth_vel >= depth_lim

# Select transect locations within the study region
dist_mask = (
    (lat_vel >= lat_bnds[0])
    & (lat_vel <= lat_bnds[1])
    & (lon_vel >= lon_bnds[0])
    & (lon_vel <= lon_bnds[1])
)
idx_dist = np.where(dist_mask)[0]

# Extract velocity data within the specified region
u_along = u_along[idx_dist, :, :]
u_along = u_along[:, :, idx_depth]

v_cross = v_cross[idx_dist, :, :]
v_cross = v_cross[:, :, idx_depth]

u_along_bar = u_along_bar[idx_dist, :]
v_cross_bar = v_cross_bar[idx_dist, :]

# -----------------------------------------------------------------------------
# Compute the time-mean velocity fields
# -----------------------------------------------------------------------------

u_along_mean = np.mean(u_along, axis=1) 
v_cross_mean = np.mean(v_cross, axis=1) 
u_along_bar_mean = np.mean(u_along_bar, axis=1) 
v_cross_bar_mean = np.mean(v_cross_bar, axis=1) 

# -----------------------------------------------------------------------------
# Compute the depth average of the decorrelation scale and the standard error
# -----------------------------------------------------------------------------

# Convert depth to positive-downward coordinates
z = np.abs(depth)

# Select depth levels within the upper 30 m
idx_depth_Lt = z <= depth_avg_max

# Extract depths and decorrelation scales
z_sel       = z[idx_depth_Lt]
Lt_sel      = np.ma.masked_invalid(Lt[:, idx_depth_Lt])
Lt_stdm_sel = np.ma.masked_invalid(Lt_stdm[:, idx_depth_Lt])

# Initialize depth-averaged decorrelation scale
Lt_depth_avg = np.ma.masked_all(len(dist))
Lt_stdm_depth_avg = np.ma.masked_all(len(dist))

# Loop through distance offshore
for idist in range(len(dist)):

    # Extract vertical profile
    Lt_prof      = Lt_sel[idist, :]
    Lt_stdm_prof = Lt_stdm_sel[idist, :]

    # Identify valid depth levels
    valid = ~np.ma.getmaskarray(Lt_prof) & ~np.ma.getmaskarray(Lt_stdm_prof)

    # Require at least two valid depth levels
    if np.sum(valid) < 2:
        continue

    # Obtain valid depths and decorrelation scales
    z_valid       = z_sel[valid]
    Lt_valid      = Lt_prof[valid]
    Lt_stdm_valid = Lt_stdm_prof[valid]

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
fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 2, 2]) 

#--- Subplot 1 ---# 
ax = fig.add_subplot(gs[0])
ax1 = ax

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
ax.grid(linestyle='--', alpha=0.1,color='k') 

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
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.05, y_text=1.035, ms= 4, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.05, y_text=1.035, ms =4, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.05, y_text=1.035, ms =4, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Remove bottom x-axis ticklabels
ax.set_xticklabels([])

#--- Subplot 2 ---# 
ax = fig.add_subplot(gs[1])
ax2 = ax

# Plot the time-mean velocity vectors in the cross and along-shore coordinates 
q = ax.quiver(dist_vel[idx_dist], np.zeros_like(dist_vel[idx_dist]), v_cross_bar_mean, u_along_bar_mean, angles='uv', scale=1.3, width=0.001)

# Set axis attributes
ax.set_xlim(0, dist[-1])
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks([])
ax.invert_xaxis()
ax.tick_params(top=True, bottom=True, left=False, right=False, direction='out', length=3.5)
ax.grid(linestyle='--', alpha=0.1,color='k')  

# Set quiver legend
ax.quiverkey(q, X=0.75, Y=0.2, U=0.1, label='0.1 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 12})

# Add coordinate system arrows 
ax.annotate('', xy=(x0 + arrow_length_x, y0), xytext=(x0, y0),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.5, headwidth=3, headlength=3))
ax.text(x0 + arrow_length_x + 0.01, y0, 'Cross-shore', transform=ax.transAxes,
        verticalalignment='center', fontsize=8)

ax.annotate('', xy=(x0, y0 + arrow_length_y), xytext=(x0, y0),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.5, headwidth=3, headlength=3))
ax.text(x0, y0 + arrow_length_y + 0.02, 'Along-shore', transform=ax.transAxes,
        horizontalalignment='center', fontsize=8)

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
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.05, y_text=1.035, ms=4, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.05, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.05, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Remove top and bottom x-axis ticklabels
ax_top.set_xticklabels([])
ax.set_xticklabels([])

#--- Subplot 3 ---# 
ax = fig.add_subplot(gs[2])
ax3 = ax

# Plot the time-mean along-shore velocity 
cf = ax.contourf(dist,abs(depth),u_along_mean.T, levels=levels_vel, cmap=cmap_vel, extend='both')

# Plot the zero-contour of the time-mean along-shore velocity 
ax.contour(dist,abs(depth),u_along_mean.T, levels=[0], colors='black', linewidths=2)

# Plot the ocean bottom depth 
ax.fill_between(dist_wd, abs(water_depth), abs(depth[-1]), color='0.4') 

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
cax = fig.add_axes([0.915, 0.3725, 0.02, 0.23])
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
cbar.set_label('Along-shore Velocity (m/s)')
cbar.set_ticks(ticks_vel)

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

# Add CCE1, CCE2, and CCE3 locations markers
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.02, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.02, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.02, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Remove top and bottom x-axis ticklabels
ax_top.set_xticklabels([])
ax.set_xticklabels([])

#--- Subplot 4 ---# 
ax = fig.add_subplot(gs[3])
ax4 = ax

# Plot the time-mean along-shore velocity 
cf = ax.contourf(dist,abs(depth),fve[:,:,0].T, levels=levels_fve, cmap=cmap_fve, extend='both')

# Plot the ocean bottom depth 
ax.fill_between(dist_wd, abs(water_depth), abs(depth[-1]), color='0.4') 

# Set axis attributes
ax.set_xlabel('Distance from shore (km)')
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
cbar.set_label(r'FVE ($\%$)')
cbar.set_ticks(ticks_fve)

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

# Add CCE1, CCE2, and CCE3 locations markers
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.02, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.02, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.02, y_text=1.035,ms=4, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Remove top x-axis ticklabels
ax_top.set_xticklabels([])

# Label each subplot
add_corner_label(ax1, [0.025,0.145], 'A', fontsize = 13)
add_corner_label(ax2, [0.025,0.145], 'B', fontsize = 13)
add_corner_label(ax3, [0.025,0.08], 'C', fontsize = 13)
add_corner_label(ax4, [0.025,0.08], 'D', fontsize = 13)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig13.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)