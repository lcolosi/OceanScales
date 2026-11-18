# =============================================================================
# Figure S12
# =============================================================================
#
# Caption:
#    
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
from matplotlib.colors import to_rgba
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
option_depth         = 9.6 
segment_months     = 6
depth_lim          = -220 
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
# Load MITgcm and CCE decorrelation scales, bathymetry, and velocity data
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
var      = nc.variables['VAR'][:]
svf      = nc.variables['SVF'][:]
svf_stdm = nc.variables['SVF_stdm'][:]

# --- Bathymetry --- # 

# Obtain filename path
filename_bathy = PATH_data / "mitgcm" / "transect" / "DEPTH_CCS_trans.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
water_depth = nc_bathy.variables['water_depth'][:]

# Set the depth at the coast to zero 
water_depth[0] = 0

# --- CCE Mooring Locations --- # 
lat1, lat2, lat3  = 33.457, 34.3075, 34.44825228022894           
lon1, lon2, lon3  = -122.52233, -120.8042, -120.53825701527784 

# --- CCE Autocorrelation Data --- # 

# Set path to processed regional MITgcm data
PATH_processed_cce1 = PATH_data / "cce" / "cce1" / "processed"
PATH_processed_cce2 = PATH_data / "cce" / "cce2" / "processed"

# Obtain filename paths
filename_cce1 = PATH_processed_cce1 / f"cce1_decor_scale_{option_data}_hrly_mooring_gaussian_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_decor_scale_{option_data}_hrly_mooring_gaussian_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc_cce1 = Dataset(filename_cce1, 'r')
nc_cce2 = Dataset(filename_cce2, 'r')

# Extract data variables
depth_cce1    = nc_cce1.variables['depth'][:]
Lt_cce1       = nc_cce1.variables['decor_scale'][:]
Lt_stdm_cce1  = nc_cce1.variables['decor_scale_stdm'][:]

depth_cce2    = nc_cce2.variables['depth'][:]
Lt_cce2       = nc_cce2.variables['decor_scale'][:]
Lt_stdm_cce2  = nc_cce2.variables['decor_scale_stdm'][:]

# Close the netcdf file
nc_cce1.close()
nc_cce2.close()

# --- CCE Spectral Data --- # 

# Obtain filename paths
filename_cce1 = PATH_processed_cce1 / f"cce1_spectra_{option_data}_hrly_mooring_gaussian_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_spectra_{option_data}_hrly_mooring_gaussian_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc_cce1 = Dataset(filename_cce1, 'r')
nc_cce2 = Dataset(filename_cce2, 'r')

# Extract data variables
var_cce1      = nc_cce1.variables['VAR'][:]
svf_cce1      = nc_cce1.variables['SVF'][:]
svf_stdm_cce1 = nc_cce1.variables['SVF_stdm'][:]

var_cce2      = nc_cce2.variables['VAR'][:]
svf_cce2      = nc_cce2.variables['SVF'][:]
svf_stdm_cce2 = nc_cce2.variables['SVF_stdm'][:]

# Close the netcdf file
nc_cce1.close()
nc_cce2.close()

# -----------------------------------------------------------------------------
# Find the 200 m and 2000 m isobath locations along the transect
# -----------------------------------------------------------------------------

# Set helper function
def find_isobath_distance(distances, bathy, target_depth):
    """
    Return the distance from shore where the transect crosses a target isobath.
    Uses the first crossing moving offshore. If no sign change is found,
    returns the location of the closest bathymetric value.
    """
    distances = np.asarray(distances, dtype=float)
    bathy = np.abs(np.asarray(bathy, dtype=float))

    # Difference from target depth
    diff = bathy - target_depth

    # Find crossings between adjacent grid cells
    idx = np.where(diff[:-1] * diff[1:] <= 0)[0]

    if idx.size > 0:
        i = idx[0]
        return np.interp(
            target_depth,
            [bathy[i], bathy[i+1]],
            [distances[i], distances[i+1]]
        )
    else:
        # Fallback: use nearest point
        return distances[np.nanargmin(np.abs(bathy - target_depth))]

# Find the distances corresponding to the 200 m and 2000 m isobaths
dist_200m  = find_isobath_distance(dist, water_depth, 200.0)
dist_2000m = find_isobath_distance(dist, water_depth, 2000.0)

# -----------------------------------------------------------------------------
# Extract data from the specified depth 
# -----------------------------------------------------------------------------

# Set depth to positive values 
depth_pos = np.abs(depth)
depth_cce1_pos = np.abs(depth_cce1)
depth_cce2_pos = np.abs(depth_cce2)

# Find the depth levels closest to the specified depth 
idx_depth = np.argmin(np.abs(depth_pos - option_depth))
idx_depth_cce1 = np.argmin(np.abs(depth_cce1_pos - option_depth))
idx_depth_cce2 = np.argmin(np.abs(depth_cce2_pos - option_depth))

# Extract data at the specified depth
Lt_depth       = Lt[:, idx_depth]
Lt_depth_cce1 = Lt_cce1[idx_depth_cce1]
Lt_depth_cce2 = Lt_cce2[idx_depth_cce2]

Lt_stdm_depth  = Lt_stdm[:, idx_depth]
Lt_stdm_depth_cce1 = Lt_stdm_cce1[idx_depth_cce1]
Lt_stdm_depth_cce2 = Lt_stdm_cce2[idx_depth_cce2]

var_depth      = var[:,idx_depth, :]
var_depth_cce1 = var_cce1[idx_depth_cce1, :]
var_depth_cce2 = var_cce2[idx_depth_cce2, :]

svf_depth      = svf[:,idx_depth, :]
svf_depth_cce1 = svf_cce1[idx_depth_cce1, :]
svf_depth_cce2 = svf_cce2[idx_depth_cce2, :]

svf_stdm_depth = svf_stdm[:,idx_depth, :]
svf_stdm_depth_cce1 = svf_stdm_cce1[idx_depth_cce1, :]
svf_stdm_depth_cce2 = svf_stdm_cce2[idx_depth_cce2, :]

# -----------------------------------------------------------------------------
# Plot decorrelation time scale, absolute variance, and SVF  
# -----------------------------------------------------------------------------

# Set plotting parameters
period_bounds = [4.0, 30.0, 182.5]
svf_bands = ['low', 'mid', 'high']
svf_colors = {
    'low': 'tab:green',
    'mid': 'tab:red',
    'high': 'tab:blue',
}
svf_labels = {
    'low': r'$\phi_\mathrm{low}$',
    'mid': r'$\phi_\mathrm{mid}$',
    'high': r'$\phi_\mathrm{high}$',
}

# Create figure
fig = plt.figure(figsize=(13,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 2]) 

#--- Subplot 1 ---# 
ax = fig.add_subplot(gs[0])
ax1 = ax

# Plot decorrelation time scale
lt_line, = ax.plot(dist, Lt_depth, color='k', linewidth=2, zorder=3, label=r'$\overline{\mathcal{T}}$')

# Plot one standard error envelope of the decorrelation time scale
lt_error = ax.fill_between(dist, Lt_depth - Lt_stdm_depth, Lt_depth + Lt_stdm_depth, color='k', alpha=0.15, linewidth=0, zorder=2, label=r'$\sigma_{\overline{\mathcal{T}}}$')

# Set axis attributes
ax.set_ylabel('Decorrelation Scale (days)')
ax.set_xlim(0, dist[-1])
ax.set_ylim(0,25) 
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(0,25+5,5)) 
ax.invert_xaxis()
ax.tick_params(top=True, bottom=True, left=True, right=False, direction='out', length=3.5)
ax.grid(axis='x', linestyle='--', alpha=0.1, color='k')

# --- Create right axis for SVF --- #
ax_svf = ax.twinx()

# Initialize list to hold handles for legend
svf_handles = []

# Loop through spectral bands
for iband, band in enumerate(svf_bands):

    # Extract SVF and its uncertainty
    values = svf_depth[:, iband]
    errors = svf_stdm_depth[:, iband]

    # Set color
    color = svf_colors[band]

    # Plot the spectral variance fraction
    line, = ax_svf.plot(
        dist,
        values,
        color=color,
        linewidth=1.8,
        label=svf_labels[band],
        zorder=4,
    )

    # Plot uncertainty envelope (+/- 1 standard error)
    ax_svf.fill_between(
        dist,
        values - errors,
        values + errors,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=3,
    )

    # Append line handle for legend
    svf_handles.append(line)

# Set axis attributes for SVF
ax_svf.set_ylabel(r'$\phi$ ($\%$)')
ax_svf.set_ylim(0, 90)
ax_svf.set_yticks(np.arange(0, 91, 10))

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

# Plot 200 m and 2000 m isobath markers
ax.axvline(dist_200m,  color='k', linestyle='--',  lw=1.0, alpha=0.7, zorder=1)
ax.axvline(dist_2000m, color='k', linestyle='--', lw=1.0, alpha=0.7, zorder=1)

# Label isobaths near the top of panel A
ax.text(
    dist_200m, 0.97, '200 m',
    transform=ax.get_xaxis_transform(),
    rotation=90, ha='right', va='top',
    fontsize=11, color='k',
)

ax.text(
    dist_2000m, 0.97, '2000 m',
    transform=ax.get_xaxis_transform(),
    rotation=90, ha='right', va='top',
    fontsize=11, color='k',
)

# Remove bottom x-axis ticklabels
ax.set_xticklabels([])

# --- Plot observational decorrelation scales and SVF --- #

# Organize observational data
moorings = [
    {
        'name': 'CCE1',
        'dist': dist1,
        'Lt': Lt_depth_cce1,
        'Lt_err': Lt_stdm_depth_cce1,
        'svf': svf_depth_cce1,
        'svf_err': svf_stdm_depth_cce1,
        'marker': '^',
        'color': 'tab:green',
    },
    {
        'name': 'CCE2',
        'dist': dist2,
        'Lt': Lt_depth_cce2,
        'Lt_err': Lt_stdm_depth_cce2,
        'svf': svf_depth_cce2,
        'svf_err': svf_stdm_depth_cce2,
        'marker': 's',
        'color': 'tab:red',
    },
]

# Initialize list for observational legend handles
obs_handles = []

# Loop through moorings
for mooring in moorings:

    # Plot decorrelation scales with error bars
    handle = ax.errorbar(
        mooring['dist'],
        mooring['Lt'],
        yerr=mooring['Lt_err'],
        fmt=mooring['marker'],
        color='k',
        markersize=7,
        markeredgecolor='k',
        markeredgewidth=0.5,
        capsize=4,
        elinewidth=1.25,
        linestyle='none',
        zorder=10,
        label=mooring['name'],
    )

    obs_handles.append(handle)

    # Loop through spectral bands
    for iband, band in enumerate(svf_bands):

        # Set spectral band color
        color = svf_colors[band]

        # Plot observational SVF with uncertainty
        ax_svf.errorbar(
            mooring['dist'],
            mooring['svf'][iband],
            yerr=mooring['svf_err'][iband],
            fmt=mooring['marker'],
            color=color,
            ecolor=to_rgba(color, alpha=0.7),
            markersize=7,
            markeredgecolor='none',
            capsize=3,
            elinewidth=1.25,
            linestyle='none',
            zorder=10,
            alpha=0.7,
        )

# Combine decorrelation scale and SVF legend handles
legend_handles = svf_handles + [lt_line, lt_error] + obs_handles

# Place legend outside the axes on the right
ax_svf.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(1.075, 0.5),
    fontsize=11,
    frameon=True,
    framealpha=0.85,
    ncol=1,
)

#--- Subplot 2 ---# 
ax = fig.add_subplot(gs[1])
ax2 = ax

# Plot absolute variances for low, mid, and high frequency bands 
ax.semilogy(dist, var_depth[:,0], '-', color='tab:green', linewidth=2, label=r'$\sigma^2_\mathrm{low}$')
ax.semilogy(dist, var_depth[:,1], '-', color='tab:red', linewidth=2, label=r'$\sigma^2_\mathrm{mid}$')
ax.semilogy(dist, var_depth[:,2], '-', color='tab:blue', linewidth=2, label=r'$\sigma^2_\mathrm{high}$')

# Set axis attributes
ax.set_xlabel('Distance from shore (km)')
ax.set_ylabel(r'$\sigma^2$ ($\mathrm{kg}^2/\mathrm{m}^6$)')
ax.set_xlim(0, dist[-1])
ax.set_ylim(10**(-4), 10**(-1))
ax.set_xticks(np.arange(0,250+25,25))
ax.invert_xaxis()
ax.tick_params(top=True, bottom=True, left=True, right=False, direction='out', length=3.5)
ax.grid(which='both', linestyle='--', alpha=0.1,color='k') 
ax.legend(loc='upper center', fontsize=11, ncol=3)

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

# Plot 200 m and 2000 m isobath markers
ax.axvline(dist_200m,  color='k', linestyle='--',  lw=1.0, alpha=0.7, zorder=1)
ax.axvline(dist_2000m, color='k', linestyle='--', lw=1.0, alpha=0.7, zorder=1)

# Remove top x-axis ticklabels
ax_top.set_xticklabels([])

# --- Plot observational decorrelation scales and SVF --- #

# Organize observational data
moorings = [
    {
        'name': 'CCE1',
        'dist': dist1,
        'var': var_depth_cce1,
        'marker': '^',
        'color': 'tab:green',
    },
    {
        'name': 'CCE2',
        'dist': dist2,
        'var': var_depth_cce2,
        'marker': 's',
        'color': 'tab:red',
    },
]

# Initialize list for observational legend handles
# obs_handles = []

# Loop through moorings
for mooring in moorings:

    # Loop through spectral bands
    for iband, band in enumerate(svf_bands):

        # Set spectral band color
        color = svf_colors[band]

        # Plot observational SVF with uncertainty
        ax.plot(
            mooring['dist'],
            mooring['var'][iband],
            mooring['marker'],
            color=color,
            markersize=7,
            markeredgecolor='none',
            linestyle='none',
            zorder=10,
            alpha=0.7,
        )

# Label each subplot
add_corner_label(ax1, [0.0225,0.075], 'A', fontsize = fontsize)
add_corner_label(ax2, [0.0225,0.95], 'B', fontsize = fontsize)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "figS12.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)


