# =============================================================================
# Figure S13
# =============================================================================
#
# Caption:
#    
#   
# Author:
#   Luke Colosi
#
# Created:
#   2026-11-17
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms

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
#                        Options: "detrend", "demean", or None
# - option_averaging: Specifies whether to compute the variance-weighted or unweighted
#                     ensemble-average of the autocorrelation.
#                      Options: "weighted" for variance-weighted average.
#                               "unweighted" for unweighted average.
# - option_plot: Specifies the whether the variance density is expressed as per unit
#                frequency or per unit logarithmic frequency.
#                Options: "log" for variance density per unit logarithmic frequency.
#                         "linear" for variance density per unit frequency.
# - segment_months : Specifies the window duration. 
# - option_depth: Specifies the water depth which the data is extracted from 
#                 at each mooring.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_detrend_seg = "detrend"
option_averaging   = "weighted"
segment_months     = 6
option_depth       = 9.6
option_slope_range = [182.5, 4.0]      

# Label segment processing 
seg_proc = "none" if option_detrend_seg is None else option_detrend_seg

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm autocorrelation and spectral data
# -----------------------------------------------------------------------------

# --- MITgcm Autocorrelation Data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_mooring_linear_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
site    = nc.variables['site'][:]
depth   = nc.variables['depth'][:]
Lt      = nc.variables['decor_scale'][:]
Lt_stdm = nc.variables['decor_scale_stdm'][:]

# Close the netcdf file
nc.close()

# --- MITgcm Spectral Data --- # 

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_mooring_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
var      = nc.variables['VAR'][:]
svf      = nc.variables['SVF'][:]
svf_stdm = nc.variables['SVF_stdm'][:]

# Close the netcdf file
nc.close()

# --- Mixed Layer Depth ---# 

# Obtain filename path
filename_mld = PATH_processed / f"mitgcm_proc_density_hrly_mooring.nc"

# Generate the nc data structure
nc = Dataset(filename_mld, 'r')

# Extract data variables
mld = nc.variables['MLD'][:]

# Close the netcdf file
nc.close()

# -----------------------------------------------------------------------------
# Compute the time mean and standard deviation mixed layer depth 
# -----------------------------------------------------------------------------

mld_mean = np.ma.mean(mld,axis=1)
mld_std = np.ma.std(mld,axis=1,ddof=1)

# -----------------------------------------------------------------------------
# Plot decorrelation time scale, SVF and frequency spectra   
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

# Set the positive depth values for plotting
depth_pos = np.abs(depth)

# Create figure
fig, axes = plt.subplots(3,2,figsize=(13,12),gridspec_kw={'width_ratios': [1, 1]}) 

# --- Subplot 1 --- # 
ax = axes[0,0]

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[0], ls='--', lw=1.25, color='#008C95', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 25], mld_mean[0] - mld_std[0], mld_mean[0] + mld_std[0], color='#008C95', alpha=0.15, edgecolor='none',linewidth=0)

# Plot the absolute variance for low, mid, and high frequency bands at CCE1
ax.semilogx(var[0,:,0], depth_pos, '-', color='tab:green', linewidth=2)
ax.semilogx(var[0,:,1], depth_pos, '-', color='tab:red', linewidth=2)
ax.semilogx(var[0,:,2], depth_pos, '-', color='tab:blue', linewidth=2)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(10**(-4), 10**(-1))
ax.set_ylim(0,200)
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xscale('log')
ax.set_xticklabels([]) 
ax.invert_yaxis()
ax.grid(which='both',linestyle='--',alpha=0.1,color='k')

# --- Subplot 2 --- # 
ax = axes[0,1]

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[0], ls='--', lw=1.25, color='#008C95', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 25], mld_mean[0] - mld_std[0], mld_mean[0] + mld_std[0], color='#008C95', alpha=0.15, edgecolor='none',linewidth=0)

# Plot the depth profile of the decorrelation time scale at CCE1
ax.plot(Lt[0,:], depth_pos, '-', color='k', linewidth=1.5, zorder=4)

# Plot standard error of the mean
ax.fill_betweenx(depth_pos, Lt[0,:] - Lt_stdm[0,:], Lt[0,:] + Lt_stdm[0,:], color='k', alpha=0.15, edgecolor='none',linewidth=0)

# Set axis attributes
ax.set_xlim(0,25)
ax.set_ylim(0, 200)
ax.set_xticks(np.arange(0,25+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xticklabels([])
ax.invert_yaxis()
ax.tick_params(top=False, 
               bottom=True, 
               left=True, 
               right=False, 
               labelleft=False,
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.3)

# --- Create top axis for SVF --- #
ax_svf = ax.twiny()

# Loop through spectral bands
for iband, band in enumerate(svf_bands):

    # Extract SVF and its uncertainty for CCE1
    values = svf[0, :, iband]
    errors = svf_stdm[0, :, iband]

    # Set color
    color = svf_colors[band]

    # Plot spectral variance fraction as a function of depth
    line, = ax_svf.plot(
        values,
        np.abs(depth),
        color=color,
        linewidth=1.8,
        label=svf_labels[band],
        zorder=4,
    )

    # Plot uncertainty envelope (+/- 1 standard error)
    ax_svf.fill_betweenx(
        np.abs(depth),
        values - errors,
        values + errors,
        color=color,
        alpha=0.15,
        linewidth=0.5,
        zorder=3,
    )

# Set axis attributes for SVF
ax_svf.set_xlim(0,90)
ax_svf.set_xticks(np.arange(0, 91, 10))
ax_svf.set_xlabel(r'$\phi$ ($\%$)')
ax_svf.xaxis.set_ticks_position('top')
ax_svf.xaxis.set_label_position('top')
ax_svf.tick_params(axis='x', 
                   which='major',
                   top=True, 
                   bottom=False,
                   direction='out', 
                   length=3.5)
ax_svf.set_ylim(0, 200)
ax_svf.invert_yaxis()

# --- Subplot 3 --- # 
ax = axes[1,0]

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[1], ls='--', lw=1.25, color='#008C95', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 25], mld_mean[1] - mld_std[1], mld_mean[1] + mld_std[1], color='#008C95', alpha=0.15, edgecolor='none',linewidth=0)

# Plot the absolute variance for low, mid, and high frequency bands at CCE2
ax.semilogx(var[1,:,0], depth_pos, '-', color='tab:green', linewidth=2)
ax.semilogx(var[1,:,1], depth_pos, '-', color='tab:red', linewidth=2)
ax.semilogx(var[1,:,2], depth_pos, '-', color='tab:blue', linewidth=2)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(10**(-4), 10**(-1))
ax.set_ylim(0,200)
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xscale('log')
ax.set_xticklabels([])
ax.invert_yaxis()
ax.grid(which='both',linestyle='--',alpha=0.1,color='k')

# --- Subplot 4 --- # 
ax = axes[1,1]

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[1], ls='--', lw=1.25, color='#008C95', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 25], mld_mean[1] - mld_std[1], mld_mean[1] + mld_std[1], color='#008C95', alpha=0.15, edgecolor='none',linewidth=0)

# Plot the depth profile of the decorrelation time scale at CCE1
ax.plot(Lt[1,:], depth_pos, '-', color='k', linewidth=1.5, zorder=4)

# Plot standard error of the mean
ax.fill_betweenx(depth_pos, Lt[1,:] - Lt_stdm[1,:], Lt[1,:] + Lt_stdm[1,:], color='k', alpha=0.15, edgecolor='none',linewidth=0)

# Set axis attributes
ax.set_xlim(0,25)
ax.set_ylim(0, 200)
ax.set_xticks(np.arange(0,25+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xticklabels([])
ax.invert_yaxis()
ax.tick_params(top=False, 
               bottom=True, 
               left=True, 
               right=False, 
               labelleft=False,
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.3)

# --- Create top axis for SVF --- #
ax_svf = ax.twiny()

# Loop through spectral bands
for iband, band in enumerate(svf_bands):

    # Extract SVF and its uncertainty for CCE2
    values = svf[1, :, iband]
    errors = svf_stdm[1, :, iband]

    # Set color
    color = svf_colors[band]

    # Plot spectral variance fraction as a function of depth
    line, = ax_svf.plot(
        values,
        np.abs(depth),
        color=color,
        linewidth=1.8,
        label=svf_labels[band],
        zorder=4,
    )

    # Plot uncertainty envelope (+/- 1 standard error)
    ax_svf.fill_betweenx(
        np.abs(depth),
        values - errors,
        values + errors,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=3,
    )

# Set axis attributes for SVF
ax_svf.set_xlim(0,90)
ax_svf.set_xticks(np.arange(0, 91, 10))
ax_svf.xaxis.set_ticks_position('top')
ax_svf.xaxis.set_label_position('top')
ax_svf.tick_params(axis='x', 
                   which='major',
                   top=True, 
                   bottom=False,
                   direction='out', 
                   length=3.5)
ax_svf.set_ylim(0, 200)
ax_svf.invert_yaxis()
ax_svf.set_xticklabels([])

# --- Subplot 5 --- # 
ax = axes[2,0]

# Plot the mean mixed layer depth 
mld_line = ax.axhline(mld_mean[2], ls='--', lw=1.25, color='#008C95', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
mld_error = ax.fill_between([0, 25], mld_mean[2] - mld_std[2], mld_mean[2] + mld_std[2], color='#008C95', alpha=0.15, edgecolor='none',linewidth=0)

# Plot the absolute variance for low, mid, and high frequency bands at CCE1
ax.semilogx(var[2,:,0], depth_pos, '-', color='tab:green', linewidth=2, label=r'$\sigma^2_\mathrm{low}$')
ax.semilogx(var[2,:,1], depth_pos, '-', color='tab:red', linewidth=2, label=r'$\sigma^2_\mathrm{mid}$')
ax.semilogx(var[2,:,2], depth_pos, '-', color='tab:blue', linewidth=2, label=r'$\sigma^2_\mathrm{high}$')

# Set axis attributes
ax.set_xlabel(r'$\sigma^2$ ($\mathrm{kg}^2/\mathrm{m}^6$)')
ax.set_ylabel('Depth (m)')
ax.set_xlim(10**(-4), 10**(-1))
ax.set_ylim(0,200)
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xscale('log')
ax.invert_yaxis()
ax.grid(which='both',linestyle='--',alpha=0.1,color='k')
ax.legend(loc='lower center', fontsize=11, ncol=3)

# --- Subplot 6 --- # 
ax = axes[2,1]

# Plot the mean mixed layer depth 
mld_line = ax.axhline(mld_mean[2], ls='--', lw=1.25, color='#008C95', alpha=1, label=r"$\overline{z}_{mld}$")

# Plot the range of mixed layer depths (1 standard deviation)
mld_error = ax.fill_between([0, 25], mld_mean[2] - mld_std[2], mld_mean[2] + mld_std[2], color='#008C95', alpha=0.15, edgecolor='none',linewidth=0, label=r"$\sigma_{z_{mld}}$")

# Plot the depth profile of the decorrelation time scale at CCE1
lt_line, = ax.plot(Lt[2,:], depth_pos, '-', color='k', linewidth=1.5, zorder=4, label=r'$\overline{\mathcal{T}}$')

# Plot standard error of the mean
lt_error = ax.fill_betweenx(depth_pos, Lt[2,:] - Lt_stdm[2,:], Lt[2,:] + Lt_stdm[2,:], color='k', alpha=0.15, label=r'$\sigma_{\overline{\mathcal{T}}}$', edgecolor='none',linewidth=0)

# Set axis attributes
ax.set_xlabel('Decorrelation scale (days)')
ax.set_xlim(0,25)
ax.set_ylim(0, 200)
ax.set_xticks(np.arange(0,25+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_yaxis()
ax.tick_params(top=False, 
               bottom=True, 
               left=True, 
               right=False, 
               labelleft=False,
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.3)

# --- Create top axis for SVF --- #
ax_svf = ax.twiny()

# Initialize list to hold handles for legend
svf_handles = []

# Loop through spectral bands
for iband, band in enumerate(svf_bands):

    # Extract SVF and its uncertainty for CCE3
    values = svf[2, :, iband]
    errors = svf_stdm[2, :, iband]

    # Set color
    color = svf_colors[band]

    # Plot spectral variance fraction as a function of depth
    line, = ax_svf.plot(
        values,
        np.abs(depth),
        color=color,
        linewidth=1.8,
        label=svf_labels[band],
        zorder=4,
    )

    # Plot uncertainty envelope (+/- 1 standard error)
    ax_svf.fill_betweenx(
        np.abs(depth),
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
ax_svf.set_xlim(0,90)
ax_svf.set_xticks(np.arange(0, 91, 10))
ax_svf.xaxis.set_ticks_position('top')
ax_svf.xaxis.set_label_position('top')
ax_svf.tick_params(axis='x', 
                   which='major',
                   top=True, 
                   bottom=False,
                   direction='out', 
                   length=3.5)
ax_svf.set_xticklabels([])

# Match shared y-axis orientation
ax_svf.set_ylim(0, 200)
ax_svf.invert_yaxis()

# Combine decorrelation scale and SVF legend handles
legend_handles = [lt_line, lt_error, mld_line, mld_error] + svf_handles

# Add a legend
ax_svf.legend(handles=legend_handles,
              loc='lower center',
              #bbox_to_anchor=(0.28, 0.19),
              fontsize=11,
              frameon=True,
              framealpha=0.85,
              title_fontsize=11, 
              ncol=2)

# Label each subplot
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten() 
add_corner_label(ax1, [0.96,0.08], 'A', fontsize = fontsize)
add_corner_label(ax2, [0.945,0.08], 'B', fontsize = fontsize)
add_corner_label(ax3, [0.96,0.08], 'C', fontsize = fontsize)
add_corner_label(ax4, [0.945,0.08], 'D', fontsize = fontsize)
add_corner_label(ax5, [0.96,0.08], 'E', fontsize = fontsize)
add_corner_label(ax6, [0.945,0.08], 'F', fontsize = fontsize)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.08, wspace=0.05)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'figS13.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)






