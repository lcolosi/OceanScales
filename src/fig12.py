# =============================================================================
# Figure 12
# =============================================================================
#
# Caption:
#   Frequency spectrogram of model potential density at (a) CCE1, (c) CCE2, and (e) 
#   CCE3. Vertical dashed lines denote the bounds of the frequency bands for the
#   spectral variance fraction estimate. Decorrelation time scales of model potential
#   density as a function of depth at (b) CCE1, (d) CCE2, and (f) CCE3. Spectral 
#   variance fraction $\phi$ for frequancy bands 2 hours to 4 days (blue), 
#   4 to 30 days (red), and 30 to 182.5 days (green) as a function of depth are also
#   shown. The seasonally averaged mixed-layer depth $\overline{z}_{mld}$ and the
#   temporal standard deviation of mixed layer depth are displayed as horizontal
#   dashed turquoise lines and light shaded regions, respectively.  
#   
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-20
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
# Load MITgcm and CCE autocorrelation and spectral data
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
freq     = nc.variables['freq'][:]
psd      = nc.variables['PSD'][:]
psd_CI   = nc.variables['PSD_CI'][:]
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
levels = np.logspace(np.log10(7e-7), np.log10(5e0), num=45) 
norm   = mcolors.LogNorm(vmin=7e-7, vmax=5e0)
label  = r'PSD (kg$^2$/m$^6$/cycles/day)'
cmap          = cmo.deep_r 
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
fig, axes = plt.subplots(3,2,figsize=(13,12),gridspec_kw={'width_ratios': [1.5, 1]}) 

# --- Subplot 1 --- # 
ax = axes[0,0]

# Plot the depth spectrogram at CCE1
ax.contourf(freq, depth_pos, psd[0,:,:], levels=levels, norm=norm, cmap=cmap, extend='both')

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(freq[1],freq[-1])
ax.set_ylim(0,200)
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xscale('log')
ax.set_xticklabels([]) 
ax.invert_yaxis()
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
ax.tick_params(axis='x',
               which='major',
               top=False, 
               bottom=True, 
               direction='out', 
               length=3.5)
ax.tick_params(axis='x',
               which='minor',
               top=False, 
               bottom=True,  
               direction='out', 
               length=2)
ax.tick_params(axis='y',
               which='major',
               left=True, 
               right=True, 
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.1,color='k')

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

# Plot the depth spectrogram at CCE1
ax.contourf(freq, depth_pos, psd[1,:,:], levels=levels, norm=norm, cmap=cmap, extend='both')

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(freq[1],freq[-1])
ax.set_ylim(0,200)
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xscale('log')
ax.set_xticklabels([])
ax.invert_yaxis()
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
ax.tick_params(axis='x',
               which='major',
               top=True, 
               bottom=True, 
               direction='out', 
               length=3.5)
ax.tick_params(axis='x',
               which='minor',
               top=True, 
               bottom=True,  
               direction='out', 
               length=2)
ax.tick_params(axis='y',
               which='major',
               left=True, 
               right=True, 
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.1,color='k')

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

# Plot the depth spectrogram at CCE1
cf = ax.contourf(freq, depth_pos, psd[2,:,:], levels=levels, norm=norm, cmap=cmap, extend='both')

# Set axis attributes
ax.set_xlabel('Frequency (cycles/day)')
ax.set_ylabel('Depth (m)')
ax.set_xlim(freq[1],freq[-1])
ax.set_ylim(0,200)
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xscale('log')
ax.invert_yaxis()
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
ax.tick_params(axis='x',
               which='major',
               top=True, 
               bottom=True, 
               direction='out', 
               length=3.5)
ax.tick_params(axis='x',
               which='minor',
               top=True, 
               bottom=True,  
               direction='out', 
               length=2)
ax.tick_params(axis='y',
               which='major',
               left=True, 
               right=True, 
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.1,color='k')

# Set colorbar
cax = plt.axes([0.15, 0.89, 0.4, 0.025])
cbar = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='both')
cbar.ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
cbar.ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1))
cbar.ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
cbar.ax.xaxis.set_minor_formatter(mticker.NullFormatter())
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_label(label) 

# --- Subplot 6 --- # 
ax = axes[2,1]

# Plot the mean mixed layer depth 
mld_line = ax.axhline(mld_mean[2], ls='--', lw=1.25, color='#008C95', alpha=1, label=r"$\overline{z}_{mld}$")

# Plot the range of mixed layer depths (1 standard deviation)
mld_error = ax.fill_between([0, 25], mld_mean[2] - mld_std[2], mld_mean[2] + mld_std[2], color='#008C95', alpha=0.15, label=r"$\sigma_{z_{mld}}$", edgecolor='none',linewidth=0)

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
              loc='center',
              bbox_to_anchor=(0.28, 0.19),
              fontsize=11,
              frameon=True,
              framealpha=0.85,
              title_fontsize=11, 
              ncol=2)

# Initialize counter
cn = 0

# Plot spectral band boundaries on the depth spectrograms
for ax in axes[:, 0]:

    # Loop through spectral band boundaries
    for period in period_bounds:

        # Convert period to frequency
        f = 1.0 / period

        # Plot spectral band boundary
        ax.axvline(
            x=f,
            color='k',
            linestyle='--',
            linewidth=0.9,
            alpha=0.7,
            zorder=5
        )

        if cn == 2:  

            # Label spectral band boundary
            ax.text(
                f, 0.01,
                rf'${period:g}$ days',
                transform=transforms.offset_copy(
                    ax.get_xaxis_transform(),
                    fig=fig,
                    x=3, y=0,
                    units='points'
                ),
                ha='left',
                va='bottom',
                rotation=90,
                fontsize=13,
                color='k',
                zorder=6
            )

    # Increment counter
    cn += 1

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
    PATH_figs / 'fig12.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)

# -----------------------------------------------------------------------------
# Find the depth where phi_low first exceeds phi_mid at CCE2 and CCE3
# -----------------------------------------------------------------------------


# --- Find the depth where phi_low first exceeds phi_mid --- #

# Identify spectral band indices
iband_low = svf_bands.index('low')
iband_mid = svf_bands.index('mid')

# Loop through mooring sites
for site_name, site_index in [("CCE2", 1), ("CCE3", 2)]:

    # Extract spectral variance fractions at this mooring
    phi_low = svf[site_index, :, iband_low]
    phi_mid = svf[site_index, :, iband_mid]

    # Compute their difference, ordered from shallowest to deepest
    z = np.asarray(np.ma.filled(depth_pos, np.nan), dtype=float)
    difference = np.asarray(
        np.ma.filled(phi_low - phi_mid, np.nan),
        dtype=float,
    )

    valid = np.isfinite(z) & np.isfinite(difference)
    z, difference = z[valid], difference[valid]

    order = np.argsort(z)
    z, difference = z[order], difference[order]

    if z.size == 0:
        print(f"{site_name}: No valid spectral variance fraction data available.")

    elif difference[0] > 0:
        print(
            f"{site_name}: phi_low already exceeds phi_mid at the "
            f"shallowest sampled depth ({z[0]:.2f} m)."
        )

    else:
        # Find first transition from phi_low <= phi_mid to phi_low > phi_mid
        crossings = np.flatnonzero(
            (difference[:-1] <= 0) & (difference[1:] > 0)
        )

        if crossings.size:
            i = crossings[0]

            # Linearly interpolate depth where phi_low equals phi_mid
            z_cross = z[i] - difference[i] * (
                z[i + 1] - z[i]
            ) / (difference[i + 1] - difference[i])

            print(
                f"{site_name}: phi_low first exceeds phi_mid "
                f"below approximately {z_cross:.2f} m."
            )

        else:
            print(
                f"{site_name}: phi_low does not exceed phi_mid "
                f"within the sampled depth range."
            )




