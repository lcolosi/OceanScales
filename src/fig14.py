# =============================================================================
# Figure 14
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
#   2026-10-23
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo
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
option_interannual = 'gaussian' 
option_detrend_seg = "detrend"
option_averaging   = "weighted"
option_plot        = "linear"
segment_months     = 6
option_depth       = 10 
option_slope_range = [182.5, 4.0]

# Set the spectral bands by their period limits (days)
period_bands = {
    r"$\phi_\mathrm{high}$": (2/24, 4.0),
    r"$\phi_\mathrm{mid}$": (4.0, 30.0),
    r"$\phi_\mathrm{low}$": (30.0, 182.5),
}

# Set time and space parameters
fmin_slope = [1/182.5]
fmax_slope = [1/4.0]       

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

# --- CCE Autocorrelation Data --- # 

# Set path to processed regional MITgcm data
PATH_processed_cce1 = PATH_data / "cce" / "cce1" / "processed"
PATH_processed_cce2 = PATH_data / "cce" / "cce2" / "processed"

# Obtain filename paths
filename_cce1 = PATH_processed_cce1 / f"cce1_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

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
filename_cce1 = PATH_processed_cce1 / f"cce1_spectra_{option_data}_hrly_mooring_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_spectra_{option_data}_hrly_mooring_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc_cce1 = Dataset(filename_cce1, 'r')
nc_cce2 = Dataset(filename_cce2, 'r')

# Extract data variables
freq_cce1     = nc_cce1.variables['freq'][:]
psd_cce1      = nc_cce1.variables['PSD'][:]
psd_CI_cce1   = nc_cce1.variables['PSD_CI'][:]

freq_cce2     = nc_cce2.variables['freq'][:]
psd_cce2      = nc_cce2.variables['PSD'][:]
psd_CI_cce2   = nc_cce2.variables['PSD_CI'][:]

# Close the netcdf file
nc_cce1.close()
nc_cce2.close()

# --- CCE Mooring latitudes --- #
cce_lat  = np.array([33.457, 34.3075, 34.44825228022894])  

# -----------------------------------------------------------------------------
# Compute spectral variance fraction (SVF) in the spectral bands
# -----------------------------------------------------------------------------

# Set helper function 
def integrate_spectral_band(frequency, spectrum, low_frequency, high_frequency):
    """Integrate one-sided PSD over a band, interpolating at its boundaries."""

    # Obtain indices of frequencies within the specified band
    inside = (frequency > low_frequency) & (frequency < high_frequency)

    # Extract the frequencies within the band and concatinate the low and high frequency boundaries
    f_band = np.r_[low_frequency, frequency[inside], high_frequency]

    # Interpolate the spectrum at the low and high frequency boundaries and concatinate with the spectrum within the band
    s_band = np.interp(f_band, frequency, spectrum)

    # Integrate the spectrum over the band using the trapezoidal rule
    var_band = np.trapezoid(s_band, f_band)

    return var_band

# --- CCE1 --- # 

# Define arrays filled with NaN values
freq_array = np.asarray(np.ma.filled(freq_cce1, np.nan), dtype=float)
psd_array = np.asarray(np.ma.filled(psd_cce1, np.nan), dtype=float)

# Create a boolean mask to identify valid (finite and positive) frequency values
valid_freq = np.isfinite(freq_array) & (freq_array > 0)

# Apply mask to data arrays
freq_array = freq_array[valid_freq]
psd_array = psd_array[:, valid_freq]

# Initialize dictionary 
svf_cce1 = {band: np.full((len(depth_cce1)), np.nan) for band in period_bands}

# Loop through water depths
for idepth in range(len(depth_cce1)):

    # Extract the spectrum at the specified depth
    spectrum = psd_array[idepth, :]

    # Compute total variance of the spectrum
    total_variance = np.trapezoid(spectrum, freq_array)

    # Loop though spectral bands
    for band, (period_short, period_long) in period_bands.items():

        # Set the low and high frequency limits of the band
        low, high = 1.0 / period_long, 1.0 / period_short

        # Compute the variance in the spectral band
        band_variance = integrate_spectral_band(freq_array, spectrum, low, high)

        # Compute the spectral variance fraction (SVF) in the spectral band
        svf_cce1[band][idepth] = 100.0 * band_variance / total_variance   

# --- CCE2 --- # 

# Define arrays filled with NaN values
freq_array = np.asarray(np.ma.filled(freq_cce2, np.nan), dtype=float)
psd_array = np.asarray(np.ma.filled(psd_cce2, np.nan), dtype=float)

# Create a boolean mask to identify valid (finite and positive) frequency values
valid_freq = np.isfinite(freq_array) & (freq_array > 0)

# Apply mask to data arrays
freq_array = freq_array[valid_freq]
psd_array = psd_array[:, valid_freq]

# Initialize dictionary 
svf_cce2 = {band: np.full((len(depth_cce2)), np.nan) for band in period_bands}

# Loop through water depths
for idepth in range(len(depth_cce2)):

    # Extract the spectrum at the specified depth
    spectrum = psd_array[idepth, :]

    # Compute total variance of the spectrum
    total_variance = np.trapezoid(spectrum, freq_array)

    # Loop though spectral bands
    for band, (period_short, period_long) in period_bands.items():

        # Set the low and high frequency limits of the band
        low, high = 1.0 / period_long, 1.0 / period_short

        # Compute the variance in the spectral band
        band_variance = integrate_spectral_band(freq_array, spectrum, low, high)

        # Compute the spectral variance fraction (SVF) in the spectral band
        svf_cce2[band][idepth] = 100.0 * band_variance / total_variance  

# -----------------------------------------------------------------------------
# Plot decorrelation time scale, SVF and frequency spectra   
# -----------------------------------------------------------------------------

# Set plotting parameters
cmap                = cmo.deep_r 
period_bounds       = [4.0, 30.0, 182.5]
cce1_standard_depth = np.array([10, 20, 30, 40, 60, 75, 150])
cce2_standard_depth = np.array([7, 15, 25, 45, 75])
psd_colors_cce1     = np.flipud(cmo.haline(np.linspace(0, 1, 7)))
psd_colors_cce2     = np.flipud(cmo.haline(np.linspace(0, 1, 5)))
svf_colors          = ['tab:blue', 'tab:red', 'tab:green']

# Set earth's rotation rate (rad/s)
omega = 7.2921159e-5

# Set the tidal and Coriolis frequencies in cpd
f_K1 = 1.00274
f_M2 = 1.93227
f_coriolis = 2 * omega * np.sin(np.deg2rad(cce_lat)) / (2 * np.pi) * 86400

# Create figure
fig, axes = plt.subplots(2,2,figsize=(14,12),gridspec_kw={'width_ratios': [1.5, 1]}) 

# --- Subplot 1 --- # 
ax = axes[0,0]

# Add tidal and Coriolis frequencies
ax.axvline(f_K1, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_M2, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_coriolis[0], color='k', linestyle='--', linewidth=1.0)

# Add frequency labels
ax.annotate(r'$\mathrm{K}_1$', xy=(f_K1, ax.get_ylim()[1]), xytext=(-8, 42), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$f$', xy=(f_coriolis[0], ax.get_ylim()[1]), xytext=(5, 43), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$\mathrm{M}_2$', xy=(f_M2, ax.get_ylim()[1]), xytext=(8, 42), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

# Loop through depths 
for i, idepth in enumerate(depth_cce1):

    # Obtain the frequency spectrum and its confidence interval at the ith depth 
    ipsd    = psd_cce1[i,:]
    ipsd_CI = psd_CI_cce1[i,:,:]

    # Plot the frequency spectra at each depth
    ax.loglog(freq_cce1[1:], ipsd[1:], '-', lw = 1, color=psd_colors_cce1[i], label=rf'z = {np.abs(idepth):.0f}')
    ax.fill_between(freq_cce1[1:], ipsd_CI[1:,0], ipsd_CI[1:,1], color=psd_colors_cce1[i], alpha=0.15, edgecolor='none',linewidth=0)

# Set axis attributes
ax.set_ylabel(r'PSD ($\mathrm{kg}^2/\mathrm{m}^6/\mathrm{cycle}/\mathrm{day}$)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(1*10**(-6),1*10**1)
ax.set_xticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
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
ax.legend(loc='center', bbox_to_anchor=(0.905, 0.725), fontsize=10)

# --- Subplot 2 --- # 
ax = axes[0,1]

# Plot the depth profile of the decorrelation time scale at CCE1
ax.plot(Lt_cce1, np.abs(depth_cce1), '.-', color='k', linewidth=1.5, zorder=4)

# Plot standard error of the mean
ax.fill_betweenx(np.abs(depth_cce1), Lt_cce1 - Lt_stdm_cce1, Lt_cce1 + Lt_stdm_cce1, color='k', alpha=0.15)

# Set left edge x-position 
x_left = ax.get_xlim()[0] -13

# Plot the sensor depths 
ax.plot(
    np.full_like(cce1_standard_depth, x_left),
    cce1_standard_depth,
    marker='d', 
    linestyle='None',
    color='k', 
    markersize=5, 
    alpha = 1, 
    clip_on=False,
    label='Standard depths'  
)

# Set axis attributes
ax.set_ylabel('Depth (m)')
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
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.3)

# --- Create top axis for SVF --- #
ax_svf = ax.twiny()

# Loop through spectral bands
for (band, values), color in zip(svf_cce1.items(), svf_colors):

    # Plot SVF as a function of depth for site 0
    line, = ax_svf.plot(values, np.abs(depth_cce1), '.-',
                        color=color, linewidth=1.8,
                        label=band, zorder=3)

# Set axis attributes for SVF
ax_svf.set_xlabel(r'$\phi$ ($\%$)')
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

# --- Subplot 3 --- # 
ax = axes[1,0]

# Add tidal and Coriolis frequencies
ax.axvline(f_K1, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_M2, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_coriolis[0], color='k', linestyle='--', linewidth=1.0)

# Add frequency labels
ax.annotate(r'$\mathrm{K}_1$', xy=(f_K1, ax.get_ylim()[1]), xytext=(-8, 42), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$f$', xy=(f_coriolis[0], ax.get_ylim()[1]), xytext=(5, 43), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$\mathrm{M}_2$', xy=(f_M2, ax.get_ylim()[1]), xytext=(8, 42), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

# Loop through depths 
for i, idepth in enumerate(depth_cce2):

    # Obtain the frequency spectrum and its confidence interval at the ith depth 
    ipsd    = psd_cce2[i,:]
    ipsd_CI = psd_CI_cce2[i,:,:]

    # Plot the frequency spectra at each depth
    ax.loglog(freq_cce2[1:], ipsd[1:], '-', lw = 1, color=psd_colors_cce2[i], label=rf'z = {np.abs(idepth):.0f}')
    ax.fill_between(freq_cce2[1:], ipsd_CI[1:,0], ipsd_CI[1:,1], color=psd_colors_cce2[i], alpha=0.15, edgecolor='none',linewidth=0)

# Set axis attributes
ax.set_xlabel('Frequency (cpd)')
ax.set_ylabel(r'PSD ($\mathrm{kg}^2/\mathrm{m}^6/\mathrm{cycle}/\mathrm{day}$)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(1*10**(-6),1*10**1)
ax.grid(True,linestyle='--',alpha=0.3)
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
ax.legend(loc='center', bbox_to_anchor=(0.91, 0.77), fontsize=10)

# --- Subplot 4 --- # 
ax = axes[1,1]

# Plot the depth profile of the decorrelation time scale at CCE1
lt_line, = ax.plot(Lt_cce2, np.abs(depth_cce2), '.-', color='k', linewidth=1.5, zorder=4, label=r'$\overline{\mathcal{T}}$')

# Plot standard error of the mean
lt_error = ax.fill_betweenx(np.abs(depth_cce2), Lt_cce2 - Lt_stdm_cce2, Lt_cce2 + Lt_stdm_cce2, color='k', alpha=0.15, label=r'$\sigma_{\overline{\mathcal{T}}}$')

# Set left edge x-position 
x_left = ax.get_xlim()[0] - 5

# Plot the sensor depths 
std_depths, = ax.plot(
    np.full_like(cce2_standard_depth, x_left),
    cce2_standard_depth,
    marker='d', 
    linestyle='None',
    color='k', 
    markersize=5, 
    alpha = 1, 
    clip_on=False,
    label='Standard depths'
)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Decorrelation Scale (days)')
ax.set_xlim(0,25)
ax.set_ylim(0, 200)
ax.set_xticks(np.arange(0,25+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_yaxis()
ax.tick_params(top=False, 
               bottom=True, 
               left=True, 
               right=False, 
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(axis='y',linestyle='--',alpha=0.3)

# --- Create top axis for SVF --- #
ax_svf = ax.twiny()

# Initialize list to hold handles for legend
svf_handles = []

# Loop through spectral bands
for (band, values), color in zip(svf_cce2.items(), svf_colors):

    # Plot SVF as a function of depth for site 0
    line, = ax_svf.plot(values, np.abs(depth_cce2), '.-',
                        color=color, linewidth=1.8,
                        label=band, zorder=3)
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
ax_svf.set_ylim(0, 200)
ax_svf.invert_yaxis()
ax_svf.set_xticklabels([])

# Combine decorrelation scale and SVF legend handles
legend_handles = [lt_line, lt_error, std_depths] + svf_handles

# Add a legend
ax_svf.legend(handles=legend_handles,
              loc='center',
              bbox_to_anchor=(0.23, 0.17),
              fontsize=11,
              frameon=True,
              framealpha=0.85,
              title_fontsize=11)

# Initialize counter
cn = 0

# Plot spectral band boundaries on the depth spectra
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

        if cn == 1:  

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
ax1, ax2, ax3, ax4 = axes.flatten() 
add_corner_label(ax1, [0.96,0.945], 'A', fontsize = fontsize)
add_corner_label(ax2, [0.9425,0.945], 'B', fontsize = fontsize)
add_corner_label(ax3, [0.96,0.945], 'C', fontsize = fontsize)
add_corner_label(ax4, [0.9425,0.945], 'D', fontsize = fontsize)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.06, wspace=0.18)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'fig14.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
