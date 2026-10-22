# =============================================================================
# Figure 12
# =============================================================================
#
# Caption:
#   (a) Decorrelation time scale (solid black) of model potential density at 9.6 meter water 
#   depth along the CalCOFI line 80.0 transect with the gray-shaded region representing
#   one standard error of the mean. Spectral variance fraction $\phi$ for frequancy
#   bands 2 hours to 4 days (blue), 4 to 30 days (red), and 30 to 182.5 days (green) 
#   along line 80.0 are also shown. Vertical dashed black lines designate the location 
#   of the 200 and 2000 meter isobaths. Colored triangles along the top axis denote
#   the location of CCE1 (green), CCE2 (red), and CCE3 (blue) mooring. (b) Frequency
#   spectrogram of model potential density along line 80.0. Vertical dashed lines again 
#   denote the 200 and 200 meter isobaths. Horizontal lines denote the bounds of the 
#   frequency bands for the spectral variance fraction estimate.    
#   
#   
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-18
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

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
# - option_slope_range: Specifies the frequency range for spectral slope (in days).
# - option_depth_average: Specifies whether to compute the depth-averaged or single-depth
#                         decorrelation scale and spectra.
# - option_depth_range: Specifies the depth range for depth-averaging (in meters).
# - option_depth: Specifies the single depth (in meters) for extracting the decorrelation scale and spectra.
# - option_plot: Specifies the whether the variance density is expressed as per unit
#                frequency or per unit logarithmic frequency.
#                Options: "log" for variance density per unit logarithmic frequency.
#                         "linear" for variance density per unit frequency.
# - segment_months : Specifies the window duration. 
# - sn_threshold : Signal-to-noise ratio threshold for the statistical significance 
#                  criteria. Represents the number of standard deviation a
#                  decorrelation scale estimate is away from the regional spatial
#                  median.
#
# ------------#

# Set processing parameters
option_data          = 'density'    
option_interannual   = 'linear' 
option_detrend_seg   = "detrend"
option_averaging     = "weighted" 
option_slope_range   = [182.5, 4.0] 
option_depth_average = False
option_depth_range   = [0.0, 10.0]
option_depth         = 9.6 
option_plot          = "linear"
segment_months       = 6
depth_lim            = -220 
lat_bnds             = [33.0, 35.0]                                          
lon_bnds             = [237.0, 240.0]

# Set the spectral bands by their period limits (days)
period_bands = {
    r"$\phi$(2 hour--4 days)": (2/24, 4.0),
    r"$\phi$(4--30 days)": (4.0, 30.0),
    r"$\phi$(30--182.5 days)": (30.0, 182.5),
}

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
freq     = nc.variables['freq'][:]
psd      = nc.variables['PSD'][:]
psd_CI   = nc.variables['PSD_CI'][:]

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

# --- Depth Average --- # 
if option_depth_average: 

    # Extract depth limits 
    zmin, zmax = sorted(option_depth_range)

    # Find the depth levels within the specified depth range
    depth_idx = np.flatnonzero(np.isfinite(depth_pos) & (depth_pos >= zmin) & (depth_pos <= zmax))

   # Check that at least one depth level was selected
    if depth_idx.size == 0:
        raise ValueError(
            f"No model depths found between {zmin} and {zmax} m. "
            f"Available depths: {depth_pos}"
        )

    # Compute depth average across selected depth levels
    Lt_depth = np.ma.mean(Lt[:, depth_idx], axis=1)
    Lt_stdm_depth = np.ma.mean(Lt_stdm[:, depth_idx], axis=1)
    psd_depth = np.ma.mean(psd[:, depth_idx, :], axis=1)
    psd_CI_depth = np.ma.mean(psd_CI[:, depth_idx, :], axis=1)

# --- Single Depth --- #   
else: 

    # Find the depth levels closest to the specified depth 
    idx_depth = np.argmin(np.abs(depth - option_depth))

    # Extract data at the specified depth
    Lt_depth      = Lt[:, idx_depth]
    Lt_stdm_depth = Lt_stdm[:, idx_depth]

    psd_depth    = psd[:, idx_depth, :]
    psd_CI_depth = psd_CI[:, idx_depth, :]

# -----------------------------------------------------------------------------
# Compute the spectral variance fraction in the spectral bands
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

# Define arrays filled with NaN values
freq_array = np.asarray(np.ma.filled(freq, np.nan), dtype=float)
psd_array = np.asarray(np.ma.filled(psd_depth, np.nan), dtype=float)

# Create a boolean mask to identify valid (finite and positive) frequency values
valid_freq = np.isfinite(freq_array) & (freq_array > 0)

# Apply mask to data arrays
freq_array = freq_array[valid_freq]
psd_array = psd_array[:, valid_freq]

# Initialize dictionary 
svf = {band: np.full(len(dist), np.nan) for band in period_bands}

# Loop through distances 
for idist, spectrum in enumerate(psd_array):

    # Compute total variance of the spectrum
    total_variance = np.trapezoid(spectrum, freq_array)

    # Loop though spectral bands
    for band, (period_short, period_long) in period_bands.items():

        # Set the low and high frequency limits of the band
        low, high = 1.0 / period_long, 1.0 / period_short

        # Compute the variance in the spectral band
        band_variance = integrate_spectral_band(freq_array, spectrum, low, high)

        # Compute the spectral variance fraction (svf) in the spectral band
        svf[band][idist] = 100.0 * band_variance / total_variance    

# -----------------------------------------------------------------------------
# Plot decorrelation time scale, SVF and frequency spectra   
# -----------------------------------------------------------------------------

# Set plotting parameters
cmap   = cmo.deep_r 
period_bounds = [4.0, 30.0, 182.5]
svf_colors = ['tab:blue', 'tab:red', 'tab:green']

# Set the plotting parameters based on the option_plot
if option_plot == "log": 
    
    # Compute variance per unit logarithmic frequency 
    psd_plot = psd_array * freq_array[np.newaxis, :]

    # Set plotting parameters 
    levels = np.logspace(np.log10(1e-5), np.log10(3e-2), num=60)  
    norm   = mcolors.LogNorm(vmin=1e-5, vmax=3e-2) 
    label  = r'$f\mathrm{S}(f)$ (kg$^2$/m$^6$)'

else: 

    # Set psd_plot variable 
    psd_plot = psd_array

    # Set plotting parameters 
    levels = np.logspace(np.log10(7e-7), np.log10(1e0), num=45) 
    norm   = mcolors.LogNorm(vmin=7e-7, vmax=1e0)
    label  = r'PSD (kg$^2$/m$^6$/cycles/day)'

# Create figure
fig = plt.figure(figsize=(13,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 2]) 

#--- Subplot 1 ---# 
ax = fig.add_subplot(gs[0])
ax1 = ax

# Plot the decorrelation scale
lt_line, =ax.plot(dist, Lt_depth, color='k', linewidth=2, zorder=3, label=r'$\overline{\mathcal{T}}$')

# Plot the uncertainty envelope (+/- 1 standard deviation) of the decorrelation scale
lt_error = ax.fill_between(dist, Lt_depth - Lt_stdm_depth, Lt_depth + Lt_stdm_depth, color='k', alpha=0.15, linewidth=0, zorder=2, label=r'$\sigma_{\overline{\mathcal{T}}}$')

# Set axis attributes
ax.set_ylabel('Decorrelation Scale (days)')
ax.set_xlim(0, dist[-1])
ax.set_ylim(0,20) 
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(0,20+5,5)) 
ax.invert_xaxis()
ax.tick_params(top=True, bottom=True, left=True, right=False, direction='out', length=3.5)
ax.grid(axis='x', linestyle='--', alpha=0.1, color='k')

# --- Create right axis for FVE --- #
ax_svf = ax.twinx()

# Initialize list to hold handles for legend
svf_handles = []

# Loop through spectral bands
for (band, values), color in zip(svf.items(), svf_colors):

    # Plot the spectral variance fraction (SVF) for each spectral band
    line, = ax_svf.plot(dist, values, color=color, linewidth=1.8,
                        label=band, zorder=4)

    # Append the line handle to the list for the legend
    svf_handles.append(line)

# Set axis attributes for SVF
ax_svf.set_ylabel(r'$\phi$ ($\%$)')
ax_svf.set_ylim(0, 80)
ax_svf.set_yticks(np.arange(0, 81, 10))

# Combine decorrelation scale and SVF legend handles
legend_handles = [lt_line, lt_error] + svf_handles

# Keep the legend compact to avoid obscuring the decorrelation curve.
ax_svf.legend(handles=legend_handles, 
              loc='center', 
              bbox_to_anchor=(0.55, 0.79), 
              fontsize=11,
              frameon=True, 
              framealpha=0.85, 
              title_fontsize=11,)

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

#--- Subplot 2 ---# 
ax = fig.add_subplot(gs[1])
ax2 = ax

# Plot the time-mean along-shore velocity 
cf = ax.contourf(dist, freq_array, psd_plot.T, levels=levels, cmap=cmap, extend='both', norm=norm)

# Plot and label spectral band boundaries
for period in period_bounds:

    # Convert period to frequency
    f = 1.0 / period

    # Plot spectral band boundary
    ax.axhline(
        y=f,
        color='k',
        linestyle='--',
        linewidth=0.9,
        alpha=0.7,
        zorder=5
    )

    # Label spectral band boundary
    ax.text(
        0.005, f,
        rf'${period:g}$ days',
        transform=ax.get_yaxis_transform(),
        ha='left',
        va='bottom',
        fontsize=11,
        color='k',
        zorder=6
    )

# Set axis attributes
ax.set_xlabel('Distance from shore (km)')
ax.set_ylabel('Frequency (cycles/day)')
ax.set_xlim(0,dist[-1])
ax.set_ylim(freq_array[0],freq_array[-1])
ax.set_yscale('log')
ax.set_xticks(np.arange(0,250+25,25))
ax.invert_xaxis()
ax.grid(axis='x', linestyle='--',alpha=0.1,color='k')

# Set colorbar
cax = fig.add_axes([0.91, 0.125, 0.02, 0.4])
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
cbar.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1))
cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
cbar.set_label(label) 

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
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.015, y_text=1.035,ms=6, fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '',y_marker=1.015, y_text=1.035,ms=6, fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.015, y_text=1.035,ms=6, fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot 200 m and 2000 m isobath markers
ax.axvline(dist_200m,  color='k', linestyle='--',  lw=1.0, alpha=0.7, zorder=1)
ax.axvline(dist_2000m, color='k', linestyle='--', lw=1.0, alpha=0.7, zorder=1)

# Remove top x-axis ticklabels
ax_top.set_xticklabels([])

# Label each subplot
add_corner_label(ax1, [0.0225,0.075], 'A', fontsize = fontsize)
add_corner_label(ax2, [0.0225,0.95], 'B', fontsize = fontsize)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.075, wspace=0.05)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'fig12.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
