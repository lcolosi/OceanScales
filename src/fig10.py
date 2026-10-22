# =============================================================================
# Figure 10
# =============================================================================
#
# Caption:
#   Scatter plot of the decorrelation time scale as a function of spectral 
#   slope for model data in the study domain at 9.66 meters water depth. 
#   Dots are colored according to bathymetric regions described in table~\ref{t1}: 
#   deep ocean ($H>2000$ m; green), intermediate depth ($200<H\leq2000$ m; red), 
#   and continental shelf ($H\leq200$ m; blue). Dashed curves show the analytically 
#   derived solution, calculated using the full integral of the autocorrelation 
#   (gray) and the maximum of the cumulative integral $\mathcal{I}(\tau)$ (black). 
#   Inset shows the same scatterplot over an expanded range of spectral slopes
#   and decorrelation time scales. Marginal probability density functions of 
#   spectral slope (top) and decorrelation time scales (right) are show for each 
#   bathymetric region, with colors corresponding to those in the scatter plot.  
#   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-07
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
from scipy.stats import gaussian_kde

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_figs = ROOT / "figs"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox 
from stats import compute_pdf_1D

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
# - segment_months : Specifies the window duration. 
# - option_depth: Specifies the water depth which the data is extracted from 
#                 at each mooring.
#
# ------------#

# Set processing parameters
option_data         = 'density'    
option_interannual  = 'linear' 
option_detrend_seg  = "detrend"
option_averaging    = "weighted"
segment_months      = 6
option_depth        = 9 
option_slope_range  = [182.5, 4.0]
option_analytic_est = "biased" 

# Label segment processing 
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
# Load MITgcm and CCE decorrelation scale and spectral data 
# -----------------------------------------------------------------------------

# --- MITgcm Decorrelation Scale Data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_reg_depth_{option_depth}m_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
lon     = nc.variables['lon'][:]
lat     = nc.variables['lat'][:]
Lt      = nc.variables['decor_scale'][:]
Lt_stdm = nc.variables['decor_scale_stdm'][:]
Lt_std  = nc.variables['decor_scale_std'][:]

# --- MITgcm Spectral Data --- # 

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_reg_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
spec_slope      = nc.variables['SPEC_slope'][:]
spec_slope_stdm = nc.variables['SPEC_slope_stde'][:]

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

# --- CCE Spectral Data --- # 

# Obtain filename paths
filename_cce1 = PATH_processed_cce1 / f"cce1_spectra_{option_data}_hrly_mooring_gaussian_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_spectra_{option_data}_hrly_mooring_gaussian_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc_cce1 = Dataset(filename_cce1, 'r')
nc_cce2 = Dataset(filename_cce2, 'r')

# Extract data variables
spec_slope_cce1      = nc_cce1.variables['SPEC_slope'][:]
spec_slope_stdm_cce1 = nc_cce1.variables['SPEC_slope_stde'][:]

spec_slope_cce2      = nc_cce2.variables['SPEC_slope'][:]
spec_slope_stdm_cce2 = nc_cce2.variables['SPEC_slope_stde'][:]

# --- Water Depth --- # 

# Set path to processed regional MITgcm data
PATH_preproc = PATH_data / "mitgcm" / "regional" 

# Obtain filename paths
filename = PATH_preproc / f"SEA-STATE_CCS_reg_background.nc"

# Generate the nc data structure
nc = Dataset(filename, 'r')

# Extract local bathymetric depth
water_depth = nc.variables["water_depth"][:]

# Mask dry cells previously set to NaN during preprocessing
water_depth_m = np.ma.masked_invalid(water_depth)

# --- Analytic Autocorrelation --- # 

# Set filename to saved the netcdf file
filename = PATH_data / "analytic" / f"analytic_decor_scale_{option_analytic_est}.nc"

# Load in data
with Dataset(filename, "r") as nc:

    # Load analytic autocorrelation and decorrelation scale
    T_tilde      = nc.variables["decor_scale"][:]
    T_tilde_maxI = nc.variables["decor_scale_maxI"][:]
    #T_tilde_upper = nc.variables["decor_scale_upper"][:]

    # Load coordinates
    T_ac            = nc.variables["duration_ac"][:]
    T_ds            = nc.variables["duration_ds"][:]
    alpha_values_ac = nc.variables["slope_ac"][:]
    alpha_values_ds = nc.variables["slope_ds"][:]

# Convert from days to months
days_per_month = 365 / 12 
T_months_ac    = T_ac / days_per_month
T_months_ds    = T_ds / days_per_month

# -----------------------------------------------------------------------------
# Preprocess spectral slope and decorrelation scale data
# -----------------------------------------------------------------------------

# Convert spectral slope to slope parameter alpha 
alpha = np.abs(spec_slope)

# Convert masked arrays to regular arrays with NaNs
alpha_data = np.ma.filled(alpha, np.nan)
slope_err  = np.ma.filled(spec_slope_stdm, np.nan)
Lt_data    = np.ma.filled(Lt, np.nan)
Lt_err     = np.ma.filled(Lt_stdm, np.nan)

# Identify valid grid points
valid = (
    np.isfinite(alpha_data) &
    np.isfinite(slope_err) &
    np.isfinite(Lt_data) &
    np.isfinite(Lt_err) &
    (slope_err >= 0) &
    (Lt_err >= 0)
)

# -----------------------------------------------------------------------------
# Plot decorrelation time scale as a function of spectral slope
# -----------------------------------------------------------------------------

# Set plotting parameters
regions = [
    (water_depth_m > 2000, 'tab:green', 'Deep Ocean'),
    ((water_depth_m > 200) & (water_depth_m <= 2000), 'tab:red', 'Intermediate Depth'),
    (water_depth_m <= 200, 'tab:blue', 'Continental Shelf'),
]
fontsize_ins = 12
alpha_edges = np.arange(0, 2.15 + 0.05, 0.05)
Lt_edges    = np.arange(4, 26 + 0.5, 0.5)

# Create figure
fig = plt.figure(figsize=(10, 8))

# Define GridSpec layout for scatter plot and marginal PDFs
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[5, 1],
    height_ratios=[1, 5],
    hspace=0.03,
    wspace=0.03
)

# Create axes
ax       = fig.add_subplot(gs[1, 0])
ax_top   = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

# Adjust figure margins
fig.subplots_adjust(
    left=0.12,
    right=0.95,
    bottom=0.13,
    top=0.95
)

# Loop through bathymetric regions
for mask, color, label in regions:

    # Select valid data points within the current region
    select = valid & np.ma.filled(mask, False)

    # Extract data for the region
    x = alpha_data[select]
    y = Lt_data[select]

    # Plot scatter for the current region
    ax.scatter(
        x,
        y,
        s=10,
        color=color,
        alpha=0.225,
        edgecolors='none',
        label=label
    )

    # Compute marginal PDFs
    _, pdf_x, x_centers = compute_pdf_1D(x, alpha_edges)
    _, pdf_y, y_centers = compute_pdf_1D(y, Lt_edges)

    # Plot top marginal PDF
    ax_top.stairs(
        pdf_x,
        alpha_edges,
        color=color,
        linewidth=1.8
    )

    ax_top.bar(
        x_centers,
        pdf_x,
        width=np.diff(alpha_edges),
        facecolor=color,
        alpha=0.15,
        edgecolor='none',
        linewidth=1.2
    )

    # Plot right marginal PDF
    ax_right.stairs(
        pdf_y,
        Lt_edges,
        orientation='horizontal',
        color=color,
        linewidth=1.8
    )

    ax_right.barh(
        y_centers,
        pdf_y,
        height=np.diff(Lt_edges),
        facecolor=color,
        alpha=0.15,
        edgecolor='none',
        linewidth=1.2
    )

# --- Top marginal PDF --- #
ax_top.tick_params(
    axis='both',
    which='both',
    bottom=True,
    left=False,
    labelbottom=False,
    labelleft=False
)

for spine in ['top', 'right', 'left']:
    ax_top.spines[spine].set_visible(False)


# --- Right marginal PDF --- #
ax_right.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=True,
    labelbottom=False,
    labelleft=False
)

for spine in ['top', 'right', 'bottom']:
    ax_right.spines[spine].set_visible(False)

# Remove the top and right spines of the main axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Obtain the depth indices for the CCE1 and CCE2 data 
idx_cce1 = np.argmin(np.abs(depth_cce1 - option_depth))
idx_cce2 = np.argmin(np.abs(depth_cce2 - option_depth))

# Extract spectral slopes and decorrelation scales
alpha_cce1 = np.abs(spec_slope_cce1[idx_cce1])
alpha_cce2 = np.abs(spec_slope_cce2[idx_cce2])
alpha_stdm_cce1 = spec_slope_stdm_cce1[idx_cce1]
alpha_stdm_cce2 = spec_slope_stdm_cce2[idx_cce2]

Lt_cce1_plot = Lt_cce1[idx_cce1]
Lt_cce2_plot = Lt_cce2[idx_cce2]
Lt_stdm_cce1_plot = Lt_stdm_cce1[idx_cce1]
Lt_stdm_cce2_plot = Lt_stdm_cce2[idx_cce2]

# Plot observations
ax.plot(
    alpha_cce1, Lt_cce1_plot,
    marker='^', linestyle='none',
    color='tab:green', markeredgecolor='black',
    markersize=8, alpha=1, label='CCE1'
)

ax.plot(
    alpha_cce2, Lt_cce2_plot,
    marker='s', linestyle='none',
    color='tab:red', markeredgecolor='black',
    markersize=8, alpha=1, label='CCE2'
)

# Plot CCE1
ax.errorbar(
    alpha_cce1,
    Lt_cce1_plot,
    xerr=alpha_stdm_cce1,
    yerr=Lt_stdm_cce1_plot,
    fmt='^',
    markersize=10,
    color='tab:green',
    markeredgecolor='k',
    markeredgewidth=1.5,
    ecolor='black',
    elinewidth=1.2,
    capsize=3,
    capthick=1.2,
    alpha=1.0,
    zorder=10,
)

# Plot CCE2
ax.errorbar(
    alpha_cce2,
    Lt_cce2_plot,
    xerr=alpha_stdm_cce2,
    yerr=Lt_stdm_cce2_plot,
    fmt='s',
    markersize=10,
    color='tab:red',
    markeredgecolor='k',
    markeredgewidth=1.5,
    ecolor='black',
    elinewidth=1.2,
    capsize=3,
    capthick=1.2,
    alpha=1.0,
    zorder=10,
)

# Obtain the indices of the segment duration 
idx_T_ds = T_months_ds == segment_months
idx_T_ac = T_months_ac == segment_months

# Plot the theoretical decorrelation scale and spectral slope relationship
ax.plot(alpha_values_ds, np.squeeze(T_tilde[idx_T_ds,:]), '--', color='gray', linewidth=1.5, label=r'$\langle T_f \rangle/\pi$')
ax.plot(alpha_values_ac, np.squeeze(T_tilde_maxI[idx_T_ac,:]), '--', color='k', linewidth=1.5, label='Analytic')

# Set axis attributes
ax.set_xlabel(r'Spectral Slope $\alpha$')
ax.set_ylabel(r'Decorrelation Scale (days)')
ax.set_xlim(0, 2.15)
ax.set_ylim(4, 26)
ax.set_xticks(np.arange(0,2+0.25,0.25))
ax.set_yticks(np.arange(4,26+2,2))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)
ax.legend(loc='lower left', fontsize=12)

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.1, 0.55, 0.4, 0.4])

# Loop through regions
for mask, color, label in regions:

    # Select valid data points within the current region
    select = valid & mask

    # Plot scatter for the current region
    axins.scatter(
        alpha_data[select],
        Lt_data[select],
        s=10,
        color=color,
        alpha=0.25,
        edgecolors='none',
        label=label
    )

# Plot the theoretical decorrelation scale and spectral slope relationship
axins.plot(alpha_values_ds, np.squeeze(T_tilde[idx_T_ds,:]), '--', color='gray', linewidth=1.5)
axins.plot(alpha_values_ac, np.squeeze(T_tilde_maxI[idx_T_ac,:]), '--', color='k', linewidth=1.5)

# Set inset axis attributes
axins.set_xlabel(r'Spectral Slope $\alpha$',fontsize=fontsize_ins)
axins.set_ylabel('Decorrelation Scale (days)',fontsize=fontsize_ins)
axins.set_xlim(0, 3)
axins.set_ylim(0, 35)
axins.set_xticks(np.arange(0,3+0.5,0.5))
axins.set_yticks(np.arange(0,35+5,5))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(which='both',direction='out',top=False,right=False,left=True,bottom=True,labelsize=fontsize_ins,length=3)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig10.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)

# -----------------------------------------------------------------------------
# Compute Pearson correlation between spectral slope and decorrelation scale
# -----------------------------------------------------------------------------

# Compute correlation across the entire study domain
r_all = np.corrcoef(alpha_data[valid], Lt_data[valid])[0, 1]

print(f'All Water Depths: r = {r_all:.3f}, N = {np.sum(valid)}')

# Compute correlation for each bathymetric region
for mask, color, label in regions:

    # Select valid grid points within the region
    select = valid & np.ma.filled(mask, False)

    # Extract spectral slope and decorrelation scale
    x = alpha_data[select]
    y = Lt_data[select]

    # Compute Pearson correlation coefficient
    r = np.corrcoef(x, y)[0, 1]

    print(f'{label}: r = {r:.3f}, N = {len(x)}')
