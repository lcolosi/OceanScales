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
#   2026-10-07
# =============================================================================

# Import libraries 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_figs = ROOT / "figs"

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
option_interannual = 'linear' 
option_detrend_seg = True
segment_months     = 6
option_depth       = 9 
option_slope_range = [182.5, 30.0] # [182.5, 14.0]

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

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
# Load MITgcm decorrelation scale and spectral data
# -----------------------------------------------------------------------------

# --- MITgcm Decorrelation Scale Data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_reg_depth_{option_depth}m_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

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

# --- Water Depth --- # 

# Set path to processed regional MITgcm data
PATH_preproc = PATH_data / "mitgcm" / "regional" 

# Obtain filename paths
filename = PATH_preproc / f"SEA-STATE_CCS_reg_background.nc"

# Generate the nc data structure
nc = Dataset(filename, 'r')

# Extract local bathymetric depth
water_depth = nc.variables["water_depth"][:]

# Close file
nc.close()

# Mask dry cells previously set to NaN during preprocessing
water_depth_m = np.ma.masked_invalid(water_depth)

# --- Analytic Autocorrelation --- # 

# Set filename to saved the netcdf file
filename = PATH_data / "analytic" / f"analytic_decor_scale.nc"

# Load in data
with Dataset(filename, "r") as nc:

    # Load analytic autocorrelation and decorrelation scale
    T_tilde      = nc.variables["decor_scale"][:]
    T_tilde_maxI = nc.variables["decor_scale_maxI"][:]

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
# Compute representative uncertainty for each bathymetric region
# -----------------------------------------------------------------------------

# Define valid masks for each bathymetric region
shelf_mask_valid = valid & (water_depth_m <= 200)
inter_mask_valid = valid & (water_depth_m > 200) & (water_depth_m <= 2000)
deep_mask_valid = valid & (water_depth_m > 2000)

# Compute median 1-sigma uncertainty in each region

# Continental Shelf
shelf_xerr = np.nanmedian(slope_err[shelf_mask_valid])
shelf_yerr = np.nanmedian(Lt_err[shelf_mask_valid])

# Intermediate Depth
inter_xerr = np.nanmedian(slope_err[inter_mask_valid])
inter_yerr = np.nanmedian(Lt_err[inter_mask_valid])

# Deep Ocean
deep_xerr = np.nanmedian(slope_err[deep_mask_valid])
deep_yerr = np.nanmedian(Lt_err[deep_mask_valid])

# -----------------------------------------------------------------------------
# Plot decorrelation time scale as a function of spectral slope
# -----------------------------------------------------------------------------

# Set plotting parameters
fontsize_g = 18
regions = [
    (water_depth_m > 2000, 'tab:green', 'Deep Ocean'),
    ((water_depth_m > 200) & (water_depth_m <= 2000), 'tab:red', 'Intermediate Depth'),
    (water_depth_m <= 200, 'tab:blue', 'Continental Shelf'),
]
fontsize_ins = 10

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Loop through regions
for mask, color, label in regions:

    # Select valid data points within the current region
    select = valid & mask

    # Plot scatter for the current region
    ax.scatter(
        alpha_data[select],
        Lt_data[select],
        s=10,
        color=color,
        alpha=0.25,
        edgecolors='none',
        label=label
    )

# Plot the theoretical decorrelation scale and spectral slope relationship
idx_T_ds = T_months_ds == 6
ax.plot(alpha_values_ds, np.squeeze(T_tilde[idx_T_ds,:]), '--', color='k', linewidth=1.5, label='Analytic')

# Set axis attributes
ax.set_xlabel(r'Spectral Slope $\alpha$', fontsize=fontsize_g)
ax.set_ylabel(r'Decorrelation Scale (days)', fontsize=fontsize_g)
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(4, 26)
ax.set_xticks(np.arange(0,2+0.25,0.25))
ax.set_yticks(np.arange(4,26+2,2))
ax.tick_params(axis='both', labelsize=fontsize_g)
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)
ax.legend(loc='lower right')

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.075, 0.55, 0.4, 0.4])

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
idx_T_ds = T_months_ds == 6
axins.plot(alpha_values_ds, np.squeeze(T_tilde[idx_T_ds,:]), '--', color='k', linewidth=1.5, label='Analytic')

# Set inset axis attributes
axins.set_xlabel(r'Spectral Slope $\alpha$',fontsize=fontsize_ins)
axins.set_ylabel('Decorrelation Scale (days)',fontsize=fontsize_ins)
axins.set_xlim(-0.1, 2.5)
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


# # Plot the representive uncertainty for the continental shelf region
# ax.errorbar(
#     x0, y0,
#     xerr=shelf_xerr, yerr=shelf_yerr,
#     fmt='o',
#     color='tab:blue',
#     ecolor='tab:blue',
#     elinewidth=1.5,
#     capsize=3,
#     markersize=4
# )
# ax.text(x0 + 0.08, y0 + 0.4, 'Continental Shelf ', va='center', fontsize=14)

# # Plot the representive uncertainty for the intermediate depth region
# ax.errorbar(
#     x0, y0 - dy,
#     xerr=inter_xerr, yerr=inter_yerr,
#     fmt='o',
#     color='tab:red',
#     ecolor='tab:red',
#     elinewidth=1.5,
#     capsize=3,
#     markersize=4
# )
# ax.text(x0 + 0.08, y0 - dy + 0.4, 'Intermediate Depth', va='center', fontsize=14)

# # Plot the representive uncertainty for the deep ocean region
# ax.errorbar(
#     x0, y0 - 2.4*dy,
#     xerr=deep_xerr, yerr=deep_yerr,
#     fmt='o',
#     color='tab:green',
#     ecolor='tab:green',
#     elinewidth=1.5,
#     capsize=3,
#     markersize=4
# )
# ax.text(x0 + 0.08, y0 - 2.4*dy + 0.4, 'Deep ocean', va='center', fontsize=14)

# x0 = 0.3 # 0.22
# y0 = 24 #22.8
# dy = 4




