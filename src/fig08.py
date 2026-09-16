# =============================================================================
# Figure 08
# =============================================================================
#
# Caption:
#   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-16
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset

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
#                        Options: True or False
# - segment_months : Specifies the window duration. 
# - option_depth: Specifies the water depth which the data is extracted from 
#                 at each mooring.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'gaussian' 
option_detrend_seg = True
segment_months     = 6
option_depth       = 10 

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

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
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
site       = nc.variables['site'][:]
depth_m    = nc.variables['depth'][:]
time_lag_m = nc.variables['lag'][:]
autocorr_m = nc.variables['autocorr'][:]
Lt_m       = nc.variables['decor_scale'][:]
Lt_stdm_m  = nc.variables['decor_scale_stdm'][:]

# --- MITgcm Spectral Data --- # 

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
freq_m     = nc.variables['freq'][:]
psd_m      = nc.variables['PSD'][:]
psd_CI_m   = nc.variables['PSD_CI'][:]

# --- CCE Autocorrelation Data --- # 

# Set path to processed regional MITgcm data
PATH_processed_cce1 = PATH_data / "cce" / "cce1" / "processed"
PATH_processed_cce2 = PATH_data / "cce" / "cce2" / "processed"

# Obtain filename paths
filename_cce1 = PATH_processed_cce1 / f"cce1_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc_cce1 = Dataset(filename_cce1, 'r')
nc_cce2 = Dataset(filename_cce2, 'r')

# Extract data variables
depth_cce1    = nc_cce1.variables['depth'][:]
time_lag_cce1 = nc_cce1.variables['lag'][:]
autocorr_cce1 = nc_cce1.variables['autocorr'][:]
Lt_cce1       = nc_cce1.variables['decor_scale'][:]
Lt_stdm_cce1  = nc_cce1.variables['decor_scale_stdm'][:]

depth_cce2    = nc_cce2.variables['depth'][:]
time_lag_cce2 = nc_cce2.variables['lag'][:]
autocorr_cce2 = nc_cce2.variables['autocorr'][:]
Lt_cce2       = nc_cce2.variables['decor_scale'][:]
Lt_stdm_cce2  = nc_cce2.variables['decor_scale_stdm'][:]

# --- CCE Spectral Data --- # 


# -----------------------------------------------------------------------------
# Obtain the time series at the specified depth for each mooring
# -----------------------------------------------------------------------------

# Set index for the depth of interest
depth_m_index    =  np.abs(np.abs(depth_m) - option_depth).argmin()
depth_cce1_index =  np.abs(np.abs(depth_cce1) - option_depth).argmin()
depth_cce2_index =  np.abs(np.abs(depth_cce2) - option_depth).argmin()

# Extract the data at the specified depth
autocorr_m_depth    = autocorr_m[:,depth_m_index,:]
autocorr_cce1_depth = autocorr_cce1[depth_cce1_index,:]
autocorr_cce2_depth = autocorr_cce2[depth_cce2_index,:]

psd_m_depth     = psd_m[:,depth_m_index,:]
#psd_cce1_depth = psd_cce1[depth_cce1_index,:]
#psd_cce2_depth = psd_cce2[depth_cce2_index,:]

psd_CI_m_depth     = psd_CI_m[:,depth_m_index,:,:]
#psd_CI_cce1_depth = psd_CI_cce1[depth_cce1_index,:,:]
#psd_CI_cce2_depth = psd_CI_cce2[depth_cce2_index,:,:]

# -----------------------------------------------------------------------------
# Plot decorrelation time scales at mooring locations 
# -----------------------------------------------------------------------------

# Find the zero lag index 
zero_lag_m    = np.abs(time_lag_m).argmin()
zero_lag_cce1 = np.abs(time_lag_cce1).argmin()
zero_lag_cce2 = np.abs(time_lag_cce2).argmin()

# Obtain non-negative lags
time_lag_m_pos    = time_lag_m[zero_lag_m:]
time_lag_cce1_pos = time_lag_cce1[zero_lag_cce1:]
time_lag_cce2_pos = time_lag_cce2[zero_lag_cce2:]

# Obtain corresponding autocorrelations
autocorr_m_depth_pos = autocorr_m_depth[:, zero_lag_m:]
autocorr_cce1_depth_pos = autocorr_cce1_depth[zero_lag_cce1:]
autocorr_cce2_depth_pos = autocorr_cce2_depth[zero_lag_cce2:]

# Set plotting parameters 
x_max = 182.5
dx = 20

# Create figure
fig, axes = plt.subplots(2,2,figsize=(15, 10))
ax_flat = axes.flatten()

# --- Subplot 1 --- # 
ax = ax_flat[0]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the mean autocorrelation at CCE1
ax.plot(time_lag_m_pos, autocorr_m_depth_pos[0,:], color='tab:green', linewidth=1.5)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.45, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

# Label each subplot
pos = [0.95, 0.91] 
add_corner_label(axes[0,0], pos, 'A', fontsize = fontsize)
add_corner_label(axes[0,1], pos, 'B', fontsize = fontsize)
add_corner_label(axes[1,0], pos, 'C', fontsize = fontsize)
add_corner_label(axes[1,1], pos, 'D', fontsize = fontsize)

# Adjust spacing 
plt.subplots_adjust(hspace=0.21, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'fig08.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)