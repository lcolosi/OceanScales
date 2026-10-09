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
from plotting import add_corner_label, plot_spectral_slope

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
option_analytic_slope = [1.0, 1.25, 1.5, 1.75, 2.0] 
option_slope_range    = [182.5, 14.0]

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
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_mooring_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

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

# --- CCE Mooring latitudes --- #
cce_lat  = np.array([33.457, 34.3075, 34.44825228022894])  

# --- Analytic Autocorrelation --- # 

# Set filename to saved the netcdf file
filename = PATH_data / "analytic" / f"analytic_decor_scale.nc"

# Load in data
with Dataset(filename, "r") as nc:

    # Load analytic autocorrelation and decorrelation scale
    rho     = nc.variables["autocorr"][:]
    T_tilde = nc.variables["decor_scale"][:]

    # Load coordinates
    tau             = nc.variables["lag"][:]
    T_ac            = nc.variables["duration_ac"][:]
    T_ds            = nc.variables["duration_ds"][:]
    alpha_values_ac = nc.variables["slope_ac"][:]
    alpha_values_ds = nc.variables["slope_ds"][:]

# Convert from days to months
days_per_month = 365 / 12 
T_months_ac    = T_ac / days_per_month
T_months_ds    = T_ds / days_per_month

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
psd_cce1_depth = psd_cce1[depth_cce1_index,:]
psd_cce2_depth = psd_cce2[depth_cce2_index,:]

psd_CI_m_depth     = psd_CI_m[:,depth_m_index,:,:]
psd_CI_cce1_depth = psd_CI_cce1[depth_cce1_index,:,:]
psd_CI_cce2_depth = psd_CI_cce2[depth_cce2_index,:,:]

# -----------------------------------------------------------------------------
# Obtain the analytic autocorrelation at a specified spectral slope and duration
# -----------------------------------------------------------------------------

# Convert duration from days to months
T_months_ac    = T_ac / days_per_month
T_months_ds    = T_ds / days_per_month

# Obtain the indices for the specified slope and duration
alpha_indices = np.isin(alpha_values_ac, option_analytic_slope)
T_indices = np.isin(T_months_ac, segment_months)

# Select the analytic autocorrelation for the specified slope and duration
autocorr_analytic = rho[:,T_indices,alpha_indices]

T_month_ac = T_months_ac[T_indices]
alpha_values_ac = alpha_values_ac[alpha_indices]

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

# Set earth's rotation rate (rad/s)
omega = 7.2921159e-5

# Set the tidal and Coriolis frequencies in cpd
f_K1 = 1.00274
f_M2 = 1.93227
f_coriolis = 2 * omega * np.sin(np.deg2rad(cce_lat)) / (2 * np.pi) * 86400

# Set plotting parameters 
x_max = 182.5
dx = 20
fmin_slope = 4*10**(-3) 
fmax_slope = 1.5*10**(1) 
fontsize_ins = 10

# Create figure
fig, axes = plt.subplots(2,2,figsize=(15, 10))
ax_flat = axes.flatten()

# --- Subplot 1 --- # 
ax = ax_flat[0]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the mean autocorrelation at CCE1 for the model and observations
ax.plot(time_lag_m_pos, autocorr_m_depth_pos[0,:], '-', color='dimgray', linewidth=1.5)
ax.plot(time_lag_cce1_pos, autocorr_cce1_depth_pos, '-', color='tab:green', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
idx_alpha = alpha_values_ac == 1.5
ax.plot(tau, autocorr_analytic[:,idx_alpha], '--', color='k', linewidth=1.5)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.45, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.35, 0.55, 0.45, 0.42])

# Plot the zero line 
axins.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=0.75)

# Plot the mean autocorrelation
axins.plot(time_lag_m_pos, autocorr_m_depth_pos[0,:], '-', color='dimgray', linewidth=1.5)
axins.plot(time_lag_cce1_pos, autocorr_cce1_depth_pos, '-', color='tab:green', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
idx_alpha = alpha_values_ac == 1.5
axins.plot(tau, autocorr_analytic[:,idx_alpha], '--', color='k', linewidth=1.5)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 20)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 20+2, 2))
axins.set_yticks(np.arange(0,1+0.25,0.25))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(which='both',direction='out',top=False,right=False,left=True,bottom=True,labelsize=fontsize_ins,length=3)

# --- Subplot 2 --- # 
ax = ax_flat[1]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the mean autocorrelation at CCE1 for the model and observations
ax.plot(time_lag_m_pos, autocorr_m_depth_pos[1,:], '-', color='dimgray', linewidth=1.5)
ax.plot(time_lag_cce2_pos, autocorr_cce2_depth_pos, '-', color='tab:red', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
idx_alpha = alpha_values_ac == 1.25
ax.plot(tau, autocorr_analytic[:,idx_alpha], '--', color='k', linewidth=1.5)

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.45, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.35, 0.55, 0.45, 0.42])

# Plot the zero line 
axins.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=0.75)

# Plot the mean autocorrelation
axins.plot(time_lag_m_pos, autocorr_m_depth_pos[1,:], '-', color='dimgray', linewidth=1.5)
axins.plot(time_lag_cce2_pos, autocorr_cce2_depth_pos, '-', color='tab:red', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
idx_alpha = alpha_values_ac == 1.25
axins.plot(tau, autocorr_analytic[:,idx_alpha], '--', color='k', linewidth=1.5)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 20)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 20+2, 2))
axins.set_yticks(np.arange(0,1+0.25,0.25))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(which='both',direction='out',top=False,right=False,left=True,bottom=True,labelsize=fontsize_ins,length=3)

# --- Subplot 3 --- # 
ax = ax_flat[2]

# Add tidal and Coriolis frequencies
ax.axvline(f_K1, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_M2, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_coriolis[0], color='k', linestyle='--', linewidth=1.0)

# Add frequency labels
ax.annotate(r'$\mathrm{K}_1$', xy=(f_K1, ax.get_ylim()[1]), xytext=(-8, 20), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$f$', xy=(f_coriolis[0], ax.get_ylim()[1]), xytext=(5, 21), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$\mathrm{M}_2$', xy=(f_M2, ax.get_ylim()[1]), xytext=(8, 20), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

# Plot the mean power spectrum at CCE1 for the model and observations
ax.loglog(freq_m[1:], psd_m_depth[0,1:], '-', color='dimgray', linewidth=1.5, label='Model')
ax.loglog(freq_cce1[1:], psd_cce1_depth[1:], '-', color='tab:green', linewidth=1.5, alpha=0.7, label='Obs')

# Plot the 95% confidence interval of the mean power spectrum 
ax.fill_between(freq_m[1:], psd_CI_m_depth[0,1:,0], psd_CI_m_depth[0,1:,1], color='dimgray', alpha=0.25)
ax.fill_between(freq_cce1[1:], psd_CI_cce1_depth[1:,0], psd_CI_cce1_depth[1:,1], color='tab:green', alpha=0.25)

# Plot the reference spectral slopes lines
plot_spectral_slope(ax, fmin=fmin_slope, fmax=fmax_slope, y_ref=2e0, slope=-1.5, color='gray', linestyle='--', linewidth=1, fontsize=11)
plot_spectral_slope(ax, fmin=fmin_slope, fmax=fmax_slope, y_ref=4e0, slope=-2, color='gray', linestyle='-.', linewidth=1, fontsize=11)

# Label spectral slopes
ax.annotate(
    r'$f^{-\frac{3}{2}}$',
    xy=(0.95, 0.37),
    xycoords='axes fraction',
    fontsize=11,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$f^{-2}$',
    xy=(0.95, 0.23),
    xycoords='axes fraction',
    fontsize=11,
    color='k',
    ha='left',
    va='center',
)

# Set axis attributes
ax.set_xlabel('Frequency (cpd)')
ax.set_ylabel(r'PSD ($\mathrm{kg}^2/\mathrm{m}^6/\mathrm{cycle}/\mathrm{day}$)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(6*10**(-10),1*10**1)
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)
ax.legend(loc='lower left', fontsize=14)

# --- Subplot 4 --- # 
ax = ax_flat[3]

# Add tidal and Coriolis frequencies
ax.axvline(f_K1, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_M2, color='k', linestyle=':', linewidth=1.0)
ax.axvline(f_coriolis[1], color='k', linestyle='--', linewidth=1.0)

# Add frequency labels
ax.annotate(r'$\mathrm{K}_1$', xy=(f_K1, ax.get_ylim()[1]), xytext=(-8, 20), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$f$', xy=(f_coriolis[1], ax.get_ylim()[1]), xytext=(5, 21), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

ax.annotate(r'$\mathrm{M}_2$', xy=(f_M2, ax.get_ylim()[1]), xytext=(8, 20), textcoords='offset points',
            rotation=0, ha='center', va='top', fontsize=10)

# Plot the mean power spectrum at CCE1 for the model and observations
ax.loglog(freq_m[1:], psd_m_depth[1,1:], '-', color='dimgray', linewidth=1.0)
ax.loglog(freq_cce2[1:], psd_cce2_depth[1:], '-', color='tab:red', linewidth=1.0, alpha=0.7)

# Plot the 95% confidence interval of the mean power spectrum 
ax.fill_between(freq_m[1:], psd_CI_m_depth[1,1:,0], psd_CI_m_depth[1,1:,1], color='dimgray', alpha=0.25)
ax.fill_between(freq_cce2[1:], psd_CI_cce2_depth[1:,0], psd_CI_cce2_depth[1:,1], color='tab:red', alpha=0.25)

# Plot the reference spectral slopes lines
plot_spectral_slope(ax, fmin=fmin_slope, fmax=fmax_slope, y_ref=3e0, slope=-1, color='gray', linestyle='--', linewidth=1, fontsize=11)
plot_spectral_slope(ax, fmin=fmin_slope, fmax=fmax_slope, y_ref=2.5e1, slope=-1.75, color='gray', linestyle='-.', linewidth=1, fontsize=11)

# Label spectral slopes
ax.annotate(
    r'$f^{-\frac{7}{4}}$',
    xy=(0.95, 0.38),
    xycoords='axes fraction',
    fontsize=11,
    color='k',
    ha='left',
    va='center',
)

ax.annotate(
    r'$f^{-1}$',
    xy=(0.95, 0.65),
    xycoords='axes fraction',
    fontsize=11,
    color='k',
    ha='left',
    va='center',
)

# Set axis attributes
ax.set_xlabel('Frequency (cpd)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(6*10**(-10),1*10**1)
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)

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

print(1/freq_m[:20])