# =============================================================================
# Figure 08
# =============================================================================
#
# Caption:
#   Autocorrelation of the the residual potential density time series at (a) CCE1,
#   (b) CCE2, and (c) CCE3 for observations at 10 m depth for CCE1 (green) and 7 m
#   depth for CCE2 (red) and model data at 9.66 m depth (gray). Corresponding 
#   frequency spectra of the residual potential density time series at (d) CCE1, 
#   (e) CCE2, and (f) CCE3. Dashed black curves in (a-c) 
#   are the autocorrelation computed using the analytic solution \ref{} using the 
#   spectral slope estimated from the model's frequency spectrum over the frequency 
#   band $0.0055 \leq f \leq 0.25$ cpd.  
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
import mpmath as mp

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
from spectra import spectral_slope
from autocorr import autocorrelation_analytic, compute_decor_scale

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
option_data        = 'density'    
option_interannual = 'gaussian' 
option_detrend_seg = "detrend"
option_averaging   = "weighted"
segment_months     = 6
option_depth       = 10 
option_slope_range = [182.5, 4.0]

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

# --- MITgcm Autocorrelation Data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

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
filename_cce1 = PATH_processed_cce1 / f"cce1_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{option_averaging}_{seg_proc}_seg_duration_{segment_months}mo.nc"

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
# Least Squares Fit the power spectrum and obtain the spectral slope
# -----------------------------------------------------------------------------

# Define datasets (adjust model PSD variable names as needed)
datasets = {
    'cce1_m': (freq_m, psd_m_depth[0, :]),
    'cce2_m': (freq_m, psd_m_depth[1, :]),
    'cce3_m': (freq_m, psd_m_depth[2, :]),
    'cce1':   (freq_cce1, psd_cce1_depth),
    'cce2':   (freq_cce2, psd_cce2_depth),
}

# Set number of frequency ranges
nrange = len(fmin_slope)

# Initialize results
spec_slope = {key: np.zeros(nrange) for key in datasets}
spec_stdm  = {key: np.zeros(nrange) for key in datasets}
spec_fit   = {key: [] for key in datasets}
f_fit      = {key: [] for key in datasets}

# Loop through frequency ranges
for i, (fmin, fmax) in enumerate(zip(fmin_slope, fmax_slope)):

    # Loop through dataset 
    for key, (freq, psd) in datasets.items():

        # Compute linear fit to the logarithm of the power spectrum
        slope, stdm, fit, freq_fit = spectral_slope(
            freq, psd, fmin, fmax
        )

        # Store results
        spec_slope[key][i] = slope
        spec_stdm[key][i]  = stdm
        spec_fit[key].append(fit)
        f_fit[key].append(freq_fit)

# -----------------------------------------------------------------------------
# Compute the analytic autocorrelation at the specified spectral slope and duration
# -----------------------------------------------------------------------------

# Set sampling interval and record duration parameters (units: days)
dt = 1/24                                         
T  = (6/12) * 365  

# Set maximum and minimum frequency in units of cpd
fmax = 1 / (2*dt)                                         
fmin = 1 / T               

# Set time lag (units: days)
tau = np.arange(0, T + dt/2, dt)

# Set precision for complex special functions
mp.dps = 25

# Initialize results
autocorr_analytic = {key: np.full((nrange, len(tau)), np.nan) for key in datasets}
Lt_analytic       = {key: np.full(nrange, np.nan) for key in datasets}

# Loop through datasets
for key in datasets:

    # Loop through spectral slope estimates
    for i, slope in enumerate(spec_slope[key]):

        # Convert spectral slope to positive exponent
        alpha = np.abs(slope)

        # Compute analytic autocorrelation at positive lags
        _, autocorr_pos, _ = autocorrelation_analytic(
            tau[1:],
            fmin,
            fmax,
            alpha,
            biased=True,
            record_duration=T
        )

        # Add exact zero-lag autocorrelation
        rho = np.concatenate(([1.0], np.asarray(autocorr_pos, dtype=float)))

        # Construct full autocorrelation and lag arrays
        rho_full = np.concatenate((rho[:0:-1], rho))
        lag_full = np.concatenate((-tau[:0:-1], tau))

        # Compute decorrelation scale by maximizing the symmetric integral
        Lt, _ = compute_decor_scale(rho_full, lag_full)

        # Store results
        autocorr_analytic[key][i, :] = rho
        Lt_analytic[key][i] = Lt

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
fmin_slope_ref = 4*10**(-3) 
fmax_slope_ref = 1.5*10**(1) 
fontsize_ins = 10

# Set helper function 
def slope_label(key, i=0):
    alpha = np.abs(spec_slope[key][i])
    stdm = spec_stdm[key][i]
    return rf'$\alpha = {alpha:.2f} \pm {stdm:.2f}$'

# Create figure
fig, axes = plt.subplots(2,3,figsize=(20, 10))
ax_flat = axes.flatten()

# --- Subplot 1 --- # 
ax = ax_flat[0]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the mean autocorrelation at CCE1 for the model and observations
ax.plot(time_lag_m_pos, autocorr_m_depth_pos[0,:], '-', color='dimgray', linewidth=1.5, label='Model')
ax.plot(time_lag_cce1_pos, autocorr_cce1_depth_pos, '-', color='tab:green', linewidth=1.5, label='Obs')

# Plot the analytic autocorrelation for the specified slope and duration
ax.plot(tau, autocorr_analytic['cce1_m'][0,:], '--', color='k', linewidth=1.5, label='Analytic')

# Set axis attributes
ax.set_xlabel('Time Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_xlim(-10,x_max)
ax.set_ylim(-0.45, 1.1)
ax.set_xticks(np.arange(0,x_max+dx,dx))
ax.set_yticks(np.arange(-0.25,1.0+ 0.25, 0.25))
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)
ax.legend(loc='lower right', fontsize=12)

# Add inset showing short-lag autocorrelation
axins = ax.inset_axes([0.35, 0.55, 0.45, 0.42])

# Plot the zero line 
axins.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=0.75)

# Plot the mean autocorrelation
axins.plot(time_lag_m_pos, autocorr_m_depth_pos[0,:], '-', color='dimgray', linewidth=1.5)
axins.plot(time_lag_cce1_pos, autocorr_cce1_depth_pos, '-', color='tab:green', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
axins.plot(tau, autocorr_analytic['cce1_m'][0,:], '--', color='k', linewidth=1.5)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 22)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 22+2, 2))
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
ax.plot(tau, autocorr_analytic['cce2_m'][0,:], '--', color='k', linewidth=1.5)

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
axins.plot(tau, autocorr_analytic['cce2_m'][0,:], '--', color='k', linewidth=1.5)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 22)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 22+2, 2))
axins.set_yticks(np.arange(0,1+0.25,0.25))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(which='both',direction='out',top=False,right=False,left=True,bottom=True,labelsize=fontsize_ins,length=3)

# --- Subplot 3 --- # 
ax = ax_flat[2]

# Plot the zero line 
ax.axhline(0, color='k', alpha = 0.8, ls='--', linewidth=1)

# Plot the mean autocorrelation at CCE1 for the model and observations
ax.plot(time_lag_m_pos, autocorr_m_depth_pos[2,:], '-', color='dimgray', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
ax.plot(tau, autocorr_analytic['cce3_m'][0,:], '--', color='k', linewidth=1.5)

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
axins.plot(time_lag_m_pos, autocorr_m_depth_pos[2,:], '-', color='dimgray', linewidth=1.5)

# Plot the analytic autocorrelation for the specified slope and duration
axins.plot(tau, autocorr_analytic['cce3_m'][0,:], '--', color='k', linewidth=1.5)

# Set inset axis attributes
axins.set_xlabel('Time Lag (days)',fontsize=fontsize_ins)
axins.set_ylabel('Autocorrelation',fontsize=fontsize_ins)
axins.set_xlim(0, 22)
axins.set_ylim(-0.1,1.05)
axins.set_xticks(np.arange(0, 22+2, 2))
axins.set_yticks(np.arange(0,1+0.25,0.25))
axins.grid(True, linestyle='--', alpha=0.3)
axins.tick_params(which='both',direction='out',top=False,right=False,left=True,bottom=True,labelsize=fontsize_ins,length=3)

# --- Subplot 4 --- # 
ax = ax_flat[3]

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
ax.loglog(freq_m[1:], psd_m_depth[0,1:], '-', color='dimgray', linewidth=1)
ax.loglog(freq_cce1[1:], psd_cce1_depth[1:], '-', color='tab:green', linewidth=1, alpha=0.7)

# Plot the 95% confidence interval of the mean power spectrum 
ax.fill_between(freq_m[1:], psd_CI_m_depth[0,1:,0], psd_CI_m_depth[0,1:,1], color='dimgray', alpha=0.15)
ax.fill_between(freq_cce1[1:], psd_CI_cce1_depth[1:,0], psd_CI_cce1_depth[1:,1], color='tab:green', alpha=0.15)

# Plot the least-squares fit of the spectral slope
ax.loglog(f_fit['cce1_m'][0], spec_fit['cce1_m'][0], '--', color='k', linewidth=1.25, label='Model: ' + slope_label('cce1_m'))
ax.loglog(f_fit['cce1'][0], spec_fit['cce1'][0], '--', color='tab:green', linewidth=1.25, label='Obs: ' + slope_label('cce1'))

# Set axis attributes
ax.set_xlabel('Frequency (cpd)')
ax.set_ylabel(r'PSD ($\mathrm{kg}^2/\mathrm{m}^6/\mathrm{cycle}/\mathrm{day}$)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(6*10**(-10),1*10**1)
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=True, left=True, bottom=True, length=5)
ax.legend(loc='lower left', fontsize=12)

# --- Subplot 5 --- # 
ax = ax_flat[4]

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
ax.loglog(freq_m[1:], psd_m_depth[1,1:], '-', color='dimgray', linewidth=1)
ax.loglog(freq_cce2[1:], psd_cce2_depth[1:], '-', color='tab:red', linewidth=1, alpha=0.7)

# Plot the 95% confidence interval of the mean power spectrum 
ax.fill_between(freq_m[1:], psd_CI_m_depth[1,1:,0], psd_CI_m_depth[1,1:,1], color='dimgray', alpha=0.15)
ax.fill_between(freq_cce2[1:], psd_CI_cce2_depth[1:,0], psd_CI_cce2_depth[1:,1], color='tab:red', alpha=0.15)

# Plot the least-squares fit of the spectral slope
ax.loglog(f_fit['cce2_m'][0], spec_fit['cce2_m'][0], '--', color='k', linewidth=1.25, label='Model: ' + slope_label('cce2_m'))
ax.loglog(f_fit['cce2'][0], spec_fit['cce2'][0], '--', color='tab:red', linewidth=1.25, label='Obs: ' + slope_label('cce2'))

# Set axis attributes
ax.set_xlabel('Frequency (cpd)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(6*10**(-10),1*10**1)
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)
ax.legend(loc='lower left', fontsize=12)

# --- Subplot 6 --- # 
ax = ax_flat[5]

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
ax.loglog(freq_m[1:], psd_m_depth[2,1:], '-', color='dimgray', linewidth=1)

# Plot the 95% confidence interval of the mean power spectrum 
ax.fill_between(freq_m[1:], psd_CI_m_depth[2,1:,0], psd_CI_m_depth[2,1:,1], color='dimgray', alpha=0.15)

# Plot the least-squares fit of the spectral slope
ax.loglog(f_fit['cce3_m'][0], spec_fit['cce3_m'][0], '--', color='k', linewidth=1.25, label='Model: ' + slope_label('cce3_m'))

# Set axis attributes
ax.set_xlabel('Frequency (cpd)')
ax.set_xlim(4*10**(-3), 1.5*10**1)
ax.set_ylim(6*10**(-10),1*10**1)
ax.set_yticklabels([])
ax.grid(True,linestyle='--',alpha=0.3)
ax.tick_params(which='both', direction='out', top=False, right=False, left=True, bottom=True, length=5)
ax.legend(loc='lower left', fontsize=12)

# Label each subplot
pos = [0.95, 0.91] 
add_corner_label(axes[0,0], pos, 'A', fontsize = fontsize)
add_corner_label(axes[0,1], pos, 'B', fontsize = fontsize)
add_corner_label(axes[0,2], pos, 'C', fontsize = fontsize)
add_corner_label(axes[1,0], pos, 'D', fontsize = fontsize)
add_corner_label(axes[1,1], pos, 'E', fontsize = fontsize)
add_corner_label(axes[1,2], pos, 'E', fontsize = fontsize)

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

# -----------------------------------------------------------------------------
# Diagnostic summary of fitted spectral slope parameters
# -----------------------------------------------------------------------------

print("\n" + "=" * 85)
print("Spectral Slope Fit Diagnostics")
print("=" * 85)

# Loop through frequency ranges
for i, (fmin, fmax) in enumerate(zip(fmin_slope, fmax_slope)):

    print(f"\nFrequency Range {i+1}: {fmin:.4f} - {fmax:.4f} cpd")
    print(f"Number of Frequency Bins: {len(f_fit['cce1_m'][i])}")

    print("-" * 85)
    print(
        f"{'Dataset':<12}"
        f"{'Slope':>12}"
        f"{'STDM':>12}"
        f"{'Alpha':>12}"
        f"{'Status':>15}"
    )
    print("-" * 85)

    # Loop through datasets
    for key in datasets:

        # Extract fitted parameters
        slope = spec_slope[key][i]
        stdm  = spec_stdm[key][i]
        alpha = -slope

        # Check fitted parameters
        if not np.isfinite(slope) or not np.isfinite(stdm):
            status = "INVALID"
        elif stdm < 0:
            status = "INVALID STDM"
        elif slope >= 0:
            status = "NON-RED"
        else:
            status = "OK"

        # Print diagnostics
        print(
            f"{key:<12}"
            f"{slope:12.4f}"
            f"{stdm:12.4f}"
            f"{alpha:12.4f}"
            f"{status:>15}"
        )

print("=" * 85)