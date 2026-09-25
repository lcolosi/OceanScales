# =============================================================================
# Figure S03
# =============================================================================
#
# Caption:
#   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-23
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo

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

# Set processing parameters
option_simulation        = "ideal"
option_random_amplitudes = False
option_normalization     = "sample"
option_detrend_seg       = False

# Set ensemble and time parameters
n_realization = 100
T             = 2 

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"
rand_amp = "random_amplitudes" if option_random_amplitudes else "random_phases"

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load Monte Carlo and Analytic autocorrelation and decorrelation-scale estimates
# -----------------------------------------------------------------------------

# --- Analytic --- # 

# Set filename to saved the netcdf file
filename = PATH_data / "analytic" / f"analytic_decor_scale.nc"

# Load in data
with Dataset(filename, "r") as nc:

    # Load analytic autocorrelation and decorrelation scale
    rho   = nc.variables["autocorr"][:]
    decor = nc.variables["decor_scale"][:]

    # Load coordinates
    tau      = nc.variables["lag"][:]
    T_ac     = nc.variables["duration_ac"][:]
    T_ds     = nc.variables["duration_ds"][:]
    alpha_ac = nc.variables["slope_ac"][:]
    alpha_ds = nc.variables["slope_ds"][:]

# Convert from days to months
days_per_month = 365 / 12 
T_ac_months    = T_ac / days_per_month
T_ds_months    = T_ds / days_per_month

# --- Monte Carlo Simulation --- # 

# Set filename to saved the netcdf file
if option_simulation == "ideal": 
    filename = PATH_data / "analytic" / f"monte_carlo_sim_decor_scale_ideal_{rand_amp}_norm_{option_normalization}_realizations_{n_realization}_{seg_proc}.nc"
else: 
    filename = PATH_data / "analytic" / f"monte_carlo_sim_decor_scale_real_{rand_amp}_norm_{option_normalization}_realizations_{n_realization}_{seg_proc}_record_length_{T}yr.nc"

# Load in data
with Dataset(filename, "r") as nc:

    # Load analytic autocorrelation and decorrelation scale
    rho_monte       = nc.variables["autocorr"][:]
    rho_std_monte   = nc.variables["autocorr_ens_std"][:]
    decor_monte     = nc.variables["decor_scale"][:]
    decor_std_monte = nc.variables["decor_scale_std"][:]
    
    # Load coordinates
    tau_monte = nc.variables["lag"][:]
    T_monte     = nc.variables["duration"][:]
    alpha_monte = nc.variables["alpha"][:]

# Convert from months to days
T_monte_days = T_monte*(365/12)

# -----------------------------------------------------------------------------
# Plot Autocorrelation and Decorrelation Scale 
# -----------------------------------------------------------------------------

# Set plotting parameters
colors_decor = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange'] 
alpha_p      = [1.5, 2, 3, 4]
T_months_p   = [3, 6, 8, 12]
x_max        = 4

# Create figure and axes 
fig,axes = plt.subplots(2,2,figsize=(12, 10))

#-------------------# 
# Subplot 1
#-------------------# 
ax = axes[0,0]

# Plot a horizontal line at rho equal to zero  
ax.axhline(0, ls = '--', lw = 1.5, alpha = 0.7, color='k')

# --- Analytic --- # 

# Find the index of the desired record duration 
idx_T_ac = np.argmin(np.abs(np.array(T_ac_months) - 12))

# Find indices of alpha values to plot
idx_alpha_p = [
    np.where(np.isclose(alpha_ac, alpha))[0][0]
    for alpha in alpha_p
]

# Loop through alpha values 
for k, idx_alpha in enumerate(idx_alpha_p): 

    # Plot the autocorrelation for ith alpha value 
    ax.plot(tau, rho[:,idx_T_ac,idx_alpha], '-', lw = 2, color=colors_decor[k], 
            label=rf"$\alpha =$ {np.round(alpha_ac[idx_alpha],1)}") 

# --- Monte Carlo Simulation --- # 

# Find the index of the desired record duration 
idx_T_ac = np.argmin(np.abs(np.array(T_monte) - 12))

# Find indices of alpha values to plot
idx_alpha_p = [
    np.where(np.isclose(alpha_monte, alpha))[0][0]
    for alpha in alpha_p
]

# Loop through alpha values 
for k, idx_alpha in enumerate(idx_alpha_p): 

    # Plot the autocorrelation for ith alpha value 
    ax.plot(tau_monte, rho_monte[idx_T_ac,idx_alpha,:], '--', lw = 2, color=colors_decor[k]) 

    # Plot the spread of autocorrelation across monte carlo realizations for ith alpha value 
    ax.fill_between(tau_monte, rho_monte[idx_T_ac,idx_alpha,:] - rho_std_monte[idx_T_ac,idx_alpha,:], rho_monte[idx_T_ac,idx_alpha,:] + rho_std_monte[idx_T_ac,idx_alpha,:], color=colors_decor[k], alpha = 0.3) 

# Set axis attributes 
ax.set_ylabel('Autocorrelation')
ax.set_xlabel(r'$\tau$ (days)')
ax.set_xticks(np.arange(0,365+10,10))
ax.set_yticks(np.arange(-0.75,1+0.25,0.25))
ax.set_xlim(0,120)
ax.set_ylim(-0.75,1)
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)
ax.legend(loc='upper right', fontsize=fontsize-4, framealpha=0.9, edgecolor='black')

#-------------------# 
# Subplot 2
#-------------------#  
ax = axes[0,1]

# Plot the zero autocorrelation line 
ax.axhline(0, ls = '--', lw = 1.5, alpha = 0.7, color='k')

# --- Analytic --- # 

# Find the index of the desired spectral slope
idx_alpha_ac = np.argmin(np.abs(np.array(alpha_ac) - 3))

# Loop through f_min values 
for k in range(0,len(T_ac_months)): 

    # Plot the ith autocorrelation function for the ith T value 
    ax.plot(tau, rho[:,k,idx_alpha_ac], '-', lw = 2, label=f"T = {int(round(T_ac_months[k]))} months", color=colors_decor[k]) 

# --- Monte Carlo Simulation --- # 

# Find the index of the desired spectral slope
idx_alpha_ac = np.argmin(np.abs(np.array(alpha_monte) - 3))

# Find indices of duration values to plot
idx_T_p = [
    np.where(np.isclose(T_monte, iT))[0]
    for iT in T_ac_months
]

# Loop through f_min values 
for k, idx_T in enumerate(idx_T_p): 

    # Obtain autocorrelation for ith duration and spectral slope 
    rho_p = np.squeeze(rho_monte[idx_T,idx_alpha_ac,:])
    rho_std_p = np.squeeze(rho_std_monte[idx_T,idx_alpha_ac,:])

    # Plot the ith autocorrelation function for the ith T value 
    ax.plot(tau_monte, rho_p, '--', lw = 2, color=colors_decor[k]) 

    # Plot the spread of autocorrelation across monte carlo realizations for ith alpha value 
    ax.fill_between(tau_monte, rho_p - rho_std_p, rho_p + rho_std_p, color=colors_decor[k], alpha = 0.3) 

# Set axis attributes 
ax.set_xlabel(r'$\tau$ (days)')
ax.set_xticks(np.arange(0,365+10,10))
ax.set_yticks(np.arange(-0.75,1+0.25,0.25))
ax.set_xlim(0,120)
ax.set_ylim(-0.75,1)
ax.set_yticklabels([])
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)
ax.legend(loc='upper right', fontsize=fontsize-4, framealpha=0.9, edgecolor='black')

#-------------------# 
# Subplot 3
#-------------------# 
ax = axes[1,0]

# Loop through a subset of T values 
for k, iT in enumerate(T_months_p): 

    # Find index value
    idx_T = np.argmin(np.abs(T_ds_months - iT))
    idx_T_monte = np.argmin(np.abs(T_monte - iT))

    # Plot the monte carlo simulation decorrelation scale 
    ax.plot(alpha_monte, decor_monte[idx_T_monte,:], '--', lw = 2, color=colors_decor[k]) 

    # Plot spread of decorrelation values across monte carlo realizations
    ax.fill_between(alpha_monte, decor_monte[idx_T_monte,:] - decor_std_monte[idx_T_monte,:], decor_monte[idx_T_monte,:] + decor_std_monte[idx_T_monte,:], color=colors_decor[k], alpha=0.3)

    # Plot the decorrelation scale as a function of spectral slope
    ax.plot(alpha_ds[::2], decor[idx_T,::2], '.-', lw = 2, label=f"T = {int(T_ds_months[idx_T])} months", color=colors_decor[k]) 

# Plot markers for decorrelation scales for alpha = 1.5, 2, 3, 4 and T = 12 months
idx_t = np.argmin(np.abs(T_ds_months - 12))
idx_a1 = np.argmin(np.abs(alpha_ds - 1.5))
idx_a2 = np.argmin(np.abs(alpha_ds - 2))
idx_a3 = np.argmin(np.abs(alpha_ds - 3))
idx_a4 = np.argmin(np.abs(alpha_ds - 4))
ax.plot(alpha_ds[idx_a1], decor[idx_t, idx_a1], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(alpha_ds[idx_a2], decor[idx_t, idx_a2], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(alpha_ds[idx_a3], decor[idx_t, idx_a3], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(alpha_ds[idx_a4], decor[idx_t, idx_a4], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)

# Set axis attributes 
ax.set_xlabel(r'Spectral Slope $\alpha$')
ax.set_ylabel('Decorrelation Scale (days)')
ax.set_xticks(np.arange(0,5+0.5,0.5))
ax.set_yticks(np.arange(0,100+10,10))
ax.set_xlim(0,5)
ax.set_ylim(0,100)
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)
ax.legend(loc='center', bbox_to_anchor=(0.185, 0.75), fontsize=fontsize-4, framealpha=0.9, edgecolor='black')

#-------------------# 
# Subplot 4
#-------------------# 
ax = axes[1,1]

# Loop through a subset of alpha values 
for k, ialpha in enumerate(alpha_p): 

    # Find index value
    idx_alpha = np.argmin(np.abs(np.array(alpha_ds) - ialpha))
    idx_alpha_monte = np.argmin(np.abs(alpha_monte - ialpha))

    # Plot the monte carlo simulation decorrelation scale 
    ax.plot(T_monte_days, decor_monte[:,idx_alpha_monte], '--', lw = 2, color=colors_decor[k]) 

    # Plot spread of decorrelation values across monte carlo realizations
    ax.fill_between(T_monte_days, decor_monte[:,idx_alpha_monte] - decor_std_monte[:,idx_alpha_monte], decor_monte[:,idx_alpha_monte] + decor_std_monte[:,idx_alpha_monte], color=colors_decor[k], alpha=0.3)

    # Plot the decorrelation scale as a function of f_min
    ax.plot(T_ds, decor[:,idx_alpha], '.-', lw = 2, label=rf"$\alpha =$ {np.round(alpha_ds[idx_alpha],1)}", color=colors_decor[k]) 

# Plot markers for decorrelation scales for alpha = 3 and T = 3, 6, 8, 12 months
idx_t3 = np.argmin(np.abs(T_ds_months - 3))
idx_t6 = np.argmin(np.abs(T_ds_months - 6))
idx_t8 = np.argmin(np.abs(T_ds_months - 8))
idx_t12 = np.argmin(np.abs(T_ds_months - 12))
idx_a = np.argmin(np.abs(alpha_ds - 3))
ax.plot(T_ds[idx_t3], decor[idx_t3, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(T_ds[idx_t6], decor[idx_t6, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(T_ds[idx_t8], decor[idx_t8, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(T_ds[idx_t12], decor[idx_t12, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1, clip_on=False)

# Set axis attributes 
ax.set_xlabel(r'Duration $T$ (days)')
ax.set_xticks(np.arange(0,360+60,60))
ax.set_yticks(np.arange(0,100+10,10))
ax.set_xlim(0,365)
ax.set_ylim(0,100)
ax.set_yticklabels([])
ax.tick_params(top=False, bottom=True, left=True, right=True,
            direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)
ax.legend(loc='center', bbox_to_anchor=(0.13, 0.75), fontsize=fontsize-4, framealpha=0.9, edgecolor='black')

# Label each subplot
ax1, ax2, ax3, ax4 = axes.flatten()
add_corner_label(ax1, [0.05,0.06], 'A', fontsize = fontsize)
add_corner_label(ax2, [0.05,0.06], 'C', fontsize = fontsize)
add_corner_label(ax3, [0.05,0.94], 'B', fontsize = fontsize)
add_corner_label(ax4, [0.05,0.94], 'D', fontsize = fontsize)

# Adjust spacing
plt.tight_layout()

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'figS03.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
