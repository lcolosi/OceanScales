# =============================================================================
# Figure 03
# =============================================================================
#
# Caption:
#   (a) Autocorrelation computed from (\ref{autocor_analytic}) as a function of time
#   lag $\tau$ for $T = 12$ months, $\mathrm{d}t = 12$ hours, and four cases of
#   spectral slope: 1.5, 2.0, 3.0, and 4.0. (b) Autocorrelation computed from 
#   (\ref{autocor_analytic}) as a function of time lag $\tau$ for $\alpha = 3.0$, 
#   $\mathrm{d}t = 12$ hours, and four cases of record duration: 3, 6, 8, and 
#   12 months. (c) Decorrelation time scale computed from (\ref{decor_scale_analytic})
#   as a function of spectral slope for four cases of record duration: 3, 6, 8,
#   and 12 months. Diamond markers denote the decorrelation scales of the
#   autocorrelation in (a). (d) Decorrelation time scale computed from 
#   (\ref{decor_scale_analytic}) as a function of record duration for four cases
#   of spectral slope: 1.5, 2.0, 3.0, and 4.0. Diamond markers denote the
#   decorrelation scales of the autocorrelation in (b).
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-12
# =============================================================================

# Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import sys
import os
import cmocean.cm as cmo
import warnings

# Set path to access additional python functions
sys.path.append('/Users/lukecolosi/Desktop/projects/graduate_research/Gille_lab/' \
                'OceanScales/tools/')

# Import plotting toolbox 
from plotting import add_corner_label
from autocorr import autocorrelation_analytic, decorrelation_scale_analytic

# -----------------------------------------------------------------------------
# Set plotting parameters 
# -----------------------------------------------------------------------------

# Set paths to figures directories
ROOT = '/Users/lukecolosi/Desktop/projects/graduate_research/Gille_lab/OceanScales/'
PATH_figs   = ROOT + 'figs/'

# Set font and fontsize using LaTeX 
fontsize=16
os.environ["PATH"] = "/usr/local/texlive/2022/bin/universal-darwin:" + os.environ["PATH"]
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Set parameters for Autocorrelation calculation
# -----------------------------------------------------------------------------

# Set sampling parameters (units: days)
dt = 1/24                                         # Sampling interval 
T_ac  = np.array([3/12, 6/12, 8/12, 1]) * (365)   # Record duration

# Set maximum and minimum frequency in units of cpd
fmax = 1/dt                                         
fmin_values_ac = 1/T_ac               

# Set spectral slope values 
alpha_values_ac = [1.5, 2.0, 3.0, 4.0, 5.0]

# Convert time to units of months 
days_per_month = 365 / 12 
T_months_ac = T_ac / days_per_month

# Set time lag (units: days)
tau = np.linspace(dt, max(T_ac), 200) 

# Set precision for complex special functions
mp.dps = 25

# -----------------------------------------------------------------------------
# Compute autocorrelation 
# -----------------------------------------------------------------------------

# Set dimension lengths
nfmin, nalpha, ntau = len(fmin_values_ac), len(alpha_values_ac), len(tau)

# Initialize array
rho = np.zeros((ntau, nfmin, nalpha))

# Loop through f_min 
for i, fmin in enumerate(fmin_values_ac):

    # Loop through spectral slope
    for j, alpha in enumerate(alpha_values_ac):

        # Compute the autocorrelation function 
        _, rho[:,i,j], _ = autocorrelation_analytic(tau, fmin, fmax, alpha)

# -----------------------------------------------------------------------------
# Set parameters for decorrelation scale calculation
# -----------------------------------------------------------------------------

# Set sampling parameters (units: days) 
dt = 1/24                                                     # Sampling interval
T_ds = np.flipud(np.arange(0.025, 1 + 0.025, 0.025) * (365))  # Record duration 

# Set maximum and minimum frequency in units of cpd
fmax = 1/dt                                         
fmin_values_ds = 1/T_ds               

# Set spectral slope values 
alpha_values_ds = np.arange(0.1,5+0.1,0.1)

# Convert time to units of months 
days_per_month = 365 / 12 
T_months_ds = T_ds / days_per_month

# -----------------------------------------------------------------------------
# Compute decorrelation scale 
# -----------------------------------------------------------------------------

# Set parameters
nfmin, nalpha = len(fmin_values_ds), len(alpha_values_ds)

# Initialize array
T_tilde = np.zeros((nfmin, nalpha))

# Loop through f_min
for i, fmin in enumerate(fmin_values_ds):

    # Loop through spectral slope
    for j, alpha in enumerate(alpha_values_ds):

        # Compute the decorrelation scale (units: days)
        T_tilde[i,j], _ = decorrelation_scale_analytic(fmin, fmax, alpha) 

# -----------------------------------------------------------------------------
# Plot Autocorrelation and Decorrelation Scale 
# -----------------------------------------------------------------------------

# Set plotting parameters
colors_decor = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange'] 
alpha_p      = [1.5, 2, 3, 4]
T_months_p   = [3, 6, 8, 12]
idx_alpha_ac = np.argmin(np.abs(np.array(alpha_values_ac) - 3))
idx_T_ac     = np.argmin(np.abs(np.array(T_months_ac) - 12))
x_max        = 4

# Create figure and axes 
fig,axes = plt.subplots(2,2,figsize=(12, 10))

#-------------------# 
# Subplot 1
#-------------------# 
ax = axes[0,0]

# Plot a horizontal line at rho equal to zero  
ax.axhline(0, ls = '--', lw = 1.5, alpha = 0.7, color='k')

# Loop through alpha values 
for k in range(0,len(alpha_values_ac[:-1])): 

    # Plot the autocorrelation for ith alpha value 
    ax.plot(tau, rho[:,idx_T_ac,k], '-', lw = 2, 
            label=rf"$\alpha =$ {np.round(alpha_values_ac[k],1)}",
              color=colors_decor[k]) 

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

# Loop through f_min values 
for k in range(0,len(T_months_ac)): 

    # Plot the ith autocorrelation function for the ith T value 
    ax.plot(tau, rho[:,k,idx_alpha_ac], '-', lw = 2, label=f"T = {int(T_months_ac[k])} months", color=colors_decor[k]) 

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
    idx_T = np.argmin(np.abs(T_months_ds - iT))

    # Plot the decorrelation scale as a function of spectral slope
    ax.plot(alpha_values_ds, T_tilde[idx_T,:], '.-', lw = 2, label=f"T = {int(T_months_ds[idx_T])} months", color=colors_decor[k]) 

# Plot markers for decorrelation scales for alpha = 1.5, 2, 3, 4 and T = 12 months
idx_t = np.argmin(np.abs(T_months_ds - 12))
idx_a1 = np.argmin(np.abs(alpha_values_ds - 1.5))
idx_a2 = np.argmin(np.abs(alpha_values_ds - 2))
idx_a3 = np.argmin(np.abs(alpha_values_ds - 3))
idx_a4 = np.argmin(np.abs(alpha_values_ds - 4))
ax.plot(alpha_values_ds[idx_a1], T_tilde[idx_t, idx_a1], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(alpha_values_ds[idx_a2], T_tilde[idx_t, idx_a2], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(alpha_values_ds[idx_a3], T_tilde[idx_t, idx_a3], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(alpha_values_ds[idx_a4], T_tilde[idx_t, idx_a4], 'd', color='tab:purple', markersize=8, markeredgecolor='black', markeredgewidth=1)

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
    idx_alpha = np.argmin(np.abs(np.array(alpha_values_ds) - ialpha))

    # Plot the decorrelation scale as a function of f_min
    ax.plot(T_ds, T_tilde[:,idx_alpha], '.-', lw = 2, label=rf"$\alpha =$ {np.round(alpha_values_ds[idx_alpha],1)}", color=colors_decor[k]) 

# Plot markers for decorrelation scales for alpha = 3 and T = 3, 6, 8, 12 months
idx_t3 = np.argmin(np.abs(T_months_ds - 3))
idx_t6 = np.argmin(np.abs(T_months_ds - 6))
idx_t8 = np.argmin(np.abs(T_months_ds - 8))
idx_t12 = np.argmin(np.abs(T_months_ds - 12))
idx_a = np.argmin(np.abs(alpha_values_ds - 3))
ax.plot(T_ds[idx_t3], T_tilde[idx_t3, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(T_ds[idx_t6], T_tilde[idx_t6, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(T_ds[idx_t8], T_tilde[idx_t8, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1)
ax.plot(T_ds[idx_t12], T_tilde[idx_t12, idx_a], 'd', color='tab:red', markersize=8, markeredgecolor='black', markeredgewidth=1, clip_on=False)

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
add_corner_label(ax2, [0.05,0.06], 'B', fontsize = fontsize)
add_corner_label(ax3, [0.05,0.94], 'C', fontsize = fontsize)
add_corner_label(ax4, [0.05,0.94], 'D', fontsize = fontsize)

# Adjust spacing
plt.tight_layout()

# Save figure in high resolution 
fig.savefig(
    PATH_figs + 'fig02.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)