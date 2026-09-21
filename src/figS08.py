# =============================================================================
# Figure S07
# =============================================================================
#
# Caption:
#   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-19
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
# - option_var : Specifies whether the mean or standard deviation will be plotted.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'gaussian' 
option_detrend_seg = True
option_var         = 'mean'

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set plotting parameters
seg_durations = [1, 2, 3, 4, 6, 8, 12]

# Path to processed data 
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"
PATH_cce1_processed = PATH_data / "cce" / "cce1" / "processed"
PATH_cce2_processed = PATH_data / "cce" / "cce2" / "processed"

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm and CCE decorrelation-scale estimates
# -----------------------------------------------------------------------------

# --- MITgcm --- # 

# Initialize array
decor_scale_m = []
decor_scale_stdm_m = []

# Loop through window durations
for seg_duration in seg_durations:

    filename = (
        PATH_processed
        / f"mitgcm_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_"
          f"seg_duration_{seg_duration}mo.nc"
    )

    with Dataset(filename, "r") as nc:

        # Append decorrelation scale to list
        decor_scale_m.append(nc.variables["decor_scale"][:])
        decor_scale_stdm_m.append(nc.variables["decor_scale_stdm"][:])

        # Save depth coordinate once
        if seg_duration == seg_durations[0]:
            depth_m = nc.variables["depth"][:]


# Stack window durations along a new first dimension
decor_scale_m = np.ma.stack(decor_scale_m, axis=0)
decor_scale_stdm_m = np.ma.stack(decor_scale_stdm_m, axis=0)

# --- Mixed Layer Depth ---# 

# Obtain filename path
filename_mld = PATH_processed / f"mitgcm_proc_density_hrly_mooring.nc"

# Generate the nc data structure
nc = Dataset(filename_mld, 'r')

# Extract data variables
mld = nc.variables['MLD'][:]

# --- CCE Moorings --- # 

# Initialize array
decor_scale_cce1 = []
decor_scale_stdm_cce1 = []
decor_scale_cce2 = []
decor_scale_stdm_cce2 = []

# Loop through window durations
for seg_duration in seg_durations:

    # --- CCE 1 --- # 

    filename1 = (
        PATH_cce1_processed
        / f"cce1_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_"
          f"seg_duration_{seg_duration}mo.nc"
    )

    with Dataset(filename1, "r") as nc:

        # Append decorrelation scale to list
        decor_scale_cce1.append(nc.variables["decor_scale"][:])
        decor_scale_stdm_cce1.append(nc.variables["decor_scale_stdm"][:])

        # Save depth coordinate once
        if seg_duration == seg_durations[0]:
            depth_cce1 = nc.variables["depth"][:]


    # --- CCE 2 --- # 

    filename2 = (
        PATH_cce2_processed
        / f"cce2_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_"
            f"seg_duration_{seg_duration}mo.nc"
    )

    with Dataset(filename2, "r") as nc:
    
        # Append decorrelation scale to list
        decor_scale_cce2.append(nc.variables["decor_scale"][:])
        decor_scale_stdm_cce2.append(nc.variables["decor_scale_stdm"][:])

        # Save depth coordinate once
        if seg_duration == seg_durations[0]:
            depth_cce2 = nc.variables["depth"][:]

    
# Stack window durations along a new first dimension
decor_scale_cce1 = np.ma.stack(decor_scale_cce1, axis=0)
decor_scale_stdm_cce1 = np.ma.stack(decor_scale_stdm_cce1, axis=0)
decor_scale_cce2 = np.ma.stack(decor_scale_cce2, axis=0)
decor_scale_stdm_cce2 = np.ma.stack(decor_scale_stdm_cce2, axis=0)

# -----------------------------------------------------------------------------
# Compute the time mean and standard deviation mixed layer depth 
# -----------------------------------------------------------------------------

mld_mean = np.ma.mean(mld,axis=1)
mld_std = np.ma.std(mld,axis=1,ddof=1)

# -----------------------------------------------------------------------------
# Plot decorrelation time scales at mooring sites
# -----------------------------------------------------------------------------

# Set depth convention to positive downward
depth_m_pos = np.abs(depth_m) 
depth_cce1_pos = np.abs(depth_cce1) 
depth_cce2_pos = np.abs(depth_cce2) 

# Set standard observational depths 
cce1_standard_depth = np.array([10, 20, 30, 40, 60, 75, 150])
cce2_standard_depth = np.array([7, 15, 25, 45, 75])

# Use a discrete colormap for the seven window durations
colors = cmo.haline_r(np.linspace(0.05, 0.95, len(seg_durations)))

# Create figure 
fig, axes = plt.subplots(2,3,figsize=(15, 10))
axes_flat = axes.flatten()

# --- Subplot 1 --- # 
ax = axes_flat[0]

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract decorrelation scale and uncertainty
    Lt = decor_scale_m[j,0,:]
    Lt_stdm = decor_scale_stdm_m[j,0,:]

    # Plot mean decorrelation scale depth profile
    ax.plot(
        Lt,
        depth_m_pos,
        '.-',
        color=colors[j],
        linewidth=2,
        markersize=5,
        label=f"{seg_duration} month"
    )

    # Plot standard error of the mean 
    ax.fill_betweenx(
        depth_m_pos,
        Lt - Lt_stdm,
        Lt + Lt_stdm,
        color=colors[j],
        alpha=0.12,
    )

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[0], ls='--', lw=1.5, color='dimgray', alpha=1, label=r"$\overline{z}_{mld}$")

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 45], mld_mean[0] - mld_std[0], mld_mean[0] + mld_std[0], color='dimgray', alpha=0.15, label=r"$\sigma_{\overline{z}_{mld}}$")

# Set left edge x-position
x_right = ax.get_xlim()[0] + 2.25  

# Plot model grid depth levels
ax.plot(np.full_like(depth_m_pos[:-1], x_right), depth_m_pos[:-1], marker='.', linestyle='None', color='k', markersize=6, alpha=0.6,clip_on=False)

# Set figure attributes
ax.set_ylabel('Depth (m)')
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xlim(0,45)
ax.set_ylim(0,200)
ax.set_xticklabels([])
ax.grid(True, ls= '--', lw=0.5, alpha=0.1,color='k')
ax.tick_params(top=False, 
            bottom=True, 
            left=True, 
            right=True, 
            labelleft=True,
            direction='out', 
            length=3.5)
ax.invert_yaxis()

# --- Subplot 2 --- # 
ax = axes_flat[1]

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract decorrelation scale and uncertainty
    Lt = decor_scale_m[j,1,:]
    Lt_stdm = decor_scale_stdm_m[j,1,:]

    # Plot mean decorrelation scale depth profile
    ax.plot(
        Lt,
        depth_m_pos,
        '.-',
        color=colors[j],
        linewidth=2,
        markersize=5,
    )

    # Plot standard error of the mean 
    ax.fill_betweenx(
        depth_m_pos,
        Lt - Lt_stdm,
        Lt + Lt_stdm,
        color=colors[j],
        alpha=0.12,
    )

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[1], ls='--', lw=1.5, color='dimgray', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 45], mld_mean[1] - mld_std[1], mld_mean[1] + mld_std[1], color='dimgray', alpha=0.15)

# Set left edge x-position
x_right = ax.get_xlim()[0] + 2.25  

# Plot model grid depth levels
ax.plot(np.full_like(depth_m_pos[:-1], x_right), depth_m_pos[:-1], marker='.', linestyle='None', color='k', markersize=6, alpha=0.6,clip_on=False)

# Set figure attributes
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xlim(0,45)
ax.set_ylim(0,200)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, ls= '--', lw=0.5, alpha=0.1,color='k')
ax.tick_params(top=False, 
            bottom=True, 
            left=True, 
            right=True, 
            labelleft=True,
            direction='out', 
            length=3.5)
ax.invert_yaxis()

# --- Subplot 3 --- # 
ax = axes_flat[2]

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract decorrelation scale and uncertainty
    Lt = decor_scale_m[j,2,:]
    Lt_stdm = decor_scale_stdm_m[j,2,:]

    # Plot mean decorrelation scale depth profile
    ax.plot(
        Lt,
        depth_m_pos,
        '.-',
        color=colors[j],
        linewidth=2,
        markersize=5,
    )

    # Plot standard error of the mean 
    ax.fill_betweenx(
        depth_m_pos,
        Lt - Lt_stdm,
        Lt + Lt_stdm,
        color=colors[j],
        alpha=0.12,
    )

# Plot the mean mixed layer depth 
ax.axhline(mld_mean[2], ls='--', lw=1.5, color='dimgray', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 45], mld_mean[2] - mld_std[2], mld_mean[2] + mld_std[2], color='dimgray', alpha=0.15)

# Set left edge x-position
x_right = ax.get_xlim()[0] + 2.25  

# Plot model grid depth levels
ax.plot(np.full_like(depth_m_pos[:-1], x_right), depth_m_pos[:-1], marker='.', linestyle='None', color='k', markersize=6, alpha=0.6,clip_on=False, label='Model depths')

# Set figure attributes
ax.set_xlabel('Decorrelation Scale (days)')
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xlim(0,45)
ax.set_ylim(0,200)
ax.set_yticklabels([])
ax.grid(True, ls= '--', lw=0.5, alpha=0.1,color='k')
ax.tick_params(top=False, 
            bottom=True, 
            left=True, 
            right=True, 
            labelleft=True,
            direction='out', 
            length=3.5)
ax.invert_yaxis()

# --- Subplot 4 --- # 
ax = axes_flat[3]

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract decorrelation scale and uncertainty
    Lt = decor_scale_cce1[j,:]
    Lt_stdm = decor_scale_stdm_cce1[j,:]

    # Plot mean decorrelation scale depth profile
    ax.plot(
        Lt,
        depth_cce1_pos,
        '.-',
        color=colors[j],
        linewidth=2,
        markersize=5,
    )

    # Plot standard error of the mean 
    ax.fill_betweenx(
        depth_cce1_pos,
        Lt - Lt_stdm,
        Lt + Lt_stdm,
        color=colors[j],
        alpha=0.12,
    )

# Set left edge x-position
x_right = ax.get_xlim()[0]  

# Plot model grid depth levels
ax.plot(np.full_like(cce1_standard_depth, x_right), cce1_standard_depth, marker='d', linestyle='None', color='k', markersize=5, alpha=1, clip_on=False, label='Standard depths' )

# Set figure attributes
ax.set_xlabel('Decorrelation Scale (days)')
ax.set_ylabel('Depth (m)')
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xlim(0,45)
ax.set_ylim(0,200)
ax.grid(True, ls= '--', lw=0.5, alpha=0.1,color='k')
ax.tick_params(top=True, 
            bottom=True, 
            left=True, 
            right=True, 
            labelleft=True,
            direction='out', 
            length=3.5)
ax.invert_yaxis()

# --- Subplot 5 --- # 
ax = axes_flat[4]

# Loop through durations
for j, seg_duration in enumerate(seg_durations):

    # Extract decorrelation scale and uncertainty
    Lt = decor_scale_cce2[j,:]
    Lt_stdm = decor_scale_stdm_cce2[j,:]

    # Plot mean decorrelation scale depth profile
    ax.plot(
        Lt,
        depth_cce2_pos,
        '.-',
        color=colors[j],
        linewidth=2,
        markersize=5
    )

    # Plot standard error of the mean 
    ax.fill_betweenx(
        depth_cce2_pos,
        Lt - Lt_stdm,
        Lt + Lt_stdm,
        color=colors[j],
        alpha=0.12,
    )

# Set left edge x-position
x_right = ax.get_xlim()[0]  

# Plot model grid depth levels
ax.plot(np.full_like(cce2_standard_depth, x_right), cce2_standard_depth, marker='d', linestyle='None', color='k', markersize=5, alpha=1, clip_on=False)

# Set figure attributes
ax.set_xlabel('Decorrelation Scale (days)')
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xlim(0,45)
ax.set_ylim(0,200)
ax.set_yticklabels([])
ax.grid(True, ls= '--', lw=0.5, alpha=0.1,color='k')
ax.tick_params(top=True, 
            bottom=True, 
            left=True, 
            right=True, 
            labelleft=True,
            direction='out', 
            length=3.5)
ax.invert_yaxis()

#--- Subplot 6 ---# 
ax = axes_flat[5]

# Turn off axis
ax.axis('off')

# Obtain the handle for the legend 
handles = []
labels = []

# Loop through axes that contain legend items
for i in [0,1,2,3]:  
    h, l = axes_flat[i].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Display legend in the position of the 6th axis
ax.legend(handles, labels, loc='center', fontsize=12)

# Label each subplot
pos = [0.94, 0.07]
add_corner_label(axes_flat[0], pos, 'A', fontsize = 16)
add_corner_label(axes_flat[1], pos, 'B', fontsize = 16)
add_corner_label(axes_flat[2], pos, 'C', fontsize = 16)
add_corner_label(axes_flat[3], pos, 'D', fontsize = 16)
add_corner_label(axes_flat[4], pos, 'E', fontsize = 16)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------

fig.savefig(
    PATH_figs / "figS07.png",
    dpi=300,
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
)

