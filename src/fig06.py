# =============================================================================
# Figure 06
# =============================================================================
#
# Caption:
#   Decorrelation time scale of potential density (solid curve) as a function of
#   depth from model data at (a) CCE1, (b) CCE2, and (c) CCE3, and observations
#   at (d) CCE1 and (e) CCE2. The standard error of the mean for the decorrelation 
#   scale is shown as the shaded regions. The seasonally averaged mixed-layer depth
#   $\overline{z}_{mld}$ and its standard deviation are displayed as horizontal
#   dashed lines and light shaded regions, respectively. For CCE1, CCE2, and CCE3
#   model data, solid black circles denote the vertical model grid. For CCE1 and CCE2
#   observations, solid back diamonds denote the depth of the CTD sensor.
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
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_detrend_seg = True
segment_months     = 8

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
# Load MITgcm decorrelation scales, bathymetry, CCE, and CalCOFI data
# -----------------------------------------------------------------------------

# --- MITgcm Data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
site    = nc.variables['site'][:]
depth_m = nc.variables['depth'][:]
Lt      = nc.variables['decor_scale'][:]
Lt_stdm = nc.variables['decor_scale_stdm'][:]
Lt_std  = nc.variables['decor_scale_std'][:]

# --- Mixed Layer Depth ---# 


# --- CCE Data --- # 

# -----------------------------------------------------------------------------
# Compute the time mean and standard deviation mixed layer depth 
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Plot decorrelation time scales at mooring locations 
# -----------------------------------------------------------------------------

# Set plotting parameters 
depth_pos_m = abs(depth_m)
depth_lim = [0,200]
cce1_sensor_depth = np.array([9, 19, 29, 39, 60, 75, 150])
cce2_sensor_depth = np.array([6, 14, 25, 44, 74])

# Create figure
fig, axes = plt.subplots(2,3,figsize=(15, 10))
ax_flat = axes.flatten()

# --- Subplot 1 --- # 
ax = ax_flat[0]

# Plot CCE1 potential density decor scale
ax.plot(Lt[0,:],depth_pos_m,'.-', color='tab:green', label='CCE1')

# Plot standard error of the mean
ax.fill_betweenx(depth_pos_m, Lt[0,:] - Lt_stdm[0,:], Lt[0,:] + Lt_stdm[0,:], color='tab:green', alpha=0.5)

# Set left edge x-position
x_right = ax.get_xlim()[0] - 13.6  

# Plot model grid depth levels
ax.plot(
    np.full_like(depth_pos_m[:-1], x_right),
    depth_pos_m[:-1],
    marker='.', 
    linestyle='None',
    color='k', 
    markersize=6, 
    alpha=0.6,
    clip_on=False,
)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlim(0,45)
ax.set_ylim(depth_lim[0], depth_lim[1])
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xticklabels([])
ax.invert_yaxis()
ax.tick_params(top=True, 
               bottom=True, 
               left=True, 
               right=True, 
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

# --- Subplot 2 --- # 
ax = ax_flat[1]

# Plot CCE1 potential density decor scale
ax.plot(Lt[1,:],depth_pos_m,'.-', color='tab:red', label='CCE1')

# Plot standard error of the mean
ax.fill_betweenx(depth_pos_m, Lt[1,:] - Lt_stdm[1,:], Lt[1,:] + Lt_stdm[1,:], color='tab:red', alpha=0.5)

# Set left edge x-position
x_left = ax.get_xlim()[0] - 10.4  

# Plot model grid depth levels
ax.plot(
    np.full_like(depth_pos_m[:-1], x_left),
    depth_pos_m[:-1],
    marker='.', 
    linestyle='None',
    color='k', 
    markersize=6, 
    alpha=0.6,
    clip_on=False,
)

# Set axis attributes
ax.set_xlim(0,45)
ax.set_ylim(depth_lim[0], depth_lim[1])
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.invert_yaxis()
ax.tick_params(top=True, 
               bottom=True, 
               left=True, 
               right=True, 
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

# --- Subplot 3 --- # 
ax = ax_flat[2]

# Plot CCE1 potential density decor scale
ax.plot(Lt[2,:],depth_pos_m,'.-', color='tab:blue', label='CCE3')

# Plot standard error of the mean
ax.fill_betweenx(depth_pos_m, Lt[2,:] - Lt_stdm[2,:], Lt[2,:] + Lt_stdm[2,:], color='tab:blue', alpha=0.5)

# Set left edge x-position
x_left = ax.get_xlim()[0] - 9 

# Plot model grid depth levels
ax.plot(
    np.full_like(depth_pos_m[:-1], x_left),
    depth_pos_m[:-1],
    marker='.', 
    linestyle='None',
    color='k', 
    markersize=6, 
    alpha=0.6,
    clip_on=False, 
    label='Model grid',
)

# Set axis attributes
ax.set_xlabel(r'Decorrelation Scale (days)')
ax.set_xlim(0,45)
ax.set_ylim(depth_lim[0], depth_lim[1])
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.tick_params(top=True, 
               bottom=True, 
               left=True, 
               right=True, 
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

#--- Subplot 4 ---# 
ax = ax_flat[3]

# Set left edge x-position 
x_left = ax.get_xlim()[0]

# Plot the sensor depths 
ax.plot(
    np.full_like(cce1_sensor_depth, x_left),
    cce1_sensor_depth,
    marker='d', 
    linestyle='None',
    color='k', 
    markersize=5, 
    alpha = 1, 
    clip_on=False,
    label='Sensor depth'  
)

# Set axis attributes
ax.set_ylabel('Depth (m)')
ax.set_xlabel(r'Decorrelation Scale (days)')
ax.set_xlim(0,45)
ax.set_ylim(depth_lim[0], depth_lim[1])
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_yaxis()
ax.tick_params(top=True, 
               bottom=True, 
               left=True, 
               right=True, 
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

#--- Subplot 5 ---# 
ax = ax_flat[4]

# Set left edge x-position 
x_left = ax.get_xlim()[0] 

# Plot the sensor depths 
ax.plot(
    np.full_like(cce2_sensor_depth, x_left),
    cce2_sensor_depth,
    marker='d', 
    linestyle='None',
    color='k', 
    markersize=5, 
    alpha = 1, 
    clip_on=False,
)

# Set axis attributes
ax.set_xlabel(r'Decorrelation Scale (days)')
ax.set_xlim(0,45)
ax.set_ylim(depth_lim[0], depth_lim[1])
ax.set_xticks(np.arange(0,45+5,5))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.tick_params(top=True, 
               bottom=True, 
               left=True, 
               right=True, 
               labelleft=True,
               direction='out', 
               length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

#--- Subplot 6 ---# 
ax = ax_flat[5]

# Turn off axis
ax.axis('off')

# Obtain the handle for the legend 
handles = []
labels = []

# Loop through axes that contain legend items
for i in [0,1,2,3]:  
    h, l = ax_flat[i].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Display legend in the position of the 6th axis
ax.legend(handles, labels, loc='center', fontsize=16)

# Label each subplot
pos = [0.94, 0.07]
add_corner_label(ax_flat[0], pos, 'A', fontsize = 16)
add_corner_label(ax_flat[1], pos, 'B', fontsize = 16)
add_corner_label(ax_flat[2], pos, 'C', fontsize = 16)
add_corner_label(ax_flat[3], pos, 'D', fontsize = 16)
add_corner_label(ax_flat[4], pos, 'E', fontsize = 16)

# Adjust figure spacing
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig04.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)


