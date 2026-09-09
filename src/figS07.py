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
option_interannual = 'linear' 
option_detrend_seg = True
option_var         = 'mean'

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# -----------------------------------------------------------------------------
# Set plotting parameters
# -----------------------------------------------------------------------------

# Set plotting parameters
seg_durations = [1, 2, 3, 4, 6, 8, 12]
site_names = ["CCE1", "CCE2", "CCE3"]

# Path to processed data 
PATH_processed = PATH_data / "mitgcm" / "mooring" / "processed"

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load decorrelation-scale estimates
# -----------------------------------------------------------------------------

# Initialize array
decor_scale = []
decor_scale_stdm = []

# Loop through window durations
for seg_duration in seg_durations:

    filename = (
        PATH_processed
        / f"mitgcm_decor_scale_density_hrly_mooring_linear_detrend_"
          f"seg_duration_{seg_duration}mo.nc"
    )

    with Dataset(filename, "r") as nc:

        # Append decorrelation scale to list
        decor_scale.append(nc.variables["decor_scale"][:])
        decor_scale_stdm.append(nc.variables["decor_scale_stdm"][:])

        # Save depth coordinate once
        if seg_duration == seg_durations[0]:
            depth = nc.variables["depth"][:]


# Stack window durations along a new first dimension
decor_scale = np.ma.stack(decor_scale, axis=0)
decor_scale_stdm = np.ma.stack(decor_scale_stdm, axis=0)

# -----------------------------------------------------------------------------
# Plot decorrelation time scales at mooring sites
# -----------------------------------------------------------------------------

# Set depth convention to positive downward
depth_pos = np.abs(depth) 

# Create figure 
fig, axes = plt.subplots(
    1,
    3,
    figsize=(10, 6),
    constrained_layout=True
)

# Use a discrete colormap for the seven window durations
colors = cmo.haline_r(
    np.linspace(0.05, 0.95, len(seg_durations))
)

# Loop through mooring sites 
for isite, ax in enumerate(axes):

    # Loop through durations
    for j, seg_duration in enumerate(seg_durations):

        # Extract decorrelation scale and uncertainty
        Lt = decor_scale[j,isite,:]

        Lt_stdm = decor_scale_stdm[j,isite,:]

        # Plot mean decorrelation-scale profile
        ax.plot(
            Lt,
            depth_pos,
            '.-',
            color=colors[j],
            linewidth=2,
            markersize=5,
            label=f"{seg_duration} month"
        )

        # Plot uncertainty envelope
        ax.fill_betweenx(
            depth_pos,
            Lt - Lt_stdm,
            Lt + Lt_stdm,
            color=colors[j],
            alpha=0.12,
        )

    # Set figure attributes
    ax.set_title(site_names[isite], fontsize=fontsize)
    ax.set_xticks(np.arange(0,45+10,10))
    ax.set_yticks(np.arange(0,200+25,25))
    ax.set_xlim(0,45)
    ax.set_ylim(0,200)
    ax.grid(True, ls= '--', lw=0.5, alpha=0.1,color='k')
    ax.tick_params(top=False, 
               bottom=True, 
               left=True, 
               right=True, 
               labelleft=True,
               direction='out', 
               length=3.5)
    ax.invert_yaxis()

    # Remove y-axis tickmarks of second and third panels
    if isite > 0:
        ax.set_yticklabels([])

# Axis labels and limits
axes[0].set_ylabel("Depth (m)")
for ax in axes:
    ax.set_xlabel("Decorrelation Scale (days)")

# Add legend
axes[-1].legend(
    loc="lower right",
    frameon=False,
    fontsize=9
)

# Label each subplot
pos = [0.925, 0.955]
add_corner_label(axes[0], pos, 'A', fontsize = fontsize-2)
add_corner_label(axes[1], pos, 'B', fontsize = fontsize-2)
add_corner_label(axes[2], pos, 'C', fontsize = fontsize-2)

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

