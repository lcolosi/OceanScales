# =============================================================================
# Figure S03
# =============================================================================
#
# Caption:
#   Decorrelation time scale along CalCOFI line 80 computed using (a) 1 month,
#   (b) 2 month, (c) 3 month, (d) 4 month, (e) 6 month, (f) 8 month, and
#   (g) 12 month window duration following the methodology for computing
#   decorrelation time scale discussed in section 3 of the paper. Gray shading
#   is the ocean bottom. Decorrelation scales that differ from the regional
#   spatial mean by less than or equal to one standard error are considered 
#   not statistically significant and are indicated by hatching. The regional 
#   spatial median (solid blue) and the  25$^{textrm{th}}$ to
#   75$^{textrm{th}}$ percentile range (blue shading) are shown in panel (h).
# 
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-26
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cmocean.cm as cmo
import matplotlib as mpl

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
#                Options: "temp", "sal", "density", "u_along", or "v_cross".
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - option_var : Specifies whether to plot the mean or standard deviation of
#                the decorrelation scale. 
#                        Options: "mean" or "std" 
# - sn_threshold : Signal-to-noise ratio threshold for the statistical significance 
#                  criteria. Represents the number of standard deviation a
#                  decorrelation scale estimate is away from the regional spatial
#                  median.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_detrend_seg = True
option_var         = 'mean'

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set uncertainty estimate parameters
sn_threshold = 1

# -----------------------------------------------------------------------------
# Set segment durations, plotting limits and plotting parameters
# -----------------------------------------------------------------------------

# Set plotting parameters
fontsize_l = 14
pos = [0.075, 0.1]
subplot_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
mpl.rcParams["hatch.linewidth"] = 0.2 

if option_var == 'mean': 
    label = 'Decorrelation Scale (days)'
elif option_var == 'std': 
    label = 'Standard deviation (days)'

# Set font and fontsize using LaTeX 
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# Segment durations to plot (months)
segment_months = [1, 2, 3, 4, 6, 8, 12]

if option_var == "mean":

    # Set colorbar limits for each segment duration
    scale_limits = {
        1:  (2, 4),
        2:  (3, 8),
        3:  (4, 12),
        4:  (4, 16),
        6:  (6, 22),
        8:  (8, 30),
        12: (10, 40),
    }

    # Contour intervals steps
    steps = {
        1:  0.05,
        2:  0.1,
        3:  0.2,
        4:  0.25,
        6:  0.5,
        8:  0.5,
        12: 1,
    }

    # Colorbar tick steps
    step_ticks = {
        1:  0.5,
        2:  1,
        3:  1,
        4:  2,
        6:  2,
        8:  5,
        12: 5,
    }

elif option_var == "std":

    # Set colorbar limits for each segment duration
    scale_limits = {
        1:  (1, 2),
        2:  (1, 3),
        3:  (1, 5),
        4:  (1, 7),
        6:  (1, 9),
        8:  (1, 10),
        12: (1, 10),
    }

    # Contour intervals steps
    steps = {
        1:  0.025,
        2:  0.05,
        3:  0.05,
        4:  0.25,
        6:  0.25,
        8:  0.25,
        12: 0.25,
    }

    # Colorbar tick steps
    step_ticks = {
        1:  0.25,
        2:  0.25,
        3:  0.5,
        4:  1,
        6:  1,
        8:  1,
        12: 1,
    }

# -----------------------------------------------------------------------------
# Load transect bathymetry
# -----------------------------------------------------------------------------

filename_depth = PATH_data / "mitgcm" / "transect" / "DEPTH_CCS_trans.nc"

with Dataset(filename_depth, "r") as nc:
    distance_wd = nc.variables["distance"][:]
    bottom_depth = nc.variables["Depth"][:]

# -----------------------------------------------------------------------------
# Plot regional decorrelation time scales
# -----------------------------------------------------------------------------

# Initialize array for median decorrelation scales
median = np.zeros(len(segment_months))
q25 = np.zeros(len(segment_months))
q75 = np.zeros(len(segment_months))

# Define processed data path
PATH_processed = PATH_data / "mitgcm" / "transect" / "processed"

# Create figure
fig, axes = plt.subplots(
    2,
    4,
    figsize=(20, 9),
    constrained_layout=True,
)

# Flatten axes 
axes = axes.ravel()

# Loop through segment durations
for i, months in enumerate(segment_months):

    # Set axis handle
    ax = axes[i]

    # Set panel-specific color scale
    vmin, vmax = scale_limits[months]
    step = steps[months]
    levels = np.arange(vmin,vmax + step,step)

    #------------------------------------------#
    # Load decorrelation-scale data
    #------------------------------------------#

    filename_mitgcm = (
        PATH_processed
        / f"mitgcm_decor_scale_{option_data}_hrly_trans_"
          f"{option_interannual}_{seg_proc}_"
          f"seg_duration_{months}mo.nc"
    )

    with Dataset(filename_mitgcm, "r") as nc:

        distance = nc.variables["dist"][:]
        depth = nc.variables["depth"][:]

        if option_var == "mean":
            data     = nc.variables["decor_scale"][:]
            data_unc = nc.variables["decor_scale_stdm"][:]

        elif option_var == "std":
            data     = nc.variables["decor_scale_std"][:]
            data_unc = nc.variables["decor_scale_stds"][:]


    #------------------------------------------#
    # Compute transect statistics
    #------------------------------------------#

    median[i] = np.ma.median(data)
    q25[i] = np.percentile(data.compressed(), 25)
    q75[i] = np.percentile(data.compressed(), 75)

    #------------------------------------------#
    # Compute statistical significance mask
    #------------------------------------------#

    # Compute spatial median
    trans_median = np.ma.median(data)

    # Compute the signal-to-noise ratio (with respect to the regional mean)
    sn_ratio =  np.abs(data - trans_median) / data_unc

    # Identify non-significant values
    significance_mask = np.ma.getmask(
        np.ma.masked_less_equal(sn_ratio, sn_threshold)
    )

    # Get land mask
    data_mask_array = np.ma.getmaskarray(data)

    # Keep only non-significant ocean points
    significance_mask = significance_mask & ~data_mask_array

    # Non-significant ocean points = 1; everything else = NaN    
    data_mask = np.where(significance_mask, 1, np.nan)

    #------------------------------------------#
    # Plot data
    #------------------------------------------#

    # Plot decorrelation time scales
    ct = ax.contourf(
        distance,
        abs(depth),
        data.T,
        levels=levels,
        cmap=cmo.amp,
        extend="both",
    )

    # Overlay statistical-significance hatching
    ax.contourf(
        distance,
        abs(depth),
        data_mask.T,
        levels=[0.5, 1.5],
        hatches=["..."],
        colors="none",
    )

    # Plot bathymetry
    ax.fill_between(
        distance_wd,
        bottom_depth,
        abs(depth).max(),
        color="0.5",
    )


    #------------------------------------------#
    # Set axis attributes
    #------------------------------------------#

    ax.set_ylim(
        abs(depth).max(),
        abs(depth).min(),
    )

    ax.set_xlim(
        distance.min(),
        distance.max(),
    )

    # Invert x axis
    ax.invert_xaxis()

    # Only show x labels on bottom row
    if i < 4:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )

    # Only show y labels on left column
    if i not in (0, 4):
        ax.tick_params(
            axis="y",
            labelleft=False,
        )

    if i in (0, 4):
        ax.set_ylabel(
            "Depth (m)",
            fontsize=14,
        )

    if i >= 4:
        ax.set_xlabel(
            "Distance from shore (km)",
            fontsize=14,
        )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.15,
    )

    # Add Panel title
    ax.set_title(
        f"{months} month"
        if months == 1
        else f"{months} months",
        fontsize=14,
    )

    # Add corner subplot label 
    add_corner_label(ax, pos, subplot_label[i], fontsize = fontsize_l)

    # Add individual colorbar
    cbar = fig.colorbar(
        ct,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.04,
    )

    cbar.set_label(
        label,
        fontsize=12,
    )

    cbar.ax.tick_params(
        labelsize=11,
    )

    cbar.set_ticks(
        np.arange(
            vmin,
            vmax + step_ticks[months],
            step_ticks[months],
        )
    )

# -----------------------------------------------------------------------------
# Plot transect median decorrelation scale
# -----------------------------------------------------------------------------

# Set axis handle
ax = axes[-1]

# Remove previous axes and replace with a new one
position = ax.get_position()
ax.remove()

ax_med = fig.add_axes([
    position.x0 + 0.07,
    position.y0 + 0.06,
    position.width,
    position.height - 0.05,
])

# Plot interquartile range
ax_med.fill_between(
    segment_months,
    q25,
    q75,
    alpha=0.2,
    color='tab:blue',
    label="25th–75th percentile",
)


# Plot transect median
ax_med.plot(
    segment_months,
    median,
    marker="o",
    linewidth=2,
    markersize=7,
    color='tab:blue',
    label="Transect Median",
)


# Set labels
ax_med.set_xlabel(
    "Window duration (months)",
    fontsize=14,
)

ax_med.set_ylabel(
    label,
    fontsize=14,
)


# Set ticks
ax_med.set_xticks(segment_months)

ax_med.tick_params(
    axis="both",
    labelsize=12,
)


# Add grid
ax_med.grid(
    True,
    linestyle="--",
    alpha=0.3,
)


# Add legend
ax_med.legend(
    loc="upper left",
    fontsize=12,
)

# Label subplot
add_corner_label(ax_med, [0.9, 0.1], 'H', fontsize = fontsize_l)

# Adjust spacing
fig.get_layout_engine().set(
    wspace=0.05,
    hspace=0.05,
)

# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------

fig.savefig(
    PATH_figs / "figS03.png",
    dpi=300,
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
)