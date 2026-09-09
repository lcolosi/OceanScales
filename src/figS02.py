# =============================================================================
# Figure S02
# =============================================================================
#
# Caption:
#   Decorrelation time scale in the study domain at 9.6 meter water depth
#   computed using (a) 1 month, (b) 2 month, (c) 3 month, (d) 4 month, 
#   (e) 6 month, (f) 8 month, and (g) 12 month window duration following the
#   methodology for computing decorrelation time scale discussed in section 3
#   of the paper. Black contour lines are the ocean topography with 200 and 2000 meter
#   isobaths highlighted as solid black lines. Decorrelation scales that differ
#   from the regional spatial mean by less than or equal to one standard error
#   are considered not statistically significant and are indicated by hatching.
#   The regional spatial median (solid blue) and the  25$^{textrm{th}}$ to
#   75$^{textrm{th}}$ percentile range (blue shading) are shown in panel (h).   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-18
# =============================================================================

# Import libraries 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import cartopy.crs as ccrs
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

# Import plotting toolbox for cartopy figures
from plotting import set_coastlines, set_grid_ticks, add_corner_label

# -----------------------------------------------------------------------------
# Set processing parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density", "uvel", "vvel", or "ssh".
# - option_depth: Depth at which the decorrelation time scale is computed
#                 (units: meters). Not used for SSH.
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - option_var : Specifies whether to plot the mean or standard deviation of
#                the decorrelation scale. 
#                        Options: "mean" or "std" ' 
# - sn_threshold : Signal-to-noise ratio threshold for the statistical significance 
#                  criteria. Represents the number of standard deviation a
#                  decorrelation scale estimate is away from the regional spatial
#                  median.
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_depth       = 9   
option_interannual = 'linear' 
option_detrend_seg = True
option_var         = 'mean'

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set uncertainty estimate parameters
sn_threshold = 1

# -----------------------------------------------------------------------------
# Set segment durations, plotting limits, and other plotting parameters
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
xticks = [-123, -122, -121, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
resolution = "10m"
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
level_is = np.arange(100, 300, 100)
levels_ms = np.arange(1000, 3000, 500)
fontsize_g = 12
fontsize_c = 7
fontsize_l = 14
pos = [0.94, 0.9]
subplot_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
cmap = cmo.amp
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

if option_var == 'mean': 

    # Set colorbar limits for each segment duration
    scale_limits = {
        1:  (2, 4),
        2:  (3, 7),
        3:  (4, 10),
        4:  (4, 16),
        6:  (6, 22),
        8:  (8, 32),
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
        8:  4,
        12: 5,
    }

elif option_var == 'std': 

    # Set colorbar limits for each segment duration
    scale_limits = {
        1:  (1, 2),
        2:  (1, 3),
        3:  (1, 5),
        4:  (1, 7),
        6:  (1, 10),
        8:  (1, 10),
        12: (1, 12),
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
# Load bathymetry, CCE, and CalCOFI data
# -----------------------------------------------------------------------------

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# --- Bathymetry --- #

filename_bathy = PATH_data / "bathymetry" / "etopo1_point_conception.nc"

with Dataset(filename_bathy, "r") as nc_bathy:
    lon_b = nc_bathy.variables["lon"][:]
    lat_b = nc_bathy.variables["lat"][:]
    bathy = nc_bathy.variables["BATHY"][:]


# --- CCE Mooring Locations --- #

lat1, lat2, lat3 = 33.457, 34.3075, 34.44825228022894
lon1, lon2, lon3 = -122.52233, -120.8042, -120.53825701527784


# --- CalCOFI Line 80.0 Positions --- #

filename = PATH_data / "calcofi" / "CalCOFIStationOrder.csv"

calCOFI_data = np.genfromtxt(
    filename,
    delimiter=",",
    skip_header=1,
    usecols=(1, 3, 7, 11),
    invalid_raise=False,
)

# Grab stations on line 80.0
calCOFI_line80 = calCOFI_data[calCOFI_data[:, 0] == 80.0]

# Parse data into separate arrays
calCOFI_lat = calCOFI_line80[:, 1]
calCOFI_lon = calCOFI_line80[:, 2]

# -----------------------------------------------------------------------------
# Plot regional decorrelation time scales
# -----------------------------------------------------------------------------

# Initialize array for regional median decorrelation scales
median = np.zeros(len(segment_months))
q25 = np.zeros(len(segment_months))
q75 = np.zeros(len(segment_months))

# Create figure
fig, axes = plt.subplots(
    2,
    4,
    figsize=(18, 9),
    subplot_kw={"projection": projection},
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
    levels = np.arange(vmin,vmax+step,step,)

    #------------------------------------------#
    # Load decorrelation-scale data
    #------------------------------------------#

    filename_mitgcm = (
        PATH_processed
        / f"mitgcm_decor_scale_{option_data}_hrly_reg_"
          f"depth_{option_depth}m_{option_interannual}_{seg_proc}_"
          f"seg_duration_{months}mo.nc"
    )

    with Dataset(filename_mitgcm, "r") as nc:

        lon = nc.variables["lon"][:]
        lat = nc.variables["lat"][:]

        if option_var == 'mean': 
            data = nc.variables["decor_scale"][:]
            data_unc = nc.variables["decor_scale_stdm"][:]
        elif option_var == 'std': 
            data = nc.variables["decor_scale_std"][:]
            data_unc = nc.variables["decor_scale_stds"][:]

    #------------------------------------------#
    # Compute regional median and inter-quartile range
    #------------------------------------------#

    median[i] = np.ma.median(data)
    q25[i] = np.percentile(data.compressed(), 25)
    q75[i] = np.percentile(data.compressed(), 75)

    #------------------------------------------#
    # Compute statistical significance mask
    #------------------------------------------#

    # Compute spatial median
    reg_median = np.ma.median(data)

    # Compute the signal-to-noise ratio (with respect to the regional mean)
    sn_ratio = np.abs(data - reg_median) / data_unc 

    # Mask non-significant grid points
    significance_mask = np.ma.getmask(
        np.ma.masked_less_equal(sn_ratio, sn_threshold)
    )

    # Get land mask
    land_mask = np.ma.getmaskarray(data)

    # Keep only non-significant ocean points
    significance_mask = significance_mask & ~land_mask

    # Non-significant ocean points = 1; everything else = NaN
    data_mask = np.where(significance_mask, 1, np.nan)

    #------------------------------------------#
    # Plot data
    #------------------------------------------#

    # Plot coastlines and land
    set_coastlines(
        ax,
        projection,
        resolution,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )

    # Plot decorrelation time scales
    ct = ax.contourf(
        lon,
        lat,
        data,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend="both",
    )

    # Overlay statistical-significance hatching
    cs = ax.contourf(
        lon,
        lat,
        data_mask,
        levels=[0.5, 1.5],
        hatches=["..."],
        colors="none",
        zorder=10,
        transform=ccrs.PlateCarree(),
    )

    # Plot CCE moorings
    ax.scatter(
        lon1,
        lat1,
        color="w",
        edgecolor="black",
        marker="^",
        s=25,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    ax.scatter(
        lon2,
        lat2,
        color="w",
        edgecolor="black",
        marker="s",
        s=25,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    ax.scatter(
        lon3,
        lat3,
        color="w",
        edgecolor="black",
        marker="o",
        s=25,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    # Plot bathymetric contours
    ax.contour(
        lon_b,
        lat_b,
        -bathy,
        levels=levels_ms,
        colors="black",
        linewidths=0.4,
        linestyles="dashed",
    )

    ax.contour(
        lon_b,
        lat_b,
        -bathy,
        levels=[2000],
        colors="black",
        linewidths=0.8,
        linestyles="solid",
    )

    ax.contour(
        lon_b,
        lat_b,
        -bathy,
        levels=level_is,
        colors="black",
        linewidths=0.4,
        linestyles="dashed",
    )

    ax.contour(
        lon_b,
        lat_b,
        -bathy,
        levels=[200],
        colors="black",
        linewidths=0.8,
        linestyles="solid",
    )

    # Plot CalCOFI Line 80
    ax.plot(
        calCOFI_lon % 360,
        calCOFI_lat,
        color="k",
        linestyle=(0, (5, 3)),
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )


    #------------------------------------------#
    # Set axis attributes
    #------------------------------------------#

    # Only show longitude labels on bottom row
    xlabels = i >= 0

    # Only show latitude labels on left column
    ylabels = i in (0, 4)

    set_grid_ticks(
        ax,
        xticks=xticks,
        yticks=yticks,
        xlabels=xlabels,
        ylabels=ylabels,
        grid=True,
        fontsize=fontsize_g,
        color="k",
        lw=1,
        ls="--",
        alpha=0.1,
    )

    # Add panel title
    ax.set_title(
        f"{months} month" if months == 1 else f"{months} months",
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
        fontsize=11,
    )

    cbar.ax.tick_params(
        labelsize=10,
    )

    cbar.set_ticks(
        np.arange(vmin,vmax+step_ticks[months],step_ticks[months])
    )


# -----------------------------------------------------------------------------
# Plot regional median decorrelation scale
# -----------------------------------------------------------------------------

# Set axis handle
ax = axes[-1]

# Remove cartopy axes and replace with a normal one
position = ax.get_position()
ax.remove()

ax_med = fig.add_axes([
    position.x0 + 0.065, 
    position.y0,
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

# Plot regional median
ax_med.plot(
    segment_months,
    median,
    marker="o",
    linewidth=2,
    markersize=7,
    color='tab:blue',
    label='Regional Median'
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
    loc='upper left',
    fontsize=12
)

# Label subplot
add_corner_label(ax_med, [0.9, 0.1], 'H', fontsize = fontsize_l)

# Adjust spacing 
fig.get_layout_engine().set(
    wspace=0.05,
    hspace=0,
)

# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------

# Save figure
fig.savefig(
    PATH_figs / "figS02.png",
    dpi=300,
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
)