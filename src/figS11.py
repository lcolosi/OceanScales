# =============================================================================
# Figure 11
# =============================================================================
#
# Caption:
#   Decorrelation time scale in the study domain at (a) 0.5, (b) 9.66, (c) 52.69, (d) 99.37 meter depth. 
#   Black contour lines are the ocean topography with 200 and 2000 meter
#   isobaths highlighted as solid black lines. Decorrelation scales that differ
#   from the regional spatial medain by less than or equal to one standard error
#   are considered not statistically significant and are indicated by hatching.
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

# Import plotting toolbox 
from plotting import set_coastlines, set_grid_ticks, set_cbar, add_scalebar, add_corner_label, plot_transect_distance_axis

# -----------------------------------------------------------------------------
# Set processing and plotting parameters
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
#                        Options: "detrend", "demean", or None
# - option_averaging: Specifies whether to compute the variance-weighted or unweighted
#                     ensemble-average of the autocorrelation.
#                      Options: "weighted" for variance-weighted average.
#                               "unweighted" for unweighted average.
# - segment_months : Specifies the window duration. 
# - sn_threshold : Signal-to-noise ratio threshold for the statistical significance 
#                  criteria. Represents the number of standard deviation a
#                  decorrelation scale estimate is away from the regional spatial
#                  median. 
#
# ------------#

# Set processing parameters
option_data        = 'density'     
option_interannual = 'linear' 
option_detrend_seg = "detrend"
option_averaging   = "weighted" 
option_plot_vel    = True
segment_months     = 6

# Label segment processing method 
seg_proc = "none" if option_detrend_seg is None else option_detrend_seg

# Set uncertainty estimate parameters
sn_threshold = 1

# -----------------------------------------------------------------------------
# Set depths, plotting limits, and other plotting parameters
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
resolution = "10m"
xticks = [-123, -122.5, -122, -121.5, -121, -120.5, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
level_is = np.arange(100, 300, 100)
levels_ms = np.arange(1000, 3000, 500)
fontsize_g = 18
fontsize_c = 10
fontsize_l = 18
pos = [0.96, 0.94]
subplot_label = ['A', 'B', 'C', 'D']
cmap = cmo.amp
mpl.rcParams["hatch.linewidth"] = 0.2 
label = 'Decorrelation Scale (days)'

# Set font and fontsize using LaTeX 
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# Depths to plot (meters)
depths = [0.5, 9, 52, 99]

# Set colorbar limits, contour interval steps, and colorbar tick steps
levels   = np.arange(6,24+0.25,0.25)
ticks_cb = np.arange(6,24+2,2) 

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

# --- CalCOFI line 80.0 Positions --- # 

# Obtain filename path
filename = PATH_data / "mitgcm" / "transect" / "processed" / "mitgcm_proc_vel_hrly_trans.nc"

# Load transect coordinates
with Dataset(filename, 'r') as nc:
    dist_tr = np.asarray(nc.variables['dist'][:])
    lon_tr  = np.asarray(nc.variables['LON'][:])
    lat_tr  = np.asarray(nc.variables['LAT'][:])

# Convert longitude to -180 to 180 degrees
lon_tr = ((lon_tr + 180) % 360) - 180

# Sort coordinates by distance from shore
idx = np.argsort(dist_tr)
dist_tr = dist_tr[idx]
lon_tr  = lon_tr[idx]
lat_tr  = lat_tr[idx]

# -----------------------------------------------------------------------------
# Plot regional decorrelation time scales
# -----------------------------------------------------------------------------

# Create figure
fig, axes = plt.subplots(
    2,
    2,
    figsize=(20, 13),
    subplot_kw={"projection": projection}
)

# Flatten axes 
axes = axes.ravel()

# Loop through segment durations
for i, depth in enumerate(depths):

    # Set axis handle
    ax = axes[i]

    #------------------------------------------#
    # Load decorrelation scale data
    #------------------------------------------#

    filename_mitgcm = (
        PATH_processed
        / f"mitgcm_decor_scale_{option_data}_hrly_reg_"
            f"depth_{depth}m_{option_interannual}_{option_averaging}_{seg_proc}_"
            f"seg_duration_{segment_months}mo.nc"
    )

    with Dataset(filename_mitgcm, "r") as nc:

        lon = nc.variables["lon"][:]
        lat = nc.variables["lat"][:]

        data = nc.variables["decor_scale"][:]
        data_unc = nc.variables["decor_scale_stdm"][:]

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
        hatches=['....'],
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

    # Plot Line 80 CalCOFI transect
    ax.plot(
        lon_tr,
        lat_tr,
        color='k',
        linestyle= (0, (3, 2)),
        linewidth=1.5,
        transform=ccrs.PlateCarree(),
        zorder=9,
    )

    # Plot distance axis along CalCOFI Line 80.0 
    plot_transect_distance_axis(
        ax,
        dist_tr,
        lon_tr,
        lat_tr,
        tick_spacing=25,
        label_spacing=50,
        tick_length=2.0,
        linewidth = 1.25,
        show_labels=False,
        max_tick_distance=275,
    )

    #------------------------------------------#
    # Set axis attributes
    #------------------------------------------#

    # Only show longitude labels on bottom row
    xlabels = i >= 0

    # Only show latitude labels on left column
    ylabels = i in (0, 1, 2, 3)

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
    ax.set_title(f"{depth} m", fontsize=20)

    # Add corner subplot label 
    add_corner_label(ax, pos, subplot_label[i], fontsize = fontsize_l)

    if i == 0:
    
        # Create colormap
        cax = plt.axes([0.92, 0.29, 0.02, 0.4])
        set_cbar(
            ct,
            cax,
            fig,
            orientation="vertical",
            extend="both",
            label=label,
            fontsize=fontsize_g,
            ticks=ticks_cb, 
            invert = False
        )
        
        # Add a 20-km scale bar
        add_scalebar(
            ax, 
            length_km=20, 
            location=(0.925, 0.78),
            linewidth=1, 
            text_kwargs=dict(fontsize=14, color='white', weight='bold')
        )

# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------

# Save figure
fig.savefig(
    PATH_figs / "figS11.png",
    dpi=300,
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
)



