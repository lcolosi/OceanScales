# =============================================================================
# Figure 09
# =============================================================================
#
# Caption: 
#   Spectral slope $\alpha$ in the study domain at 9.66 meter water depth. Black
#   contour lines are the ocean bathymetry, with 200- and 2000-meter isobaths
#   highlighted as solid black lines. Dashed bathymetric contours are shown at
#   100-m intervals around the 200-m isobath and at 500-m intervals around the
#   2000-m isobath.
#   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-16
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
from plotting import set_coastlines, set_grid_ticks, set_cbar, add_scalebar, add_corner_label

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
option_interannual = 'linear' 
option_detrend_seg = True
segment_months     = 6
option_depth       = 9 
option_slope_range = [182.5, 4.0] 

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set uncertainty estimate parameters
sn_threshold = 1

# Set font and fontsize using LaTeX 
fontsize=16
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm decorrelation scale and spectral data
# -----------------------------------------------------------------------------

# --- MITgcm Spectral Data --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_spectra_{option_data}_hrly_reg_{option_interannual}_slope_{option_slope_range[0]}_{option_slope_range[1]}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
lon             = nc.variables['lon'][:]
lat             = nc.variables['lat'][:]
spec_slope      = nc.variables['SPEC_slope'][:]
spec_slope_stdm = nc.variables['SPEC_slope_stde'][:]
fve             = nc.variables['FVE'][:]

# Convert spectral slope to slope parameter alpha 
alpha = np.abs(spec_slope)

# --- Bathymetry --- # 

# Obtain filename path
filename_bathy = PATH_data / "bathymetry" / "etopo1_point_conception.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
lon_b  = nc_bathy.variables['lon'][:]
lat_b  = nc_bathy.variables['lat'][:]
bathy  = nc_bathy.variables['BATHY'][:]

# --- CCE Mooring Locations --- # 
lat1, lat2, lat3  = 33.457, 34.3075, 34.44825228022894           
lon1, lon2, lon3  = -122.52233, -120.8042, -120.53825701527784 

# --- CalCOFI Line 80.0 Positions --- # 

# Obtain filename path
filename = PATH_data / "calcofi" / "CalCOFIStationOrder.csv"

# Load csv file 
calCOFI_data = np.genfromtxt(
    filename,
    delimiter=",",
    skip_header=1,
    usecols=(1, 3, 7, 11),
    invalid_raise=False
)

# Grab stations on line 80.0
calCOFI_line80 = calCOFI_data[calCOFI_data[:, 0] == 80.0] 

# Parse data into separate arrays
calCOFI_lat   = calCOFI_line80[:, 1]
calCOFI_lon   = calCOFI_line80[:, 2]

# -----------------------------------------------------------------------------
# Compute the relative uncertainty of the spectral slope
# -----------------------------------------------------------------------------

# Compute spatial median
slope_reg_median = np.ma.median(spec_slope)
    
# Compute the signal-to-noise ratio (with respect to the regional mean)
slope_sn_ratio = np.abs(spec_slope - slope_reg_median) / spec_slope_stdm

# Mask not statistically significant grid points
slope_mask = np.ma.getmask(np.ma.masked_less_equal(slope_sn_ratio, sn_threshold))

# Get land mask from spectral slope
land_mask = np.ma.getmaskarray(spec_slope)

# Combine statistical significance and land masks
slope_mask = slope_mask & ~land_mask

# Create a mask array where non-significant ocean points = 1, others = NaN
data_mask = np.where(slope_mask, 1, np.nan)

# -----------------------------------------------------------------------------
# Plot regional spectral slope and FVE 
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
resolution = "10m"
xticks = [-123, -122.5, -122, -121.5, -121, -120.5, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
levels_fve = np.linspace(40, 82, 43)
ticks_fve  = np.linspace(40, 80, 5)
levels_slope = np.linspace(1, 2, 101)
ticks_slope = np.linspace(1, 2, 6)
levels_is = np.arange(100,300+100,100)
levels_ms = np.arange(1000,3000+500,500)
fontsize_g = 18
fontsize_c = 10
cmap_fve = cmo.rain
cmap_slope = cmo.rain
mpl.rcParams["hatch.linewidth"] = 0.2 

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 12), subplot_kw={"projection": projection})

# --- Subplot 1 --- # 
ax = axes[0]

# Plot coastlines and land 
set_coastlines(
    ax, 
    projection, 
    resolution, 
    lon_min=lon_min, 
    lon_max=lon_max, 
    lat_min=lat_min, 
    lat_max=lat_max
) 

# Plot the spectral slope 
ct = ax.contourf(
    lon, 
    lat, 
    alpha, 
    levels=levels_slope,
    transform=ccrs.PlateCarree(),
    cmap=cmap_slope, 
    extend='both'
)

# Overlay a contourf with hatching for the non-significant regions
ax.contourf(
    lon,
    lat,
    data_mask,
    levels=[0.5, 1.5],      
    hatches=['....'],   
    colors='none',          
    zorder=10,              
    transform=ccrs.PlateCarree()
)

# Plot the CCE1 mooring point
ax.scatter(
    lon1, 
    lat1, 
    color='w',
    edgecolor='black', 
    marker='^', 
    s=40, 
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE1'
)

# Plot the CCE2 mooring point
ax.scatter(
    lon2, 
    lat2, 
    color='w',  
    edgecolor='black', 
    marker='s', 
    s=40,  
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE2'
)

# Plot the CCE3 mooring point
ax.scatter(
    lon3, 
    lat3, 
    color= 'w',  
    edgecolor='black', 
    marker='o', 
    s=40,  
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE3'
)

# Plot depth contour lines
ct1 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=levels_ms, 
    colors='black', 
    linewidths=0.5, 
    linestyles='dashed'
)
ct2 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=[2000], 
    colors='black', 
    linewidths=1, 
    linestyles='solid'
)
ct3 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=levels_is, 
    colors='black', 
    linewidths=0.5, 
    linestyles='dashed'
)
ct4 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=[200], 
    colors='black', 
    linewidths=1, 
    linestyles='solid'
)

# Plot Line 80 CalCOFI Stations
ax.plot(
    calCOFI_lon % 360, 
    calCOFI_lat,
    color='k',
    linestyle=(0, (5, 3)),  
    linewidth=1.5,
    transform=ccrs.PlateCarree(),
)

# Set grid ticks 
set_grid_ticks(
    ax,
    xticks=xticks,
    yticks=yticks,
    xlabels=False,
    ylabels=True,
    grid=True,
    fontsize=fontsize_g,
    color='k',
    lw=1,
    ls='--',
    alpha=0.1
)

# Create colormap
cax = plt.axes([0.8075, 0.535, 0.02, 0.3])
cbar = set_cbar(
    ct,
    cax,
    fig,
    orientation="vertical",
    extend="both",
    label=r'Spectral Slope $\alpha$',
    fontsize=fontsize_g,
    ticks=ticks_slope, 
    invert = False
)

# Set legend
ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 0.905),
    fontsize=10,
    framealpha=0.9,
    edgecolor='black'
)

# --- Subplot 2 --- # 
ax = axes[1]

# Plot coastlines and land 
set_coastlines(
    ax, 
    projection, 
    resolution, 
    lon_min=lon_min, 
    lon_max=lon_max, 
    lat_min=lat_min, 
    lat_max=lat_max
) 

# Plot FVE
ct = ax.contourf(
    lon, 
    lat, 
    fve[:,:,0], 
    levels=levels_fve,
    transform=ccrs.PlateCarree(),
    cmap=cmap_fve, 
    extend='both'
)

# Plot the CCE1 mooring point
ax.scatter(
    lon1, 
    lat1, 
    color='w',
    edgecolor='black', 
    marker='^', 
    s=40, 
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE1'
)

# Plot the CCE2 mooring point
ax.scatter(
    lon2, 
    lat2, 
    color='w',  
    edgecolor='black', 
    marker='s', 
    s=40,  
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE2'
)

# Plot the CCE3 mooring point
ax.scatter(
    lon3, 
    lat3, 
    color= 'w',  
    edgecolor='black', 
    marker='o', 
    s=40,  
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE3'
)

# Plot depth contour lines
ct1 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=levels_ms, 
    colors='black', 
    linewidths=0.5, 
    linestyles='dashed'
)
ct2 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=[2000], 
    colors='black', 
    linewidths=1, 
    linestyles='solid'
)
ct3 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=levels_is, 
    colors='black', 
    linewidths=0.5, 
    linestyles='dashed'
)
ct4 = ax.contour(
    lon_b, 
    lat_b, 
    -1*(bathy),
    levels=[200], 
    colors='black', 
    linewidths=1, 
    linestyles='solid'
)

# Plot Line 80 CalCOFI Stations
ax.plot(
    calCOFI_lon % 360, 
    calCOFI_lat,
    color='k',
    linestyle=(0, (5, 3)),  
    linewidth=1.5,
    transform=ccrs.PlateCarree(),
)

# Set grid ticks 
set_grid_ticks(
    ax,
    xticks=xticks,
    yticks=yticks,
    xlabels=True,
    ylabels=True,
    grid=True,
    fontsize=fontsize_g,
    color='k',
    lw=1,
    ls='--',
    alpha=0.1
)

# Show x-axis ticks on both the top and bottom axes,
# but display tick labels only on the bottom axis
ax.xaxis.set_ticks_position('both')

ax.tick_params(
    axis='x',
    which='major',
    top=True,
    bottom=True,
    labeltop=False,
    labelbottom=True,
    direction='out',
    length=5,
)

# Create colormap
cax = plt.axes([0.8075, 0.14, 0.02, 0.3])
cbar = set_cbar(
    ct,
    cax,
    fig,
    orientation="vertical",
    extend="both",
    label=r'FVE ($\%$)',
    fontsize=fontsize_g,
    ticks=ticks_fve, 
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

# Label each subplot
pos = [0.96, 0.94]
add_corner_label(axes[0], pos, 'A', fontsize = 16)
add_corner_label(axes[1], pos, 'B', fontsize = 16)

# Adjust spacing 
plt.subplots_adjust(hspace=0.075, wspace=0.05)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig09.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)




