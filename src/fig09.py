# =============================================================================
# Figure 09
# =============================================================================
#
# Caption:
#   Decorrelation time scale in the study domain at 9 meter water depth. Black
#   contour lines are the ocean topography with 200 and 2000 meter isobaths 
#   highlighted as solid black lines. Decorrelation scales less than or equal to 
#   one standard error are considered not statistically significant and are indicated
#   with a hatched overlay.
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

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_figs = ROOT / "figs"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox for cartopy figures
from plotting import set_coastlines, set_grid_ticks, set_cbar, add_scalebar

# -----------------------------------------------------------------------------
# Set processing and plotting parameters
# -----------------------------------------------------------------------------

# ------------#
# --- Note ---#
# ------------#
#
# - option_data: Data variable to analyze.
#                Options: "temp", "sal", "density", "uvel", "vvel", or "ssh".
#
# - option_depth: Depth at which the decorrelation time scale is computed
#                 (units: meters). Not used for SSH.
# - option_interannual: Specifies the model of the interannual variability. 
#                       Options include: 'linear' or 'gaussian'
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - ns : Noise to signal ratio. 
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_depth       = 9   
option_interannual = 'linear' 
option_detrend_seg = True

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set uncertainty estimate parameters
ns = 1

# Set font and fontsize using LaTeX 
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm decorrelation scales, bathymetry, CCE, and CalCOFI data
# -----------------------------------------------------------------------------

# --- Decorrelation Time Scales --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_reg_depth_{option_depth}m_{option_interannual}_{seg_proc}.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
lon     = nc.variables['lon'][:]
lat     = nc.variables['lat'][:]
Lt      = nc.variables['decor_scale'][:]
Lt_stdm = nc.variables['decor_scale_stdm'][:]
Lt_std  = nc.variables['decor_scale_std'][:]

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
# Compute the relative uncertainty of the decorrelation scale
# -----------------------------------------------------------------------------

# Compute spatial mean
Lt_reg_mean = np.ma.mean(Lt)

# Compute the relative uncertainty (with respect to the regional mean)
Lt_rel_unc = Lt_stdm / np.abs(Lt - Lt_reg_mean)

# Mask not statistically significant grid points
Lt_mask = np.ma.getmask(np.ma.masked_greater_equal(Lt_rel_unc, ns))

# Get land mask from Lt
land_mask = np.ma.getmaskarray(Lt)

# Combine statistical significance and land masks
Lt_mask = Lt_mask & ~land_mask

# Create a mask array where non-significant ocean points = 1, others = NaN
data_mask = np.where(Lt_mask, 1, np.nan)

# -----------------------------------------------------------------------------
# Plot regional decorrelation time scales  
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
xticks = [-123, -122, -121, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
resolution = "10m"
bounds = np.arange(0,360+40,40)
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
levels = np.arange(5,20+0.5,0.5) 
level_is = np.arange(100,300,100)
levels_ms = np.arange(1000,3000,500)
fontsize_g = 18
fontsize_c = 10
cmap = cmo.amp

# Create figure
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": projection})

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

# Plot decorrelation time scales
ct = ax.contourf(
    lon, 
    lat, 
    Lt, 
    levels=levels,
    transform=ccrs.PlateCarree(),
    cmap=cmap, 
    extend='both'
)

# Overlay a contourf with hatching for the non-significant regions
ax.contourf(
    lon,
    lat,
    data_mask,
    levels=[0.5, 1.5],      
    hatches=['..'],        
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
    levels=level_is, 
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
plt.clabel(ct1, fontsize=fontsize_c)
plt.clabel(ct2, fontsize=fontsize_c)
plt.clabel(ct3, fontsize=fontsize_c)
plt.clabel(ct4, fontsize=fontsize_c)

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

# Create colormap
cax = plt.axes([0.91, 0.3, 0.02, 0.4])
set_cbar(
    ct,
    cax,
    fig,
    orientation="vertical",
    extend="both",
    label='Decorrelation Scale (days)',
    fontsize=fontsize_g,
    ticks=np.arange(5,20+5,5), 
    invert = False
)

# Set legend
ax.legend(
    loc='upper right',
    fontsize=14,
    framealpha=0.9,
    edgecolor='black'
)

# Add a 20-km scale bar
add_scalebar(
    ax, 
    length_km=20, 
    location=(0.925, 0.78),
    linewidth=1, 
    text_kwargs=dict(fontsize=14, color='white', weight='bold')
)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig09.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)

