# =============================================================================
# Figure 01 
# =============================================================================
#
# Caption:
#   Map of the Point Conception study region with ocean bathymetry. 200 and 2000 meter
#   isobaths are shown as solid black lines. Triangle, square, and circle icons mark
#   the positions of the CCE1, CCE2, and CCE3 moorings, respectively. The red-dashed
#   line denotes the CalCOFI line 80.0 cross-shelf transect.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-12
# =============================================================================

# Import python libraries
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
from plotting import set_coastlines, set_grid_ticks, set_cbar

# -----------------------------------------------------------------------------
# Set plotting parameters
# -----------------------------------------------------------------------------

# Set font and fontsize using LaTeX 
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load CCE, CalCOFI, and bathymetry data 
# -----------------------------------------------------------------------------

#------------------------------------------# 
# CCE Mooring Locations  
#------------------------------------------# 
lat1, lat2, lat3  = 33.457, 34.3075, 34.44825228022894           
lon1, lon2, lon3  = -122.52233, -120.8042, -120.53825701527784 

#------------------------------------------# 
# Bathymetry  
#------------------------------------------# 

# Obtain filename path
filename_bathy = PATH_data / "bathymetry" / "etopo1_point_conception.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
lon    = nc_bathy.variables['lon'][:]
lat    = nc_bathy.variables['lat'][:]
bathy  = nc_bathy.variables['BATHY'][:]

#------------------------------------------# 
# CalCOFI Line 80.0 Positions  
#------------------------------------------# 

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
# Plot bathymetry 
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
xticks = [-123, -122, -121, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
resolution = "10m"
bounds = np.arange(0,360+40,40)
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
levels = np.arange(0, 4500, 100) 
level_is = np.arange(100,300,100)
levels_ms = np.arange(1000,3000,500)
fontsize_g = 25

# Create figure
fig, ax = plt.subplots(figsize=(18, 20), subplot_kw={"projection": projection})

# Plot coastlines and land 
set_coastlines(ax, projection, resolution, lon_min=lon_min, lon_max=lon_max, 
               lat_min=lat_min, lat_max=lat_max) 

# Plot bathymetry
ct = ax.contourf(
    lon, lat, abs(bathy), levels=levels,
    transform=ccrs.PlateCarree(),
    cmap=cmo.deep, 
    extend = 'max'
)

# Plot the CCE1 mooring point
ax.scatter(
    lon1, lat1, 
    color='w',
    edgecolor='black', marker='^', s=180, 
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE1'
)

# Plot the CCE2 mooring point
ax.scatter(
    lon2, lat2, 
    color='w',  
    edgecolor='black', marker='s', s=180,  
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE2'
)

# Plot the CCE3 mooring point
ax.scatter(
    lon3, lat3, 
    color= 'w',  
    edgecolor='black', marker='o', s=180,  
    transform=ccrs.PlateCarree(),
    zorder=10, 
    label='CCE3'
)

# Plot depth contour lines
ct1 = ax.contour(lon, lat, -1*(bathy),levels=levels_ms, colors='black', linewidths=1, linestyles='dashed')
ct2 = ax.contour(lon, lat, -1*(bathy),levels=[2000], colors='black', linewidths=2, linestyles='solid')
ct3 = ax.contour(lon, lat, -1*(bathy),levels=level_is, colors='black', linewidths=1, linestyles='dashed')
ct4 = ax.contour(lon, lat, -1*(bathy),levels=[200], colors='black', linewidths=2, linestyles='solid')
plt.clabel(ct1, fontsize=fontsize)
plt.clabel(ct2, fontsize=fontsize)
plt.clabel(ct3, fontsize=fontsize)
plt.clabel(ct4, fontsize=fontsize)

# Plot Line 80 CalCOFI Stations
ax.plot(
    calCOFI_lon % 360, calCOFI_lat,
    color='tab:red',
    linestyle=(0, (5, 3)),  
    linewidth=3,
    transform=ccrs.PlateCarree(),
    label='CalCOFI \n Line 80.0'
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
    alpha=0.3
)

# Create colormap
cax = plt.axes([0.91, 0.315, 0.02, 0.35])
set_cbar(
    ct,
    cax,
    fig,
    orientation="vertical",
    extend="both",
    label='Depth (m)',
    fontsize=fontsize_g,
    ticks=np.arange(0,4500+500,500), 
    invert = True
)

# Set legend
ax.legend(
    loc='upper right',
    fontsize=fontsize,
    framealpha=0.9,
    edgecolor='black'
)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig01.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)

