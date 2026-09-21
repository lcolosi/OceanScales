# =============================================================================
# Figure S03
# =============================================================================
#
# Caption:
#   (a) First baroclinic Rossby Radius of deformation and (b) Root-Mean-Square 
#   velocity with time-mean velocity vectors in the study domain. Black contour 
#   lines are the ocean topography with 200 and 2000 meter isobaths highlighted as
#   solid black lines. 
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-09
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

# Import plotting toolbox 
from plotting import set_coastlines, set_grid_ticks, set_cbar, add_scalebar, add_corner_label

# -----------------------------------------------------------------------------
# Set processing and plotting parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - option_depth_avg: Depth averaging option for the velocity fields. Options are:
#                     "full" for full-water-column depth average, or "upper" for 
#                     upper-ocean depth average to a specified depth.
# - option_rms: RMS velocity calculation option. Options are:
#               "with_mean" for RMS velocity including the mean flow, or
#               "without_mean" for RMS velocity excluding the mean flow.
# - depth_avg_threshold: Specifies the maximum depth at which the depth-average
#                        velocity is computed to.  
#
# ------------# 

# Set processing parameters
option_depth_avg    = 'upper'    
depth_avg_threshold = 200  
option_rms          = 'with_mean' 

# Set font and fontsize using LaTeX 
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm Rossby radius/RMS velocity, bathymetry, CCE, and CalCOFI data
# -----------------------------------------------------------------------------

# --- Rossby Radius and RMS Velocity --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_advection_time_scale_reg_{option_depth_avg}_{depth_avg_threshold}m_{option_rms}.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
lon     = nc.variables['lon'][:]
lat     = nc.variables['lat'][:]
Rd      = nc.variables['ROSSBY_RADIUS_full'][:]
U_rms   = nc.variables['U_RMS_full'][:]
u_mean  = nc.variables['UVEL_full'][:]
v_mean  = nc.variables['VVEL_full'][:]

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
# Plot regional Rossby Radius of Deformation and RMS velocity 
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
resolution = "10m"
xticks = [-123, -122.5, -122, -121.5, -121, -120.5, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
levels_Rd = np.arange(0,30+0.5,0.5) 
ticks_Rd  = np.arange(0,30+5,5) 
levels_U_rms = np.arange(0.10,0.22+0.0025,0.0025)
ticks_U_rms  = np.arange(0.10,0.22+0.02,0.02)
skip = 2
levels_is = np.arange(100,300+100,100)
levels_ms = np.arange(1000,3000+500,500)
fontsize_g = 18
fontsize_c = 10
cmap_Rd = cmo.tempo
cmap_U_rms = cmo.speed

# Create figure
fig, axes = plt.subplots(1,2,figsize=(18, 12), subplot_kw={"projection": projection})

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

# Plot First Baroclinic Rossby Radius of deformation 
ct = ax.contourf(
    lon, 
    lat, 
    Rd[1,:,:], 
    levels=levels_Rd,
    transform=ccrs.PlateCarree(),
    cmap=cmap_Rd, 
    extend='max'
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

# Create colormap
cax = plt.axes([0.18, 0.685, 0.25, 0.02])
cbar = set_cbar(
    ct,
    cax,
    fig,
    orientation="horizontal",
    extend="max",
    label='Rossby Deformation Radius (km)',
    fontsize=fontsize_g,
    ticks=ticks_Rd, 
    invert = False
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")

# Add a 20-km scale bar
add_scalebar(
    ax, 
    length_km=20, 
    location=(0.925, 0.78),
    linewidth=1, 
    text_kwargs=dict(fontsize=12, color='white', weight='bold')
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

# Plot Root-Mean-Square Velocity 
ct = ax.contourf(
    lon, 
    lat, 
    U_rms, 
    levels=levels_U_rms,
    transform=ccrs.PlateCarree(),
    cmap=cmap_U_rms, 
    extend='both'
)

# Plot the Root-Mean-Square velocity vectors 
q = ax.quiver(
    lon[::skip], 
    lat[::skip], 
    u_mean[::skip, ::skip], 
    v_mean[::skip, ::skip],
    transform=ccrs.PlateCarree(), 
    linewidth=0.5, 
    scale=5, 
    width=0.001, 
    color='k', 
    zorder=5,
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

# Create colormap
cax = plt.axes([0.6, 0.685, 0.25, 0.02])
cbar = set_cbar(
    ct,
    cax,
    fig,
    orientation="horizontal",
    extend="both",
    label='Root-Mean-Square Velocity (m/s)',
    fontsize=fontsize_g,
    ticks=ticks_U_rms, 
    invert = False
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")

# Plot a quiver key for the velocity vectors
ax.quiverkey(
    q,
    X=0.88,
    Y=0.8,
    U=0.1,
    label=r"$0.1\ \mathrm{m}/\mathrm{s}$",
    labelpos="E",
    coordinates="axes",
    fontproperties={"size": 12}
)

# Label each subplot
pos = [0.96, 0.94]
add_corner_label(axes[0], pos, 'A', fontsize = 16)
add_corner_label(axes[1], pos, 'B', fontsize = 16)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "figS03.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)

