# =============================================================================
# Figure 05
# =============================================================================
#
# Caption:
#   Fraction of variance explained by unweighted annual and semi-annual
#   least-squares fit for each analysis: CCE1 (a), CCE2 (b), and CCE3 (c)
#   mooring locations (solid and dashed lines denote model and observed data,
#   respectively); CalCOFI line 80.0 cross-shelf transect (d); Point Conception
#   regional domain (e). Gray shading in (d-e) is the land. Black contour lines
#   are ocean bathymetry with 200 and 2000 meters isobaths highlighted as solid
#   black lines.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-21
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
import matplotlib.gridspec as gridspec

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_figs = ROOT / "figs"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox 
from plotting import add_corner_label, add_x_axis_marker, set_coastlines, set_grid_ticks, set_cbar, add_scalebar

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
#                        Options: True or False
# - segment_months : Specifies the window duration. 
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
segment_months     = 6

# Label segment processing 
seg_proc = "detrend" if option_detrend_seg else "demean"

# Set uncertainty estimate parameters
sn_threshold = 1

# Set font and fontsize using LaTeX 
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,         
    "text.usetex": True,           
    "font.family": "serif",       
    "text.latex.preamble": r"\usepackage{amsmath}" 
})

# -----------------------------------------------------------------------------
# Load MITgcm and CCE fraction of variance explained, bathymetry, CCE, and CalCOFI data
# -----------------------------------------------------------------------------

# --- Regional --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "regional" / "processed"

# Obtain filename paths
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_reg_depth_{option_depth}m_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
lon_r = nc.variables['lon'][:]
lat_r = nc.variables['lat'][:]
fve_r = nc.variables['FVE'][:]

# --- Transect --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "transect" / "processed"

# Obtain filename path
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_trans_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
dist_t  = nc.variables['dist'][:]
depth_t = nc.variables['depth'][:]
lon_t   = nc.variables['LON'][:]
lat_t   = nc.variables['LAT'][:]
fve_t   = nc.variables['FVE'][:]

# --- Mooring --- # 

# Set path to processed regional MITgcm data
PATH_processed      = PATH_data / "mitgcm" / "mooring" / "processed"
PATH_processed_cce1 = PATH_data / "cce" / "cce1" / "processed"
PATH_processed_cce2 = PATH_data / "cce" / "cce2" / "processed"

# Obtain filename paths
filename_m    = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce1 = PATH_processed_cce1 / f"cce1_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"
filename_cce2 = PATH_processed_cce2 / f"cce2_decor_scale_{option_data}_hrly_mooring_{option_interannual}_{seg_proc}_seg_duration_{segment_months}mo.nc"

# Generate the nc data structure
nc_m    = Dataset(filename_m, 'r')
nc_cce1 = Dataset(filename_cce1, 'r')
nc_cce2 = Dataset(filename_cce2, 'r')

# Extract data variables
site_m  = nc_m.variables['site'][:]
depth_m = nc_m.variables['depth'][:]
fve_m   = nc_m.variables['FVE'][:]

fve_cce1   = nc_cce1['FVE'][:]
depth_cce1 = nc_cce1['depth'][:]
fve_cce2   = nc_cce2['FVE'][:]
depth_cce2 = nc_cce2['depth'][:]

# --- Mixed Layer Depth --- # 

# Obtain filename path
filename_mld_t = PATH_data / "mitgcm" / "transect" / "processed" / "mitgcm_proc_density_hrly_trans.nc"
filename_mld_m = PATH_data / "mitgcm" / "mooring" / "processed" / "mitgcm_proc_density_hrly_mooring.nc"

# Generate the nc data structure
nc_t = Dataset(filename_mld_t, 'r')
nc_m = Dataset(filename_mld_m, 'r')

# Extract data variables
mld_t = nc_t.variables['MLD'][:]
mld_m = nc_m.variables['MLD'][:]

# --- Bathymetry --- # 

# Obtain filename path
filename_bathy = PATH_data / "bathymetry" / "etopo1_point_conception.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
lon_b  = nc_bathy.variables['lon'][:]
lat_b  = nc_bathy.variables['lat'][:]
bathy  = nc_bathy.variables['BATHY'][:]

# Obtain filename path
filename_bathy = PATH_data / "mitgcm" / "transect" / "DEPTH_CCS_trans.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
dist_wd     = nc_t.variables['dist'][:]
water_depth = nc_bathy.variables['water_depth'][:]

# Set the depth at the coast to zero 
water_depth[0] = 0

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
# Compute the time mean and standard deviation mixed layer depth 
# -----------------------------------------------------------------------------

mld_t_mean = np.ma.mean(mld_t,axis=1)
mld_t_std = np.ma.std(mld_t,axis=1,ddof=1)

mld_m_mean = np.ma.mean(mld_m,axis=1)
mld_m_std = np.ma.std(mld_m,axis=1,ddof=1)

# -----------------------------------------------------------------------------
# Compute precentage of variance explained
# -----------------------------------------------------------------------------

# Convert FVE to precent
fve_r    = fve_r * 100
fve_t    = fve_t * 100
fve_m    = fve_m * 100
fve_cce1 = fve_cce1 * 100
fve_cce2 = fve_cce2 * 100

# Compute the positive downward 
depth_t_pos    = np.abs(depth_t)
depth_m_pos    = np.abs(depth_m) 
depth_cce1_pos = np.abs(depth_cce1) 
depth_cce2_pos = np.abs(depth_cce2) 
water_depth    = np.abs(water_depth)

# -----------------------------------------------------------------------------
# Plot fraction of variance explained
# -----------------------------------------------------------------------------

# Set plotting parameters
projection = ccrs.PlateCarree(central_longitude=0.0)
resolution = "10m"
xticks = [-123, -122.5, -122, -121.5, -121, -120.5, -120]
yticks = [33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75, 35.00]
lon_min, lon_max = -123, -120
lat_min, lat_max = 33, 35
levels_t = np.arange(0,100+2.5,2.5)
levels_r = np.arange(35,90+1,1)
levels_is = np.arange(100,300+100,100)
levels_ms = np.arange(1000,3000+500,500)
fontsize_legend = 14
cmap = cmo.haline
axes = []

# Create figure
fig = plt.figure(figsize=(12, 14))

# Set axes for subplots
gs = gridspec.GridSpec(
    nrows=3,
    ncols=3,
    height_ratios=[1, 1, 1.5],
    hspace=0.4,
    wspace=0.2
)

# --- Subplot 1 --- # 
ax = fig.add_subplot(gs[0, 0])

# Append axis to axes array
axes.append(ax)

# Plot the mean mixed layer depth 
ax.axhline(mld_m_mean[0], ls='--', lw=1.5, color='k', alpha=1, label=r"$\overline{z}_{mld}$")

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 100], mld_m_mean[0] - mld_m_std[0], mld_m_mean[0] + mld_m_std[0], color='k', alpha=0.15, label=r"$\sigma_{\overline{z}_{mld}}$")

# Plot the FVE for model and observations at CCE1 
ax.plot(fve_m[0,:], depth_m_pos, '.-', color='tab:green', linewidth=1,label='Model')
ax.plot(fve_cce1, depth_cce1_pos, '.--', color='tab:green', linewidth=1, label = 'Obs')

# Set figure attributes
ax.set_xlabel(r'FVE ($\%$)')
ax.set_ylabel("Depth (m)")
ax.set_xlim(0,100)
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,100+20,20))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_yaxis()
ax.tick_params(top=False, bottom=True, left=True, right=True, direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)
ax.legend(loc='right', fontsize=fontsize_legend-3, framealpha=0.9, edgecolor='black')

# --- Subplot 2 --- # 
ax = fig.add_subplot(gs[0, 1])

# Append axis to axes array
axes.append(ax)

# Plot the mean mixed layer depth 
ax.axhline(mld_m_mean[1], ls='--', lw=1.5, color='k', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 100], mld_m_mean[1] - mld_m_std[1], mld_m_mean[1] + mld_m_std[1], color='k', alpha=0.15)

# Plot the FVE for CCE1 
ax.plot(fve_m[1,:], depth_m_pos, '.-', color='tab:red', linewidth=1)
ax.plot(fve_cce2, depth_cce2_pos, '.--', color='tab:red', linewidth=1)

# Set figure attributes
ax.set_xlabel(r'FVE ($\%$)')
ax.set_xlim(0,100)
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,100+20,20))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.tick_params(top=False, bottom=True, left=True, right=True, direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

# --- Subplot 3 --- # 
ax = fig.add_subplot(gs[0, 2])

# Append axis to axes array
axes.append(ax)

# Plot the mean mixed layer depth 
ax.axhline(mld_m_mean[2], ls='--', lw=1.5, color='k', alpha=1)

# Plot the range of mixed layer depths (1 standard deviation)
ax.fill_between([0, 100], mld_m_mean[2] - mld_m_std[2], mld_m_mean[2] + mld_m_std[2], color='k', alpha=0.15)

# Plot the FVE for CCE3
ax.plot(fve_m[2,:], depth_m_pos, '.-', color='tab:blue', linewidth=1)

# Set figure attributes
ax.set_xlabel(r'FVE ($\%$)')
ax.set_xlim(0,100)
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,100+20,20))
ax.set_yticks(np.arange(0,200+25,25))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.tick_params(top=False, bottom=True, left=True, right=True, direction='out', length=3.5)
ax.grid(True,linestyle='--',alpha=0.3)

#--- Subplot 4 ---# 
ax = fig.add_subplot(gs[1, :])

# Append axis to axes array
axes.append(ax)

# Plot FVE along the CalCOFI transect
cf = ax.contourf(dist_t,depth_t_pos,fve_t.T, levels=levels_t, cmap=cmap, extend='neither')

# Plot the ocean bottom depth 
ax.fill_between(dist_wd, water_depth, depth_t_pos[-1], color='0.4') 

# Plot the time-mean mixed layer depth 
ax.plot(dist_wd, mld_t_mean, '-', color='k', lw=2)

# Plot the standard deviation of the mixed layer depth 
ax.fill_between(dist_wd, mld_t_mean - mld_t_std, mld_t_mean + mld_t_std, color='k', alpha=0.15)

# Set axis attributes
ax.set_xlabel('Distance from shore (km)')
ax.set_ylabel('Depth (m)')
ax.set_xlim(0,dist_t[-1])
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,250+25,25))
ax.set_yticks(np.arange(0,200+25,25))
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(linestyle='--',alpha=0.3,color='grey')

# Set colorbar
cax = fig.add_axes([0.92, 0.45, 0.025, 0.175])                              
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='neither')
cbar.set_label(r'FVE ($\%$)')
cbar.set_ticks(np.arange(0,100 + 20, 20))

# --- Create top axis for longitude --- #
ax_top = ax.twiny()

# Make sure limits match
ax_top.set_xlim(ax.get_xlim())

# Choose where you want longitude ticks (same positions as distance ticks)
dist_ticks = ax.get_xticks()

# Interpolate longitude at those distance values
lon_180 = ((lon_t + 180) % 360) - 180
lon_ticks = np.interp(dist_ticks, dist_t, lon_180)

# Create labels but only keep every other one
labels = [
    f"{abs(x):.1f}°W" if i % 2 == 0 else ""
    for i, x in enumerate(lon_ticks)
]

# Set ticks and labels
ax_top.set_xticks(dist_ticks)
ax_top.set_xticklabels(labels) 

sort_idx = np.argsort(lon_180)
lon_sorted = lon_180[sort_idx]
dist_sorted = dist_t[sort_idx]

# Interpolate longtiude onto distance coordinates 
dist1 = np.interp(lon1, lon_sorted, dist_sorted)
dist2 = np.interp(lon2, lon_sorted, dist_sorted)
dist3 = np.interp(lon3, lon_sorted, dist_sorted)

# Add CCE1, CCE2, and CCE3 locations markers
add_x_axis_marker(ax_top, dist1, 'v', '', y_marker=1.02, y_text=1.035,fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
add_x_axis_marker(ax_top, dist2, 'v', '', y_marker=1.02, y_text=1.035,fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
add_x_axis_marker(ax_top, dist3, 'v', '', y_marker=1.02, y_text=1.035,fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

#--- Subplot 5 ---# 
ax = fig.add_subplot(gs[2, 0:3], projection = projection)

# Append axis to axes array
axes.append(ax)

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

# Plot FVE within the study region 
ct = ax.contourf(
    lon_r, 
    lat_r, 
    fve_r, 
    levels=levels_r,
    transform=ccrs.PlateCarree(),
    cmap=cmap, 
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
    fontsize=fontsize,
    color='k',
    lw=1,
    ls='--',
    alpha=0.1
)

# Create colorbar
cax = fig.add_axes([0.83, 0.105, 0.025, 0.25]) 
cbar = fig.colorbar(ct, cax=cax, orientation='vertical', extend='both') 
cbar.set_label(r'FVE ($\%$)')
cbar.set_ticks(np.arange(40, 90+10, 10))

# Set legend
ax.legend(
    loc='upper right',
    fontsize=fontsize_legend,
    framealpha=0.9,
    edgecolor='black'
)

# Expand width manually
ax.set_position([0.08, 0.07, 0.9, 0.32])

# Label each subplot
add_corner_label(axes[0], [0.9175, 0.095], 'A', fontsize=fontsize)
add_corner_label(axes[1], [0.9175, 0.095], 'B', fontsize=fontsize)
add_corner_label(axes[2], [0.9175, 0.095], 'C', fontsize=fontsize)
add_corner_label(axes[3], [0.025, 0.09], 'D', fontsize=fontsize)
add_corner_label(axes[4], [0.04, 0.06], 'E', fontsize=fontsize)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / 'figS01.png',
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)





