# =============================================================================
# Figure 05
# =============================================================================
#
# Caption:
#   Decorrelation time scale along CalCOFI line 80. Gray shading is the ocean bottom.
#   Decorrelation scales less than or equal to one standard error are considered
#   not statistically significant and are indicated with a hatched overlay. Solid
#   black curve is the seasonally averaged mixed-layer depth, and light black shading
#   represents its standard deviation.
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
from plotting import set_cbar

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
# - ns : Noise to signal ratio. 
#
# ------------#

# Set processing parameters
option_data        = 'density'    
option_interannual = 'linear' 
option_detrend_seg = False

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
# Load MITgcm decorrelation scales, bathymetry, CCE, and mixed-layer depth data
# -----------------------------------------------------------------------------

# --- Decorrelation Time Scales --- # 

# Set path to processed regional MITgcm data
PATH_processed = PATH_data / "mitgcm" / "transect" / "processed"

# Obtain filename path
filename_mitgcm = PATH_processed / f"mitgcm_decor_scale_{option_data}_hrly_trans_{option_interannual}_{seg_proc}.nc"

# Generate the nc data structure
nc = Dataset(filename_mitgcm, 'r')

# Extract data variables
dist    = nc.variables['dist'][:]
depth   = nc.variables['depth'][:]
lon     = nc.variables['LON'][:]
Lt      = nc.variables['decor_scale'][:]
Lt_stdm = nc.variables['decor_scale_stdm'][:]
Lt_std  = nc.variables['decor_scale_std'][:]

# --- Mixed Layer Depth --- # 

# Obtain filename path
filename_mld = PATH_processed / f"mitgcm_proc_density_hrly_trans.nc"

# Generate the nc data structure
nc = Dataset(filename_mld, 'r')

# Extract data variables
mld  = nc.variables['MLD'][:]

# --- Bathymetry --- # 

# Obtain filename path
filename_bathy = PATH_data / "mitgcm" / "transect" / "DEPTH_CCS_trans.nc"

# Generate the nc data structure
nc_bathy = Dataset(filename_bathy, 'r')

# Extract data variables
water_depth = nc_bathy.variables['water_depth'][:]

# --- CCE Mooring Locations --- # 
lat1, lat2, lat3  = 33.457, 34.3075, 34.44825228022894           
lon1, lon2, lon3  = -122.52233, -120.8042, -120.53825701527784 

# -----------------------------------------------------------------------------
# Compute the time mean and standard deviation mixed layer depth 
# -----------------------------------------------------------------------------
mld_mean = np.ma.mean(mld,axis=1)
mld_std = np.ma.var(mld,axis=1,ddof=1)

# -----------------------------------------------------------------------------
# Compute the relative uncertainty of the decorrelation scale
# -----------------------------------------------------------------------------

# Compute spatial mean
Lt_reg_mean = np.ma.mean(Lt)

# Compute the relative uncertainty (with respect to the regional mean)
Lt_rel_unc = Lt_stdm / (Lt - Lt_reg_mean)

# Mask not statistically significant grid points
Lt_mask = np.ma.getmask(np.ma.masked_greater_equal(Lt_rel_unc, ns))

# Get land mask from Lt
land_mask = np.ma.getmaskarray(Lt)

# Combine statistical significance and land masks
Lt_mask = Lt_mask & ~land_mask

# Create a mask array where non-significant ocean points = 1, others = NaN
data_mask = np.where(Lt_mask, 1, np.nan)

# -----------------------------------------------------------------------------
# Plot the CalCOFI line 80.0 transect decorrelation time scales  
# -----------------------------------------------------------------------------

# Set plotting parameters
level = np.arange(5,40+1,1)
cmap = cmo.amp

# Create figure
fig, ax = plt.subplots(figsize=(10,5))

# Plot decorrelation time scale
cf = ax.contourf(dist,abs(depth),Lt.T, levels=level, cmap=cmap, extend='both')

# Overlay a contourf with hatching for the non-significant regions
ax.contourf(
    dist,
    abs(depth),
    data_mask.T,
    levels=[0.5, 1.5],      
    hatches=['///'],       
    colors='none',          
    linewidths=0,
    zorder=10,              
)

# Plot the ocean bottom depth 
ax.fill_between(dist, abs(water_depth), abs(depth[-1]), color='0.4') 

# Set axis attributes
ax.set_xlabel('Distance from shore (km)')
ax.set_ylabel('Depth (m)')
ax.set_ylim(0,200)
ax.set_xticks(np.arange(0,250+25,25))
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(linestyle='--',alpha=0.3,color='grey')

# Set colorbar
cax = fig.add_axes([0.96, 0.16, 0.025, 0.73])
cbar = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
cbar.set_label('Decorrelation Scale (days)')
cbar.set_ticks(np.arange(5,40+5,5))

# --- Create top axis for longitude --- #
ax_top = ax.twiny()

# Make sure limits match
ax_top.set_xlim(ax.get_xlim())

# Choose where you want longitude ticks (same positions as distance ticks)
dist_ticks = ax.get_xticks()

# Interpolate longitude at those distance values
lon_180 = ((lon + 180) % 360) - 180
lon_ticks = np.interp(dist_ticks, dist, lon_180)

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
dist_sorted = dist[sort_idx]

# Interpolate longtiude onto distance coordinates 
dist1 = np.interp(lon1, lon_sorted, dist_sorted)
dist2 = np.interp(lon2, lon_sorted, dist_sorted)
dist3 = np.interp(lon3, lon_sorted, dist_sorted)

# Add CCE1, CCE2, and CCE3 locations 
# add_freq_marker(ax_top, dist1, '', y_marker=1.02, y_text=1.035,fontsize=14,markerfacecolor='tab:green',markeredgecolor='tab:green')
# add_freq_marker(ax_top, dist2, '', y_marker=1.02, y_text=1.035,fontsize=14,markerfacecolor='tab:red',markeredgecolor='tab:red')
# add_freq_marker(ax_top, dist3, '', y_marker=1.02, y_text=1.035,fontsize=14,markerfacecolor='tab:blue',markeredgecolor='tab:blue')

# Plot vertical lines at CCE1, CCE2, and CCE3 locations
ax.axvline(dist1, color='tab:green', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist2, color='tab:red', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(dist3, color='tab:blue', linestyle='--', lw=1.5, alpha=0.7)

# Save figure in high resolution 
fig.savefig(
    PATH_figs / "fig05.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)



