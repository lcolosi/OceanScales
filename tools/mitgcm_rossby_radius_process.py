# =============================================================================
# Processing MITgcm data for the Rossby Radius Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the 
#   Rossby Radius Analysis. These include: 
# 
#       (1) The Baroclinic and Barotropic Rossby Deformation Radii 
#       (2) The Root-Mean-Square Depth-average Velocity  
#       (3) The Advection Time Scale 
#       (4) The Froude Number 
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-01
# =============================================================================

# Import python libraries 
import os
import sys
from pathlib import Path
import xarray as xr
import numpy as np
from netCDF4 import Dataset
import gsw

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox 
from ocean_analysis import compute_rossby_modes

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - option_depth_avg: Specifies the depth threshold at which the depth-average
#                     velocity is computed to.  
# - nmode : Specifies the number of vertical modes to compute in the Rossby
#           deformation radius calculation. For example, nmode = 4 computes the
#           first four vertical modes: 
#                    mode 0 = barotropic
#                    mode 1 = first baroclinic
#                    mode 2 = second baroclinic
#                    mode 3 = third baroclinic
# - nz_mode : Specifies the number of vertical points used in finite-difference
#             eigenproblem.   
#
# ------------# 

# Set processing parameters
option_depth_avg = 200   
nmode            = 4
nz_mode          = 256 

# Set path to regional pre-processed data directory
PATH_preproc = PATH_data / "mitgcm" / "regional"

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------
print("Loading time-mean and seasonal-mean fields...")

# Obtain filename paths
filename = PATH_preproc / f"MITgcm_CCS_rossby_radius_background_upper_{option_depth_avg}m.nc"

# Generate the nc data structure
nc = Dataset(filename, 'r')

# Extract coordinate variables
depth  = nc.variables['Z'][:]
lon    = nc.variables['XC'][:]
lat    = nc.variables['YC'][:]
season = nc.variables['season'][:]

# Extract time-mean hydrographic variables
SA_mean      = nc.variables['SA_mean'][:]
CT_mean      = nc.variables['CT_mean'][:]
sigma0_mean  = nc.variables['sigma0_mean'][:]

# Extract seasonal hydrographic variables
SA_season      = nc.variables['SA_season'][:]
CT_season      = nc.variables['CT_season'][:]
sigma0_season  = nc.variables['sigma0_season'][:]

# Extract time-mean velocity variables
uvel_full_mean  = nc.variables['uvel_full_mean'][:]
vvel_full_mean  = nc.variables['vvel_full_mean'][:]
uvel_upper_mean = nc.variables['uvel_upper_mean'][:]
vvel_upper_mean = nc.variables['vvel_upper_mean'][:]

# Extract seasonal velocity variables
uvel_full_season  = nc.variables['uvel_full_season'][:]
vvel_full_season  = nc.variables['vvel_full_season'][:]
uvel_upper_season = nc.variables['uvel_upper_season'][:]
vvel_upper_season = nc.variables['vvel_upper_season'][:]

# Extract local bathymetric depth
water_depth = nc.variables["water_depth"][:]

# Close input file
nc.close()

# Mask dry cells previously set to NaN during preprocessing
SA_mean_m     = np.ma.masked_invalid(SA_mean)
CT_mean_m     = np.ma.masked_invalid(CT_mean)
sigma0_mean_m = np.ma.masked_invalid(sigma0_mean)

SA_season_m     = np.ma.masked_invalid(SA_season)
CT_season_m     = np.ma.masked_invalid(CT_season)
sigma0_season_m = np.ma.masked_invalid(sigma0_season)

uvel_full_mean_m  = np.ma.masked_invalid(uvel_full_mean)
vvel_full_mean_m  = np.ma.masked_invalid(vvel_full_mean)
uvel_upper_mean_m = np.ma.masked_invalid(uvel_upper_mean)
vvel_upper_mean_m = np.ma.masked_invalid(vvel_upper_mean)

uvel_full_season_m  = np.ma.masked_invalid(uvel_full_season)
vvel_full_season_m  = np.ma.masked_invalid(vvel_full_season)
uvel_upper_season_m = np.ma.masked_invalid(uvel_upper_season)
vvel_upper_season_m = np.ma.masked_invalid(vvel_upper_season)

water_depth_m = np.ma.masked_invalid(water_depth)

# -----------------------------------------------------------------------------
# Compute the Buoyancy Frequency 
# -----------------------------------------------------------------------------
print("Computing time-mean and seasonal-mean buoyancy frequency...")

# Number of vertical levels
ndepth = len(depth)

#------------------------------------------#
# Background Buoyancy Frequency
#------------------------------------------#

# Compute the pressure field 
pressure = gsw.p_from_z(
    depth[:, None, None],
    lat[None, :, None]
)

# Set latitude array with dimensions compatible with the time-mean fields
lat_mean = lat[None, :, None]

# Compute the background buoyancy frequency from time-average
Nsquare_mean, pressure_mid_mean = gsw.Nsquared(
    SA_mean_m,
    CT_mean_m,
    pressure,
    lat=lat_mean,
    axis=0,
)

# Mask invalid values
Nsquare_mean = np.ma.masked_invalid(Nsquare_mean)

# Convert midpoint pressure back to vertical position
depth_mid_mean = gsw.z_from_p(
    pressure_mid_mean,
    lat_mean
)

depth_mid_mean = np.ma.masked_invalid(depth_mid_mean)

#------------------------------------------#
# Seasonal Background Buoyancy Frequency
#------------------------------------------#

# Add a dimension for seasons
pressure_season = pressure[None, :, :, :]
lat_season      = lat[None, None, :, None]

# Compute the seasonal background buoyancy frequency from the seasonal average
Nsquare_season, pressure_mid_season = gsw.Nsquared(
    SA_season_m,
    CT_season_m,
    pressure_season,
    lat=lat_season,
    axis=1,
)

# Mask invalid values
Nsquare_season = np.ma.masked_invalid(Nsquare_season)

# Convert midpoint pressure back to vertical position
depth_mid_season = gsw.z_from_p(
    pressure_mid_season,
    lat_season
)

depth_mid_season = np.ma.masked_invalid(depth_mid_season)

#------------------------------------------#
# Compute Background N(z) 
#------------------------------------------#

# Mask statically unstable values before taking the square root
N_mean = np.ma.sqrt(
    np.ma.masked_less(Nsquare_mean, 0.0)
)

N_season = np.ma.sqrt(
    np.ma.masked_less(Nsquare_season, 0.0)
)

# -----------------------------------------------------------------------------
# Compute the Rossby Deformation Radius 
# -----------------------------------------------------------------------------
print("Computing time-mean and seasonal-mean Rossby deformation radii...")

# Set the mode number vector 
mode = np.arange(nmode)

# Set the number of horizontal grid points and seasons
nlat = len(lat)
nlon = len(lon)
nseason = len(season)

# Initialize arrays 
phase_speed_mean     = np.full((nmode, nlat, nlon), np.nan, dtype=float)
rossby_radius_mean   = np.full((nmode, nlat, nlon), np.nan, dtype=float)
phase_speed_season   = np.full((nseason, nmode, nlat, nlon), np.nan, dtype=float)
rossby_radius_season = np.full((nseason, nmode, nlat, nlon), np.nan, dtype=float)

# ------------------------------------------#
# Time-mean Rossby Deformation Radius
# ------------------------------------------#

# Loop through latitude 
for ilat in range(nlat): 

    # Print progress statement
    print(
        f"  Time mean: latitude {ilat + 1}/{nlat}",
        end="\r",
        flush=True,
    )

    # Loop through longitude 
    for ilon in range(nlon): 

        # Skip dry cells 
        if np.ma.ismasked(water_depth[ilat,ilon]): 
            continue

        # Obtain water depth
        H = float(water_depth(ilat,ilon))

        # Compute the 

# -----------------------------------------------------------------------------
# Compute the RMS velocity 
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Compute the Advection Time Scale 
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Compute the Froude Number
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------



