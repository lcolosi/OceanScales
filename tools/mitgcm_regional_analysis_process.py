# =============================================================================
# Processing MITgcm data for the Regional Analysis
# =============================================================================
#
# Description:
#   Slice the MITgcm data to extract the temperature, salinity, and 
#   velocity variables for the Regional Analysis. The script will read in the MITgcm 
#   output files, extract the necessary variables, and save them in netcdf format for 
#   further analysis.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-10
# =============================================================================

# Import python libraries 
import sys
import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - delta_t: Model time step in seconds (time increments of the diagnostics can differ).
# - depth: Depth level extracted for regional analysis(units: meters).
# - lat_bnds: Latitude bounds setting the region of interest.
# - lon_bnds: Longitude bounds setting the region of interest.
# - encoding: Start time of the model run.
# - PATH_GRID: Directory containing the model grid.
# - PATH_OUTPUT: Directory containing model diagnostics.
# - PATH_nc: Directory where netCDF files are saved.
# - file_dim: Diagnostic file dimension (3D for T, S, drhodr, and velocity; 2D for etan).

# Model parameters 
delta_t = 150  

# Set time and space parameters  
depth = 9                                                         
lat_bnds  = [33.0, 35.0]                                          
lon_bnds  = [237.0, 240.0]                                         
encoding  = {'time': {'units': 'seconds since 2015-12-01 2:00'}}  

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                    
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'     
PATH_nc     = '/data/SO3/lcolosi/mitgcm/SWOT_MARA_RUN4_LY/'  
file_dim    = '2D'                                         

# -----------------------------------------------------------------------------
# Load the grid and diagnostics data into a python structure
# -----------------------------------------------------------------------------

#------------#  
#--- Note ---#
#------------# 
#
# - PATH_OUTPUT: Directory containing model output (.data and .meta files).
# - PATH_GRID: Directory containing the model grid.
# - iters: Load all available model iterations.
# - delta_t: Model time step in seconds.
# - ignore_unknown_vars: Do not ignore unrecognized variables.
# - prefix: Load diagnostics corresponding to the specified file dimension.
# - ref_date: Start time of the simulation, including model spin-up.
# - geometry: Model grid uses spherical-polar coordinates.

# Create dataset 
ds = open_mdsdataset(
    PATH_OUTPUT,                    
    PATH_GRID,                      
    iters='all',                    
    delta_t=delta_t, 
    ignore_unknown_vars=False,     
    prefix=['diags_' + file_dim],   
    ref_date="2015-01-01 02:00:00", 
    geometry='sphericalpolar'       
)

# Convert all variables and coordinates in the dataset to little-endian

# --- Variables --- #
for var in ds.data_vars:
    if ds[var].dtype.byteorder == '>' or (ds[var].dtype.byteorder == '=' and sys.byteorder == "big"):  
        ds[var] = ds[var].astype(ds[var].dtype.newbyteorder('<'))

# --- Coordinates --- # 
for coord in ds.coords:
    if ds[coord].dtype.byteorder == '>'or (ds[coord].dtype.byteorder == '=' and sys.byteorder == "big"):  
        ds[coord] = ds[coord].astype(ds[coord].dtype.newbyteorder('<'))

# -----------------------------------------------------------------------------
# Slice array based on longitude and latitude bounds of the region
# -----------------------------------------------------------------------------

if file_dim == '3D':
    print(f"Extracting 3D fields...")

    # Obtain the depth coordinate 
    depth_levels = abs(ds['Z'].values)

    # Find the index of the depth level closest to depth
    depth_idx = np.argmin(np.abs(depth_levels - depth))

    # Check the actual depth value you're selecting
    actual_depth = depth_levels[depth_idx]
    print(f"Selected depth: {actual_depth} m at index {depth_idx}")

    # Extract scalar fields 
    theta = ds['THETA'].isel(Z=depth_idx).sel(YC=slice(*lat_bnds), 
                                              XC=slice(*lon_bnds))
    salt  = ds['SALT'].isel(Z=depth_idx).sel(YC=slice(*lat_bnds), 
                                             XC=slice(*lon_bnds))
    uvel  = ds['UVEL'].isel(Z=depth_idx).sel(YC=slice(*lat_bnds), 
                                             XG=slice(*lon_bnds))
    vvel  = ds['VVEL'].isel(Z=depth_idx).sel(YG=slice(*lat_bnds), 
                                             XC=slice(*lon_bnds))

elif file_dim == '2D':
    print(f"Extracting 2D fields...")

    # Extract scalar fields 
    etan = ds['ETAN'].sel(YC=slice(*lat_bnds), 
                          XC=slice(*lon_bnds))

# -----------------------------------------------------------------------------
# Save data in netcdf files
# -----------------------------------------------------------------------------

# Set the dictionary of variables to save
if file_dim == '3D':
    vars_to_save = {
        'THETA': theta,
        'SALT': salt,
        'UVEL': uvel,
        'VVEL': vvel
    }
elif file_dim == '2D':
    vars_to_save = {
        'ETAN': etan
    }

# Loop through each variable and save efficiently
for var_name, da in vars_to_save.items():

    # Print status
    print(f"Saving {var_name}...")
    
    # Chunk along time for faster write 
    if 'time' in da.dims:
        da = da.chunk({'time': 1000})
    
    # Load into memory before saving 
    da = da.load()

    # Save to NetCDF file
    if file_dim == '3D':
        da.to_netcdf(
            f"{PATH_nc}{var_name}_CCS_hrly_reg_depth_{abs(int(actual_depth))}m.nc",
            engine='netcdf4',                
            format='NETCDF4',         
            encoding=encoding            
        )
    elif file_dim == '2D':
        da.to_netcdf(
            f"{PATH_nc}{var_name}_CCS_hrly_reg.nc",
            engine='netcdf4',                
            format='NETCDF4',         
            encoding=encoding            
        )

