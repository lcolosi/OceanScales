# =============================================================================
# Processing MITgcm data for the Mooring Analysis
# =============================================================================
#
# Description:
#   Slice the MITgcm data to extract the temperature, salinity, and 
#   velocity variables for the mooring analysis. The script will read in 
#   the MITgcm output files, extract the necessary variables at CCE mooring sites, 
#   and save them in netcdf format for further 
#   analysis.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-11
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
# - lat_cce: Latitude locations of the CCE mooring sites.
# - lon_cce: Longitude locations of the CCE mooring sites.
# - encoding: Start time of the model run.
# - PATH_GRID: Directory containing the model grid.
# - PATH_OUTPUT: Directory containing model diagnostics.
# - PATH_nc: Directory where netCDF files are saved.
# - file_dim: Diagnostic file dimension (3D for T, S, drhodr, and velocity; 2D for etan).


# Model parameters 
delta_t = 150  

# Set time and space parameters  
lat_cce  = [33.457, 34.3075, 34.44825228022894]                  
lon_cce  = [-122.52233, -120.8042, -120.53825701527784]          
encoding  = {'time': {'units': 'seconds since 2015-12-01 2:00'}} 

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                   
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'     
PATH_nc     = '/data/SO3/lcolosi/mitgcm/SWOT_MARA_RUN4_LY/' 
file_dim    = '3D'     

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

#--- Variables ---#
for var in ds.data_vars:
    if ds[var].dtype.byteorder == '>' or (ds[var].dtype.byteorder == '=' and sys.byteorder == "big"):  
        ds[var] = ds[var].astype(ds[var].dtype.newbyteorder('<'))

#--- Coordinates ---# 
for coord in ds.coords:
    if ds[coord].dtype.byteorder == '>'or (ds[coord].dtype.byteorder == '=' and sys.byteorder == "big"):  
        ds[coord] = ds[coord].astype(ds[coord].dtype.newbyteorder('<'))

# -----------------------------------------------------------------------------
# Slice array based on longitude and latitude of CCE moorings
# -----------------------------------------------------------------------------

# Get 2D coordinate fields
lat_YC = ds['YC']
lon_XC = ds['XC']
lat_YG = ds['YG']
lon_XG = ds['XG']

# Set dictionary to hold extracted profiles per variable
all_profiles = {var: [] for var in ds.data_vars}

# Loop through CCE sites
for i, (lat_target, lon_target) in enumerate(zip(lat_cce, lon_cce)):

    # Obtain indicies of the closest grid point to the target latitude and longitude 
    
    # --- Center-point variables (XC/YC) --- #
    dist_sq_center = (lat_YC - lat_target)**2 + (lon_XC - lon_target)**2
    j_YC, i_XC = np.unravel_index(np.argmin(dist_sq_center.values), dist_sq_center.shape)
    
    # --- U-point variables (XG/YC) --- #
    dist_sq_u = (lat_YC - lat_target)**2 + (lon_XG - lon_target)**2
    j_YC_u, i_XG = np.unravel_index(np.argmin(dist_sq_u.values), dist_sq_u.shape)
    
    # --- V-point variables (XC/YG) --- #
    dist_sq_v = (lat_YG - lat_target)**2 + (lon_XC - lon_target)**2
    j_YG, i_XC_v = np.unravel_index(np.argmin(dist_sq_v.values), dist_sq_v.shape)
    
    # Loop through all variables 
    for var in ds.data_vars:
        da = ds[var]

        # Select appropriate index and extract data 
        if {'YC', 'XC'}.issubset(da.dims):
            sel = da.isel(YC=j_YC, XC=i_XC)
        elif {'YC', 'XG'}.issubset(da.dims):
            sel = da.isel(YC=j_YC_u, XG=i_XG)
        elif {'YG', 'XC'}.issubset(da.dims):
            sel = da.isel(YG=j_YG, XC=i_XC_v)
        else:
            # For unexpected variable shapes
            print(f"Skipping {var}: unknown coordinate configuration.")
            continue
        
        # Add site dimension
        sel = sel.expand_dims(site=[f"CCE{i+1}"])
        all_profiles[var].append(sel)

# Combine all variables into datasets per variable
profiles_ds = xr.Dataset({var: xr.concat(all_profiles[var], dim='site') for var in all_profiles})

# -----------------------------------------------------------------------------
# Save data in netcdf files
# -----------------------------------------------------------------------------

# Loop through each variable in the profiles dataset
for var in profiles_ds.data_vars:

    # Print status to monitor progress
    print(f"Saving {var}...")
    
    # Select the data array corresponding to the current variable
    da = profiles_ds[var]
    
    # Chunk along time for faster write
    if 'time' in da.dims:
        da = da.chunk({'time': 1000})

    # Load into memory before saving 
    da = da.load()
    
    # Save to NetCDF file
    da.to_netcdf(
        f"{PATH_nc}{var}_CCS_hrly_mooring.nc",  
        engine='netcdf4',                
        format='NETCDF4',                      
        encoding=encoding
    )
