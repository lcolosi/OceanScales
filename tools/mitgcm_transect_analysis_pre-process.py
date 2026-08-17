# =============================================================================
# Processing MITgcm data for the Transect Analysis
# =============================================================================
#
# Description:
#   Slice the MITgcm data to extract the temperature, salinity, and 
#   velocity variables for the cross-shelf transect Analysis. The script will read in 
#   the MITgcm output files, extract the necessary variables along the calCOFI 
#   line 80.0 cross-shelf transect, and save them in netcdf format for further 
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
from geopy.distance import geodesic
import xgcm

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - delta_t: Model time step in seconds (time increments of the diagnostics can differ).
# - lat_bnds: Latitude bounds setting the region of interest.
# - lon_bnds: Longitude bounds setting the region of interest.
# - encoding: Start time of the model run.
# - PATH_GRID: Directory containing the model grid.
# - PATH_OUTPUT: Directory containing model diagnostics.
# - PATH_nc: Directory where netCDF files are saved.
# - file_dim: Diagnostic file dimension (3D for T, S, drhodr, and velocity; 2D for etan).
#
# ------------# 

# Model parameters 
delta_t = 150  

# Set time and space parameters  
lat_bnds  = [33.0, 35.0]                                          
lon_bnds  = [237.0, 240.0]                                        
encoding  = {'time': {'units': 'seconds since 2015-12-01 2:00'}}  

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                                     
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'                     
PATH_nc     = '/data/SO3/lcolosi/mitgcm/SWOT_MARA_RUN4_LY/spatial/transect/' 
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
#
# ------------# 

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
# Interpolate the velocity grids on the (XC, YC) grid
# -----------------------------------------------------------------------------

# Define the grid object (says which dimensions are 'center' and which are 'left')
grid = xgcm.Grid(ds, coords={'X': {'center': 'XC', 'left': 'XG'}, 
                             'Y': {'center': 'YC', 'left': 'YG'}, 
                             'Z': {'center': 'Z',  'left': 'Zl'}}, 
                 periodic=False, boundary='extend') 

# Interpolate to the centers
ds['U_center'] = grid.interp(ds.UVEL, axis='X') # Interpolate from X-face to center
ds['V_center'] = grid.interp(ds.VVEL, axis='Y') # Interpolate from Y-face to center
ds['W_center'] = grid.interp(ds.WVEL, axis='Z') # Interpolate from Z-face (Zl) to center

# -----------------------------------------------------------------------------
# Set CalCOFI station locations
# -----------------------------------------------------------------------------

# Manually read in station locations
calcofi_lat = np.array([34.46667, 34.45, 34.31667, 34.15, 33.81667, 
                        33.48333, 33.15, 32.81667])
calcofi_lon = np.array([-120.48906, -120.5239, -120.80245, -121.15, -121.84304, 
                        -122.53335, -123.22099, -123.90599])

# Sort stations from shore outward
idx = np.argsort(calcofi_lon)
calcofi_lon = calcofi_lon[idx]
calcofi_lat = calcofi_lat[idx] 

# -----------------------------------------------------------------------------
# Compute cumulative diastance along line 80.0 
# -----------------------------------------------------------------------------

# Initialize array 
dist = np.zeros(len(calcofi_lon))

# Loop through stations 
for i in range(1,len(calcofi_lon)): 

    # Define i and i + 1 points along transect
    pt1 = (calcofi_lat[i-1], calcofi_lon[i-1])
    pt2 = (calcofi_lat[i],   calcofi_lon[i])

    # Compute distance in kilometers along transect
    dist[i] = dist[i-1] + geodesic(pt1, pt2).km

# -----------------------------------------------------------------------------
# Create a denser distance axis (near the resolution of the model grid) 
# -----------------------------------------------------------------------------

# Set spacing (units: kilometer)
dr = 2 

# Generate a denser array 
dist_dense = np.arange(0, dist[-1], dr)

# Interpolate lat and longitude along this denser line
calcofi_lat_dense = np.interp(dist_dense, dist, calcofi_lat)
calcofi_lon_dense = np.interp(dist_dense, dist, calcofi_lon)

# Convert the calcofi longitude to span from 0 to 360 
calcofi_lon_dense = (calcofi_lon_dense + 360) % 360

# -----------------------------------------------------------------------------
# Interpolate model onto transect 
# -----------------------------------------------------------------------------

# Apply land mask using hFacC (wet-dry mask) to avoid blending of zeros (fill value) near the ocean bottom
theta_masked = ds['THETA'].where(ds['hFacC'] > 0)
salt_masked  = ds['SALT'].where(ds['hFacC'] > 0)
uvel_masked  = ds['U_center'].where(ds['hFacC'] > 0)
vvel_masked  = ds['V_center'].where(ds['hFacC'] > 0)

# Define your transect DataArrays
lat_da = xr.DataArray(calcofi_lat_dense, dims="distance")
lon_da = xr.DataArray(calcofi_lon_dense, dims="distance")

# Interpolate on transect
theta = theta_masked.interp(YC=lat_da, XC=lon_da)
salt = salt_masked.interp(YC=lat_da, XC=lon_da)
uvel = uvel_masked.interp(YC=lat_da, XC=lon_da)
vvel = vvel_masked.interp(YC=lat_da, XC=lon_da)

# -----------------------------------------------------------------------------
# Assign distance coordinate
# -----------------------------------------------------------------------------

theta = theta.assign_coords(
    distance=("distance", dist_dense)
)

salt = salt.assign_coords(
    distance=("distance", dist_dense)
)

uvel = uvel.assign_coords(
    distance=("distance", dist_dense)
)

vvel = vvel.assign_coords(
    distance=("distance", dist_dense)
)

# -----------------------------------------------------------------------------
# Create dataset for land mask 
# -----------------------------------------------------------------------------

# Define wet-dry array for the transect (e.g., 1 for ocean, 0 for land) and depth array for the transect
hfac = theta['hFacC']
depth = theta['Z']

# Set boolean wet mask
wet = hfac > 0.99 

# Count number of wet cells per column
bottom_index = wet.sum(dim='Z') - 1

# Prevent negative indices (in case of full land columns)
bottom_index = bottom_index.clip(min=0)

# Set bottom depth using the bottom index
bottom_depth = depth.isel(Z=bottom_index)

# Create a new dataset to store the ocean depth data
bottom_ds = xr.Dataset(
    data_vars=dict(
        bottom_depth=('distance', bottom_depth.values)
    ),
    coords=dict(
        distance=theta.coords['distance'].values
    ),
    attrs=dict(
        description="Ocean bottom depth derived from hFacC",
        units="meters"
    )
)

# -----------------------------------------------------------------------------
# Save data in netcdf files
# -----------------------------------------------------------------------------

# --- Ocean Bottom --- #
bottom_ds.to_netcdf(
    f"{PATH_nc}DEPTH_CCS_trans.nc",
    engine='netcdf4',                
    format='NETCDF4'           
)

# --- Sea State Variables --- #

# Set the dictionary of variables to save
vars_to_save = {
    'THETA': theta,
    'SALT': salt,
    'UVEL': uvel,
    'VVEL': vvel
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
    da.to_netcdf(
        f"{PATH_nc}{var_name}_CCS_hrly_trans.nc",
        engine='netcdf4',                
        format='NETCDF4',        
        encoding=encoding            
    )
