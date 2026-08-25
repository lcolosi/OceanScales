# =============================================================================
# Pre-Processing MITgcm data for the Mooring Analysis
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
from pathlib import Path
import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
import xgcm
import warnings

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox for cartopy figures
from plotting import status

# Suppress the interpolation warning message from xgcm
warnings.filterwarnings(
    "ignore",
    message=r"The return type of `Dataset\.dims` will be changed",
    category=FutureWarning,
)

status(f"Starting MITgcm pre-processing for the Mooring Analysis")

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# - delta_t: Model time step in seconds (time increments of the diagnostics can differ).
# - lat_cce: Latitude locations of the CCE mooring sites.
# - lon_cce: Longitude locations of the CCE mooring sites.
# - encoding: Start time of the model run.
# - PATH_GRID: Directory containing the model grid.
# - PATH_OUTPUT: Directory containing model diagnostics.
# - PATH_nc: Directory where netCDF files are saved.
# - file_dim: Diagnostic file dimension (3D for T, S, drhodr, and velocity; 2D for etan).
#
# ------------ # 

# Model parameters 
delta_t = 150  

# Set time and space parameters  
lat_cce  = [33.457, 34.3075, 34.44825228022894]                  
lon_cce = np.array([-122.52233,-120.8042,-120.53825701527784]) % 360         
encoding  = {'time': {'units': 'seconds since 2015-12-01 2:00'}} 

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                   
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'     
PATH_nc     = '/data/SO3/lcolosi/OceanScales/mitgcm/mooring/' 
file_dim    = '3D'     

# -----------------------------------------------------------------------------
# Load the grid and diagnostics data into a python structure
# -----------------------------------------------------------------------------
status(f"Loading the grid and diagnostics data...")

# ------------ #  
# --- Note --- # 
# ------------ # 
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
# ------------ # 

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
# Interpolate the velocity grids on the (XC, YC) grid
# -----------------------------------------------------------------------------
status(f"Interpolating the velocity grid...")

# Define the grid object (says which dimensions are 'center' and which are 'left')
grid = xgcm.Grid(ds, 
                 coords={'X': {'center': 'XC', 'left': 'XG'}, 
                         'Y': {'center': 'YC', 'left': 'YG'}, 
                         'Z': {'center': 'Z',  'left': 'Zl'}}, 
                 periodic=False, 
                 boundary='extend'
                 ) 

# Interpolate to the centers
ds['U_center'] = grid.interp(ds["UVEL"], axis='X') # Interpolate from X-face to center
ds['V_center'] = grid.interp(ds["VVEL"], axis='Y') # Interpolate from Y-face to center
ds['W_center'] = grid.interp(ds["WVEL"], axis='Z') # Interpolate from Z-face (Zl) to center

# -----------------------------------------------------------------------------
# Select model data at CCE mooring locations ()
# -----------------------------------------------------------------------------
status(f"Selecting 3D fields and masking dry-cells at mooring locations...")

# Get 2D coordinate fields
lat_YC = ds['YC']
lon_XC = ds['XC']

# Define center-cell ocean mask
wet = ds["hFacC"] > 0

# Variables to extract (mask dry cells)
variables = {
    "THETA": ds["THETA"].where(wet),
    "SALT": ds["SALT"].where(wet),
    "UVEL": ds["U_center"].where(wet),
    "VVEL": ds["V_center"].where(wet),
}

# Initialize profiles
all_profiles = {var: [] for var in variables}

# Loop through CCE moorings
for i, (lat_target, lon_target) in enumerate(zip(lat_cce, lon_cce)):

    # Compute the distance from the ith CCE mooring 
    distance = (
        (lat_YC - lat_target)**2
        + (lon_XC - lon_target)**2
    )

    # Find nearest model grid point
    j, k = np.unravel_index(
        np.argmin(distance.values),
        distance.shape,
    )

    # Verify the model grid cell location extracted
    status(
    f"CCE{i + 1}: "
    f"target=({lat_target:.3f}, {lon_target:.3f}), "
    f"model=({ds['YC'].values[j]:.3f}, "
    f"{ds['XC'].values[k]:.3f})"
)

    # Loop through data variables 
    for var, da in variables.items():

        # Extract data at the mooring site 
        profile = da.isel(YC=j, XC=k)

        # Add site dimension 
        profile = profile.expand_dims(site=[f"CCE{i + 1}"])
        all_profiles[var].append(profile)

# Combine profiles into a single dataset
profiles_ds = xr.Dataset({
    var: xr.concat(profiles, dim="site")
    for var, profiles in all_profiles.items()
})

# -----------------------------------------------------------------------------
# Save data in NetCDF files
# -----------------------------------------------------------------------------

# Loop through each variable in the profiles dataset
for var_name, da in profiles_ds.data_vars.items():

    # Print status to monitor progress
    status(f"Saving {var_name} to {var_name}_CCS_hrly_mooring.nc ...")

    # Chunk along time
    if "time" in da.dims:
        da = da.chunk({"time": 1000})

    # Load into memory
    da = da.load()

    # Save to NetCDF
    da.to_netcdf(
        f"{PATH_nc}{var_name}_CCS_hrly_mooring.nc",
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )

status("MITgcm mooring pre-processing complete!")
