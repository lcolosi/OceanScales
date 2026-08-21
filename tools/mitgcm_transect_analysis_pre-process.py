# =============================================================================
# Pre-Processing MITgcm data for the Transect Analysis
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
from pathlib import Path
import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
from geopy.distance import geodesic
import xgcm

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox for cartopy figures
from plotting import status
from process_mitgcm_data import compute_bearing_angle

status(f"Starting MITgcm pre-processing for the Regional Analysis")

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
# - coastal_threshold : Wet-cell threshold for determining the location of the coastline.
#
# ------------# 

# Model parameters 
delta_t = 150  

# Set time and space parameters  
lat_bnds  = [33.0, 35.0]                                          
lon_bnds  = [237.0, 240.0]                                        
encoding  = {'time': {'units': 'seconds since 2015-12-01 2:00'}}  
coastal_threshold = 0.5

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                                     
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'                     
PATH_nc     = '/data/SO3/lcolosi/OceanScales/mitgcm/transect/'
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
# Set CalCOFI station locations
# -----------------------------------------------------------------------------
status(f"Interpolating data onto the CalCOFI line 80.0 starting at the shore...")

# Define station locations
calcofi_lat = np.array([
    34.46667,
    34.45000,
    34.31667,
    34.15000,
    33.81667,
    33.48333,
    33.15000,
    32.81667,
])

calcofi_lon = np.array([
    -120.48906,
    -120.52390,
    -120.80245,
    -121.15000,
    -121.84304,
    -122.53335,
    -123.22099,
    -123.90599,
])

# Sort stations from shore outward (here, the largest longitude is closest to shore)
station_order = np.argsort(calcofi_lon)[::-1]
calcofi_lon = calcofi_lon[station_order]
calcofi_lat = calcofi_lat[station_order] 

# -----------------------------------------------------------------------------
# Estimate native MITgcm horizontal grid resolution
# -----------------------------------------------------------------------------

lon = ds["XC"].values
lat = ds["YC"].values

# Native grid spacing in degrees
dlon = np.median(np.diff(lon))
dlat = np.median(np.diff(lat))

# Representative model latitude
lat_ref = np.median(lat)

# Convert grid spacing from degrees to kilometers
dx_native = 111.32 * np.cos(np.deg2rad(lat_ref)) * dlon
dy_native = 111.32 * dlat

# Set the distance interval for along-transect interpolation (rounded to the nearest integer)
dr = round(min(dx_native, dy_native))

# -----------------------------------------------------------------------------
# Extend CalCOFI Line 80 toward the coast
# -----------------------------------------------------------------------------

# Define the two most nearshore CalCOFI stations
station_1 = (calcofi_lat[0], calcofi_lon[0])
station_2 = (calcofi_lat[1], calcofi_lon[1])

# Determine the shoreward direction of Line 80
shoreward_bearing = compute_bearing_angle(station_2, station_1)

# Create distances for searching along the shoreward extension (units: km)
search_spacing = 0.25  
search_distance = 100  

shoreward_distance = np.arange(
    0,
    search_distance + search_spacing,
    search_spacing,
)

# Convert the distances along the shoreward extension to lat and lon positions
points = [
    geodesic(kilometers=d).destination(
        station_1,
        shoreward_bearing,
    )
    for d in shoreward_distance
]

shoreward_lat = np.array([point.latitude for point in points])
shoreward_lon = np.array([point.longitude for point in points]) % 360

# -----------------------------------------------------------------------------
# Locate coastline along the shoreward extension
# -----------------------------------------------------------------------------

# Extract surface wet-cell mask
surface_hfac = ds["hFacC"].isel(Z=0)

# Interpolate mask onto shoreward search points
shoreward_hfac = surface_hfac.interp(
    YC=xr.DataArray(shoreward_lat, dims="coast_search"),
    XC=xr.DataArray(shoreward_lon, dims="coast_search"),
).values

# Find first land point
dry_index = np.where(shoreward_hfac <= coastal_threshold)[0][0]

# Last ocean point before the coast
wet_index = dry_index - 1

# -----------------------------------------------------------------------------
# Determine coastline position
# -----------------------------------------------------------------------------

# Interpolate the distance where hFacC crosses the coastline threshold
coast_distance = np.interp(
    coastal_threshold,
    [shoreward_hfac[dry_index], shoreward_hfac[wet_index]],
    [shoreward_distance[dry_index], shoreward_distance[wet_index]],
)

# Convert coastline distance to latitude and longitude
coast_point = geodesic(
    kilometers=coast_distance
).destination(
    station_1,
    shoreward_bearing,
)

coast_lat = coast_point.latitude
coast_lon = coast_point.longitude

# -----------------------------------------------------------------------------
# Construct Line 80 path from coastline to offshore and its cumulative distance
# -----------------------------------------------------------------------------

# Add coastline point to beginning of CalCOFI station coordinates
transect_lat = np.concatenate(([coast_lat], calcofi_lat))
transect_lon = np.concatenate(([coast_lon], calcofi_lon))

# Calculate distance between consecutive points
segment_distance = np.array([
    geodesic(
        (transect_lat[i - 1], transect_lon[i - 1]),
        (transect_lat[i], transect_lon[i]),
    ).km
    for i in range(1, len(transect_lat))
])

# Calculate cumulative distance from coastline
transect_distance = np.concatenate((
    [0.0],
    np.cumsum(segment_distance),
))

# -----------------------------------------------------------------------------
# Interpolate Line 80 to the native model resolution
# -----------------------------------------------------------------------------

# Create regularly spaced distances along the transect
dist_dense = np.arange(
    0.0,
    transect_distance[-1] + dr,
    dr,
)

# Do not extend beyond the final station
dist_dense = dist_dense[dist_dense <= transect_distance[-1]]

# Interpolate latitude and longitude onto the regular distance grid
lat_dense = np.interp(
    dist_dense,
    transect_distance,
    transect_lat,
)

lon_dense = np.interp(
    dist_dense,
    transect_distance,
    transect_lon,
) % 360.0

# Convert transect coordinates to xarray DataArrays
lat_da = xr.DataArray(
    lat_dense,
    dims="distance",
    coords={"distance": dist_dense},
)

lon_da = xr.DataArray(
    lon_dense,
    dims="distance",
    coords={"distance": dist_dense},
)

# -----------------------------------------------------------------------------
# Interpolate MITgcm data onto Line 80
# -----------------------------------------------------------------------------

# Mask land points for each variable  
wet = ds["hFacC"] > 0

ds_transect = xr.Dataset({
    "THETA": ds["THETA"].where(wet),
    "SALT": ds["SALT"].where(wet),
    "UVEL": ds["U_center"].where(wet),
    "VVEL": ds["V_center"].where(wet),
})

# Interpolate model data onto the transect
ds_transect = ds_transect.interp(
    YC=lat_da,
    XC=lon_da,
)

# Add latitude and longitude as functions of distance from shore
ds_transect = ds_transect.assign_coords(
    latitude=("distance", lat_dense),
    longitude=("distance", lon_dense),
)

# Add coordinate metadata
ds_transect["distance"].attrs = {
    "long_name": "distance from shore",
    "units": "km",
}

ds_transect["latitude"].attrs = {
    "long_name": "latitude",
    "units": "degrees_north",
}

ds_transect["longitude"].attrs = {
    "long_name": "longitude",
    "units": "degrees_east",
}

# -----------------------------------------------------------------------------
# Determine water depth along the transect
# -----------------------------------------------------------------------------
status(f"Obtain water depth along the transect...")

# Count wet cells in each vertical water column
wet_count = wet.sum(dim="Z")

# Index of deepest wet cell (handles completely dry columns)
bottom_index = (wet_count - 1).clip(min=0)

# Water depth on the native grid
water_depth = np.abs(ds["Z"].isel(Z=bottom_index))

# Mask fully dry columns
water_depth = water_depth.where(wet_count > 0)

# Interpolate water depth onto Line 80
water_depth = water_depth.interp(
    YC=lat_da,
    XC=lon_da,
)

# Add water depth to the dataset
ds_transect["water_depth"] = water_depth

ds_transect["water_depth"].attrs = {
    "long_name": "ocean water depth derived from hFacC",
    "units": "m",
    "positive": "down",
}

# -----------------------------------------------------------------------------
# Save data in netcdf files
# -----------------------------------------------------------------------------

# Save ocean water depth
ds_transect["water_depth"].to_netcdf(
    PATH_nc / "DEPTH_CCS_trans.nc",
    engine="netcdf4",
    format="NETCDF4",
)

# Set the list of variables to save
vars_to_save = ["THETA", "SALT", "UVEL", "VVEL"]

# Loop through each variable 
for var in vars_to_save:

    # Print status
    status(f"Saving {var} to {var}_CCS_hrly_trans.nc ...")

    # Obtain variable data array
    da = ds_transect[var]
    
    # Chunk along time for faster write
    if 'time' in da.dims:
        da = da.chunk({'time': 1000})
    
    # Load into memory before saving 
    da = da.load()

    # Save to NetCDF file
    da.to_netcdf(
        f"{PATH_nc}{var}_CCS_hrly_trans.nc",
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )

status("MITgcm transect preprocessing complete!")
