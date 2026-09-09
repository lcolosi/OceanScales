# =============================================================================
# Pre-Processing MITgcm data for the Rossby Radius Analysis
# =============================================================================
#
# Description:
#   Extract MITgcm hydrographic and velocity fields from the Point Conception
#   region. Compute Absolute Salinity, Conservative Temperature, and potential
#   density, along with full-water-column and upper-ocean depth-averaged
#   velocity. Time-mean and seasonal-mean background fields are saved for
#   subsequent Rossby deformation radius and circulation analyses.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-01
# =============================================================================

# Import python libraries 
import sys
from pathlib import Path
import numpy as np
from xmitgcm import open_mdsdataset
import xarray as xr
import gsw
import xgcm
import warnings

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import plotting toolbox
from plotting import status

# Suppress the interpolation warning message from xgcm
warnings.filterwarnings(
    "ignore",
    message=r"The return type of `Dataset\.dims` will be changed",
    category=FutureWarning,
)

status(f"Starting MITgcm pre-processing for the Rossby Radius Analysis")

# -----------------------------------------------------------------------------
# Set data parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - option_depth_avg: Depth averaging option for the velocity fields. Options are:
#                     "full" for full-water-column depth average, or "upper" for 
#                     upper-ocean depth average to a specified depth.
# - delta_t: Model time step in seconds (time increments of the diagnostics can differ).
# - max_depth : Depth threshold for the depth average velocity calculation (units: m). 
# - lat_bnds: Latitude bounds setting the region of interest.
# - lon_bnds: Longitude bounds setting the region of interest.
# - halo_cells : Number of data points to extend the boundaries to ensure data is
#                present at the boundaries the study doamin when plotting
#                with contourf.
# - encoding: Start time of the model run.
# - PATH_GRID: Directory containing the model grid.
# - PATH_OUTPUT: Directory containing model diagnostics.
# - PATH_nc: Directory where netCDF files are saved.
# - file_dim: Diagnostic file dimension (3D for T, S, drhodr, and velocity; 2D for etan).
#
# ------------# 

# Set processing parameters
option_depth_avg = "upper" 

# Model parameters 
delta_t = 150  

# Set time and space parameters  
max_depth  = 200.0                                                      
lat_bnds   = [33.0, 35.0]                                          
lon_bnds   = [237.0, 240.0]
halo_cells = 3    
encoding   = {'time': {'units': 'seconds since 2015-12-01 2:00'}}                                        

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                    
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'     
PATH_nc     = '/data/SO3/lcolosi/OceanScales/mitgcm/regional/'  

# Validate processing parameters
if option_depth_avg not in {"full", "upper"}:
    raise ValueError(
        "option_depth_avg must be either 'full' or 'upper'"
    )

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
    prefix=['diags_3D'],   
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
# Slice array based on longitude and latitude bounds of the region
# -----------------------------------------------------------------------------
status(f"Selecting 3D fields in study domain...")

# Compute the median longitude and latitude spatial resolution
dlon = float(np.median(np.abs(np.diff(ds["XC"].values))))
dlat = float(np.median(np.abs(np.diff(ds["YC"].values))))

# Set the longitude and latitude slicing vectors
lon_slice = slice(
    lon_bnds[0] - halo_cells * dlon,
    lon_bnds[1] + halo_cells * dlon,
)

lat_slice = slice(
    lat_bnds[0] - halo_cells * dlat,
    lat_bnds[1] + halo_cells * dlat,
)

# Extract scalar fields 
theta = ds['THETA'].sel(YC=lat_slice, 
                        XC=lon_slice)
salt  = ds['SALT'].sel(YC=lat_slice, 
                       XC=lon_slice)
uvel  = ds['U_center'].sel(YC=lat_slice, 
                           XC=lon_slice)
vvel  = ds['V_center'].sel(YC=lat_slice, 
                           XC=lon_slice)
water_depth  = ds['Depth'].sel(YC=lat_slice, 
                               XC=lon_slice)

# -----------------------------------------------------------------------------
# Mask dry cells
# -----------------------------------------------------------------------------
status("Masking dry cells...")

# Extract center-cell wet fraction over the study region
hfac = ds["hFacC"].sel(
    YC=lat_slice,
    XC=lon_slice,
)

# Identify wet cells
wet = hfac > 0

# Identify the wet cells at just the surface
wet_surface = hfac.isel(Z=0) > 0

# Mask dry cells
theta = theta.where(wet)
salt  = salt.where(wet)
uvel  = uvel.where(wet)
vvel  = vvel.where(wet)
water_depth = water_depth.where(wet_surface)

# -----------------------------------------------------------------------------
# Compute Absolute Salinity, Conservative Temperature, and Potential Density
# -----------------------------------------------------------------------------
status("Computing time-average background hydrographic fields...")

# Set coordinates
depth = salt["Z"]
lat   = salt["YC"]
lon   = salt["XC"]

# Compute pressure as a function of depth and latitude
pressure = xr.DataArray(
    gsw.p_from_z(
        depth.values[:, None],
        lat.values[None, :],
    ),
    dims=("Z", "YC"),
    coords={"Z": depth, "YC": lat},
)

# Compute Absolute Salinity
SA = xr.apply_ufunc(
    gsw.SA_from_SP,
    salt,
    pressure,
    lon,
    lat,
    dask="parallelized",
    output_dtypes=[np.float64],
)

# Compute Conservative Temperature
CT = xr.apply_ufunc(
    gsw.CT_from_pt,
    SA,
    theta,
    dask="parallelized",
    output_dtypes=[np.float64],
)

# Compute Potential Density Anomaly referenced to the surface
sigma0 = xr.apply_ufunc(
    gsw.sigma0,
    SA,
    CT,
    dask="parallelized",
    output_dtypes=[np.float64],
)

# Compute time-mean background fields
SA_mean     = SA.mean(dim="time")
CT_mean     = CT.mean(dim="time")
sigma0_mean = sigma0.mean(dim="time")

# -----------------------------------------------------------------------------
# Compute depth-averaged velocity fields
# -----------------------------------------------------------------------------
status("Computing depth-averaged velocity fields...")

# Compute effective wet-cell thickness
dz = ds["drF"] * hfac

# Set the surface ocean mask for 2-D fields
wet_surface = wet.isel(Z=0)

# Define depth average function 
def depth_average(var, dz, max_depth=None):
    """
    Compute the thickness-weighted vertical average of a variable.

    Parameters
    ----------
    var : xarray.DataArray
        Variable to vertically average. Must contain the ``Z`` dimension.
    dz : xarray.DataArray
        Effective wet-cell thickness.
    max_depth : float, optional
        Maximum averaging depth in meters. If None, average over the
        full water column.

    Returns
    -------
    var_avg : xarray.DataArray
        Thickness-weighted vertical average.

    """

    # Limit cell thickness to the specified maximum depth
    if max_depth is not None:

        # Depth of the top of each model cell
        cell_top = np.abs(var["Z"]) - 0.5 * ds["drF"]

        # Thickness of each cell lying above max_depth
        dz_max = (max_depth - cell_top).clip(
            min=0.0,
            max=ds["drF"],
        )

        # Account for max_depth and partial bottom cells
        dz = xr.where(dz < dz_max, dz, dz_max)

    # Exclude cells without valid data
    weights = dz.where(var.notnull())

    # Compute numerator and denominator
    numerator   = (var * weights).sum(dim="Z")
    denominator = weights.sum(dim="Z")

    # Avoid division by zero over land
    denominator = denominator.where(denominator > 0)

    # Compute thickness-weighted vertical average
    return numerator / denominator

if option_depth_avg == "full":

    # Compute full-water-column velocity 
    uvel_depth_avg = depth_average(uvel, dz).where(wet_surface)
    vvel_depth_avg = depth_average(vvel, dz).where(wet_surface)

elif option_depth_avg == "upper":

    # Compute upper-ocean velocity from the surface to the specified depth  
    uvel_depth_avg = depth_average(uvel, dz, max_depth=max_depth).where(wet_surface)
    vvel_depth_avg = depth_average(vvel, dz, max_depth=max_depth).where(wet_surface)

# -----------------------------------------------------------------------------
# Compute seasonal-mean background fields
# -----------------------------------------------------------------------------
status("Computing seasonal background hydrographic fields...")

# Hydrographic fields
SA_season     = SA.groupby("time.season").mean(dim="time")
CT_season     = CT.groupby("time.season").mean(dim="time")
sigma0_season = sigma0.groupby("time.season").mean(dim="time")

# Set chronological order of seasons 
seasons = ["DJF", "MAM", "JJA", "SON"]

# Arrange seasons chronologically
SA_season     = SA_season.sel(season=seasons)
CT_season     = CT_season.sel(season=seasons)
sigma0_season = sigma0_season.sel(season=seasons)

# -----------------------------------------------------------------------------
# Save background and velocity fields to NetCDF
# -----------------------------------------------------------------------------
status("Saving time-mean and seasonal-mean background fields and depth-average velocity fields...")

# Remove unnecessary auxiliary grid coordinates
def clean_coords(da):
    """Remove auxiliary coordinates while retaining dimension coordinates."""
    return da.reset_coords(drop=True)

# ------------------------------------------ #
# Create datasets
# ------------------------------------------ #

# --- Background --- # 
ds_background = xr.Dataset(
    data_vars={
        # Time-mean hydrographic fields
        "SA_mean": clean_coords(SA_mean),
        "CT_mean": clean_coords(CT_mean),
        "sigma0_mean": clean_coords(sigma0_mean),

        # Seasonal-mean hydrographic fields
        "SA_season": clean_coords(SA_season),
        "CT_season": clean_coords(CT_season),
        "sigma0_season": clean_coords(sigma0_season),

        # Water Depth 
        "water_depth": clean_coords(water_depth),
    }
)

# Create depth-averaged velocity dataset
ds_uvel = xr.Dataset(
    data_vars={
        "uvel_depth_avg": clean_coords(uvel_depth_avg),
    }
)   

ds_vvel = xr.Dataset(
    data_vars={
        "vvel_depth_avg": clean_coords(vvel_depth_avg),
    }
)  

# ------------------------------------------ #
# Add variable metadata
# ------------------------------------------ #

# --- Background --- # 
ds_background["SA_mean"].attrs.update(
    long_name="Time-mean Absolute Salinity",
    units="g kg-1",
)

ds_background["SA_season"].attrs.update(
    long_name="Seasonal-mean Absolute Salinity",
    units="g kg-1",
)

ds_background["CT_mean"].attrs.update(
    long_name="Time-mean Conservative Temperature",
    units="degC",
)

ds_background["CT_season"].attrs.update(
    long_name="Seasonal-mean Conservative Temperature",
    units="degC",
)

ds_background["sigma0_mean"].attrs.update(
    long_name="Time-mean potential density anomaly referenced to 0 dbar",
    units="kg m-3",
)

ds_background["sigma0_season"].attrs.update(
    long_name="Seasonal-mean potential density anomaly referenced to 0 dbar",
    units="kg m-3",
)
 
ds_background["water_depth"].attrs.update(
    long_name="ocean water depth",
    units="m",
    positive="down",
)

# --- Velocity --- # 
if option_depth_avg == "full":

    long_name_uvel = "Full-water-column depth-averaged zonal velocity"
    long_name_vvel = "Full-water-column depth-averaged meridional velocity"

elif option_depth_avg == "upper":

    long_name_uvel = f"Upper-{max_depth:g}-m depth-averaged zonal velocity"
    long_name_vvel = f"Upper-{max_depth:g}-m depth-averaged meridional velocity"

ds_uvel["uvel_depth_avg"].attrs.update(
    long_name=long_name_uvel,
    units="m s-1",
)

ds_vvel["vvel_depth_avg"].attrs.update(
    long_name=long_name_vvel,
    units="m s-1",
)

# ------------------------------------------ #
# Add global metadata
# ------------------------------------------ #

# --- Background --- # 
ds_background.attrs.update(
    title="MITgcm CCS background hydrographic fields for Rossby radius analysis",
    description=(
        "Time-mean and seasonal-mean hydrographic fields from a high resolution MITgcm"
        " regional simulation."
    ),
)

# --- Velocity --- # 
ds_uvel.attrs.update(
    title="MITgcm CCS depth-average zonal velocity",
    description=(
        "Depth-average zonal velocity fields from a high resolution MITgcm"
        " regional simulation."
    ),
)

ds_vvel.attrs.update(
    title="MITgcm CCS depth-average meridional velocity",
    description=(
        "Depth-average meridional velocity fields from a high resolution MITgcm"
        " regional simulation."
    ),
)

# Save averaging depth as metadata for upper-ocean averages
if option_depth_avg == "upper":

    ds_uvel.attrs["max_depth_m"] = max_depth
    ds_vvel.attrs["max_depth_m"] = max_depth

# Set output filename
filename_background = "SEA-STATE_CCS_reg_background.nc"

if option_depth_avg == "full":

    filename_uvel = "UVEL_CCS_hrly_reg_depth_avg_full_water_column.nc"

    filename_vvel = "VVEL_CCS_hrly_reg_depth_avg_full_water_column.nc"

elif option_depth_avg == "upper":

    filename_uvel = f"UVEL_CCS_hrly_reg_depth_avg_upper_{max_depth:g}m.nc"

    filename_vvel = f"VVEL_CCS_hrly_reg_depth_avg_upper_{max_depth:g}m.nc"

# ------------------------------------------ #
# Save background hydrographic fields 
# ------------------------------------------ #

# Save to NetCDF
ds_background.to_netcdf(
    Path(PATH_nc) / filename_background,
    engine="netcdf4",
    format="NETCDF4",
)

status(f"Saved {filename_background} to {PATH_nc}.")

# ------------------------------------------#
# Save depth average velocity fields 
# ------------------------------------------#

# Save to NetCDF
ds_uvel.to_netcdf(
    Path(PATH_nc) / filename_uvel,
    engine="netcdf4",
    format="NETCDF4",
    encoding=encoding,
)

status(f"Saved {filename_uvel} to {PATH_nc}.")

ds_vvel.to_netcdf(
    Path(PATH_nc) / filename_vvel,
    engine="netcdf4",
    format="NETCDF4",
    encoding=encoding,
)

status(f"Saved {filename_vvel} to {PATH_nc}.")
status("MITgcm Rossby radius pre-processing complete!")



