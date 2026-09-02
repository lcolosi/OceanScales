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
# - delta_t: Model time step in seconds (time increments of the diagnostics can differ).
# - max_depth : Depth threshold for the depth average velocity calculation (units: m). 
# - lat_bnds: Latitude bounds setting the region of interest.
# - lon_bnds: Longitude bounds setting the region of interest.
# - halo_cells : Number of data points to extend the boundaries to ensure data is
#                present at the boundaries the study doamin when plotting
#                with contourf.
# - PATH_GRID: Directory containing the model grid.
# - PATH_OUTPUT: Directory containing model diagnostics.
# - PATH_nc: Directory where netCDF files are saved.
# - file_dim: Diagnostic file dimension (3D for T, S, drhodr, and velocity; 2D for etan).
#
# ------------# 

# Model parameters 
delta_t = 150  

# Set time and space parameters  
max_depth  = 200.0                                                      
lat_bnds   = [33.0, 35.0]                                          
lon_bnds   = [237.0, 240.0]
halo_cells = 3                                          

# Set path to project directory
PATH_GRID   = '/data/SO2/SWOT/GRID/BIN/'                    
PATH_OUTPUT = '/data/SO2/SWOT/MARA/RUN4_LY/DIAGS_HRLY/'     
PATH_nc     = '/data/SO3/lcolosi/OceanScales/mitgcm/regional/'  

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

# Mask dry cells
theta = theta.where(wet)
salt  = salt.where(wet)
uvel  = uvel.where(wet)
vvel  = vvel.where(wet)

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

# Compute potential density anomaly referenced to the surface
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
theta_mean  = theta.mean(dim="time")

# -----------------------------------------------------------------------------
# Compute depth-averaged velocity fields
# -----------------------------------------------------------------------------
status("Computing depth-averaged velocity fields...")

# Compute effective wet-cell thickness
dz = ds["drF"] * hfac

# Set the surface ocean mask for 2-D fields
wet_surface = wet.isel(Z=0)

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

# Compute full-water-column velocity
uvel_full = depth_average(uvel, dz).where(wet_surface)
vvel_full = depth_average(vvel, dz).where(wet_surface)

# Compute upper-ocean velocity
uvel_upper = depth_average(uvel, dz, max_depth=max_depth).where(wet_surface)
vvel_upper = depth_average(vvel, dz, max_depth=max_depth).where(wet_surface)

# Compute the time-average of the depth-averaged velocity fields
uvel_full_mean = uvel_full.mean(dim="time")
vvel_full_mean = vvel_full.mean(dim="time")

uvel_upper_mean = uvel_upper.mean(dim="time")
vvel_upper_mean = vvel_upper.mean(dim="time")

# -----------------------------------------------------------------------------
# Compute seasonal-mean background fields
# -----------------------------------------------------------------------------
status("Computing seasonal background hydrographic and velocity fields...")

# Hydrographic fields
SA_season     = SA.groupby("time.season").mean(dim="time")
CT_season     = CT.groupby("time.season").mean(dim="time")
sigma0_season = sigma0.groupby("time.season").mean(dim="time")
theta_season  = theta.groupby("time.season").mean(dim="time")

# Depth-averaged velocity fields
uvel_full_season = uvel_full.groupby("time.season").mean(dim="time")
vvel_full_season = vvel_full.groupby("time.season").mean(dim="time")

uvel_upper_season = uvel_upper.groupby("time.season").mean(dim="time")
vvel_upper_season = vvel_upper.groupby("time.season").mean(dim="time")

# Set chronological order of seasons 
seasons = ["DJF", "MAM", "JJA", "SON"]

# Arrange seasons chronologically
SA_season     = SA_season.sel(season=seasons)
CT_season     = CT_season.sel(season=seasons)
sigma0_season = sigma0_season.sel(season=seasons)
theta_season  = theta_season.sel(season=seasons)

uvel_full_season = uvel_full_season.sel(season=seasons)
vvel_full_season = vvel_full_season.sel(season=seasons)

uvel_upper_season = uvel_upper_season.sel(season=seasons)
vvel_upper_season = vvel_upper_season.sel(season=seasons)

# -----------------------------------------------------------------------------
# Save background fields to NetCDF
# -----------------------------------------------------------------------------
status("Saving time-mean and seasonal-mean background fields...")

# Remove unnecessary auxiliary grid coordinates
def clean_coords(da):
    """Remove auxiliary coordinates while retaining dimension coordinates."""
    return da.reset_coords(drop=True)

# Create output dataset
ds_out = xr.Dataset(
    data_vars={
        # Time-mean hydrographic fields
        "SA_mean": clean_coords(SA_mean),
        "CT_mean": clean_coords(CT_mean),
        "sigma0_mean": clean_coords(sigma0_mean),
        "theta_mean": clean_coords(theta_mean),

        # Seasonal-mean hydrographic fields
        "SA_season": clean_coords(SA_season),
        "CT_season": clean_coords(CT_season),
        "sigma0_season": clean_coords(sigma0_season),
        "theta_season": clean_coords(theta_season),

        # Time-mean depth-averaged velocity fields
        "uvel_full_mean": clean_coords(uvel_full_mean),
        "vvel_full_mean": clean_coords(vvel_full_mean),
        "uvel_upper_mean": clean_coords(uvel_upper_mean),
        "vvel_upper_mean": clean_coords(vvel_upper_mean),

        # Seasonal-mean depth-averaged velocity fields
        "uvel_full_season": clean_coords(uvel_full_season),
        "vvel_full_season": clean_coords(vvel_full_season),
        "uvel_upper_season": clean_coords(uvel_upper_season),
        "vvel_upper_season": clean_coords(vvel_upper_season),
    }
)

# Add variable metadata
ds_out["SA_mean"].attrs.update(
    long_name="Time-mean Absolute Salinity",
    units="g kg-1",
)

ds_out["SA_season"].attrs.update(
    long_name="Seasonal-mean Absolute Salinity",
    units="g kg-1",
)

ds_out["CT_mean"].attrs.update(
    long_name="Time-mean Conservative Temperature",
    units="degC",
)

ds_out["CT_season"].attrs.update(
    long_name="Seasonal-mean Conservative Temperature",
    units="degC",
)

ds_out["sigma0_mean"].attrs.update(
    long_name="Time-mean potential density anomaly referenced to 0 dbar",
    units="kg m-3",
)

ds_out["sigma0_season"].attrs.update(
    long_name="Seasonal-mean potential density anomaly referenced to 0 dbar",
    units="kg m-3",
)

ds_out["theta_mean"].attrs.update(
    long_name="Time-mean potential temperature",
    units="degC",
)

ds_out["theta_season"].attrs.update(
    long_name="Seasonal-mean potential temperature",
    units="degC",
)

# Velocity metadata
ds_out["uvel_full_mean"].attrs.update(
    long_name="Time-mean full-water-column depth-averaged zonal velocity",
    units="m s-1",
)

ds_out["vvel_full_mean"].attrs.update(
    long_name="Time-mean full-water-column depth-averaged meridional velocity",
    units="m s-1",
)

ds_out["uvel_upper_mean"].attrs.update(
    long_name=f"Time-mean upper-{max_depth:g}-m depth-averaged zonal velocity",
    units="m s-1",
)

ds_out["vvel_upper_mean"].attrs.update(
    long_name=f"Time-mean upper-{max_depth:g}-m depth-averaged meridional velocity",
    units="m s-1",
)

ds_out["uvel_full_season"].attrs.update(
    long_name="Seasonal-mean full-water-column depth-averaged zonal velocity",
    units="m s-1",
)

ds_out["vvel_full_season"].attrs.update(
    long_name="Seasonal-mean full-water-column depth-averaged meridional velocity",
    units="m s-1",
)

ds_out["uvel_upper_season"].attrs.update(
    long_name=f"Seasonal-mean upper-{max_depth:g}-m depth-averaged zonal velocity",
    units="m s-1",
)

ds_out["vvel_upper_season"].attrs.update(
    long_name=f"Seasonal-mean upper-{max_depth:g}-m depth-averaged meridional velocity",
    units="m s-1",
)

# Add global metadata
ds_out.attrs.update(
    title="MITgcm CCS background fields for Rossby radius analysis",
    description=(
        "Time-mean and seasonal-mean hydrographic and depth-averaged "
        "velocity fields from the MITgcm regional simulation."
    ),
    upper_velocity_max_depth_m=max_depth,
)

# Set output filename
filename = (
    f"MITgcm_CCS_rossby_radius_background_"
    f"upper_{max_depth:g}m.nc"
)

# Load reduced dataset into memory before writing
ds_out = ds_out.load()

# Save to NetCDF
ds_out.to_netcdf(
    Path(PATH_nc) / filename,
    engine="netcdf4",
    format="NETCDF4",
)

status(f"Saved {filename}")
status("MITgcm Rossby radius pre-processing complete!")



