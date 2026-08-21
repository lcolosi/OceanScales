# =============================================================================
# Processing MITgcm data for the Regional Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the regional 
#   decorrelation time scale analysis. These include: 
# 
#       (1) Conservative Temperature
#       (2) Absolute Salinity
#       (3) Potential Density (referenced to the surface)
#       (4) Interpolated horizontal velocity components (u, v)
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-11
# =============================================================================

# Import python libraries 
import os
from pathlib import Path
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import gsw

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - option_proc: Specifies which data set will be processed. 
#                Options include: 'vel', 'density', or 'ssh'
# - option_depth: Specifies the depth (in meters) at which to extract data.
#
# ------------# 

# Set processing parameters
option_proc  = 'density' 
option_depth = 9   

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set path to project data directory
PATH_data = ROOT / "data" / "mitgcm" / "regional"

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------

# --- Velocity Processing --- # 
if option_proc == 'vel':

    # Obtain filename paths
    filename_u = PATH_data / f"UVEL_CCS_hrly_reg_depth_{option_depth}m.nc"
    filename_v = PATH_data / f"VVEL_CCS_hrly_reg_depth_{option_depth}m.nc"

    # Generate the nc data structure
    nc_u = Dataset(filename_u, 'r')
    nc_v = Dataset(filename_v, 'r')

    # Extract data variables
    depth  = nc_u.variables['Z'][:]
    lon_XC = nc_v.variables['XC'][:]
    lat_YC = nc_u.variables['YC'][:]
    time   = num2date(nc_u.variables['time'][:], nc_u.variables['time'].units)

    u_raw  = nc_u.variables['UVEL'][:]
    v_raw  = nc_v.variables['VVEL'][:]

    # Mask data at fill values (zero for the MITgcm output)
    u_m = np.ma.masked_where(u_raw == 0, u_raw)
    v_m = np.ma.masked_where(v_raw == 0, v_raw)

# --- Density Processing --- # 
elif option_proc == 'density':

    # Obtain filename paths
    filename_temp = PATH_data / f"THETA_CCS_hrly_reg_depth_{option_depth}m.nc"
    filename_salt = PATH_data / f"SALT_CCS_hrly_reg_depth_{option_depth}m.nc"

    # Generate the nc data structure
    nc_temp = Dataset(filename_temp, 'r')
    nc_salt = Dataset(filename_salt, 'r')

    # Extract data variables
    depth = nc_temp['Z'][:]
    lon   = nc_temp.variables['XC'][:]
    lat   = nc_temp.variables['YC'][:]
    time  =  num2date(nc_temp.variables['time'][:], nc_temp.variables['time'].units)

    T     = nc_temp.variables['THETA'][:]
    S     = nc_salt.variables['SALT'][:]

    # Mask data at fill values (zero for the MITgcm output)
    T_m = np.ma.masked_where(T == 0, T)
    S_m = np.ma.masked_where(S == 0, S)

# --- SSH Processing --- # 
elif option_proc == 'ssh':

    # Obtain filename paths
    filename_ssh = PATH_data / "ETAN_CCS_hrly_reg.nc"

    # Generate the nc data structure
    nc_ssh = Dataset(filename_ssh, 'r')

    # Extract data variables
    lon  = nc_ssh.variables['XC'][:]
    lat  = nc_ssh.variables['YC'][:]
    time =  num2date(nc_ssh.variables['time'][:], nc_ssh.variables['time'].units)

    ssh  = nc_ssh.variables['ETAN'][:]

    # Mask data at fill values (zero for the MITgcm output)
    ssh_m = np.ma.masked_where(ssh == 0, ssh)

# Convert cftime.DatetimeGregorian to Python datetime objects
time_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time])

# -----------------------------------------------------------------------------
# Process Horizontal Velocity Components (u, v)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Process Density Variables (T, S, rho, sigma0)
# -----------------------------------------------------------------------------

if option_proc == 'density': 

    # Set the dimensions of the array
    ntime, nlat, nlon = T_m.shape

    # Compute pressure once for each latitude
    pressure_lat = np.array([gsw.p_from_z(depth, lat[i]) for i in range(nlat)])  

    # Broadcast pressure, lon, and lat to shape of full array
    pressure = np.broadcast_to(pressure_lat[None, :, None], (ntime, nlat, nlon))
    lon3d = np.broadcast_to(lon[None, None, :], (ntime, nlat, nlon))
    lat3d = np.broadcast_to(lat[None, :, None], (ntime, nlat, nlon))

    # Compute Absolute Salinity
    SA = gsw.SA_from_SP(S_m, pressure, lon3d, lat3d)

    # Compute Conservative Temperature
    CT = gsw.CT_from_pt(SA, T_m)

    # Compute potential density anomaly (sigma0)
    sigma0 = gsw.sigma0(SA, CT)

# -----------------------------------------------------------------------------
# Process sea surface height (ssh)
# -----------------------------------------------------------------------------

if option_proc == 'ssh':

    # Set the dimensions of the array
    ntime, nlat, nlon = ssh_m.shape 

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Velocity --- # 
if option_proc == 'vel': 

    # --- Coordinates --- # 
    Depth = xr.DataArray(data=depth, 
                        dims=(),
                        attrs=dict(
                            description='Depth level of sea-state variables.',
                            units='m'
                            )
    )

    # --- Velocity Components --- #
    u = xr.DataArray(data=u_m,
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='The x-component (zonal) of velocity interpolated onto (XC,YC) grid.',
                            units='m/s'
                        )
    )

    v = xr.DataArray(data=v_m,
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='The y-component (meridional) of velocity interpolated onto (XC,YC) grid.',
                            units='m/s'
                        )
    )

    # Create data set from data arrays 
    data = xr.Dataset({'Depth':Depth,'u':u,'v':v,})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / f"mitgcm_proc_vel_hrly_reg_depth_{option_depth}m.nc"


# --- Density --- # 
if option_proc == 'density': 

    # --- Coordinates --- # 
    Depth = xr.DataArray(data=depth, 
                        dims=(),
                        attrs=dict(
                            description='Depth level of sea-state variables.',
                            units='m'
                            )
    )

    # --- Sea State Varibles --- # 
    Pressure = xr.DataArray(data=pressure, 
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='Pressure regional map off point conception, CA.',
                            units='dbar'
                            )
    )

    SIG = xr.DataArray(data=sigma0, 
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='Potential Density anomaly regional map off point conception, CA referenced to the pressure at the sea surface.',
                            units='kg/m^3'
                            )
    ) 

    CTemp = xr.DataArray(data=CT, 
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='Conservative temperature regional map off point conception, CA.',
                            units='degrees Celcius'
                            )
    ) 

    ASal = xr.DataArray(data=SA, 
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='Absolute Salinity regional map off point conception, CA.',
                            units='g/kg'
                            )
    )

    # Create data set from data arrays 
    data = xr.Dataset({'Depth':Depth,'Pressure':Pressure,'SIG':SIG,'CTemp':CTemp,'ASal':ASal})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / f"mitgcm_proc_density_hrly_reg_depth_{option_depth}m.nc"

# --- Sea Surface Height --- # 
if option_proc == 'ssh':

    # --- Sea Surface Height --- #
    ssh = xr.DataArray(data=ssh_m,
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='Sea surface height regional map off point conception, CA.',
                            units='m'
                        )
     )

    # Create data set from data arrays 
    data = xr.Dataset({'ssh':ssh})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / "mitgcm_proc_ssh_hrly_reg.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
