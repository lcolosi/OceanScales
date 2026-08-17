# =============================================================================
# Intermediate Processing MITgcm data for the Regional Analysis
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
import sys
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import os
from scipy.interpolate import interp1d
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
option_depth = 0.5   

# Set path to project directory
ROOT = '/Users/lukecolosi/Desktop/projects/graduate_research/Gille_lab/OceanScales/'
PATH = ROOT + 'data/mitgcm/regional/'

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------

# --- Velocity Processing --- # 
if option_proc == 'vel':

    # Obtain filename paths
    filename_u = PATH + "UVEL_CCS_hrly_reg_depth_" + str(option_depth) + "m.nc"
    filename_v = PATH + "VVEL_CCS_hrly_reg_depth_" + str(option_depth) + "m.nc"

    # Generate the nc data structure
    nc_u = Dataset(filename_u, 'r')
    nc_v = Dataset(filename_v, 'r')

    # Extract data variables
    depth  = nc_u.variables['Z'][:]
    lon_XG    = nc_u.variables['XG'][:]
    lat_YC    = nc_u.variables['YC'][:]
    lon_XC    = nc_v.variables['XC'][:]
    lat_YG    = nc_v.variables['YG'][:]
    time   =  num2date(nc_u.variables['time'][:], nc_u.variables['time'].units)

    u_raw  = nc_u.variables['UVEL'][:]
    v_raw  = nc_v.variables['VVEL'][:]

    # Mask data at fill values (zero for the MITgcm output)
    u_m = np.ma.masked_where(u_raw == 0, u_raw)
    v_m = np.ma.masked_where(v_raw == 0, v_raw)

# --- Density Processing --- # 
elif option_proc == 'density':

    # Obtain filename paths
    filename_temp = PATH + "THETA_CCS_hrly_reg_depth_" + str(option_depth) + "m.nc"
    filename_salt = PATH + "SALT_CCS_hrly_reg_depth_" + str(option_depth) + "m.nc"

    # Generate the nc data structure
    nc_temp = Dataset(filename_temp, 'r')
    nc_salt = Dataset(filename_salt, 'r')

    # Extract data variables
    depth = nc_temp['Z'][:]
    lon = nc_temp.variables['XC'][:]
    lat = nc_temp.variables['YC'][:]
    time =  num2date(nc_temp.variables['time'][:], nc_temp.variables['time'].units)

    T = nc_temp.variables['THETA'][:]
    S = nc_salt.variables['SALT'][:]

    # Mask data at fill values (zero for the MITgcm output)
    T_m = np.ma.masked_where(T == 0, T)
    S_m = np.ma.masked_where(S == 0, S)

# --- SSH Processing --- # 
elif option_proc == 'ssh':

    # Obtain filename paths
    filename_ssh = PATH + "ETAN_CCS_hrly_reg.nc"

    # Generate the nc data structure
    nc_ssh = Dataset(filename_ssh, 'r')

    # Extract data variables
    lon = nc_ssh.variables['XC'][:]
    lat = nc_ssh.variables['YC'][:]
    time =  num2date(nc_ssh.variables['time'][:], nc_ssh.variables['time'].units)

    ssh = nc_ssh.variables['ETAN'][:]

    # Mask data at fill values (zero for the MITgcm output)
    ssh_m = np.ma.masked_where(ssh == 0, ssh)

# Convert cftime.DatetimeGregorian to Python datetime objects
time_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time])

# -----------------------------------------------------------------------------
# Process Horizontal Velocity Components (u, v)
# -----------------------------------------------------------------------------

if option_proc == 'vel':

    # Convert to a ndarray
    u_m = np.asarray(u_m)
    v_m = np.asarray(v_m)
    lon_XG = np.asarray(lon_XG)
    lat_YC = np.asarray(lat_YC)
    lon_XC = np.asarray(lon_XC)
    lat_YG = np.asarray(lat_YG)

    # Slice lon_XG and lat_YG to match lon_XC and lat_YC bounds respectively 
    lon_min, lon_max = np.min(lon_XC), np.max(lon_XC)
    lat_min, lat_max = np.min(lat_YC), np.max(lat_YC)
    idx_lon = (lon_XG >= lon_min) & (lon_XG <= lon_max)
    idx_lat = (lat_YG >= lat_min) & (lat_YG <= lat_max)
    lon_XG_c = lon_XG[idx_lon]
    lat_YG_c = lat_YG[idx_lat]

    # Apply the same slicing operation to u_raw and v_raw (recall: dim(u_raw) =  (time,lat_YC,lon_XG) and dim(v_raw) =  (time,lat_YG,lon_XC))
    u_raw_c = u_m[:,:,idx_lon]
    v_raw_c = v_m[:,idx_lat,:]

    # Set processing parameters
    ntime,_,_ = np.shape(u_raw_c)
    nlat,nlon = np.size(lat_YC),np.size(lon_XC)
    lon       = lon_XC
    lat       = lat_YC 

    # Initalize arrays
    u_int  = np.zeros((ntime,nlat,nlon)) 
    v_int  = np.zeros((ntime,nlat,nlon))

    # Loop through time
    for itime in range(0,ntime): 

        # Set progress bar
        progress = (itime + 1) / (len(time))
        sys.stdout.write(f"\rProgress: {progress:.1%}")
        sys.stdout.flush()

        # Grab the ith time frame 
        u_i = np.squeeze(u_raw_c[itime,:,:])
        v_i = np.squeeze(v_raw_c[itime,:,:])

        # Interpolate u_z from YC,XG grid onto the YC,XC grid 
        # Interpolate each row along columns (axis=1)
        u_int[itime,:,:] = np.array([
                            interp1d(lon_XG_c, row, kind='linear', bounds_error=False)(lon)
                            for row in u_i
        ])

        # Interpolate v_z from YG,XC grid onto the YC,XC grid 
        v_int[itime,:,:] = np.array([
                            interp1d(lat_YG_c, col, kind='linear', bounds_error=False)(lat)
                            for col in v_i.T
        ]).T 


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
    u = xr.DataArray(data=u_int,
                        dims=['time','lat','lon'],
                        coords=dict(time=time_dt,lat=lat,lon=lon),
                        attrs=dict(
                            description='The x-component (zonal) of velocity interpolated onto (XC,YC) grid.',
                            units='m/s'
                        )
    )

    v = xr.DataArray(data=v_int,
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
    file_path = PATH + "/intermediate_proc/mitgcm_proc_vel_hrly_reg_depth_" + str(option_depth) + "m.nc"


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
    file_path = PATH + "/intermediate_proc/mitgcm_proc_density_hrly_reg_depth_" + str(option_depth) + "m.nc"

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
    file_path = PATH + "/intermediate_proc/mitgcm_proc_ssh_hrly_reg.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
