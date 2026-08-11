# =============================================================================
# Intermediate Processing MITgcm data for the Transect Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the cross-
#   shelf transect decorrelation time scale analysis. These include: 
#       (1) Conservative Temperature
#       (2) Absolute Salinity
#       (3) Potential Density (referenced to the surface)
#       (4) Interpolated and rotatedhorizontal velocity components
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-11
# =============================================================================

# Import python libraries 
import sys
import os
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import gsw

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# - option_proc: Specifies which data set will be processed. 
#                Options include: 'vel' or 'density'
# - option_depth_mask: Specifies whether there is a depth mask applied to the data for 
#                      computing the depth average velocity. 
# - depth_thresh: Specify the lower depth limit of depth average if option_mask is true.
#                 Units: meters. 
# - R_earth: Specify the radius of the Earth. Units: kilometers.
# - g: Specify the acceleration due to gravity. Units: m/s^2.
#
# ------------ # 

# Set processing parameters
option_proc          = 'vel'        
option_depth_mask    = 1      

# Set physical parameters 
depth_thresh = 400   
R_earth      = 6371 
g            = 9.81 

# Set path to project directory
ROOT = '/Users/lukecolosi/Desktop/projects/graduate_research/Gille_lab/OceanScales/'
PATH = ROOT + 'data/mitgcm/transect/'

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------

# --- Velocity Processing --- # 
if option_proc == 'vel':

    # Obtain filename paths
    filename_u = PATH + "UVEL_CCS_hrly_trans.nc"
    filename_v = PATH + "VVEL_CCS_hrly_trans.nc"

    # Generate the nc data structure
    nc_u = Dataset(filename_u, 'r')
    nc_v = Dataset(filename_v, 'r')

    # Extract data variables
    depth  = nc_u.variables['Z'][:]
    lon    = nc_v.variables['XC'][:]
    lat    = nc_u.variables['YC'][:]
    dist   = nc_u.variables['distance'][:]
    time   =  num2date(nc_u.variables['time'][:], nc_u.variables['time'].units)

    u_raw  = nc_u.variables['U_center'][:]
    v_raw  = nc_v.variables['V_center'][:]

    # Mask data at fill values (zero for the MITgcm output)
    u_m = np.ma.masked_where(u_raw == 0, u_raw)
    v_m = np.ma.masked_where(v_raw == 0, v_raw)

    # Rearrange dimensions of data to (distance, time, depth)
    u_m = np.transpose(u_m, (2, 0, 1))
    v_m = np.transpose(v_m, (2, 0, 1))

# --- Density Processing --- # 
elif option_proc == 'density':

    # Obtain filename paths
    filename_temp = PATH + "THETA_CCS_hrly_trans.nc"
    filename_salt = PATH + "SALT_CCS_hrly_trans.nc"

    # Generate the nc data structure
    nc_temp = Dataset(filename_temp, 'r')
    nc_salt = Dataset(filename_salt, 'r')

    # Extract data variables
    depth = nc_temp['Z'][:]
    lon   = nc_temp.variables['XC'][:]
    lat   = nc_temp.variables['YC'][:]
    dist   = nc_temp.variables['distance'][:]
    time  =  num2date(nc_temp.variables['time'][:], nc_temp.variables['time'].units)

    T = nc_temp.variables['THETA'][:]
    S = nc_salt.variables['SALT'][:]

    # Mask data at fill values (zero for the MITgcm output)
    T_m = np.ma.masked_where(T == 0, T)
    S_m = np.ma.masked_where(S == 0, S)

    # Rearrange dimensions of data to (distance, time, depth)
    T_m = np.transpose(T_m, (2, 0, 1))
    S_m = np.transpose(S_m, (2, 0, 1))

# Convert cftime.DatetimeGregorian to Python datetime objects
time_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time])

# -----------------------------------------------------------------------------
# Process horizontal velocity components (u,v)
# -----------------------------------------------------------------------------

if option_proc == 'vel':

    #------------------------------------------# 
    # Compute depth average horizontal velocity 
    #------------------------------------------# 

    # Take absolute value of depth
    depth_pos = np.abs(depth)

    # Ensure increasing order
    if not np.all(np.diff(depth_pos) > 0):
        depth_pos = depth_pos[::-1]
        u_m = u_m[:, :, ::-1]
        v_m = v_m[:, :, ::-1]

    # Mask depth levels below threshold if requested
    if option_depth_mask == 1:

        # Mask depth levels deeper than depth_thresh
        mask_depth = depth_pos <= depth_thresh

        # Select shallower depths
        depth_sel = depth_pos[mask_depth]
        u_sel = u_m[:, :, mask_depth]
        v_sel = v_m[:, :, mask_depth]

        # Depth range
        H = depth_sel[-1] - depth_sel[0]

        # Compute weighted average
        u_bar_tmp = np.trapezoid(u_sel.filled(np.nan), depth_sel, axis=2) / H
        v_bar_tmp = np.trapezoid(v_sel.filled(np.nan), depth_sel, axis=2) / H

    else:

        # Depth range
        H = depth_pos[-1] - depth_pos[0]

        # Compute weighted average
        u_bar_tmp = np.trapezoid(u_m.filled(np.nan), depth_pos, axis=2) / H
        v_bar_tmp = np.trapezoid(v_m.filled(np.nan), depth_pos, axis=2) / H

    # Convert back to masked arrays
    u_bar = np.ma.masked_invalid(u_bar_tmp)
    v_bar = np.ma.masked_invalid(v_bar_tmp)

    #------------------------------------------# 
    # Rotate velocity vectors into the coordinate system of the transect 
    #------------------------------------------# 

    # ------------ # 
    # --- Note --- # 
    # ------------ #
    #
    # Along transect (cross-shelf direction) 
    # - The along-transect component of velocity is defined such that onshore is 
    #   positive and offshore is negative.
    # Cross transect (along-shelf direction) 
    # - The cross-transect component of velocity is defined such that upcoast is 
    #   positive and downcoast is negative.
    #
    # ------------ # 

    # Convert to radians
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)

    # Compute local cartesian displacements in the easting and northing directions (km)
    dx = R_earth * np.cos(lat_r[:-1]) * np.diff(lon_r)
    dy = R_earth * np.diff(lat_r)

    # Compute angle of transect relative to east
    theta = np.arctan2(np.mean(dy), np.mean(dx))

    # Construct rotation matrix (Counter-clockwise rotation)
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    
    # Rotate velocity components counterclockwiseto along and cross-transect directions

    # --- Depth-dependent velocities --- #
    u_along_tmp = R[0, 0] * u_m + R[0, 1] * v_m
    v_cross_tmp = R[1, 0] * u_m + R[1, 1] * v_m

    # Mask zeros
    u_along = np.ma.masked_where(u_along_tmp == 0, u_along_tmp)
    v_cross = np.ma.masked_where(v_cross_tmp == 0, v_cross_tmp)

    # --- Depth-averaged velocities --- #
    u_along_tmp = R[0, 0] * u_bar + R[0, 1] * v_bar
    v_cross_tmp = R[1, 0] * u_bar + R[1, 1] * v_bar

    # Mask zeros
    u_along_bar = np.ma.masked_where(u_along_tmp == 0, u_along_tmp)
    v_cross_bar = np.ma.masked_where(v_cross_tmp == 0, v_cross_tmp)

# -----------------------------------------------------------------------------
# Process Density Variables (T, S, rho, sigma0)
# -----------------------------------------------------------------------------

if option_proc == 'density': 

    #------------------------------------------# 
    # Compute Potential Density  
    #------------------------------------------# 

    # Set the dimensions of the array
    ndist, ntime, ndepth = T_m.shape

    # Compute pressure once for each distance and depth
    pressure_dist_depth = np.array([gsw.p_from_z(depth, lat[i]) for i in range(ndist)]) 

    # Broadcast pressure, lon, and lat to shape of full array
    pressure = np.broadcast_to(pressure_dist_depth[:, None, :], (ndist, ntime, ndepth))
    lon3d = np.broadcast_to(lon[:, None, None], (ndist, ntime, ndepth))
    lat3d = np.broadcast_to(lat[:, None, None], (ndist, ntime, ndepth))

    # Compute Absolute Salinity
    SA = gsw.SA_from_SP(S_m, pressure, lon3d, lat3d)

    # Compute Conservative Temperature
    CT = gsw.CT_from_pt(SA, T_m)

    # Compute in-situ density
    density = gsw.rho(SA, CT, pressure)

    # Compute potential density anomaly (sigma0)
    sigma0 = gsw.sigma0(SA, CT)

    # Mask ocean bottom depths
    SA = np.ma.masked_where(SA == 0, SA)
    CT = np.ma.masked_where(CT == 0, CT)
    density = np.ma.masked_where(density == 0, density)
    sigma0 = np.ma.masked_where(sigma0 == 0, sigma0)

    #------------------------------------------# 
    # Compute Buoyancy Frequency using the Neutral Density Gradient Method method 
    #------------------------------------------# 
    
    # Compute the potential density anomaly
    rho_theta = sigma0 + 1000

    # Set the dimensions of the array
    ndist, ntime, ndepth = np.shape(sigma0)

    # Compute the mean density in the upper 500 m for reference density
    rho0 = np.ma.mean(rho_theta[:,:,(depth <= depth[0]) & (depth >= -500)]) 

    # Compute the time-mean fields 
    SA_mean = np.mean(SA, axis=1)        
    T_mean  = np.mean(T_m, axis=1)
    p_mean  = np.mean(pressure, axis=1)

    # Initalize arrays 
    depth_mid = np.zeros((ndepth-1))
    Nsquare = np.zeros((ndist,ndepth-1))

    # Loop through distance along the transect 
    for idist in range(0,ndist):

        # Set progress bar
        progress = (idist + 1) / (ndist)
        sys.stdout.write(f"\rProgress: {progress:.1%}")
        sys.stdout.flush()

        # Loop through depth pairs 
        for k in range(0,len(depth)-1):

            # Compute the midpoint standard depth 
            z_half = (depth[k] + depth[k+1]) / 2

            # Convert standard depth to a reference pressure 
            p_half = gsw.conversions.z_from_p(z_half,lat[idist])

            # Compute the potential density referenced to p_half pressure
            sigma_ref_top  = gsw.pot_rho_t_exact(SA_mean[idist,k], T_mean[idist,k], p_mean[idist,k], p_half)
            sigma_ref_bottom  = gsw.pot_rho_t_exact(SA_mean[idist,k+1], T_mean[idist,k+1], p_mean[idist,k+1], p_half)

            # Compute N^2(z) profile 
            Nsquare[idist,k] = (-g/rho0) * ((sigma_ref_top - sigma_ref_bottom)/(depth[k] - depth[k+1]))

            # Save the midpoints of the depth bins 
            if idist == 0:
                depth_mid[k] = z_half

    # Compute instaneous buoyancy frequency in units of cycles/hour
    Nz = np.sqrt(Nsquare) * (60/1) * (60/1) 

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Velocity --- # 
if option_proc == 'vel': 

    # --- Coordinates --- # 
    LON = xr.DataArray(data=lon, 
                        dims=['dist'],
                        coords=dict(dist=dist),
                        attrs=dict(
                            description='Longitude along the CalCOFI line 80 transect.',
                            units='degrees'
                            )
    )

    LAT = xr.DataArray(data=lat, 
                        dims=['dist'],
                        coords=dict(dist=dist),
                        attrs=dict(
                            description='Latitude along the CalCOFI line 80 transect.',
                            units='degrees'
                            )
    )

    DIST = xr.DataArray(data=dist, 
                    dims=['dist'],
                    coords=dict(dist=dist),
                    attrs=dict(
                        description='Distance offshore along CalCOFI line 80 transect.',
                        units='kilometers'
                        )
    )

    # --- Depth-dependent Velocity Components --- #
    u = xr.DataArray(data=u_m,
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='The x-component (zonal) of velocity.',
                            units='m/s'
                        )
    )

    v = xr.DataArray(data=v_m,
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='The y-component (meridional) of velocity.',
                            units='m/s'
                        )
    )

    u_along = xr.DataArray(data=u_along,
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='The along-transect component of velocity with onshore being in the postive direction and offshore being in the negative direction.',
                            units='m/s'
                        )
    )

    v_cross = xr.DataArray(data=v_cross,
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='The cross-transect component of velocity with upcoast in the positive direction and downcoast in the negative direction.',
                            units='m/s'
                        )
    )
    

    # --- Depth-averaged Velocity Components --- #
    u_bar = xr.DataArray(data=u_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description='The depth averaged x-component (zonal) of velocity  to ' + str(depth_thresh) + ' meters.',
                            units='m/s'
                        )
    )

    v_bar = xr.DataArray(data=v_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description='The depth averaged y-component (meridional) of velocity  to ' + str(depth_thresh) + ' meters.',
                            units='m/s'
                    )
    )

    u_along_bar = xr.DataArray(data=u_along_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description='The depth averaged (integrated to ' + str(depth_thresh) + ' meters) along-transect component of velocity with onshore being in the postive direction and offshore being in the negative direction.',
                            units='m/s'
                        )
    )

    v_cross_bar = xr.DataArray(data=v_cross_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description='The depth averaged (integrated to ' + str(depth_thresh) + ' meters) cross-transect component of velocity with upcoast in the positive direction and downcoast in the negative direction.',
                            units='m/s'
                    )
    )

    # Create data set from data arrays 
    data = xr.Dataset({'LON':LON,'LAT':LAT,'DIST':DIST,'u':u,'v':v,'u_along':u_along,'v_cross':v_cross,'u_bar':u_bar,'v_bar':v_bar,'u_along_bar':u_along_bar,'v_cross_bar':v_cross_bar})

    # Set file path for saving the netcdf file
    file_path = PATH + "/intermediate_proc/mitgcm_proc_vel_hrly_trans.nc"

# --- Density --- # 
elif option_proc == 'density': 

    # --- Coordinates --- # 
    LON = xr.DataArray(data=lon, 
                        dims=['dist'],
                        coords=dict(dist=dist),
                        attrs=dict(
                            description='Longitude along CalCOFI line 80 transect.',
                            units='degrees'
                            )
    )

    LAT = xr.DataArray(data=lat, 
                        dims=['dist'],
                        coords=dict(dist=dist),
                        attrs=dict(
                            description='Latitude along CalCOFI line 80 transect.',
                            units='degrees'
                            )
    )

    DIST = xr.DataArray(data=dist, 
                    dims=['dist'],
                    coords=dict(dist=dist),
                    attrs=dict(
                        description='Distance offshore along CalCOFI line 80 transect.',
                        units='kilometers'
                        )
    )

    # --- Sea State Varibles --- # 
    Pressure = xr.DataArray(data=pressure, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Pressure profile time series along the CalCOFI line 80 transect.',
                            units='dbar'
                            )
    )

    Density = xr.DataArray(data=density, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='In-situ Density profile time series along the CalCOFI line 80 transect.',
                            units='kg/m^3'
                            )
    ) 

    SIG = xr.DataArray(data=sigma0, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Potential Density anomaly profile time series along the CalCOFI line 80 transect referenced to the pressure at the sea surface.',
                            units='kg/m^3'
                            )
    ) 

    CTemp = xr.DataArray(data=CT, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Conservative temperature profile time series along the CalCOFI line 80 transect.',
                            units='degrees Celcius'
                            )
    ) 

    ASal = xr.DataArray(data=SA, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Absolute Salinity profile time series along the CalCOFI line 80 transect.',
                            units='g/kg'
                            )
    ) 

    NZ = xr.DataArray(data=Nz, 
                    dims=['dist','depth_mid'],
                    coords=dict(dist=dist,depth_mid=depth_mid),
                    attrs=dict(
                            description='Background buoyancy frequency profile time series along the CalCOFI line 80 transect.',
                            units='cycles/hour'
                            )
    )

    # Create data set from data arrays
    data = xr.Dataset({'LON':LON,'LAT':LAT,'DIST':DIST,'Pressure':Pressure,'Density':Density,'SIG':SIG,'CTemp':CTemp,'ASal':ASal, 'NZ':NZ})

    # Set file path for saving the netcdf file
    file_path = PATH + "/intermediate_proc/mitgcm_proc_density_hrly_trans.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
