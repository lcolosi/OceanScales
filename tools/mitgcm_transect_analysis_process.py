# =============================================================================
# Processing MITgcm data for the Transect Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the cross-
#   shelf transect decorrelation time scale analysis. These include: 
# 
#       (1) Conservative Temperature
#       (2) Absolute Salinity
#       (3) Potential Density (referenced to the surface)
#       (4) Buoyancy Frequency 
#       (5) Mixed Layer Depth 
#       (6) Rotated horizontal velocity components 
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
from pathlib import Path
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import gsw
from tqdm import tqdm

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
#                      Options: True or False 
# - vel_depth_thresh: Specify the lower depth limit of velocity depth average if 
#                     option_mask is true. Units: meters. 
# - R_earth: Specify the radius of the Earth. Units: kilometers.
# - phi : Specifies the potential energy anomaly threshold for computing the mixed
#         depth. 
#
# ------------ # 

# Set processing parameters
option_proc          = 'vel'        
option_depth_mask    = True      

# Set physical parameters 
vel_depth_thresh = 400   
R_earth          = 6371 
phi              = 100

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set path to project data directory
PATH_data = ROOT / "data" / "mitgcm" / "transect"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import analysis functions 
from ocean_analysis import compute_mld

# Parameter verification
if option_proc not in ('vel', 'density'):
    raise ValueError(
        f"Invalid option_proc: {option_proc}"
    )

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------

# --- Velocity --- # 
if option_proc == 'vel':

    # Obtain filename paths
    filename_u = PATH_data / "UVEL_CCS_hrly_trans.nc"
    filename_v = PATH_data / "VVEL_CCS_hrly_trans.nc"

    # Generate the nc data structure
    nc_u = Dataset(filename_u, 'r')
    nc_v = Dataset(filename_v, 'r')

    # Extract data variables
    depth  = nc_u.variables['Z'][:]
    lon    = nc_v.variables['XC'][:]
    lat    = nc_u.variables['YC'][:]
    dist   = nc_u.variables['distance'][:]
    time   =  num2date(nc_u.variables['time'][:], nc_u.variables['time'].units)

    u = nc_u.variables['UVEL'][:]
    v  = nc_v.variables['VVEL'][:]

    # Mask dry cells previously set to NaN during preprocessing
    u_m = np.ma.masked_invalid(u)
    v_m = np.ma.masked_invalid(v)

    # Rearrange dimensions of data to (distance, time, depth)
    u_m = np.transpose(u_m, (2, 0, 1))
    v_m = np.transpose(v_m, (2, 0, 1))

# --- Density --- # 
elif option_proc == 'density':

    # Obtain filename paths
    filename_temp = PATH_data / "THETA_CCS_hrly_trans.nc"
    filename_salt = PATH_data / "SALT_CCS_hrly_trans.nc"

    # Generate the nc data structure
    nc_temp = Dataset(filename_temp, 'r')
    nc_salt = Dataset(filename_salt, 'r')

    # Extract data variables
    depth = nc_temp['Z'][:]
    lon   = nc_temp.variables['XC'][:]
    lat   = nc_temp.variables['YC'][:]
    dist  = nc_temp.variables['distance'][:]
    time  =  num2date(nc_temp.variables['time'][:], nc_temp.variables['time'].units)

    T     = nc_temp.variables['THETA'][:]
    S     = nc_salt.variables['SALT'][:]

    # Mask dry cells previously set to NaN during preprocessing
    T_m = np.ma.masked_invalid(T)
    S_m = np.ma.masked_invalid(S)

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
    # Compute depth-average horizontal velocity
    #------------------------------------------#

    # Take absolute value of depth
    depth_pos = np.abs(depth)

    # Ensure depth increases downward
    if not np.all(np.diff(depth_pos) > 0):
        depth_pos = depth_pos[::-1]
        u_m = u_m[:, :, ::-1]
        v_m = v_m[:, :, ::-1]

    # Mask depth levels below threshold if requested
    if option_depth_mask:

        # Mask depth levels deeper than vel_depth_thresh
        mask_depth = depth_pos <= vel_depth_thresh

        # Select shallower depths
        depth_sel = depth_pos[mask_depth]
        u_sel = u_m[:, :, mask_depth]
        v_sel = v_m[:, :, mask_depth]

    else:

        # Set depth and velocity components for calculation
        depth_sel = depth_pos
        u_sel = u_m
        v_sel = v_m

    # Set space and time dimensions
    ndist, ntime, ndepth = u_sel.shape

    # Initialize depth-averaged velocities as masked arrays
    u_bar = np.ma.masked_all((ndist, ntime))
    v_bar = np.ma.masked_all((ndist, ntime))

    # Loop through distance from shore
    for idist in tqdm(range(ndist), desc="Computing Depth Average Velocity", unit="distance"):

        # Loop through time 
        for itime in range(ntime):

            # Extract vertical profiles
            u_prof = u_sel[idist, itime, :]
            v_prof = v_sel[idist, itime, :]

            # Identify valid depth levels
            valid_u = ~np.ma.getmaskarray(u_prof)
            valid_v = ~np.ma.getmaskarray(v_prof)

            # Verify if the profile has two or more data points  
            if np.sum(valid_u) >= 2:

                # Obtain velocity and depths at depth levels not masked  
                z_u = depth_sel[valid_u]
                u_valid = u_prof[valid_u]

                # Compute the depth range
                H_u = z_u[-1] - z_u[0]

                # Compute depth-averaged u velocity
                if H_u > 0:
                    u_bar[idist, itime] = (
                        np.trapezoid(u_valid, z_u) / H_u
                    )

            # Verify if the profile has two or more data points 
            if np.sum(valid_v) >= 2:

                # Obtain velocity and depths at depth levels not masked  
                z_v = depth_sel[valid_v]
                v_valid = v_prof[valid_v]

                # Compute the depth range 
                H_v = z_v[-1] - z_v[0]

                # Compute depth-averaged v velocity
                if H_v > 0:
                    v_bar[idist, itime] = (
                        np.trapezoid(v_valid, z_v) / H_v
                    )

    #------------------------------------------#
    # Rotate velocity vectors into the transect coordinates
    #------------------------------------------#

    # ------------ #
    # --- Note --- #
    # ------------ #
    #
    # Along-transect (cross-shelf direction)
    # - Positive: onshore
    # - Negative: offshore
    #
    # Cross-transect (along-shelf direction)
    # - Positive: upcoast
    # - Negative: downcoast
    #
    # The transect coordinates are ordered offshore -> onshore.
    #
    # ------------ #

    # Convert longitude and latitude to radians
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)

    # Compute latitude at midpoint between adjacent transect points
    lat_mid = 0.5 * (lat_r[:-1] + lat_r[1:])

    # Compute local Cartesian displacements
    # x: positive eastward
    # y: positive northward
    dx = R_earth * np.cos(lat_mid) * np.diff(lon_r)
    dy = R_earth * np.diff(lat_r)

    # Compute total displacement from offshore to onshore
    dx_total = np.sum(dx)
    dy_total = np.sum(dy)

    # Compute overall transect angle relative to east.
    # Because the transect is ordered offshore -> onshore,
    # theta points in the positive onshore direction.
    theta = np.arctan2(
        dy_total,
        dx_total
    )

    # Define unit vector pointing onshore
    e_onshore = np.array([
        np.cos(theta),
        np.sin(theta)
    ])

    # Define unit vector pointing upcoast.
    # This is 90 degrees counterclockwise from the onshore direction.
    e_upcoast = np.array([
        -np.sin(theta),
        np.cos(theta)
    ])

    # Project velocities onto transect directions

    # --- Depth-dependent velocities --- #

    # Along-transect velocity: positive onshore
    u_along = (
        e_onshore[0] * u_m + 
        e_onshore[1] * v_m
    )

    # Cross-transect velocity: positive upcoast
    v_cross = (
        e_upcoast[0] * u_m + 
        e_upcoast[1] * v_m
    )

    # --- Depth-averaged velocities --- #

    # Along-transect velocity: positive onshore
    u_along_bar = (
        e_onshore[0] * u_bar +
        e_onshore[1] * v_bar
    )

    # Cross-transect velocity: positive upcoast
    v_cross_bar = (
        e_upcoast[0] * u_bar +
        e_upcoast[1] * v_bar
    )

# -----------------------------------------------------------------------------
# Process Density Variables (T, S, sigma0)
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

    # Compute potential density anomaly (sigma0)
    sigma0 = gsw.sigma0(SA, CT)

    # Mask ocean bottom depths
    SA = np.ma.masked_invalid(SA)
    CT = np.ma.masked_invalid(CT)
    sigma0 = np.ma.masked_invalid(sigma0)

    #------------------------------------------# 
    # Compute Buoyancy Frequency using TEOS-10 GSW
    #------------------------------------------# 

    # Compute the time-mean fields 
    SA_mean = np.ma.mean(SA, axis=1)        
    CT_mean  = np.ma.mean(CT, axis=1)

    # Initalize arrays 
    Nsquare   = np.ma.masked_all((ndist,ndepth-1))
    depth_mid = np.ma.masked_all((ndist, ndepth-1))

    # Loop through mooring sites
    for idist in tqdm(range(ndist), desc="Computing Buoyancy Frequency", unit="distance"):

        # Compute buoyancy frequency and midpoint pressure
        Nsquare[idist, :], p_mid = gsw.Nsquared(
            SA_mean[idist, :],
            CT_mean[idist, :],
            pressure_dist_depth[idist, :],
            lat[idist],
        )

        # Convert midpoint pressure to vertical position 
        depth_mid[idist,:] = gsw.z_from_p(p_mid,lat[idist])

    # --- Interpolate N^2 onto common depth grid --- # 

    # Define common midpoint-depth coordinate
    depth_mid_reg = 0.5 * (depth[:-1] + depth[1:])

    # Initialize regularly gridded N^2
    Nsquare_reg = np.ma.masked_all((ndist, depth_mid_reg.size))

    # Loop through distance off shore
    for idist in range(ndist):

        # Extract local profile
        z_prof = depth_mid[idist, :]
        N2_prof = Nsquare[idist, :]

        # Identify valid values (wet-cells)
        valid = ~(np.ma.getmaskarray(z_prof) | np.ma.getmaskarray(N2_prof))

        # Check if there are two or more data points 
        if np.sum(valid) < 2:
            continue

        # Extract non-masked data points
        z_valid = z_prof[valid]
        N2_valid = N2_prof[valid]

        # Ensure vertical coordinate is increasing
        sort_idx = np.argsort(z_valid)

        z_valid = z_valid[sort_idx]
        N2_valid = N2_valid[sort_idx]

        # Find common depths contained within this profile
        inside = (
            (depth_mid_reg >= z_valid[0])
            & (depth_mid_reg <= z_valid[-1])
        )

        # Interpolate N^2
        Nsquare_reg[idist, inside] = np.interp(
            depth_mid_reg[inside],
            z_valid,
            N2_valid
        )

    # Mask unstable values before taking square root
    Nsquare_reg = np.ma.masked_less_equal(Nsquare_reg,0)

    # Compute bouyancy frequency in units of cycles/hour
    N = np.sqrt(Nsquare_reg) / (2 * np.pi) * 3600 

    #------------------------------------------# 
    # Compute Mixed Layer Depth
    #------------------------------------------# 

    # Set the positive downward depth vector
    depth_pos = np.abs(depth)

    # Initialize arrays 
    mld = np.ma.masked_all((ndist, ntime))

    # Loop through distance off shore 
    for idist in tqdm(range(ndist), desc="Computing Mixed Layer Depth", unit="distance"):

        # Loop through time 
        for itime in range(ntime):

            # Extract density profile
            sigma0_prof = sigma0[idist, itime, :]

            # Skip fully masked profiles
            if np.ma.getmaskarray(sigma0_prof).all():
                continue

            # Compute MLD using the potential energy anomaly method
            mld[idist, itime] = compute_mld(depth_pos, density=sigma0_prof, method='potential_energy', phi=phi)

    # Mask nan values
    mld = np.ma.masked_invalid(mld)

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# Set velocity depth threshold documentation
if option_depth_mask:
    depth_avg_desc = f"upper {vel_depth_thresh} meters"
else:
    depth_avg_desc = "full water column"

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
                            description=f'The depth averaged x-component (zonal) of velocity over the {depth_avg_desc}',
                            units='m/s'
                        )
    )

    v_bar = xr.DataArray(data=v_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description=f'The depth averaged y-component (meridional) of velocity  over the {depth_avg_desc}',
                            units='m/s'
                    )
    )

    u_along_bar = xr.DataArray(data=u_along_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description=f'The depth averaged (over the {depth_avg_desc}) along-transect component of velocity with onshore being in the postive direction and offshore being in the negative direction.',
                            units='m/s'
                        )
    )

    v_cross_bar = xr.DataArray(data=v_cross_bar,
                        dims=['dist','time'],
                        coords=dict(dist=dist,time=time_dt),
                        attrs=dict(
                            description=f'The depth averaged (over the {depth_avg_desc}) cross-transect component of velocity with upcoast in the positive direction and downcoast in the negative direction.',
                            units='m/s'
                    )
    )

    # Create data set from data arrays 
    data = xr.Dataset({'LON':LON,'LAT':LAT,'DIST':DIST,'u':u,'v':v,'u_along':u_along,'v_cross':v_cross,'u_bar':u_bar,'v_bar':v_bar,'u_along_bar':u_along_bar,'v_cross_bar':v_cross_bar})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / "mitgcm_proc_vel_hrly_trans.nc"

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
    SIG = xr.DataArray(data=sigma0, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Potential Density anomaly profile time series along the CalCOFI line 80 transect referenced to 0 dbar.',
                            units='kg/m^3'
                            )
    ) 

    CTemp = xr.DataArray(data=CT, 
                        dims=['dist','time','depth'],
                        coords=dict(dist=dist,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Conservative temperature profile time series along the CalCOFI line 80 transect.',
                            units='degrees Celsius'
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

    N = xr.DataArray(data=N, 
                    dims=['dist','depth_mid'],
                    coords=dict(dist=dist,depth_mid=depth_mid_reg),
                    attrs=dict(
                            description='Background buoyancy frequency profile time series along the CalCOFI line 80 transect.',
                            units='cycles/hour'
                            )
    )

    MLD = xr.DataArray(data=mld, 
                    dims=['dist','time'],
                    coords=dict(dist=dist,time=time_dt),
                    attrs=dict(
                            description='Mixed layer depth along the CalCOFI line 80 transect computed using the potential energy anomaly method.',
                            units='meters'
                            )
    )

    # Create data set from data arrays
    data = xr.Dataset({'LON':LON,'LAT':LAT,'DIST':DIST,'SIG':SIG,'CTemp':CTemp,'ASal':ASal,'N':N,'MLD':MLD})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / "mitgcm_proc_density_hrly_trans.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
