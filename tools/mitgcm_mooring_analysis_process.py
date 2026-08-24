# =============================================================================
# Processing MITgcm data for the Mooring Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the mooring
#   decorrelation time scale analysis. These include: 
# 
#       (1) Conservative Temperature
#       (2) Absolute Salinity
#       (3) Potential Density (referenced to the surface)
#       (4) Rotated horizontal velocity components
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
from scipy.interpolate import PchipInterpolator, interp1d

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# - option_proc: Specifies which data set will be processed. 
#                Options include: 'vel' or 'density'
# - option_interp: Specifies the interpolation method for transforming to isopycnal 
#                  surfaces. Options include: 'linear' or 'Pchip'
# - vel_depth_thresh: Specify the lower depth limit of velocity depth average if 
#                     option_mask is true. Units: meters. 
# - sig_depth_thresh: Specify the lower depth limit of density depth average. 
#                     Units: meters.
# - g: Specify the acceleration due to gravity. Units: m/s^2.
# - rmsd_thresh: Threshold for significant overturn. Units: kg/m^3. 
# - threshold_frac: Threshold for continuity of isopycnal surfaces. 
#                   Units: fraction of time series.
#
# ------------ # 

# Set processing parameters
option_proc          = 'density'
option_interp        = 'Pchip'

# Set physical parameters 
rmsd_thresh      = 1e-3
threshold_frac   = 0.75

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set path to project data directory
PATH_data = ROOT / "data" / "mitgcm" / "mooring"

# Parameter verification
if option_proc not in ('vel', 'density'):
    raise ValueError(
        f"Invalid option_proc: {option_proc}"
    )

if option_interp not in ('Pchip', 'linear'):
    raise ValueError(
        f"Invalid option_interp: {option_interp}"
    )

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------

# --- Velocity --- # 
if option_proc == 'vel':

    # Obtain filename paths
    filename_u = PATH_data / "UVEL_CCS_hrly_mooring.nc"
    filename_v = PATH_data / "VVEL_CCS_hrly_mooring.nc"

    # Generate the nc data structure
    nc_u = Dataset(filename_u, 'r')
    nc_v = Dataset(filename_v, 'r')

    # Extract data variables
    site   = nc_u['site'][:]
    depth  = nc_u.variables['Z'][:]
    lon    = nc_u.variables['XC'][:]
    lat    = nc_u.variables['YC'][:]
    time   =  num2date(nc_u.variables['time'][:], nc_u.variables['time'].units)

    u  = nc_u.variables['UVEL'][:]
    v  = nc_v.variables['VVEL'][:]

    # Mask dry cells previously set to NaN during preprocessing
    u_m = np.ma.masked_invalid(u)
    v_m = np.ma.masked_invalid(v)

# --- Density --- # 
elif option_proc == 'density':

    # Obtain filename paths
    filename_temp = PATH_data / "THETA_CCS_hrly_mooring.nc"
    filename_salt = PATH_data / "SALT_CCS_hrly_mooring.nc"

    # Generate the nc data structure
    nc_temp = Dataset(filename_temp, 'r')
    nc_salt = Dataset(filename_salt, 'r')

    # Extract data variables
    site = nc_temp['site'][:]
    depth = nc_temp['Z'][:]
    lon = nc_temp.variables['XC'][:]
    lat = nc_temp.variables['YC'][:]
    time =  num2date(nc_temp.variables['time'][:], nc_temp.variables['time'].units)

    T = nc_temp.variables['THETA'][:]
    S = nc_salt.variables['SALT'][:]

    # Mask dry cells previously set to NaN during preprocessing
    T_m = np.ma.masked_invalid(T)
    S_m = np.ma.masked_invalid(S)

# Convert cftime.DatetimeGregorian to Python datetime objects
time_dt = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time])

# -----------------------------------------------------------------------------
# Process horizontal velocity components (u,v)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Process Density Variables (T, S, sigma0)
# -----------------------------------------------------------------------------

if option_proc == 'density': 

    #------------------------------------------# 
    # Compute Potential Density  
    #------------------------------------------# 

    # Set the dimensions of the array
    nsite, ntime, ndepth = T_m.shape

    # Compute pressure once for each site and depth
    pressure_site_depth = np.array([gsw.p_from_z(depth, lat[i]) for i in range(nsite)]) 

    # Broadcast pressure, lon, and lat to shape of full array
    pressure = np.broadcast_to(pressure_site_depth[:, None, :], (nsite, ntime, ndepth))
    lon3d = np.broadcast_to(lon[:, None, None], (nsite, ntime, ndepth))
    lat3d = np.broadcast_to(lat[:, None, None], (nsite, ntime, ndepth))

    # Compute Absolute Salinity
    SA = gsw.SA_from_SP(S_m, pressure, lon3d, lat3d)

    # Compute Conservative Temperature
    CT = gsw.CT_from_pt(SA, T_m)

    # Compute potential density anomaly (sigma0)
    sigma0 = gsw.sigma0(SA, CT)

    #------------------------------------------# 
    # Compute Buoyancy Frequency using TEOS-10 GSW
    #------------------------------------------# 

    # Compute time-mean hydrographic profiles
    SA_mean = np.ma.mean(SA, axis=1)
    CT_mean = np.ma.mean(CT, axis=1)

    # Initialize arrays
    Nsquare   = np.ma.masked_all((nsite, ndepth - 1))
    depth_mid = np.ma.masked_all((nsite, ndepth - 1))

    # Loop through mooring sites
    for isite in range(nsite):

        # Compute buoyancy frequency
        Nsquare[isite, :], p_mid = gsw.Nsquared(
            SA_mean[isite, :],
            CT_mean[isite, :],
            pressure_site_depth[isite, :],
            lat[isite],
        )

        # Compute the mid-depth 
        depth_mid[isite, :] = gsw.z_from_p(p_mid,lat[isite])

    # Compute bouyancy frequency in units of cycles/hour
    N = np.sqrt(Nsquare) / (2 * np.pi) * 3600 

    #------------------------------------------# 
    # Transform to Isopycnal Surfaces
    #------------------------------------------# 

    # Find the index of the top 200 meters 
    idx_depth    = depth >= -200
    
    # Extract CT, SA, and sigma0 for the top 200 meters
    CT_upper     = CT[:,:,idx_depth]
    SA_upper     = SA[:,:,idx_depth]
    sigma0_upper = sigma0[:,:,idx_depth]
    depth_upper  = depth[idx_depth]

    # Set the limits for the sigma levels based on the min and max values of sigma0 in 
    # the upper 200 meters (rounded to the nearest 0.1)
    sigma_lims = [
    np.floor(sigma0_upper.min() * 10) / 10,
    np.ceil(sigma0_upper.max() * 10) / 10,
    ]

    # Create the uniform sigma levels for interpolation
    sigma_levels = np.arange(
        sigma_lims[0],
        sigma_lims[1] + 0.1,
        0.1,
    )

   # Set the dimensions 
    ncce, ntime, nsigma = CT_upper.shape[0], CT_upper.shape[1], len(sigma_levels)

    # Initalize arrays
    z_on_sigma = np.ma.masked_all((ncce, ntime, nsigma))
    T_on_sigma = np.ma.masked_all((ncce, ntime, nsigma))
    S_on_sigma = np.ma.masked_all((ncce, ntime, nsigma))

    # Quality control flag (1 = overturn detected, 0 = stable)
    overturn_flag = np.zeros((ncce, ntime), dtype=int)

    # Loop through sites
    for isite in range(ncce): 

        # Loop over time
        for it in range(ntime):

            # Extract depth profiles 
            sigma_prof = sigma0_upper[isite,it,:]
            temp_prof  = CT_upper[isite,it,:]
            sal_prof   = SA_upper[isite,it,:]
            depth_prof = depth_upper

            # Build a valid mask (common to all variables)
            valid_mask = ~(
                np.ma.getmaskarray(sigma_prof)
                | np.ma.getmaskarray(temp_prof)
                | np.ma.getmaskarray(sal_prof)
            )

            if valid_mask.sum() < 3:
                # Not enough valid points to interpolate
                continue

            # Apply mask
            sigma_prof = sigma_prof[valid_mask]
            temp_prof  = temp_prof[valid_mask]
            sal_prof   = sal_prof[valid_mask]
            depth_prof = depth_prof[valid_mask]

            # Sort profiles by density to remove overturning
            sort_idx = np.argsort(sigma_prof)
            sigma_sorted = sigma_prof[sort_idx]
            temp_sorted  = temp_prof[sort_idx]
            sal_sorted   = sal_prof[sort_idx]
            depth_sorted = depth_prof[sort_idx]

            # Quality-control checking for overturning 
            rmsd = np.sqrt(np.nanmean((sigma_prof - sigma_sorted)**2))
            if rmsd > rmsd_thresh:
                # Mark as containing overturns
                overturn_flag[isite,it] = 1 

            # --- Pchip interpolation --- # 
            if option_interp == 'Pchip': 
                z_interp = PchipInterpolator(sigma_sorted, depth_sorted, extrapolate=False)
                T_interp = PchipInterpolator(sigma_sorted, temp_sorted, extrapolate=False)
                S_interp = PchipInterpolator(sigma_sorted, sal_sorted, extrapolate=False)
            # --- Linear interpolation --- # 
            else: 
                z_interp = interp1d(sigma_sorted, depth_sorted, bounds_error=False, fill_value=np.nan)
                T_interp = interp1d(sigma_sorted, temp_sorted, bounds_error=False, fill_value=np.nan)
                S_interp = interp1d(sigma_sorted, sal_sorted, bounds_error=False, fill_value=np.nan)

            # Evaluate at sigma levels
            z_on_sigma[isite,it,:] = np.ma.masked_invalid(z_interp(sigma_levels))
            T_on_sigma[isite,it,:] = np.ma.masked_invalid(T_interp(sigma_levels))
            S_on_sigma[isite,it,:] = np.ma.masked_invalid(S_interp(sigma_levels))

    # Mask nans values in arrays 
    z_on_sigma = np.ma.masked_invalid(z_on_sigma)
    T_on_sigma = np.ma.masked_invalid(T_on_sigma)
    S_on_sigma = np.ma.masked_invalid(S_on_sigma)

    # Initialize array
    z_on_sigma_cont = []
    T_on_sigma_cont = []
    S_on_sigma_cont = []
    isopycnal       = []

    # Loop through cce moorings
    for im in range(ncce): 

        # Count valid points along time for each sigma level
        valid_counts = np.sum(~z_on_sigma.mask[im,:,:], axis=0)

        # Find sigma levels that meet threshold
        valid_levels_idx = np.where(valid_counts >= threshold_frac * ntime)[0]

        # Set limits of continuous region and print limits
        if valid_levels_idx.size > 0:
            sigma_min = sigma_levels[valid_levels_idx[0]]
            sigma_max = sigma_levels[valid_levels_idx[-1]]
            print(f"Continuous sigma range at site {im}: "
                  f"{sigma_min:.1f} - {sigma_max:.1f}")
        else:
            raise ValueError(f"No sigma level at site {im} meets "
                             f"the continuity threshold of {threshold_frac:.2f}.")

        # Find the indices of sigma_levels within the continuous range
        sigma_idx = np.where((sigma_levels >= sigma_min) & (sigma_levels <= sigma_max))[0]

        # Extract data from T_on_sigma, z_on_sigma and S_on_sigma to keep only the 
        # continuous sigma-levels meeting the threshold criteria
        T_on_sigma_cont.append(T_on_sigma[im][:, sigma_idx])
        S_on_sigma_cont.append(S_on_sigma[im][:, sigma_idx])
        z_on_sigma_cont.append(z_on_sigma[im][:, sigma_idx])

        # Set the isopycnals levels 
        isopycnal.append(sigma_levels[sigma_idx])


# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Velocity --- # 
if option_proc == 'vel': 

    #--- Mooring Positions ---# 
    LON = xr.DataArray(data=lon, 
                        dims=['site'],
                        coords=dict(site=site),
                        attrs=dict(
                            description='Longitude for the three CCE mooring sites.',
                            units='degrees'
                            )
    )

    LAT = xr.DataArray(data=lat, 
                        dims=['site'],
                        coords=dict(site=site),
                        attrs=dict(
                            description='Latitude for the three CCE mooring sites.',
                            units='degrees'
                            )
    )

    #--- Velocity Components ---#
    u = xr.DataArray(data=u_m,
                        dims=['site','time','depth'],
                        coords=dict(site=site,time=time_dt,depth=depth),
                        attrs=dict(
                            description='The x-component (zonal) of velocity.',
                            units='m/s'
                        )
    )

    v = xr.DataArray(data=v_m,
                        dims=['site','time','depth'],
                        coords=dict(site=site,time=time_dt,depth=depth),
                        attrs=dict(
                            description='The y-component (meridional) of velocity.',
                            units='m/s'
                        )
    )

    # Create data set from data arrays
    data = xr.Dataset({'LON':LON,'LAT':LAT,'u':u,'v':v})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / "mitgcm_proc_vel_hrly_mooring.nc"

# --- Density --- # 
if option_proc == 'density': 

    # --- Mooring Positions --- # 
    LON = xr.DataArray(data=lon, 
                        dims=['site'],
                        coords=dict(site=site),
                        attrs=dict(
                            description='Longitude for the three CCE mooring sites.',
                            units='degrees'
                            )
    )

    LAT = xr.DataArray(data=lat, 
                        dims=['site'],
                        coords=dict(site=site),
                        attrs=dict(
                            description='Latitude for the three CCE mooring sites.',
                            units='degrees'
                            )
    )

    # --- Sea State Variables --- # 
    Pressure = xr.DataArray(data=pressure, 
                        dims=['site','time','depth'],
                        coords=dict(site=site,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Pressure profile time series for the three CCE mooring sites.',
                            units='dbar'
                            )
    )

    SIG = xr.DataArray(data=sigma0, 
                        dims=['site','time','depth'],
                        coords=dict(site=site,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Potential Density anomaly profile time series for the three CCE mooring sites referenced to the pressure at the sea surface.',
                            units='kg/m^3'
                            )
    ) 

    CTemp = xr.DataArray(data=CT, 
                        dims=['site','time','depth'],
                        coords=dict(site=site,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Conservative temperature profile time series for the three CCE mooring sites.',
                            units='degrees Celcius'
                            )
    ) 

    ASal = xr.DataArray(data=SA, 
                        dims=['site','time','depth'],
                        coords=dict(site=site,time=time_dt,depth=depth),
                        attrs=dict(
                            description='Absolute Salinity profile time series for the three CCE mooring sites.',
                            units='g/kg'
                            )
    )

    N1 = xr.DataArray(data=N[0,:], 
                      dims=['depth_mid_cce1'],
                      coords=dict(depth_mid_cce1=depth_mid[0,:]),
                      attrs=dict(
                            description='Background Buoyancy Frequency profile at the CCE1 mooring sites.',
                            units='cycles/hour'
                            )
    )

    N2 = xr.DataArray(data=N[1,:], 
                      dims=['depth_mid_cce2'],
                      coords=dict(depth_mid_cce2=depth_mid[1,:]),
                      attrs=dict(
                            description='Background Buoyancy Frequency profile at the CCE2 mooring sites.',
                            units='cycles/hour'
                            )
    )

    N3 = xr.DataArray(data=N[2,:], 
                      dims=['depth_mid_cce3'],
                      coords=dict(depth_mid_cce3=depth_mid[2,:]),
                      attrs=dict(
                            description='Background Buoyancy Frequency profile at the CCE3 mooring sites.',
                            units='cycles/hour'
                            )
    )

    CTemp1_sig = xr.DataArray(data=T_on_sigma_cont[0], 
                        dims=['time','isopycnal1'],
                        coords=dict(time=time_dt,isopycnal1=isopycnal[0]),
                        attrs=dict(
                            description='Conservative Temperature profiles time series in isopycnal coordinates for CCE 1 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='deg C'
                            )
    ) 

    CTemp2_sig = xr.DataArray(data=T_on_sigma_cont[1], 
                        dims=['time','isopycnal2'],
                        coords=dict(time=time_dt,isopycnal2=isopycnal[1]),
                        attrs=dict(
                            description='Conservative Temperature profiles time series in isopycnal coordinates for CCE 2 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='deg C'
                            )
    ) 

    CTemp3_sig = xr.DataArray(data=T_on_sigma_cont[2], 
                        dims=['time','isopycnal3'],
                        coords=dict(time=time_dt,isopycnal3=isopycnal[2]),
                        attrs=dict(
                            description='Conservative Temperature profiles time series in isopycnal coordinates for CCE 3 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='deg C'
                            )
    ) 

    ASal1_sig = xr.DataArray(data=S_on_sigma_cont[0], 
                        dims=['time','isopycnal1'],
                        coords=dict(time=time_dt,isopycnal1=isopycnal[0]),
                        attrs=dict(
                            description='Absolute Salinity profiles time series in isopycnal coordinates for CCE 1 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='g/kg'
                            )
    ) 

    ASal2_sig = xr.DataArray(data=S_on_sigma_cont[1], 
                        dims=['time','isopycnal2'],
                        coords=dict(time=time_dt,isopycnal2=isopycnal[1]),
                        attrs=dict(
                            description='Absolute Salinity profiles time series in isopycnal coordinates for CCE 2 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='g/kg'
                            )
    )

    ASal3_sig = xr.DataArray(data=S_on_sigma_cont[2], 
                        dims=['time','isopycnal3'],
                        coords=dict(time=time_dt,isopycnal3=isopycnal[2]),
                        attrs=dict(
                            description='Absolute Salinity profiles time series in isopycnal coordinates for CCE 3 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='g/kg'
                            )
    )

    Z1_sig = xr.DataArray(data=z_on_sigma_cont[0], 
                        dims=['time','isopycnal1'],
                        coords=dict(time=time_dt,isopycnal1=isopycnal[0]),
                        attrs=dict(
                            description='Isopycnal depth profiles time series in isopycnal coordinates for CCE 1 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                            units='m'
                            )
    ) 

    Z2_sig = xr.DataArray(data=z_on_sigma_cont[1], 
                            dims=['time','isopycnal2'],
                            coords=dict(time=time_dt,isopycnal2=isopycnal[1]),
                            attrs=dict(
                                description='Isopycnal depth profiles time series in isopycnal coordinates for CCE 2 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                                units='m'
                                )
        )
    
    Z3_sig = xr.DataArray(data=z_on_sigma_cont[2], 
                            dims=['time','isopycnal3'],
                            coords=dict(time=time_dt,isopycnal3=isopycnal[2]),
                            attrs=dict(
                                description='Isopycnal depth profiles time series in isopycnal coordinates for CCE 3 with ' + str(threshold_frac*100) + ' percent of the time series containing data.',
                                units='m'
                                )
        )

    # Create data set from data arrays
    data = xr.Dataset({'LON':LON,'LAT':LAT,'Pressure':Pressure,'SIG':SIG,'CTemp':CTemp,'ASal':ASal,'N1':N1,'N2':N2,'N3':N3,'CTemp1_sig':CTemp1_sig, 'CTemp2_sig':CTemp2_sig, 'CTemp3_sig':CTemp3_sig, 'ASal1_sig':ASal1_sig, 'ASal2_sig':ASal2_sig, 'ASal3_sig':ASal3_sig, 'Z1_sig':Z1_sig, 'Z2_sig':Z2_sig, 'Z3_sig':Z3_sig})

    # Set file path for saving the netcdf file
    file_path = PATH_data / "processed" / "mitgcm_proc_density_hrly_mooring.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')