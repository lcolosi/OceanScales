# =============================================================================
# Processing MITgcm data for the Rossby Radius Analysis
# =============================================================================
#
# Description:
#   Computes intermediate derived variables from the model diagnostics for the 
#   Rossby Radius Analysis. These include: 
# 
#       (1) The Baroclinic and Barotropic Rossby Deformation Radii 
#       (2) The Root-Mean-Square Depth-average Velocity  
#       (3) The Advection Time Scale 
#       (4) The Froude Number 
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-01
# =============================================================================

# Import python libraries 
import os
import sys
from pathlib import Path
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
import gsw
from tqdm import tqdm

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set paths to project directories
PATH_data = ROOT / "data"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import python toolboxes
from ocean_analysis import compute_rossby_modes
from plotting import status

status(f"Starting MITgcm pre-processing for the Rossby Radius Analysis...")
# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------# 
# --- Note ---# 
# ------------#
#
# - option_depth_avg: Depth averaging option for the velocity fields. Options are:
#                     "full" for full-water-column depth average, or "upper" for 
#                     upper-ocean depth average to a specified depth.
# - option_rms: RMS velocity calculation option. Options are:
#               "with_mean" for RMS velocity including the mean flow, or
#               "without_mean" for RMS velocity excluding the mean flow.
# - option_froude: Froude number calculation method. Options are:
#                  "eigenvalue" for Froude number based on the first baroclinic Rossby wave phase speed, or
#                  "dispersion" for Froude number based on the long-wave limit of the Rossby wave dispersion relation.
# - depth_avg_threshold: Specifies the maximum depth at which the depth-average
#                        velocity is computed to.  
# - nmode : Specifies the number of vertical modes to compute in the Rossby
#           deformation radius calculation. For example, nmode = 4 computes the
#           first four vertical modes: 
#                    mode 0 = barotropic
#                    mode 1 = first baroclinic
#                    mode 2 = second baroclinic
#                    mode 3 = third baroclinic
# - nz_mode : Specifies the number of vertical points used in finite-difference
#             eigenproblem.   
# - Omega : Earth's rotation rate (rad/s)
# - Re : Earth's mean radius (m)
#
# ------------# 

# Set processing parameters
option_depth_avg    = 'upper'  
option_rms          = 'with_mean'
option_froude       = 'eigenvalue'
depth_avg_threshold = 200 
nmode               = 4
nz_mode             = 512 

# Set physical constants 
Omega = 7.2921*10**(-5) 
Re    = 6.371*10**(6)   

# Set path to regional pre-processed data directory
PATH_preproc = PATH_data / "mitgcm" / "regional"

# Validate processing parameters
if option_depth_avg not in {"full", "upper"}:
    raise ValueError(
        "option_depth_avg must be either 'full' or 'upper'"
    )

# Validate processing parameters
if option_rms not in {"with_mean", "without_mean"}:
    raise ValueError(
        "option_rms must be either 'with_mean' or 'without_mean'"
    )

# -----------------------------------------------------------------------------
# Load mitgcm data netcdf files 
# -----------------------------------------------------------------------------
status("Loading time/seasonal-mean hydrographic and depth average fields...")

# ------------------------------------------ #
# Background Temperature, Salinity and Density Fields
# ------------------------------------------ #

# Obtain filename paths
filename = PATH_preproc / f"SEA-STATE_CCS_reg_background.nc"

# Generate the nc data structure
nc = Dataset(filename, 'r')

# Extract coordinate variables
depth  = nc.variables['Z'][:]
lon    = nc.variables['XC'][:]
lat    = nc.variables['YC'][:]
season = nc.variables['season'][:]

# Extract time-mean hydrographic variables
SA_mean      = nc.variables['SA_mean'][:]
CT_mean      = nc.variables['CT_mean'][:]
sigma0_mean  = nc.variables['sigma0_mean'][:]

# Extract seasonal hydrographic variables
SA_season      = nc.variables['SA_season'][:]
CT_season      = nc.variables['CT_season'][:]
sigma0_season  = nc.variables['sigma0_season'][:]

# Extract local bathymetric depth
water_depth = nc.variables["water_depth"][:]

# Close input file
nc.close()

# ------------------------------------------ #
# Depth-Averaged Velocity Fields
# ------------------------------------------ #

# Obtain filename paths
if option_depth_avg == 'full':
    filename_uvel = PATH_preproc / f"UVEL_CCS_hrly_reg_full_depth_avg.nc"
    filename_vvel = PATH_preproc / f"VVEL_CCS_hrly_reg_full_depth_avg.nc"
elif option_depth_avg == 'upper':
    filename_uvel = PATH_preproc / f"UVEL_CCS_hrly_reg_depth_avg_upper_{depth_avg_threshold:g}m.nc"
    filename_vvel = PATH_preproc / f"VVEL_CCS_hrly_reg_depth_avg_upper_{depth_avg_threshold:g}m.nc"

# Generate the nc data structure
nc_uvel = Dataset(filename_uvel, 'r')
nc_vvel = Dataset(filename_vvel, 'r')

# Extract coordinate variables
time = num2date(nc_uvel.variables["time"][:],units=nc_uvel.variables["time"].units)

# Extract time-mean velocity variables
u_depth_avg = nc_uvel.variables['uvel_depth_avg'][:]
v_depth_avg = nc_vvel.variables['vvel_depth_avg'][:]

# Close input file
nc_uvel.close()
nc_vvel.close()

# Mask dry cells previously set to NaN during preprocessing
SA_mean_m     = np.ma.masked_invalid(SA_mean)
CT_mean_m     = np.ma.masked_invalid(CT_mean)
sigma0_mean_m = np.ma.masked_invalid(sigma0_mean)

SA_season_m     = np.ma.masked_invalid(SA_season)
CT_season_m     = np.ma.masked_invalid(CT_season)
sigma0_season_m = np.ma.masked_invalid(sigma0_season)

water_depth_m = np.ma.masked_invalid(water_depth)

u_depth_avg_m  = np.ma.masked_invalid(u_depth_avg)
v_depth_avg_m  = np.ma.masked_invalid(v_depth_avg)

# -----------------------------------------------------------------------------
# Compute the Buoyancy Frequency 
# -----------------------------------------------------------------------------
status("Computing time-mean and seasonal-mean buoyancy frequency...")

# Number of vertical levels
ndepth = len(depth)

#------------------------------------------#
# Background Buoyancy Frequency
#------------------------------------------#

# Compute the pressure field 
pressure = gsw.p_from_z(
    depth[:, None, None],
    lat[None, :, None]
)

# Set latitude array with dimensions compatible with the time-mean fields
lat_mean = lat[None, :, None]

# Compute the background buoyancy frequency from time-average
Nsquare_mean, pressure_mid_mean = gsw.Nsquared(
    SA_mean_m,
    CT_mean_m,
    pressure,
    lat=lat_mean,
    axis=0,
)

# Mask invalid values
Nsquare_mean = np.ma.masked_invalid(Nsquare_mean)

# Convert midpoint pressure back to vertical position
depth_mid_mean = gsw.z_from_p(
    pressure_mid_mean,
    lat_mean
)

depth_mid_mean = np.ma.masked_invalid(depth_mid_mean)

#------------------------------------------#
# Seasonal Background Buoyancy Frequency
#------------------------------------------#

# Add a dimension for seasons
pressure_season = pressure[None, :, :, :]
lat_season      = lat[None, None, :, None]

# Compute the seasonal background buoyancy frequency from the seasonal average
Nsquare_season, pressure_mid_season = gsw.Nsquared(
    SA_season_m,
    CT_season_m,
    pressure_season,
    lat=lat_season,
    axis=1,
)

# Mask invalid values
Nsquare_season = np.ma.masked_invalid(Nsquare_season)

# Convert midpoint pressure back to vertical position
depth_mid_season = gsw.z_from_p(
    pressure_mid_season,
    lat_season
)

depth_mid_season = np.ma.masked_invalid(depth_mid_season)

#------------------------------------------#
# Compute Background N(z) 
#------------------------------------------#

# Mask statically unstable values before taking the square root
N_mean = np.ma.sqrt(
    np.ma.masked_less(Nsquare_mean, 0.0)
)

N_season = np.ma.sqrt(
    np.ma.masked_less(Nsquare_season, 0.0)
)

# -----------------------------------------------------------------------------
# Compute the Rossby Deformation Radius 
# -----------------------------------------------------------------------------
status("Computing time-mean and seasonal-mean Rossby deformation radii...")

# Set the mode number vector 
mode = np.arange(nmode)

# Set the number of horizontal grid points and seasons
nlat = len(lat)
nlon = len(lon)
nseason = len(season)

# Get land/dry-cell mask
water_depth_mask = np.ma.getmaskarray(water_depth_m)

# Initialize arrays 
phase_speed_mean     = np.full((nmode, nlat, nlon), np.nan, dtype=float)
rossby_radius_mean   = np.full((nmode, nlat, nlon), np.nan, dtype=float)
phase_speed_season   = np.full((nseason, nmode, nlat, nlon), np.nan, dtype=float)
rossby_radius_season = np.full((nseason, nmode, nlat, nlon), np.nan, dtype=float)

# ------------------------------------------#
# Time-mean Rossby Deformation Radius
# ------------------------------------------#

# Loop through longitude 
for ilon in tqdm(range(nlon), desc="Computing Time-mean Rossby Deformation Radius", unit="lon"):

    # Loop through latitude 
    for ilat in range(nlat): 

        # Skip dry cells 
        if water_depth_mask[ilat,ilon]: 
            continue

        # Set input variables for Rossby wave solver 
        z_in   = depth_mid_mean[:,ilat,ilon]
        N_in   = N_mean[:,ilat,ilon]
        lat_in = lat[ilat]
        H_in   = float(water_depth[ilat,ilon])

        # Run the linear Rossby wave solver 
        result = compute_rossby_modes(
                                      z_in,
                                      N_in,
                                      lat_in,
                                      depth_bottom=H_in,
                                      nmodes=nmode - 1,
                                      nz=nz_mode,
                                      return_modes=False,
                                ) 

        # Save variables 
        phase_speed_mean[:,ilat,ilon] = np.concatenate((
            [result["c_barotropic"]],
            result["c_baroclinic"],
        ))

        rossby_radius_mean[:,ilat,ilon] = np.concatenate((
            [result["Rd_barotropic"]],
            result["Rd_baroclinic"],
        ))

# ------------------------------------------ #
# Seasonal-mean Rossby Deformation Radius
# ------------------------------------------ #

# Loop through seasons
for iseason in tqdm(range(nseason), desc="Computing Seasonal-mean Rossby Deformation Radius", unit="season"):

    # Loop through longitude 
    for ilon in range(nlon): 

        # Loop through latitude 
        for ilat in range(nlat):

            # Skip dry cells 
            if water_depth_mask[ilat,ilon]: 
                continue

            # Set input variables for Rossby wave solver 
            z_in   = depth_mid_season[iseason, :, ilat, ilon]
            N_in   = N_season[iseason,:,ilat,ilon]
            lat_in = lat[ilat]
            H_in   = float(water_depth[ilat,ilon])

            # Run the linear Rossby wave solver 
            result = compute_rossby_modes(
                                            z_in,
                                            N_in,
                                            lat_in,
                                            depth_bottom=H_in,
                                            nmodes=nmode - 1,
                                            nz=nz_mode,
                                            return_modes=False,
                                    ) 
    
            # Save variables 
            phase_speed_season[iseason,:,ilat,ilon] = np.concatenate((
                [result["c_barotropic"]],
                result["c_baroclinic"],
            ))
    
            rossby_radius_season[iseason,:,ilat,ilon] = np.concatenate((
                [result["Rd_barotropic"]],
                result["Rd_baroclinic"],
            ))

# Convert outputs to masked arrays 
phase_speed_mean_m     = np.ma.masked_invalid(phase_speed_mean)
rossby_radius_mean_m   = np.ma.masked_invalid(rossby_radius_mean)
phase_speed_season_m   = np.ma.masked_invalid(phase_speed_season)
rossby_radius_season_m = np.ma.masked_invalid(rossby_radius_season)

# Convert Rossby deformation radius from m to km
rossby_radius_mean_m   = rossby_radius_mean_m / 1000.0
rossby_radius_season_m = rossby_radius_season_m / 1000.0

# -----------------------------------------------------------------------------
# Compute the RMS velocity 
# -----------------------------------------------------------------------------
status("Computing full-record and seasonal RMS velocity...")

# ------------------------------------------#
# Full-record RMS Velocity
# ------------------------------------------#

# Compute the time-mean of the depth-average velocity components 
u_mean = np.ma.mean(u_depth_avg_m, axis=0)
v_mean = np.ma.mean(v_depth_avg_m, axis=0) 

# Compute mean speed 
U_mean = np.sqrt(u_mean**2 + v_mean**2)

# Compute the root-mean-square of velocity components 
if option_rms == 'with_mean':

    u_rms = np.sqrt(np.ma.mean(u_depth_avg_m**2,axis=0))
    v_rms = np.sqrt(np.ma.mean(v_depth_avg_m**2,axis=0))

elif option_rms == 'without_mean':

    u_rms = np.sqrt(np.ma.mean((u_depth_avg_m - u_mean)**2,axis=0))
    v_rms = np.sqrt(np.ma.mean((v_depth_avg_m - v_mean)**2,axis=0))

# Compute the speed from the RMS velocity components
U_rms = np.sqrt(u_rms**2 + v_rms**2)

# Convert RMS speed from m/s to km/day
U_rms_kmday = U_rms * 86400.0 / 1000.0

# ------------------------------------------#
# Seasonal RMS Velocity
# ------------------------------------------#

# Convert time coordinate to month number
month = np.array([t.month for t in time])

# Define months belonging to each season
season_months = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

# Define the season label list
season_labels = []

# Loop through seasons
for s in season:

    # Decode season label if it is a bytes object
    if isinstance(s, bytes):
        season_labels.append(s.decode())

    else:
        season_labels.append(str(s))

# Initialize seasonal arrays
u_mean_season = np.ma.masked_all((nseason, nlat, nlon),dtype=float)
v_mean_season = np.ma.masked_all((nseason, nlat, nlon),dtype=float)
u_rms_season = np.ma.masked_all((nseason, nlat, nlon),dtype=float)
v_rms_season = np.ma.masked_all((nseason, nlat, nlon),dtype=float)

# Loop through seasons
for iseason, season_name in enumerate(season_labels):

    # Obtain months corresponding to the current season
    months = season_months[season_name]

    # Find hourly samples belonging to the current season
    idx_season = np.isin(month,months)

    # Extract seasonal velocity components
    u_season = u_depth_avg_m[idx_season, :, :]
    v_season = v_depth_avg_m[idx_season, :, :]

    # Compute seasonal-mean velocity components
    u_mean_season[iseason, :, :] = np.ma.mean(u_season,axis=0)
    v_mean_season[iseason, :, :] = np.ma.mean(v_season,axis=0)

    # Compute seasonal RMS velocity components
    if option_rms == 'with_mean':

        u_rms_season[iseason, :, :] = np.sqrt(np.ma.mean(u_season**2,axis=0))
        v_rms_season[iseason, :, :] = np.sqrt(np.ma.mean(v_season**2,axis=0))

    elif option_rms == 'without_mean':

        u_rms_season[iseason, :, :] = np.sqrt(np.ma.mean((u_season - u_mean_season[iseason, :, :])**2,axis=0))
        v_rms_season[iseason, :, :] = np.sqrt(np.ma.mean((v_season - v_mean_season[iseason, :, :])**2,axis=0))

# Compute seasonal mean speed
U_mean_season = np.sqrt(
    u_mean_season**2
    + v_mean_season**2
)

# Compute seasonal RMS speed
U_rms_season = np.sqrt(
    u_rms_season**2
    + v_rms_season**2
)

# Convert RMS speeds from m/s to km/day
U_rms_season_kmday = U_rms_season * 86400.0 / 1000.0

# -----------------------------------------------------------------------------
# Compute the Advection Time Scale 
# -----------------------------------------------------------------------------
status("Computing full-record and seasonal advection time scale...")

# Set first baroclinic Rossby deformation radius
Rd1        = rossby_radius_mean_m[1, :, :]
Rd1_season = rossby_radius_season_m[:, 1, :, :]

# Compute advection time scale in days
T_adv        = Rd1 / U_rms_kmday
T_adv_season = Rd1_season / U_rms_season_kmday

# -----------------------------------------------------------------------------
# Compute the Froude Number
# -----------------------------------------------------------------------------
status("Computing full-record and seasonal Froude number...")

if option_froude == 'eigenvalue':

    # Set the First-baroclinic internal gravity-wave phase speed (m/s)
    c1 = phase_speed_mean_m[1, :, :]
    c1_season = phase_speed_season_m[:, 1, :, :]

    # Compute the Froude number
    Fr = U_rms / c1
    Fr_season = U_rms_season / c1_season

elif option_froude == 'dispersion':

    # Convert first-baroclinic deformation radius from km to m
    Rd1_m = Rd1 * 1000.0
    Rd1_season_m = Rd1_season * 1000.0

    # Convert latitude to radians
    phi = np.deg2rad(lat)[:, None]

    # Compute the beta parameter (m^-1 s^-1)
    beta = 2.0 * Omega * np.cos(phi) / Re

    # Compute the long-wave limit of the Rossby wave phase speed (m/s)
    rossby_phase_velocity = -beta * Rd1_m**2
    rossby_phase_velocity_season = -beta[None, :, :] * Rd1_season_m**2

    # Convert phase speed magnitude from m/s to km/day
    c_p_kmday = np.ma.abs(rossby_phase_velocity) * 86400.0 / 1000.0
    c_p_season_kmday = np.ma.abs(rossby_phase_velocity_season) * 86400.0 / 1000.0

    # Compute the Froude number 
    Fr = U_rms_kmday / c_p_kmday
    Fr_season = U_rms_season_kmday / c_p_season_kmday

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Rossby Deformation Radius and Phase Speed --- # 
ROSSBY_RADIUS_full = xr.DataArray(data=rossby_radius_mean_m,
                        dims=['mode','lat','lon'],
                        coords=dict(mode=mode, lat=lat, lon=lon),
                        attrs=dict(
                            description=('Rossby deformation radii of long mode-m gravity waves computed from solving the linear Rossby wave eigenproblem. Rossby radii computed from the full record time-mean buoyancy frequency profile.'),
                            units='km'
                        )
)

ROSSBY_RADIUS_season = xr.DataArray(data=rossby_radius_season_m,
                        dims=['season','mode','lat','lon'],
                        coords=dict(season=season, mode=mode, lat=lat, lon=lon),
                        attrs=dict(
                            description=('Rossby deformation radii  of long mode-m gravity waves computed from solving the linear Rossby wave eigenproblem. Rossby radii computed from the seasonal-mean buoyancy frequency profile.'),
                            units='km'
                        )
)

PHASE_SPEED_full = xr.DataArray(data=phase_speed_mean_m,
                        dims=['mode','lat','lon'],
                        coords=dict(mode=mode, lat=lat, lon=lon),
                        attrs=dict(
                            description=('Phase speed of long mode-m gravity waves computed from solving the linear Rossby wave eigenproblem. Phase speed computed from the full record time-mean buoyancy frequency profile.'),
                            units='m/s'
                        )
)

PHASE_SPEED_season = xr.DataArray(data=phase_speed_season_m,
                        dims=['season','mode','lat','lon'],
                        coords=dict(season=season, mode=mode, lat=lat, lon=lon),
                        attrs=dict(
                            description=('Phase speed of long mode-m gravity waves computed from solving the linear Rossby wave eigenproblem. Phase speed computed from the seasonal-mean buoyancy frequency profile.'),
                            units='m/s'
                        )
)

# --- Root-Mean-Square Speed and Mean Velocity --- # 
U_RMS_full = xr.DataArray(data=U_rms,
                        dims=['lat','lon'],
                        coords=dict(lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Root-mean-square (RMS) speed computed from the full record depth-average velocity field. RMS speed computed {"including" if option_rms == "with_mean" else "excluding"} the time-mean flow.'),
                            units='m/s'
                        )
)

U_RMS_season = xr.DataArray(data=U_rms_season,
                        dims=['season','lat','lon'],
                        coords=dict(season=season, lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Root-mean-square (RMS) speed computed from the seasonal depth-average velocity field. RMS speed computed {"including" if option_rms == "with_mean" else "excluding"} the seasonal-mean flow.'),
                            units='m/s'
                        )
)

UVEL_full = xr.DataArray(data=u_mean,
                        dims=['lat','lon'],
                        coords=dict(lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Time-mean Zonal velocity component.'),
                            units='m/s'
                        )
)

VVEL_full = xr.DataArray(data=v_mean,
                        dims=['lat','lon'],
                        coords=dict(lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Time-mean Meridional velocity component.'),
                            units='m/s'
                        )
)

UVEL_season = xr.DataArray(data=u_mean_season,
                        dims=['season','lat','lon'],
                        coords=dict(season=season, lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Seasonal-mean Zonal velocity component.'),
                            units='m/s'
                        )
)

VVEL_season = xr.DataArray(data=v_mean_season,
                        dims=['season','lat','lon'],
                        coords=dict(season=season, lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Seasonal-mean Meridional velocity component.'),
                            units='m/s'
                        )
)

# --- Advection Time Scale --- # 
T_ADV_full = xr.DataArray(data=T_adv,
                        dims=['lat','lon'],
                        coords=dict(lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Advection time scale computed from the full record depth-average velocity field and the first baroclinic Rossby deformation radius.'),
                            units='days'
                        )
)

T_ADV_season = xr.DataArray(data=T_adv_season,
                        dims=['season','lat','lon'],
                        coords=dict(season=season, lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Advection time scale computed from the seasonal depth-average velocity field and the first baroclinic Rossby deformation radius.'),
                            units='days'
                        )
)

# --- Froude Number --- # 
Fr_full = xr.DataArray(data=Fr,
                        dims=['lat','lon'],
                        coords=dict(lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Froude number computed from the full record depth-average velocity field and the first baroclinic Rossby deformation radius.'),
                            units='unitless'
                        )
)

Fr_season = xr.DataArray(data=Fr_season,
                        dims=['season','lat','lon'],
                        coords=dict(season=season, lat=lat, lon=lon),
                        attrs=dict(
                            description=(f'Froude number computed from the seasonal depth-average velocity field and the first baroclinic Rossby deformation radius.'),
                            units='unitless'
                        )
)

# Create data set from data arrays 
data = xr.Dataset({'ROSSBY_RADIUS_full':ROSSBY_RADIUS_full,'ROSSBY_RADIUS_season':ROSSBY_RADIUS_season, 'PHASE_SPEED_full':PHASE_SPEED_full, 'PHASE_SPEED_season':PHASE_SPEED_season, 'U_RMS_full':U_RMS_full, 'U_RMS_season':U_RMS_season, 'UVEL_full':UVEL_full, 'VVEL_full':VVEL_full, 'UVEL_season':UVEL_season, 'VVEL_season':VVEL_season, 'T_ADV_full':T_ADV_full, 'T_ADV_season':T_ADV_season, 'Fr_full':Fr_full, 'Fr_season':Fr_season})

# Set global variables to document the processing parameters used 
data.attrs.update({
    "Depth_average_range": option_depth_avg,
    "Depth_average_threshold": depth_avg_threshold,
    "RMS_mean_inclusion": option_rms,
    "Froude_number_computation": option_froude,
    "Number_of_Rossby_modes": nmode,
    "Mode_solver_vertical_points": nz_mode,
})

# Set file path for saving the netcdf file
file_path = (
    PATH_preproc
    / "processed" / f"mitgcm_advection_time_scale_reg_{option_depth_avg}_{depth_avg_threshold}m_{option_rms}.nc"
)

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')

status("MITgcm Rossby radius processing complete!")
