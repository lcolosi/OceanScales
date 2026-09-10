# =============================================================================
# Processing CCE Mooring Observation data for the Mooring Analysis
# =============================================================================
#
# Description:
#   Gathers data from multiple files across multiple deployments, vertically
#   interploates onto a regular grid and computes intermediate derived
#   variables for the mooring decorrelation time scale analysis. These variables
#   include: 
# 
#       (1) Conservative Temperature
#       (2) Absolute Salinity
#       (3) Potential Density (referenced to the surface)
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-09
# =============================================================================

# Import python libraries 
import os
from pathlib import Path
import xarray as xr
import numpy as np
from netCDF4 import Dataset, num2date
import gsw

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# - option_mooring: Specifies which cce mooring will be processed. 
#                   Options include: "cce1" or "cce2"
# - depl_start: Specifies the first deployment to analyze. 
# - depl_end: Specifies the last deployment to analyze. 
# - time_bnds: Specifies the time bounds for the analysis period. 
# - lat_mooring : Specifies the mooring latitude position in degrees. 
# - lon_mooring : Specifies the mooring longitude position in degrees. 
# - z_grid: Specifies the depth levels for vertical interpolation of mooring
#           data.
# - z_tol: Specifies the tolerance window for each nominal depth level in
#          z_grid (+/- meters).
#
# ------------ # 

# Set processing parameters
option_mooring = "cce1"   

# Set mooring-specific parameters
if option_mooring == "cce1":

    depl_start = 7
    depl_end = 13

    time_bnds = [
        np.datetime64("2014-01-01T00:00"),
        np.datetime64("2019-12-31T23:00"),
    ]

    lat_mooring = 33.457
    lon_mooring = -122.52233

    z_grid = -np.array(
        [10, 20, 30, 40, 60, 75, 150,
         300, 500, 750, 1000, 2200, 3950],
        dtype=float,
    )

    z_tol = np.array(
        [3, 3, 3, 3, 5, 5, 5,
         10, 10, 20, 30, 30, 50],
        dtype=float,
    )

elif option_mooring == "cce2":

    depl_start = 4
    depl_end = 8

    time_bnds = [
        np.datetime64("2013-01-01T00:00"),
        np.datetime64("2016-12-31T23:00"),
    ]

    lat_mooring = 34.3075
    lon_mooring = -120.8042

    z_grid = -np.array(
        [7, 15, 25, 45, 75, 710],
        dtype=float,
    )

    z_tol = np.array(
        [2, 3, 3, 5, 5, 10],
        dtype=float,
    )

else:
    raise ValueError(
        f"Invalid option_mooring: {option_mooring}"
    )

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set path to project data directory
PATH_data = ROOT / "data" / "cce" 

# Set path to selected CCE mooring
PATH_mooring = PATH_data / option_mooring

# -----------------------------------------------------------------------------
# Process CCE Mooring Data 
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# For each deployment, we loop through all files with the associated deployment
# number. Files either have CTD or CAT in their filename. CAT refers to either 
# MICROCAT or SEACAT which are both CTD instruments manufactored by 
# Seabird-electronics. The MicroCAT is model SBE37 and just measures T, C,
# and optionally p. The 'SeaCAT' is model SBE16 and additionally has external
# ports for plugging in other sensors (such as the pmel instrument). For earlier
# deployments, each instrument at a given depth had a separate file saved to
# the server (i.e., the CAT files). Later deployments had all instruments combined
# into one file (i.e., the CTD files).   
# 
# Additionally, note that the QC flags used to mark bad data include: 
#     0 = no QC performed
#     3 = probably bad
#     4 = bad
#     9 = missing value
#
# ------------ # 

# Initialize lists for concatenating deployments
te_grid = []
sa_grid = []
ti_grid = []

# Loop through deployment files 
for m in range(depl_start, depl_end + 1): 

    # Set Deployment number
    depl = f"{m:02d}"

    # Print Progress statement
    print(f"Processing {option_mooring.upper()} deployment {depl}...")

    # ------------------------------------------ # 
    # Find OceanSITES CTD and CAT files 
    # ------------------------------------------ # 

    # Collect CTD and CAT file names for the ith deployment
    ctd_files = list(PATH_mooring.glob(f"OS_{option_mooring.upper()}_{depl}_*_CTD.nc"))
    cat_files = list(PATH_mooring.glob(f"OS_{option_mooring.upper()}_{depl}_*_*CAT*.nc"))

    # Combine and sort file lists
    nc_files = sorted(set(ctd_files + cat_files))

    # Print no files found if no files exist and skip rest of processing 
    if len(nc_files) == 0:
        print(f"No files found for deployment {depl}. Skipping...")
        continue

    # Print number of files found 
    print(f"Found {len(nc_files)} files.")

    # ------------------------------------------ # 
    # Load instrument data  
    # ------------------------------------------ # 

    # Initialize lists 
    ti = []
    te = []
    sa = []
    depth = []

    # Loop through files   
    for filename in nc_files:

        # Load data for instrument(s)
        with Dataset(filename, "r") as nc:

            # --- Time --- # 
            time_k = num2date(nc.variables["TIME"][:],nc.variables["TIME"].units)

            # Convert to numpy datetime64
            time_k = np.array(time_k,dtype="datetime64[s]")

            # --- Temperature and Salinity --- # 
            temp_var = nc.variables["TEMP"]
            psal_var = nc.variables["PSAL"]

            # Turn off automatic masking because the _FillValue attribute
            # cannot be safely cast to the float32 variable dtype
            temp_var.set_auto_mask(False)
            psal_var.set_auto_mask(False)

            # Read data
            temp_k = temp_var[:].astype(float).squeeze()
            psal_k = psal_var[:].astype(float).squeeze()

            # --- QC flags --- # 
            temp_qc = nc.variables["TEMP_QC"][:].astype(int).squeeze()
            psal_qc = nc.variables["PSAL_QC"][:].astype(int).squeeze()

            # --- Instrument depth --- # 
            depth_k = nc.variables["DEPTH"][:].astype(float)

            # Make sure depth is always a 1D array 
            depth_k = np.atleast_1d(depth_k)

        # ------------------------------------------ # 
        # Arange temperature/salinity into (depth,time) form  
        # ------------------------------------------ # 

        # Single-depth CAT file
        if temp_k.ndim == 1:

            temp_k = temp_k[np.newaxis, :]
            psal_k = psal_k[np.newaxis, :]

            temp_qc = temp_qc[np.newaxis, :]
            psal_qc = psal_qc[np.newaxis, :]

        # Multi-depth CTD file: currently (time, depth)
        elif temp_k.shape == (len(time_k), len(depth_k)):

            temp_k = temp_k.T
            psal_k = psal_k.T

            temp_qc = temp_qc.T
            psal_qc = psal_qc.T

        # Check resulting dimensions
        if temp_k.shape != (len(depth_k), len(time_k)):
            raise ValueError(
                f"Unexpected TEMP shape in {filename.name}: "
                f"{temp_k.shape}"
            )

        # ------------------------------------------ # 
        # Apply QC flags
        # ------------------------------------------ # 

        # Define QC flags that mark missing/bad data  
        bad_qc = [0, 3, 4, 9]

        # Apply QC
        temp_k[np.isin(temp_qc, bad_qc)] = np.nan
        psal_k[np.isin(psal_qc, bad_qc)] = np.nan

        # Some files use negative fill values instead of QC flags
        temp_k[temp_k < 0.0] = np.nan
        psal_k[psal_k < 0.0] = np.nan

        # ------------------------------------------ # 
        # Store each depth as an individual time series
        # ------------------------------------------ #

        # Loop through depths
        for idepth in range(len(depth_k)):

            # Append time array
            ti.append(time_k)

            # Append temperature, salinity, and depth to list
            te.append(temp_k[idepth, :])
            sa.append(psal_k[idepth, :])
            depth.append(depth_k[idepth])

        # ------------------------------------------ # 
        # Check native temporal resolution  
        # ------------------------------------------ # 

        if len(time_k) > 1:

            # Compute the median dt in units of hours
            dt_native = np.nanmedian(np.diff(np.sort(time_k))/ np.timedelta64(1, "h"))

            # Print results
            print(
                f"  {filename.name}: "
                f"{len(depth_k)} depths, "
                f"dt = {dt_native:.2f} hr"
            )

    # ------------------------------------------ # 
    # Merge depths onto the native time array 
    # ------------------------------------------ #

    # Merge all unique observation time 
    time = np.unique(np.concatenate(ti))

    # Initialize arrays
    T = np.full((len(depth), len(time)),np.nan)
    S = np.full((len(depth), len(time)),np.nan)

    # Loop through depths
    for idepth in range(len(depth)):

        # Find locations of the instruments timestamps
        ii = np.searchsorted(time,ti[idepth])

        # Insert observations at ith depth at observation time steps
        T[idepth, ii] = te[idepth]
        S[idepth, ii] = sa[idepth]

    # ------------------------------------------ # 
    # Sort depths from shallow to deep  
    # ------------------------------------------ #

    # Convert to an array 
    depth = np.asarray(depth,dtype=float)

    # Sort shallow to deep
    ii = np.argsort(depth)

    depth = depth[ii]
    T = T[ii, :]
    S = S[ii, :]

    # Convert to negative-downward z coordinate
    depth = -depth

    # ------------------------------------------ #
    # Compute hourly averages
    # ------------------------------------------ #

    # Assign each observation to its corresponding hour
    time_hour = time.astype("datetime64[h]")

    # Find unique hourly time steps
    time_avg = np.unique(time_hour)

    # Initialize hourly averaged arrays
    T_avg = np.full((len(depth), len(time_avg)),np.nan)
    S_avg = np.full((len(depth), len(time_avg)),np.nan)

    # Loop through hourly time steps
    for n in range(len(time_avg)):

        # Find observations within the current hour
        idx_time = time_hour == time_avg[n]

        # Loop through depths
        for idepth in range(len(depth)):

            # Obtain temperature observations during current hour
            temp = T[idepth, idx_time]

            # Average within the hour bin
            if np.any(np.isfinite(temp)):
                T_avg[idepth, n] = np.nanmean(temp)

            # Obtain salinity observations during current hour
            psal = S[idepth, idx_time]

            # Average within the hour bin
            if np.any(np.isfinite(psal)):
                S_avg[idepth, n] = np.nanmean(psal)

    # ------------------------------------------ #
    # Nudge shallowest/deepest depths
    # ------------------------------------------ #

    # ------------ # 
    # --- Note --- # 
    # ------------ #
    #
    # Nudge the shallowest and deepest observations to nearby nominal depths
    # when they fall within the specified tolerance. This preserves the endpoint
    # measurements without extrapolating beyond the observed vertical range, which
    # would require assuming that the vertical T/S gradient continues outside the
    # range of the measurements. 
    #
    # ------------ # 

    # Find nominal depth closest to shallowest observation
    distance = np.abs(depth[0] - z_grid)
    b = np.argmin(distance)

    # Nudge shallowest observation to nominal depth if within tolerance
    if distance[b] <= z_tol[b] and depth[0] < z_grid[b]:
        depth[0] = z_grid[b]

    # Find nominal depth closest to deepest observation
    distance = np.abs(depth[-1] - z_grid)
    b = np.argmin(distance)

    # Nudge deepest observation to nominal depth if within tolerance
    if distance[b] <= z_tol[b] and depth[-1] > z_grid[b]:
        depth[-1] = z_grid[b]

    # ------------------------------------------ #
    # Interpolate onto nominal depth grid
    # ------------------------------------------ #

    # Initialize arrays
    tei = np.full((len(z_grid), len(time_avg)), np.nan)
    sai = np.full((len(z_grid), len(time_avg)), np.nan)

    # Loop through hourly time steps
    for n in range(len(time_avg)):

        # --- Temperature --- #

        # Find valid temperature profile observations for ith time step
        valid = np.isfinite(T_avg[:, n])

        # Check if more than one valid depth exists
        if np.sum(valid) > 1:

            # Grab non-nan depth and temperature data points
            z_valid = depth[valid]
            t_valid = T_avg[valid, n]

            # Sort depths (interpolation requires the depth coordinate to increase)
            ii = np.argsort(z_valid)

            # Linearly interpolate the ith temperature profile (no extrapolation)
            tei[:, n] = np.interp(
                z_grid,
                z_valid[ii],
                t_valid[ii],
                left=np.nan,
                right=np.nan
            )

        # Check if only one observation exists 
        elif np.sum(valid) == 1:

            # Grab depth and temperature data point
            z_obs = depth[valid][0]
            t_obs = T_avg[valid, n][0]

            # Find nominal depth closest to the observation
            distance = np.abs(z_obs - z_grid)
            b = np.argmin(distance)

            # If within depth tolerance, nudge data point to nearest nominal depth
            if distance[b] <= z_tol[b]:
                tei[b, n] = t_obs


        # --- Salinity --- #

        # Find valid salinity profile observations for ith time step
        valid = np.isfinite(S_avg[:, n])

        # Check if more than one valid depth exists
        if np.sum(valid) > 1:

            # Grab non-nan depth and salinity data points
            z_valid = depth[valid]
            s_valid = S_avg[valid, n]

            # Sort depths (interpolation requires the depth coordinate to increase)
            ii = np.argsort(z_valid)

            # Linearly interpolate the ith salinity profile (no extrapolation)
            sai[:, n] = np.interp(
                z_grid,
                z_valid[ii],
                s_valid[ii],
                left=np.nan,
                right=np.nan
            )

        # Check if only one observation exists 
        elif np.sum(valid) == 1:

            # Grab depth and salinity data point
            z_obs = depth[valid][0]
            s_obs = S_avg[valid, n][0]

            # Find nominal depth closest to the observation
            distance = np.abs(z_obs - z_grid)
            b = np.argmin(distance)

            # If within depth tolerance, nudge data point to nearest nominal depth
            if distance[b] <= z_tol[b]:
                sai[b, n] = s_obs


    # ------------------------------------------ #
    # Concatenate deployment
    # ------------------------------------------ #

    # Append interpolated data and time
    te_grid.append(tei)
    sa_grid.append(sai)
    ti_grid.append(time_avg)


# -----------------------------------------------------------------------------
# Concatenate all deployments
# -----------------------------------------------------------------------------

# Concatenate deployments along time dimension
te_grid = np.concatenate(te_grid, axis=1)
sa_grid = np.concatenate(sa_grid, axis=1)
ti_grid = np.concatenate(ti_grid)

# -----------------------------------------------------------------------------
# Create continuous hourly time coordinate
# -----------------------------------------------------------------------------

# Define hourly temporal resolution
dt = np.timedelta64(1, "h")

# Create continuous hourly time coordinate
t = np.arange(np.min(ti_grid), np.max(ti_grid) + dt,dt)

# Initialize final arrays
T = np.full((len(z_grid), len(t)),np.nan)
S = np.full((len(z_grid), len(t)),np.nan)

# ------------------------------------------ #
# Find non-overlapping deployment times
# ------------------------------------------ #

# ------------ # 
# --- Note --- # 
# ------------ #
#
# Overlapping observations from adjacent deployments are excluded rather than
# averaged or offset-corrected. Differences between deployments may reflect
# instrument offsets, small changes in mooring location, or real ocean
# variability, so combining them would require assumptions that are difficult
# to justify. Removing duplicated overlap times provides a conservative way to
# avoid introducing additional artifacts associated with combining the two
# deployment records.
#
# ------------ # 

# Find unique observation times and number of occurrences
unique_time, first_index, counts = np.unique(
    ti_grid,
    return_index=True,
    return_counts=True
)

# Create a mask for removing repeated time steps 
single = counts == 1

# Remove all time steps that appear more than once 
unique_time = unique_time[single]
first_index = first_index[single]

# ------------------------------------------ #
# Insert observations onto regular hourly grid
# ------------------------------------------ #

# Find positions of observation times in regular hourly time array
time_index = np.searchsorted(t,unique_time)

# Insert temperature and salinity
T[:, time_index] = te_grid[:, first_index]
S[:, time_index] = sa_grid[:, first_index]

# Rename nominal depth coordinate
z = z_grid

# -----------------------------------------------------------------------------
# Extract analysis time window
# -----------------------------------------------------------------------------

# Find observations within specified time bounds
idx_time = ((t >= time_bnds[0]) & (t <= time_bnds[1]))

# Extract time, temperature, and salinity
t = t[idx_time]
T = T[:, idx_time]
S = S[:, idx_time]

# -----------------------------------------------------------------------------
# Compute TEOS-10 hydrographic variables
# -----------------------------------------------------------------------------

# Compute pressure at each nominal depth
p = gsw.p_from_z(
    z,
    lat_mooring
)

# Add time dimension for broadcasting with (depth, time) arrays
p = p[:, np.newaxis]

# Compute Absolute Salinity from Practical Salinity
SA = gsw.SA_from_SP(
    S,
    p,
    lon_mooring,
    lat_mooring
)

# Compute Conservative Temperature from in-situ temperature
CT = gsw.CT_from_t(
    SA,
    T,
    p
)

# Compute potential density anomaly referenced to 0 dbar
sigma0 = gsw.sigma0(
    SA,
    CT
)

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# Create DataArrays
CTemp = xr.DataArray(data=CT.T, 
                        dims=['time','depth'],
                        coords=dict(time=t,depth=z),
                        attrs=dict(
                            description=f'Conservative temperature profile time series for the {option_mooring.upper()} mooring sites.',
                            units='kg/m^3'
                            )
    ) 

ASal = xr.DataArray(data=SA.T, 
                        dims=['time','depth'],
                        coords=dict(time=t,depth=z),
                        attrs=dict(
                            description=f'Absolute Salinity profile time series for the {option_mooring.upper()} mooring sites.',
                            units='g/kg'
                            )
    )

SIG = xr.DataArray(data=sigma0.T, 
                        dims=['time','depth'],
                        coords=dict(time=t,depth=z),
                        attrs=dict(
                            description=f'Potential Density anomaly profile time series for the {option_mooring.upper()} mooring sites referenced to the pressure at the sea surface.',
                            units='degree_Celsius'
                            )
    ) 

# Create data set from data arrays
data = xr.Dataset({'CTemp':CTemp,'ASal':ASal,'SIG':SIG})

# Add coordinate metadata
data["depth"].attrs = {
    "long_name": "Nominal depth",
    "units": "m",
    "positive": "up"
}

data["time"].attrs = {
    "long_name": "Time"
}

# Set file path for saving the netcdf file
file_path = PATH_data / "processed" / f"{option_mooring}_proc_density_hrly_mooring.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')
