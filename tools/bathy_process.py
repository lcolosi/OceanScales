# =============================================================================
# Process Bathymetry Data for the Point Conception Study Region 
# =============================================================================
#
# Description:
#   Extracts bathymetry data from the ETOPO1 dataset for the Point Conception study 
#   region and saves data in a smaller netCDF file.
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-12
# =============================================================================

# Import python libraries 
import os
import xarray as xr
import numpy as np

# Set paths to data directory
ROOT = '/Users/lukecolosi/Desktop/projects/graduate_research/Gille_lab/OceanScales/'
PATH_data  = ROOT + 'data/bathymetry/'

# Load bathymetry data 
filename = PATH_data + "topo_25.1.nc"
ds_bathy = xr.open_dataset(filename, engine="netcdf4")

# Extract data variables
lon_b = ds_bathy["lon"]  
lat_b = ds_bathy["lat"]  
bathy = ds_bathy["z"]    

# Convert convesion of longitude (neg from the prime meridian to positive definite wrapping around the earth)
lon_n = lon_b % 360

# Set longitude and latitude bounds for the region of interest
lat_bnds   = [33, 35]                     
lon_bnds   = [-123 % 360, -120 % 360]

# Grid spacing
dlon = float(np.abs(lon_n[1] - lon_n[0]))
dlat = float(np.abs(lat_b[1] - lat_b[0]))

# Add one grid cell around plotting domain
lon_min = lon_bnds[0] - dlon
lon_max = lon_bnds[1] + dlon
lat_min = lat_bnds[0] - dlat
lat_max = lat_bnds[1] + dlat

# Extract data from bathymetry 
lon_grid = lon_n[(lon_n >= lon_min) & (lon_n <= lon_max)]
lat_grid = lat_b[(lat_b >= lat_min) & (lat_b <= lat_max)]
bathy_grid   = bathy[(lat_b >= lat_min) & (lat_b <= lat_max),(lon_n >= lon_min) & (lon_n <= lon_max)]

# Create data arrays
LON = xr.DataArray(data=lon_grid, 
                    dims=['lon'],
                    coords=dict(lon=lon_grid),
                    attrs=dict(
                        description='Longitude coordinate vector for bathymetry.',
                        units='degrees'
                        )
)

LAT = xr.DataArray(data=lat_grid, 
                    dims=['lat'],
                    coords=dict(lat=lat_grid),
                    attrs=dict(
                        description='Latitude coordinate vector for bathymetry.',
                        units='degrees'
                        )
)

BATHY = xr.DataArray(data=bathy_grid, 
                    dims=['lat','lon'],
                    coords=dict(lat=lat_grid,lon=lon_grid),
                    attrs=dict(
                        description='Bathymetry in the Point Conception study region.',
                        units='meters'
                        )
)

# Create data set from data arrays 
data = xr.Dataset({'LON':LON,'LAT':LAT,'BATHY':BATHY})

# Set file path for saving the netcdf file
file_path = PATH_data + "etopo1_point_conception.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')

