# =============================================================================
# Processing Functions  
# =============================================================================
#
# Description:
#   Functions for processing mitgcm data.   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-14
# =============================================================================

# Import libraries 
import numpy as np

# --- Transect line Bearing Angle --- # 
def compute_bearing_angle(point1, point2):
    """
    Calculate the bearing angle from point1 to point2 along a transect.

    Parameters
    ----------
    point1, point2 : tuple
        Coordinates in (latitude, longitude) format.

    Returns
    -------
    bearing_angle
        Bearing in degrees clockwise from north.
    """

    # Convert from degrees to radians 
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)

    # Compute difference between station 1 and 2 
    delta_lon = lon2 - lon1

    # Compute the east-west component of the initial great-circle direction
    x = np.sin(delta_lon) * np.cos(lat2)

    # Compute the north-south component  of the initial great-circle direction
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

    # Compute the bearing angle of the transect
    bearing_angle = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

    return bearing_angle
