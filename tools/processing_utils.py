# =============================================================================
# Processing Functions  
# =============================================================================
#
# Description:
#   Functions for processing mitgcm and CCE data.   
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

# --- Find Longest Masked data gaps --- # 
def longest_masked_gap(mask, dt):
    """
    Find the longest consecutive run of masked samples.

    Parameters
    ----------
    mask : array_like of bool
        True where data are missing.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    gap_duration : float
        Duration of the longest gap in seconds.
    gap_samples : int
        Number of consecutive missing samples in the longest gap.
    """

    # Convert mask to a boolean array 
    mask = np.asarray(mask, dtype=bool)

    # Return zeros if no missing data is present
    if not np.any(mask):
        return 0.0, 0

    # Pad with False so gaps at either end are identified (True = missing data)
    padded = np.concatenate(([False], mask, [False]))

    # Convert padding array to integers (0s and 1s)
    padded_int = padded.astype(int)

    # Compute the difference between adjacent points to identify entering (+1)
    # and exiting (-1) data daps
    changes = np.diff(padded_int)

    # Locate starts and ends of contiguous masked regions
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    # Compute the number of missing samples in each gap
    gap_lengths = stops - starts

    # Find the longest gap
    gap_samples = int(np.max(gap_lengths))

    # Convert the longest gap into time units
    gap_duration = gap_samples * dt

    return gap_duration, gap_samples

