# =============================================================================
# Filtering Functions  
# =============================================================================
#
# Description:
#   Functions 1D filtering of time series.   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-14
# =============================================================================

# Import libraries 
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --- Low-Pass Gaussian Filter --- # 
def gaussian_low_pass_filter(data, fwhm_days=365, dt_hours=1, truncate=4, mode="constant"):

    """
    Apply a low-pass Gaussian filter to a one-dimensional masked time series.

    Missing values are accounted for by smoothing both the data and a
    corresponding weight array, then normalizing the filtered data by the
    filtered weights. This prevents missing values from biasing the smoothed
    time series toward zero.

    Parameters
    ----------
    data : array_like or numpy.ma.MaskedArray
        One-dimensional input time series. NaNs and masked values are ignored
        during the smoothing operation.

    fwhm_days : float, optional
        Full width at half maximum (FWHM) of the Gaussian kernel in days.
        Internally, the FWHM is converted to the Gaussian standard deviation
        required by ``scipy.ndimage.gaussian_filter1d`` using

            sigma = FWHM / (2 * sqrt(2 * ln(2)))

        so that the Gaussian kernel has the specified full width at half maximum.
        Default is 365 days.

    dt_hours : float, optional
        Sampling interval of the input data in hours. Default is 1 hour.

    truncate : float, optional
        Truncate the Gaussian kernel at ``truncate`` standard deviations from
        its center. This determines the extent of the kernel; for example,
        ``truncate=4`` gives a kernel extending approximately from -4std to +4std.
        Default is 4.
    
    mode : str, optional
        Method used to extend the data beyond the boundaries of the time
        series. Options include "constant", "nearest", "reflect", "mirror",
        and "wrap". Default is "constant".

    Returns
    -------
    smoothed : numpy.ma.MaskedArray
        Low pass Gaussian-filtered time series with the same shape as the input.
        Locations where no valid observations contribute to the Gaussian
        kernel remain masked.
    """

    # Ensure input is a masked array
    data_ma = np.ma.masked_invalid(data)

    # Convert FWHM to the Gaussian standard deviation (days)
    sigma_days = fwhm_days / (2 * np.sqrt(2 * np.log(2)))

    # Convert from days to samples
    sigma_samples = sigma_days * 24 / dt_hours

    # Replace masked values with zeros
    y = data_ma.filled(0.0)

    # Weight array (1 = valid, 0 = missing)
    weights = (~np.ma.getmaskarray(data_ma)).astype(float)

    # Smooth both the data and the weights
    y_smooth = gaussian_filter1d(
        y,
        sigma=sigma_samples,
        mode=mode,
        cval=0.0,
        truncate=truncate,
    )

    w_smooth = gaussian_filter1d(
        weights,
        sigma=sigma_samples,
        mode=mode,
        cval=0.0,
        truncate=truncate,
    )

    # Normalize to remove the influence of missing values
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = y_smooth / w_smooth

    # Mask locations where no valid observations contributed
    smoothed = np.ma.masked_where(w_smooth <= 0, smoothed)

    return smoothed