# =============================================================================
# Autocorrelation Functions  
# =============================================================================
#
# Description:
#   Functions computing autocorrelation and the integral time scale.  
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-12
# =============================================================================

# --- Biased Estimate of the Autocorrelation --- #
def compute_autocorr_biased(data,t):

    """
    Compute the normalized, biased autocorrelation of a one-dimensional series.

    The autocorrelation is calculated from the anomaly time series (i.e., 
    time-mean removed) using FFT-based convolution through the function
    scipy.signal.correlate.

    For an input time series

        x = [x_0, x_1, ..., x_{N-1}],

    the function first forms the anomaly series

        x'_t = x_t - mean(x).

    It then computes the biased sample autocovariance at lag m as

        R(m) = (1 / N) * sum_t x'_t x'_{t+m},

    where the summation includes only index pairs that overlap at the specified
    lag. The same denominator, ``N``, is used at every lag, which makes this the
    biased autocovariance estimator.

    The autocovariance is converted to the autocorrelation by dividing by its
    zero-lag value:

        rho(m) = R(m) / R(0).

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional array containing the time-series observations.

        This function assumes that all entries are finite. Missing values
        represented by NaN or infinite values should be removed or handled
        before calling the function.

    t : numpy.ndarray
        One-dimensional array containing the time coordinate. 

    Returns
    -------
    autocorrelation : numpy.ndarray
        One-dimensional array of length 2*N - 1, where N = len(x).

        The array contains autocorrelation estimates for positive and negative lags.

    lag : numpy.ndarray
        One-dimensional integer array containing the time lag corresponding to
        each element of autocorrelation.

    Notes
    -----

    The use of method="fft" generally provides better computational
    efficiency than direct summation for long time series. The approximate
    computational complexity is O(N log N), compared with O(N^2) for a direct
    calculation of the full correlation sequence.

    No correction is made for missing observations, irregular sampling,
    trends, or other forms of nonstationarity. The input should therefore
    represent a regularly sampled series for which a lag-based
    autocorrelation is meaningful.

    """

    # Import libraries 
    import numpy as np
    from scipy.signal import correlate

    # Convert the input to a floating-point NumPy array
    data = np.asarray(data, dtype=float)

    # Obtain number of data points
    n_samples = data.size

    # The data must be a 1-dimensional array
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional.")

    # At least two observations are required for a meaningful time-series
    # covariance calculation
    if n_samples < 2:
        raise ValueError("data must contain at least two samples.")

    # Compute anomaly series x'_t = x_t - x_bar
    data_anomaly = data - np.mean(data)

    # Compute the complete linear correlation (i.e.,non-normalized summed products) using the fft approach 
    autocovariance = correlate(
        data_anomaly,
        data_anomaly,
        mode="full",
        method="fft",
    )

    # Apply the biased autocovariance normalization
    autocovariance /= n_samples

    # In a full correlation array of length 2*N - 1, zero lag is at index N - 1.
    zero_lag_index = n_samples - 1
    zero_lag_value = autocovariance[zero_lag_index]

    # Normalize by the zero-lag autocovariance to obtain autocorrelation
    autocorrelation = autocovariance / zero_lag_value

    # Construct the lag array corresponding to the full autocorrelation
    lag_number = np.arange(-(n_samples - 1), n_samples)
    lag = lag_number * abs(t[1]-t[0])

    return autocorrelation, lag


# --- Decorrelation Scale --- #
def compute_decor_scale(autocorrelation,lag):

    """
    Estimate the decorrelation scale from a full autocorrelation (positive and 
    negative lags).

    The autocorrelation is integrated over progressively wider symmetric
    intervals about zero lag. The decorrelation scale is defined as the
    maximum cumulative integral. The function also returns the lag number 
    corresponding to the maximum symmetric integral.

    Parameters
    ----------
    autocorrelation : numpy.ndarray
        Full autocorrelation containing negative and positive lags.

    lag : numpy.ndarray
        Lag coordinate corresponding to each autocorrelation value.

    Returns
    -------
    decor_scale : float
        Maximum symmetric integral of the autocorrelation.

    M_lag : float
        Lag number corresponding to the maximum symmetric integral.
    """

    # Import libraries
    import numpy as np
    from scipy.integrate import cumulative_trapezoid

    # Convert inputs to NumPy arrays
    autocorrelation = np.asarray(autocorrelation, dtype=float)
    lag = np.asarray(lag, dtype=float)

    # Verify that the inputs are compatible
    if autocorrelation.ndim != 1 or lag.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if autocorrelation.size != lag.size:
        raise ValueError("Inputs must have the same length.")

    if not np.all(np.diff(lag) > 0):
        raise ValueError("lag must be strictly increasing.")

    # Locate the zero-lag element
    zero_indices = np.flatnonzero(np.isclose(lag, 0.0))
    center = zero_indices[0]

    # Determine the largest symmetric interval that can be constructed about
    # zero lag
    maximum_radius = min(
        center,
        lag.size - center - 1,
    )

    # Compute the cumulative integral of the autocorrelation over the entire
    # lag domain
    cumulative_integral = cumulative_trapezoid(
        autocorrelation,
        lag,
        initial=0.0,
    )

    # Construct all symmetric integration intervals
    # [-L, L] centered on zero lag
    radius = np.arange(maximum_radius + 1)

    # Compute the integral over every symmetric interval using differences of
    # the cumulative integral
    symmetric_integrals = (
        cumulative_integral[center + radius]
        - cumulative_integral[center - radius]
    )

    # Compute the decorrelation scale as the maximum symmetric integral
    decor_scale = float(np.max(symmetric_integrals))

    # Obtain the lag number which corresponds to the maximum symmetric integral
    M_lag = np.argmax(symmetric_integrals)

    return decor_scale, M_lag

