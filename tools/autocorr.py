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

# Import libraries 
import numpy as np
from scipy.signal import correlate
from scipy.integrate import cumulative_trapezoid
import mpmath as mp
from datetime import timedelta

# --- Biased Estimate of the Autocorrelation --- #
def compute_autocorr_biased(
    data,
    t,
):

    """
    Function for computing the normalized, biased autocorrelation of a one-dimensional
    series.

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

    # Convert the input to a floating-point NumPy array
    data = np.asarray(data, dtype=float)

    # Obtain number of data points
    n_samples = data.size

    # The data must be a 1-dimensional array
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional.")

    # Mask data set to identify missing data points
    data_ma = np.ma.masked_invalid(data)

    # FFT autocorrelation method requires regularly sampled data (no missing data)
    if np.any(np.ma.getmaskarray(data_ma)):
        raise ValueError(
            "FFT autocorrelation requires a complete regularly sampled segment."
        )

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
def compute_decor_scale(
    autocorrelation,
    lag,
):

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

# --- Uncertainty Estimate of the Decorrelation Scale --- # 
def compute_decor_scale_unc(
    autocorr_mean,
    autocorr_seg,
    M_lag,
    dt,
    overlap=0.5,
):
    """
    Estimate the uncertainty of a decorrelation scale computed from the
    ensemble-mean autocorrelation.

    Parameters
    ----------
    autocorr_mean : array-like
        Mean autocorrelation. May have shape (nlags,), (nlags, 1), or
        (1, nlags). The autocorrelation is assumed to contain negative
        and positive lags, with zero lag at the center.

    autocorr_seg : array-like
        Autocorrelations from individual time-series segments. May have
        shape (nlags, nseg) or (nseg, nlags).

    M_lag : int
        Index of the cutoff lag used to compute the decorrelation scale,
        relative to zero lag.

    dt : float
        Sampling interval.

    overlap : float, optional
        Fractional overlap between adjacent segments. Must satisfy
        0 <= overlap < 1. Default is 0.5.

    Returns
    -------
    decor_scale_stdm : float
        Standard error of the decorrelation scale computed from the mean
        autocorrelation, accounting approximately for dependence between
        overlapping segments.

    decor_scale_std : float
        Standard deviation of decorrelation scales for individual
        realizations, estimated from the projected autocorrelation
        deviations.
    """

    # Convert inputs to arrays
    autocorr_mean = np.asarray(autocorr_mean).squeeze()
    autocorr_seg = np.asarray(autocorr_seg)

    # Check dimensions
    if autocorr_mean.ndim != 1:
        raise ValueError(
            "autocorr_mean must contain a single autocorrelation."
        )

    if autocorr_seg.ndim != 2:
        raise ValueError("autocorr_seg must be a 2D array.")

    # Check overlap
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    # Number of lags in the full two-sided autocorrelation
    nlags = autocorr_mean.size

    # Ensure the mean autocorrelation has an odd number of lags
    if nlags % 2 == 0:
        raise ValueError(
            "autocorr_mean must contain a two-sided autocorrelation with "
            "an odd number of lags."
        )

    # Ensure autocorr_seg has dimensions (nlags, nseg)
    if autocorr_seg.shape[0] == nlags:
        pass
    elif autocorr_seg.shape[1] == nlags:
        autocorr_seg = autocorr_seg.T
    else:
        raise ValueError(
            "Neither dimension of autocorr_seg matches the length of "
            "autocorr_mean."
        )

    # Determine number of segments
    nseg = autocorr_seg.shape[1]

    if nseg < 2:
        raise ValueError(
            "At least two segments are required to estimate uncertainty."
        )

    # Zero-lag index of a two-sided autocorrelation
    zero_lag_index = (nlags - 1) // 2

    # Extract zero and positive lags
    autocorr_mean_pos = autocorr_mean[zero_lag_index:]
    autocorr_seg_pos = autocorr_seg[zero_lag_index:, :]

    # Number of non-negative lags
    ntime_seg = autocorr_mean_pos.size

    # Ensure cutoff index is valid
    cutoff_index = int(M_lag)

    if not 0 <= cutoff_index < ntime_seg:
        raise ValueError(
            f"M_lag must be between 0 and {ntime_seg - 1}."
        )

    # Construct weights for the doubled trapezoidal integral
    weights = np.zeros(ntime_seg, dtype=float)

    if cutoff_index > 0:
        weights[0] = dt
        weights[cutoff_index] = dt
        weights[1:cutoff_index] = 2.0 * dt

    # Compute deviations from the mean autocorrelation
    autocorr_delta = autocorr_seg_pos - autocorr_mean_pos[:, None]

    # Compute the integral of each segment's deviation from the mean
    q = weights @ autocorr_delta

    # Compute the variance and standard deviation across segments
    projected_var = np.var(q, ddof=1)
    decor_scale_std = np.sqrt(projected_var)

    # Approximate effective number of independent segments
    nseg_eff = nseg / (
        1.0
        + 2.0
        * (1.0 - 1.0 / nseg)
        * overlap
    )

    # Compute standard error of the decorrelation scale from the mean
    decor_scale_stdm = decor_scale_std / np.sqrt(nseg_eff)

    return decor_scale_stdm, decor_scale_std


# --- Analytic Solution of the Autocorrelation --- # 
def autocorrelation_analytic(
    tau, 
    fmin, 
    fmax, 
    alpha,
):
    """
    Function for computing the analytic autocovariance and 
    autocorrelation for a power-law spectrum using the 
    upper incomplete gamma function.

    This combines:
        - Upper incomplete gamma evaluation (using analytic continuation for s < 1)
        - Autocovariance computation
        - Autocorrelation computation 

    Parameters
    ----------
    tau : array-like
        Time lags.
    fmin : float
        Minimum frequency.
    fmax : float
        Maximum frequency.
    alpha : float
        Power-law exponent.

    Returns
    -------
    R : ndarray (complex)
        Autocovariance function.
    rho : ndarray (float)
        Autocorrelation function.
    R0 : float
        Variance (R at zero lag).
    """

    #------------------------------------------# 
    # Upper Incomplete Gamma Function 
    #------------------------------------------# 
    def upper_incomplete_gamma(s, x):
        """
        Function for computing the upper incomplete gamma function Gamma(s,x).

        Using the generalize incomplete gamma function, which allows you to specify both
        the upper and lower limits of integration for the incomplete gamma function, 
        we are evaluating: 

        Gamma(s,z_1,z_2) = int_{z_1}^{z_2} t^{s - 1} e^{-t} dt

        where z_2 = infty (mp.inf), and z_1 = x (function parameter). Parameter s can
        be any real or complex value. For Re(s) < 1, the incomplete gamma function
        does not converge so analytic continuation is used to evaluate the integral.
        Singulatiries exist at zero and negative integers for Gamma(s) and gamma(s,x),
        but not for Gamma(s,x) because singluarities for Gamma(s) and gamma(s,x)
        cancel each other.   

        Gamma(s)         = (complete) Gamma function
        Gamma(s,x)       = Upper incomplete gamma function 
        gamma(s,x)       = Lower incomplete gamma function 
        Gamma(s,z_1,z_2) = generalized incomplete gamma function 

        Parameters
        ----------
        s : float or complex
            Shape parameter.
        x : float or complex
            Lower limit of integration.

        Returns
        -------
        val : complex
            Value of the upper incomplete gamma function.

        """

        # Compute upper incomplete gamma function for given s and x parameters
        upper_gamma = mp.gammainc(s, x, mp.inf)

        return upper_gamma

    #------------------------------------------# 
    # Evaulation of Analytic Integral
    #------------------------------------------# 
    def analytic_integral(f_lim, tau):
        """
        Function for evaluating the analytic solution of the autocovariance function:

            int_{-2 pi i tau f_{lim}}^{infty} f^{-alpha} e^{2 pi i tau f} df

        Parameters
        ----------
        f_lim : float
            Frequency limit.
        tau : float
            Time lag.
        alpha : float
            Power-law exponent.

        Returns
        -------
        R : complex
            Value of the integral.
            
        """

        # Compute inputs for upper incomplete gamma function 
        s = 1 - alpha                        # Shape Parameter
        x =  -2 * np.pi * 1j * tau * f_lim   # Lower limit of integration 

        # Evaluate integral 
        R = ((-2 * np.pi * 1j * tau) ** (alpha - 1)) * upper_incomplete_gamma(s, x)

        return R

    #------------------------------------------# 
    # Compute Autocovariance R(tau)
    #------------------------------------------# 

    # Initialize autocovariance function
    R = []

    # Loop through time lags 
    for itau in tau:

        # Compute the integral using the upper incomplete gamma function for f_min and f_max lower limits
        R_fmax = analytic_integral(fmax, itau)
        R_fmin = analytic_integral(fmin, itau)

        # Evaluate the integral between f_max and f_min for the ith time lag
        R.append(R_fmin - R_fmax)

    # Convert to a numpy array
    R = np.array([complex(r) for r in R])

    #------------------------------------------# 
    # Compute R(0) (Variance)
    #------------------------------------------# 
    if np.isclose(alpha, 1.0):
        R0 = np.log(fmax / fmin)
    else:
        R0 = (fmax**(1 - alpha) - fmin**(1 - alpha)) / (1 - alpha)

    #------------------------------------------# 
    # Compute Autocorrelaton rho(tau)
    #------------------------------------------# 
    rho = np.real(R) / R0

    return R, rho, R0

# --- Analytic Solution of the Decorrelation Scale --- #
def decorrelation_scale_analytic(
    fmin,
    fmax, 
    alpha,
):
    """
    Function for computing the analytic two-sided decorrelation scale for a power-law
    spectrum.  

    Parameters
    ----------
    fmin : float
        Minimum frequency.
    fmax : float
        Maximum frequency.
    alpha : float
        Power-law exponent.

    Returns
    -------
    T : ndarray (complex)
        Decorrelation scale.
    R0 : float
        Variance (R at zero lag).
    """

    # Compute R(0) (Variance)
    if np.isclose(alpha, 1.0):
        R0 = np.log(fmax / fmin)
    else:
        R0 = (fmax**(1 - alpha) - fmin**(1 - alpha)) / (1 - alpha)

    # Compute decorrelaton scale
    T = 2 * (1/(-2 * np.pi * 1j * alpha * R0)) * (fmin**(-alpha) - fmax**(-alpha)) 

    return np.imag(T), R0

# --- Segment time series --- # 
def segment_time_series(
    time, 
    data, 
    duration=1, 
    overlap=0.5,
):

    """
    Split a time series into overlapping segments.

    Parameters
    ----------
    time : array-like of datetime.datetime
        One-dimensional time coordinate containing Python datetime objects.
    data : ndarray
        Data vector (1D array aligned with `time`).
    duration : float, optional
        Length of each segment in units of 365-day years (default is 1).
        Can be fractional (e.g., 0.5 = 6 months).
    overlap : float, optional
        Fraction of overlap between consecutive segments (0–1).
        For example, 0.5 means 50% overlap.

    Returns
    -------
    segments : list of tuples
        Each entry is (time_segment, data_segment), where:
        - time_segment : ndarray of datetimes for that segment
        - data_segment : ndarray of data values for that segment
    """

    # Convert inputs to arrays
    time = np.asarray(time)
    data = np.asanyarray(data)

    # Validate inputs
    if time.ndim != 1 or data.ndim != 1:
        raise ValueError("time and data must be one-dimensional.")

    if time.size != data.size:
        raise ValueError("time and data must have the same length.")

    if time.size == 0:
        raise ValueError("time and data must not be empty.")

    if duration <= 0:
        raise ValueError("duration must be greater than zero.")

    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    # Set start and end times of the full time series
    start_time = time[0]
    end_time = time[-1]

    # Set step size between the starts of consecutive segments (in years)
    step = duration * (1 - overlap)

    # Store the (time, data) pairs for each segment
    segments = []

    # Initialize the first segment start
    seg_start = start_time

    while True:

        # Define the end time for this segment
        seg_end = seg_start + timedelta(days=365*duration)

        # If the segment would extend beyond the available record, stop
        if seg_end > end_time:
            break

        # Create a mask to select time points within this segment
        mask = (time >= seg_start) & (time < seg_end)

        # Append the selected time and data as one segment
        segments.append((time[mask], data[mask]))

        # Move the start time forward by the step (handles overlap)
        seg_start += timedelta(days=365*step)

    return segments

