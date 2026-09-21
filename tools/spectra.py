# =============================================================================
# Spectral Analysis Functions 
# =============================================================================
#
# Description:
#   Functions computing 1D spectra (frequency or wavenummber) and its
#   uncertainty.  
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-15
# =============================================================================

# Import libraries 
import numpy as np
from scipy.signal.windows import hann
from scipy.signal import detrend
from scipy.stats import chi2

#--- Spectral Uncertainties ---#
def spectral_uncertainty(
    alpha, 
    psd, 
    estimator, 
    nseg=None, 
    N=None, 
    M=None
):

    """
    Computes confidence intervals for a power spectral density estimate.

    Parameters
    ----------
    alpha : float
        Significance level between 0 and 1. For a 95% confidence
        interval, set alpha = 0.05.

    psd : numpy.ndarray
        Power spectral density estimate.

    estimator : str
        Spectral estimator used to compute the PSD.

        Options:
            'fft'     : Welch spectral estimate using 50%-overlapping
                        Hann-windowed segments.
            'autocov' : Spectrum computed from the autocovariance.

    nseg : int, optional
        Total number of overlapping Welch segments. Required when
        estimator='fft'.

    N : int, optional
        Number of data points in the full record. Required when
        estimator='autocov'.

    M : int, optional
        Half-width used in the autocovariance spectral estimator.
        Required when estimator='autocov'.

    Returns
    -------
    CI : numpy.ndarray
        Confidence interval with shape (nfreq, 2), where

            CI[:, 0] = lower confidence bound
            CI[:, 1] = upper confidence bound
    """

    # Convert PSD to floating-point array
    psd = np.asarray(psd, dtype=float)

    # Check significance level
    if not 0 < alpha < 1:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")

    # -------------------------------------------------------------------------
    # Compute effective degrees of freedom
    # -------------------------------------------------------------------------

    if estimator == "fft":

        # Check number of Welch segments
        if nseg is None:
            raise ValueError(
                "nseg must be provided when estimator='fft'."
            )

        if not isinstance(nseg, (int, np.integer)) or nseg < 1:
            raise ValueError("nseg must be a positive integer.")

        # Effective degrees of freedom for 50%-overlapping
        # Hann-windowed segments
        nu = 36 * nseg**2 / (19 * nseg - 1)

    elif estimator == "autocov":

        # Check required parameters
        if N is None or M is None:
            raise ValueError(
                "N and M must be provided when estimator='autocov'."
            )

        if N <= 0 or M <= 0:
            raise ValueError("N and M must be positive.")

        # Effective degrees of freedom for autocovariance estimator
        nu = (8 / 3) * (N / M)

    else:

        raise ValueError(
            "estimator must be either 'fft' or 'autocov'."
        )

    # -------------------------------------------------------------------------
    # Compute confidence interval
    # -------------------------------------------------------------------------

    # Multiplicative confidence bounds
    ci_low  = nu / chi2.ppf(1 - alpha / 2, nu)
    ci_high = nu / chi2.ppf(alpha / 2, nu)

    # Scale confidence bounds by PSD
    CI = np.column_stack((
        ci_low * psd,
        ci_high * psd,
    ))

    return CI

#--- 1D Power Spectrum with the Welch Method ---#
def compute_spectrum1D(
    data,
    dt,
    M,
    units,
    segment_preprocess="detrend",
):

    """
    Computes the 1D power spectral density using the Welch method.

    The input record is divided into M non-overlapping base segments.
    Additional segments are added with approximately 50% overlap,
    producing 2*M - 1 total Welch segments. Each segment is optionally
    detrended, demeaned, or left unchanged before applying a normalized
    Hann window.

    The function is written notationally for a time series, but can
    also be applied to evenly spaced spatial data.

    Parameters
    ----------
    data : array_like
        One-dimensional time or spatial data series. Data must be
        evenly spaced and contain no NaNs or masked values.

    dt : float
        Time or spatial interval between measurements.

    M : int
        Number of non-overlapping base segments. With 50% overlap,
        the final number of Welch segments is 2*M - 1.

    units : str
        Frequency units.

        Options:
            'cyclic'    : Cyclic frequency.
            'angular' : Angular frequency.

    segment_preprocess : str, optional
        Processing applied independently to each segment before
        windowing.

        Options:
            'detrend' : Remove a least-squares linear trend.
            'demean'  : Remove the segment mean.
            'none'    : Do not remove a trend or mean.

        Default is 'detrend'.

    Returns
    -------
    psd : numpy.ndarray
        One-sided power spectral density.

    f : numpy.ndarray
        Non-negative frequency vector in the units specified by
        ``units``.

    CI : numpy.ndarray
        95% confidence interval with shape (nfreq, 2).

    variance : numpy.ndarray
        Mean-square power computed in the time and frequency domains:

            variance[0] = time-domain mean-square power
            variance[1] = frequency-domain integrated PSD

        These quantities should agree to numerical precision.
    """

    # -------------------------------------------------------------------------
    # Check inputs
    # -------------------------------------------------------------------------

    # Check for masked values
    if np.ma.isMaskedArray(data):

        if np.any(np.ma.getmaskarray(data)):
            raise ValueError(
                "data must not contain masked values."
            )

    # Convert data to floating point
    data = np.asarray(data, dtype=float)

    # Check data dimensions
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional.")

    # Check for NaNs or infinite values
    if not np.all(np.isfinite(data)):
        raise ValueError(
            "data must not contain NaN or infinite values."
        )

    # Check sampling interval
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            "dt must be a positive finite scalar."
        )

    # Check number of base segments
    if not isinstance(M, (int, np.integer)) or M < 1:
        raise ValueError(
            "M must be a positive integer."
        )

    # Check frequency units
    if units not in ("cyclic", "angular"):
        raise ValueError(
            "units must be either 'cyclic' or 'angular'."
        )

    # Check segment preprocessing option
    if segment_preprocess not in ("detrend", "demean", "none"):
        raise ValueError(
            "segment_preprocess must be "
            "'detrend', 'demean', or 'none'."
        )

    # -------------------------------------------------------------------------
    # Set fundamental parameters for computing spectrum
    # -------------------------------------------------------------------------

    # Set number of data points in entire record
    N = len(data)

    # Set number of data points within each segment
    p = N // M

    # Require at least two data points per segment
    if p < 2:
        raise ValueError(
            "Each segment must contain at least two data points. "
            "Decrease M."
        )

    # Compute non-negative cyclic frequencies
    f = np.fft.rfftfreq(p, d=dt)

    # Convert to angular frequency if requested
    if units == "angular":
        f *= 2 * np.pi

    # Set number of non-negative frequencies
    L = len(f)

    # Compute frequency resolution
    df = f[1] - f[0]

    # -------------------------------------------------------------------------
    # Segment data with 50% overlap
    # -------------------------------------------------------------------------

    # Number of segments including 50% overlap
    nseg_target = 2 * M - 1

    # Compute starting index of each segment
    ind_start = (np.arange(nseg_target) * p) // 2

    # Segment data
    data_seg_n = np.column_stack([
        data[ind:ind + p]
        for ind in ind_start
    ])

    # Determine actual number of segments
    nseg = data_seg_n.shape[1]

    # -------------------------------------------------------------------------
    # Preprocess and window each segment
    # -------------------------------------------------------------------------

    # Compute periodic Hann window for FFT-based spectral analysis
    window = hann(p, sym=False)

    # Normalize window so that its mean-square value is unity (preserve mean-square power)
    window *= np.sqrt(p / np.sum(window**2))

    # Preallocate processed and windowed segments
    data_seg_w = np.empty(data_seg_n.shape,dtype=float)

    # Loop through segments
    for iseg in range(nseg):

        # Extract segment
        data_seg = data_seg_n[:, iseg]

        # Remove linear trend
        if segment_preprocess == "detrend":

            data_seg = detrend(data_seg,type="linear")

        # Remove mean
        elif segment_preprocess == "demean":

            data_seg = data_seg - np.mean(data_seg)

        # Apply normalized Hann window
        data_seg_w[:, iseg] = data_seg * window

    # -------------------------------------------------------------------------
    # Compute 1D power spectral density
    # -------------------------------------------------------------------------

    # Preallocate spectral estimates for each segment
    psd_seg = np.zeros((L, nseg),dtype=float)

    # Preallocate time-domain mean-square power (equivalent to the variance here)
    var_seg_time = np.zeros(nseg,dtype=float)

    # Loop through segments
    for iseg in range(nseg):

        # Compute one-sided Fourier transform
        fft_data_seg = np.fft.rfft(data_seg_w[:, iseg])

        # Compute squared Fourier amplitudes
        amp = np.abs(fft_data_seg)**2

        # Convert to power spectral density (PSD)
        amp_norm = amp / p**2 / df

        # Convert two-sided spectral power to a one-sided PSD
        if p % 2 == 0:
            amp_norm[1:-1] *= 2
        else:
            amp_norm[1:] *= 2

        # Save segment PSD
        psd_seg[:, iseg] = amp_norm

        # Compute mean-square power in the time domain
        var_seg_time[iseg] = np.mean(data_seg_w[:, iseg]**2)

    # -------------------------------------------------------------------------
    # Average spectra and compute diagnostics
    # -------------------------------------------------------------------------

    # Compute mean Welch spectrum
    psd = np.mean(psd_seg,axis=1)

    # Initialize variance array
    variance = np.zeros(2)

    # Mean time-domain power across segments
    variance[0] = np.mean(var_seg_time)

    # Integrated spectral power
    variance[1] = np.sum(psd) * df

    # -------------------------------------------------------------------------
    # Compute 95% confidence interval
    # -------------------------------------------------------------------------

    CI = spectral_uncertainty(
        alpha=0.05,
        psd=psd,
        estimator="fft",
        nseg=nseg,
    )

    return psd, f, CI, variance


#--- Spectral Slope ---#
def spectral_slope(
    f, 
    psd, 
    fmin, 
    fmax
):

    """
    Compute the spectral slope of a power spectral density over a specified
    frequency range using an unweighted least-squares fit in log-log space.

    The fitted model is

        log10(S) = b + m * log10(f)

    where m is the spectral slope and S is the power spectral density function.

    Parameters
    ----------
    f : array_like
        Cyclic frequencies corresponding to the PSD.

    psd : array_like
        Power spectral density.

    fmin : float
        Lower frequency bound of the fitting range.

    fmax : float
        Upper frequency bound of the fitting range.

    Returns
    -------
    slope : float
        Spectral slope in log-log space.

    slope_stde : float
        Standard error of the spectral slope based on the residual variance
        of the unweighted least-squares fit.

    fit : ndarray
        Fitted PSD values in linear space.

    f_range : ndarray
        Frequencies used in the fit, in linear space.
    """

    # Convert input arrays to floating point
    f = np.asarray(f, dtype=float)
    psd = np.asarray(psd, dtype=float)

    # Check input dimensions
    if f.ndim != 1 or psd.ndim != 1:
        raise ValueError("f and psd must be one-dimensional.")

    # Check input lengths
    if len(f) != len(psd):
        raise ValueError("f and psd must have the same length.")

    # Check frequency bounds
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise ValueError("fmin and fmax must be finite.")

    if fmin <= 0:
        raise ValueError("fmin must be greater than zero.")

    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin.")

    # -------------------------------------------------------------------------
    # Select frequency range and transform to log-log space
    # -------------------------------------------------------------------------

    # Find valid data points for fitting
    idx = (
        (f >= fmin)
        & (f <= fmax)
        & np.isfinite(f)
        & np.isfinite(psd)
        & (f > 0)
        & (psd > 0)
    )

    # Extract selected frequencies and PSD values
    f_range   = f[idx]
    psd_range = psd[idx]

    # Number of spectral points used in fit
    nfit = len(f_range)

    if nfit < 3:
        raise ValueError(
            "At least three valid spectral points are required "
            "to estimate the slope and its uncertainty."
        )

    # Transform to log-log space 
    x = np.log10(f_range)
    y = np.log10(psd_range)

    # -------------------------------------------------------------------------
    # Compute unweighted least-squares fit
    # -------------------------------------------------------------------------

    # Design matrix for y = b + m*x
    A = np.column_stack((np.ones(nfit), x,))

    # Solve least-squares problem
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # Extract the spectral slope
    slope = coef[1]

    # Compute fitted spectrum in log space
    fit_log = A @ coef

    # Convert fitted spectrum back to linear space
    fit = 10**fit_log

    # -------------------------------------------------------------------------
    # Compute slope uncertainty
    # -------------------------------------------------------------------------

    # ------------
    # --- Note ---
    # ------------
    #
    # The spectral slope is estimated from a linear least-squares fit in
    # log-log space:
    #
    #     y = b + m*x
    #
    # where
    #
    #     x = log10(f)
    #     y = log10(E)
    #
    # and m is the spectral slope. The residual for each spectral point is
    #
    #     e_i = y_i - yfit_i
    #
    # The residual variance about the fitted line is estimated as
    #
    #     var_fit = sum(e_i**2) / (nfit - 2)
    #
    # where nfit - 2 is the number of residual degrees of freedom because
    # two parameters, the intercept and slope, are estimated from the data.
    #
    # The spread of the frequencies in log space is quantified by
    #
    #     Sxx = sum((x_i - mean(x))**2)
    #
    # A larger Sxx provides greater leverage for constraining the slope.
    # The standard error of the fitted spectral slope is therefore
    #
    #     slope_stde = sqrt(var_fit / Sxx)
    #
    # Thus, the slope uncertainty increases when the spectral points have
    # greater scatter about the fitted power law and decreases when the
    # fitted frequency range spans a broader range in log-frequency space.
    # 
    # The standard error of the fitted log-log spectral slope 
    # quantifies how uncertain the fitted spectral slope is because the 
    # spectral points do not lie perfectly on a straight line in log-log space.
    #
    # ------------

    # Set degrees of freedom
    dof = nfit - 2

    # Compute the residual between the log data and the log fit
    residual = y - fit_log

    # Compute the residual variance (vertical scatter)
    var_res = np.sum(residual**2) / dof

    # Compute the spread of the frequencies in log-frequency space
    Sxx = np.sum((x - np.mean(x))**2)

    # Compute the standard error of the fitted log-log spectral slope 
    slope_stde = np.sqrt(var_res / Sxx)

    return slope, slope_stde, fit, f_range


#--- Compute spectral diagnostics ---#
def spectral_diags(
    psd, 
    f, 
    f_cutoff=None,
):

    """
    Compute spectral moments, partitioned variance ratios, and mean period.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density.

    f : array_like
        Frequency vector corresponding to the PSD.

    f_cutoff : float, optional
        Frequency separating the low- and high-frequency bands used to
        compute the partitioned variance ratio.

    Returns
    -------
    moments : ndarray
        Spectral moments [m0, m1, m2, m3], where

            mn = integral(f**n * PSD(f) df)

    FVE : ndarray
        Fraction of the total variance contained in the low- and
        high-frequency bands:

            variance_ratio[0] = low-frequency variance / total variance
            variance_ratio[1] = high-frequency variance / total variance

        If f_cutoff is None, both values are NaN.

    mean_period : float
        Mean spectral period computed from the zeroth and first
        spectral moments:

            mean_period = m0 / m1

        This corresponds to the inverse of the spectral centroid (mean spectral frequency).
    """

    # Convert input arrays to floating point
    psd = np.asarray(psd, dtype=float)
    f = np.asarray(f, dtype=float)

    # Check input dimensions
    if psd.ndim != 1 or f.ndim != 1:
        raise ValueError("psd and f must be one-dimensional.")

    # Check input lengths
    if len(psd) != len(f):
        raise ValueError("psd and f must have the same length.")

    # Check for invalid values
    if not np.all(np.isfinite(psd)) or not np.all(np.isfinite(f)):
        raise ValueError("psd and f must contain only finite values.")

    # Check for negative PSD values
    if np.any(psd < 0):
        raise ValueError("psd must be non-negative.")

    # Require at least two frequencies
    if len(f) < 2:
        raise ValueError("At least two frequency bins are required.")

    # Check that frequencies are strictly increasing
    if np.any(np.diff(f) <= 0):
        raise ValueError("f must be strictly increasing.")

    # Compute frequency resolution
    df = f[1] - f[0]

    # Check for uniform frequency spacing
    if not np.allclose(np.diff(f), df):
        raise ValueError("f must be uniformly spaced.")

    # -------------------------------------------------------------------------
    # Compute spectral moments
    # -------------------------------------------------------------------------

    # Zeroth through third spectral moments
    m0 = np.sum(psd) * df
    m1 = np.sum(f * psd) * df
    m2 = np.sum((f**2) * psd) * df
    m3 = np.sum((f**3) * psd) * df

    # Combine spectral moments
    moments = np.array([m0, m1, m2, m3])

    # -------------------------------------------------------------------------
    # Compute partitioned variance ratio
    # -------------------------------------------------------------------------

    # Initialize fraction of variance explained
    FVE = np.full(2, np.nan)

    if f_cutoff is not None:

        # Check frequency cutoff
        if not np.isfinite(f_cutoff):
            raise ValueError("f_cutoff must be finite.")

        if f_cutoff < f[0] or f_cutoff >= f[-1]:
            raise ValueError(
                "f_cutoff must lie within the frequency range."
            )

        # Define low- and high-frequency bands
        low_mask = f <= f_cutoff
        high_mask = f > f_cutoff

        # Compute variance in each frequency band
        var_low = np.sum(psd[low_mask]) * df
        var_high = np.sum(psd[high_mask]) * df

        # Normalize by total variance
        if m0 > 0:
            FVE[0] = (var_low / m0) * 100
            FVE[1] = (var_high / m0) * 100

    # -------------------------------------------------------------------------
    # Compute mean spectral period
    # -------------------------------------------------------------------------

    if m1 > 0:

        # Compute the energy weighted mean frequency 
        mean_freq = m1/m0

        # Compute the associated mean spectral period
        mean_period = 1/mean_freq

    else:

        mean_period = np.nan

    return moments, FVE, mean_period

#--- Generate 1D data from a power law spectrum ---# 
def generate_powerlaw_data(
    N=2**12, 
    alpha=2.0, 
    random_state=None, 
    dt=1.0,
):
    
    """
    Generate a synthetic data record with a power-law spectrum S(f) ~ f^(-alpha).
    Normalized PSD so that the variance of the time series matches Parseval's theorem.

    Parameters
    ----------
    N : int
        Length of the time series (preferably a power of 2 for FFT efficiency).
    alpha : float
        Spectral slope (e.g., alpha=0 white noise, alpha=1 pink noise, alpha=2 red noise).
    random_state : int or None
        Seed for reproducibility.
    dt : float
        Sampling interval (arbitrary units).

    Returns
    -------
    t : ndarray
        Time or spatial array (0..N-1).
    x : ndarray
        Generated data record.
    f : ndarray
        Frequencies corresponding to PSD (cycles per unit)
    psd : ndarray
        Power spectral density of the generated series.
    """

    #-----------------------------------------------------------------------
    # Set frequencies, amplitudes, and phases for Fourier Coefficients 
    #-----------------------------------------------------------------------

    # Create a new random number generator object for phases (for reproducable results)
    rng = np.random.default_rng(random_state)

    # Set frequencies for FFT (nonnegative with length N//2 + 1 from 0 to nyquist frequency)
    freqs = np.fft.rfftfreq(N, d=dt)  # assume unit sampling interval

    # Avoid divide-by-zero at f=0
    freqs[0] = 1e-6  

    # Power-law amplitude scaling 
    amplitude = freqs**(-alpha / 2.0)

    ###################
    # Note
    # ----
    # We need scale the Fourier amplitudes so that when squared (for computing the power spectrum), 
    # they follow the desired f^(-alpha) power law. Recall the power spectrum is square of the 
    # Fourier coefficients
    # 
    # S(f) = |X(f)|^2
    # 
    # Therefore, in order for S(f) ~ f^(-alpha), we need: 
    # 
    # |X(f)|^2 = f^(-alpha)  ->  |X(f)| = (f^(-alpha))^1/2 = f^(-alpha/2)
    ###################

    # Generate random phases uniformly distributed [0, 2pi)
    phases = rng.uniform(0, 2*np.pi, size=freqs.shape)

    ###################
    # Note
    # ----
    # The power spectrum S(f) tells us how much variance lives at each frequency but it does
    # not tell us what the waveform looks like. To actually construct a time series, you need
    # the complex Fourier coefficients: 
    # 
    # X(f) = |X(f)| e^(i phi(f)) = |X(f)| (cos(phi(f)) + i * sin(phi(f)))
    # 
    # where |X(f)| are the amplitudes of the Fourier coefficients and phi(f) are the phases. 
    # The phases must be randomized to ensure that energy is spread out in time in a
    # realistic, stochastic way. That is, to ensure create a statistically stationary time series
    # that has no artificial coherence (e.g., if the phases were fixed at the same value, at 
    # at the beginning of the record, there would be a perfectly aligned sum of sinusoids that
    # might look like a standing wave). 
    ###################

    #-----------------------------------------------------------------------
    # Compute Fourier Coefficients and build spectrum 
    #-----------------------------------------------------------------------

    # Complex Fourier coefficients
    fourier_coeffs = amplitude * np.exp(1j * phases)

    # Enforce reality conditions
    fourier_coeffs[0] = amplitude[0]               # DC component real
    if N % 2 == 0:
        fourier_coeffs[-1] = amplitude[-1]         # Nyquist real

    #-----------------------------------------------------------------------
    # Compute data record
    #-----------------------------------------------------------------------

    # Generate time or space vector
    t = np.arange(N)

    # Inverse FFT to time or space domain
    x = np.fft.irfft(fourier_coeffs, n=N)

    # Normalize to unit variance and zero mean 
    x = (x - np.mean(x)) / np.std(x)
    
    ###################
    # Note
    # ----
    # We normalize to unit variance so the data record's realizations are comparable. 
    # We can then rescale to whatever variance we need based on observations. 
    ###################

    #-----------------------------------------------------------------------
    # Compute Variance preserving PSD
    #-----------------------------------------------------------------------

    # Set spectral parameters
    df = 1 / (N * dt)
    L = N // 2 + 1 if N % 2 == 0 else (N + 1) // 2

    # Detrend time series
    x_dt = x - np.mean(x) #detrend(x)
    
    # Compute FFT of the time series
    fft_data = np.fft.fft(x_dt) 
    
    # Take squared modulus of the Fourier coefficients
    amp = np.abs(fft_data) ** 2
    
    # Grab positive frequencies for single-sided PSD
    amp_pos = amp[:L]
    
    # Double the amplitude for positive frequencies to conserve variance
    if N % 2 == 0:
        amp_pos[1:-1] *= 2
    else:
        amp_pos[1:] *= 2
    
    # Normalize power spectral density
    psd = amp_pos / (N**2 * df)

    return t, x, freqs, psd

