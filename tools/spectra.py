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
    random_amplitudes=False,
    normalization="sample",
    target_variance=1.0,
):
    
    """
    Generate a synthetic data record with a power-law spectrum S(f) ~ f^(-alpha).

    The record is normalized to zero mean and unit variance, and the returned
    one-sided PSD is variance preserving (sum(psd) * df = var(x), Parseval's theorem).

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
    random_amplitudes : bool
        If False, use fixed Fourier amplitudes and random phases such that the
        spectrum of every realization follows the target power law exactly.
        If True, generate independent Gaussian real and imaginary Fourier
        components such that the target power law is the ensemble-mean spectrum.
    normalization : {"sample", "expected", None}
        Method used to scale the variance of the generated record.
        If "sample", each realization is normalized by its own sample
        standard deviation so that its variance equals target_variance
        exactly. If "expected", all realizations are scaled by the same
        deterministic factor so that the ensemble-expected variance equals
        target_variance while allowing finite-realization variance to
        fluctuate naturally. If None, no variance normalization is applied.
    target_variance : float
        Desired variance of the generated process when normalization is
        "sample" or "expected". For "sample", every realization has exactly
        this variance. For "expected", this specifies the ensemble-expected
        variance, while individual realizations may differ from it. Ignored
        when normalization is None.

    Returns
    -------
    t : ndarray
        Time or spatial coordinate, from 0 to (N-1)*dt.
    x : ndarray
        Generated data record.
    f : ndarray
        Frequencies corresponding to PSD (cycles per unit), from 0 to Nyquist.
    psd : ndarray
        One-sided power spectral density of the generated series.

    Notes
    -----
    Two methods are available for generating the Fourier coefficients:

    1. Fixed amplitudes with random phases (random_amplitudes=False)

       Only the Fourier phases are random. The amplitudes are fixed at

           |X(f)| = f^(-alpha/2),

       so the periodogram of every realization follows the target power law
       exactly, apart from the final normalization to unit variance.

    2. Gaussian Fourier coefficients (random_amplitudes=True)

       The real and imaginary parts of each Fourier coefficient are independent
       Gaussian random variables. The coefficients are constructed as

           X(f) = sqrt(S(f)/2) * (a + i*b),

       where a and b are independent standard normal random variables and

           S(f) = f^(-alpha).

       This gives

           E[|X(f)|^2] = S(f),

       so the target power law represents the ensemble-mean spectrum rather
       than the exact spectrum of each realization. Individual realizations
       therefore contain the expected chi-squared sampling variability in
       Fourier power.

    In both cases, the spectrum is band-limited to
    f = 1/(N*dt), ..., 1/(2*dt), with no power at f = 0. The resulting record
    is periodic over its length and contains no unresolved variability at
    periods longer than the record length.

    """

    #-----------------------------------------------------------------------
    # Set frequencies and target power-law spectrum
    #-----------------------------------------------------------------------

    # Create a new random number generator object (for reproducible results)
    rng = np.random.default_rng(random_state)

    # Set non-negative frequencies for FFT 
    freqs = np.fft.rfftfreq(N, d=dt)

    # Initialize target power spectrum
    spectrum = np.zeros_like(freqs)

    # Compute power-law spectrum (leaving zero-frequency with zero power)
    spectrum[1:] = freqs[1:]**(-alpha)

    ###################
    # Note
    # ----
    # The target power spectrum is
    #
    # S(f) ~ f^(-alpha).
    #
    # Because the power at a given Fourier frequency is proportional to the
    # squared magnitude of the Fourier coefficient,
    #
    # S(f) = |X(f)|^2,
    #
    # the corresponding characteristic Fourier amplitude is
    #
    # |X(f)| = sqrt(S(f)) = f^(-alpha/2).
    #
    # For the fixed-amplitude method, this amplitude is imposed exactly at
    # every frequency. For the Gaussian method, S(f) instead specifies the
    # expected squared magnitude of the Fourier coefficient:
    #
    # E[|X(f)|^2] = S(f).
    ###################

    #-----------------------------------------------------------------------
    # Compute Fourier Coefficients and build spectrum 
    #-----------------------------------------------------------------------

    if random_amplitudes:

        # Generate independent standard-normal real and imaginary components
        a = rng.standard_normal(freqs.size)
        b = rng.standard_normal(freqs.size)

        ###################
        # Note
        # ----
        # To generate Gaussian Fourier coefficients, let
        #
        # a, b ~ N(0, 1)
        #
        # be independent standard normal random variables and define
        #
        # X(f) = sqrt(S(f)/2) * (a + i*b).
        #
        # The standard normal variables provide the random Gaussian part,
        # while sqrt(S(f)/2) sets the variance at each frequency according
        # to the target spectrum.
        #
        # Since
        #
        # Re[X(f)] = sqrt(S(f)/2) * a
        # Im[X(f)] = sqrt(S(f)/2) * b,
        #
        # each component has variance S(f)/2. Therefore,
        #
        # E[|X(f)|^2]
        #     = E[Re(X)^2] + E[Im(X)^2]
        #     = S(f)/2 + S(f)/2
        #     = S(f).
        #
        # Both the amplitude and phase therefore vary among realizations.
        # The phase is uniformly distributed from 0 to 2*pi, while the
        # Fourier power fluctuates around S(f).
        ###################

        # Generate complex Gaussian Fourier coefficients
        fourier_coeffs = np.sqrt(spectrum / 2.0) * (a + 1j * b)

        # Enforce zero power at zero frequency
        fourier_coeffs[0] = 0.0

        # The Nyquist coefficient must be real for an even-length real-valued
        # time series. Use variance S rather than S/2 because there is only
        # one independent real component at this frequency.
        if N % 2 == 0:
            fourier_coeffs[-1] = np.sqrt(spectrum[-1]) * a[-1]

    else:

        # Compute fixed Fourier amplitudes
        amplitude = np.sqrt(spectrum)

        # Generate random phases uniformly distributed over [0, 2*pi)
        phases = rng.uniform(0, 2*np.pi, size=freqs.shape)

        ###################
        # Note
        # ----
        # The power spectrum S(f) tells us how much variance lives at each
        # frequency but does not determine the waveform in time. To construct
        # a time series, we need complex Fourier coefficients:
        #
        # X(f) = |X(f)| exp(i*phi(f))
        #      = |X(f)| [cos(phi(f)) + i*sin(phi(f))].
        #
        # Here the Fourier amplitudes are fixed:
        #
        # |X(f)| = sqrt(S(f)) = f^(-alpha/2),
        #
        # while the phases are independently randomized. Therefore,
        #
        # |X(f)|^2 = S(f)
        #
        # for every realization. Different realizations have different
        # waveforms because their phases differ, but their Fourier powers
        # are identical before normalization.
        ###################

        # Generate complex Fourier coefficients
        fourier_coeffs = amplitude * np.exp(1j * phases)

        # Enforce zero power at zero frequency
        fourier_coeffs[0] = 0.0

        # The Nyquist coefficient must be real for an even-length real-valued
        # time series
        if N % 2 == 0:
            fourier_coeffs[-1] = amplitude[-1]

    #-----------------------------------------------------------------------
    # Compute data record
    #-----------------------------------------------------------------------

    # Generate time or space vector
    t = np.arange(N) * dt

    # Inverse FFT to time or space domain
    x = np.fft.irfft(fourier_coeffs, n=N)

    #-----------------------------------------------------------------------
    # Normalize data record
    #-----------------------------------------------------------------------

    # Remove any residual numerical mean
    x = x - np.mean(x)

    if normalization == "sample":

        # Normalize each realization by its own sample variance
        x = x * np.sqrt(target_variance) / np.std(x)

        ###################
        # Note
        # ----
        # Sample normalization forces every realization to have exactly the
        # specified target variance. This is useful when we are interested in
        # differences in temporal or spectral structure rather than differences
        # in the total variance among realizations.
        #
        # Because each realization is divided by its own standard deviation,
        # realization-to-realization variability in the total variance is removed.
        ###################


    elif normalization == "expected":

        # Compute expected variance from the target Fourier spectrum
        if N % 2 == 0:

            # Interior positive frequencies occur as positive/negative pairs,
            # while the Nyquist frequency occurs only once.
            variance_expected = (
                2.0 * np.sum(spectrum[1:-1])
                + spectrum[-1]
            ) / N**2

        else:

            # For odd N, all positive frequencies have corresponding
            # negative-frequency partners.
            variance_expected = (
                2.0 * np.sum(spectrum[1:])
            ) / N**2

        # Apply the same deterministic scaling to every realization
        x *= np.sqrt(target_variance / variance_expected)

        ###################
        # Note
        # ----
        # Expected-variance normalization scales the process so that its
        # ensemble-expected variance equals target_variance:
        #
        # E[var(x)] ~ target_variance.
        #
        # Importantly, the scaling factor is determined only from the prescribed
        # target spectrum and is therefore the same for every realization.
        # Individual finite realizations are NOT forced to have exactly the target
        # variance.
        #
        # For random_amplitudes=True, this preserves the natural sampling
        # variability in total variance among realizations while placing the
        # ensemble on a physically meaningful variance scale.
        ###################


    elif normalization is None:

        ###################
        # Note
        # ----
        # No variance normalization is applied. The variance is determined
        # directly by the numerical magnitude of the prescribed Fourier spectrum.
        #
        # Because S(f) = f^(-alpha) contains no physical normalization constant,
        # the absolute variance in this case generally has arbitrary units and
        # depends on N, dt, alpha, and the frequency range.
        ###################

        pass


    else:
        raise ValueError(
            "normalization must be 'sample', 'expected', or None"
        )

    #-----------------------------------------------------------------------
    # Compute variance-preserving PSD
    #-----------------------------------------------------------------------

    # Set spectral parameters
    df = 1 / (N * dt)

    # Demean time series
    x_dt = x - np.mean(x)
    
    # Compute one-sided FFT of the time series
    fft_data = np.fft.rfft(x_dt) 
    
    # Take squared modulus of the Fourier coefficients
    amp_pos = np.abs(fft_data)**2
    
    # Double power at positive frequencies to conserve variance
    if N % 2 == 0:
        amp_pos[1:-1] *= 2
    else:
        amp_pos[1:] *= 2
    
    # Normalize power spectral density
    psd = amp_pos / (N**2 * df)

    return t, x, freqs, psd
