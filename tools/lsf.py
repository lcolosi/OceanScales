# =============================================================================
# Least Squares Fitting (LSF) Functions  
# =============================================================================
#
# Description:
#   Functions preforming a least-squares fit to a time series.  
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-13
# =============================================================================

# Import libraries 
import numpy as np 

# --- Unweighted Least Squares fit Function --- #
def unweighted_lsf(data, x, parameters, freqs=None, sigma=None, linear_trend=False):
    
    """
    Function for computing an unweighted least squares fit to 1D data using
    an optional linear trend and optional sinusoidal harmonics.

    The fitted model has the form:

        h(x) = a0 + a1*x + sum_n [bn*sin(wn*x) + cn*cos(wn*x)]

    where the linear trend term a1*x is included only if linear_trend=True.

    Parameters
    ----------
    data : masked array
        Data record as a 1D masked array. This data may contain masked values
        at missing data points.

    x : array
        x-coordinate values associated with data. Must have the same length as
        data and cannot contain masked values. The fit is computed only at
        valid data points, but hfit is evaluated on the full x grid.

    parameters : int
        Number of sinusoidal frequencies to fit.

        parameters = 0:
            Fit only a constant if linear_trend=False.
            Fit a linear trend if linear_trend=True.

        parameters = 1:
            Fit one sinusoidal frequency.

        parameters = 2:
            Fit two sinusoidal frequencies.

        etc.

    freqs : array-like or None
        Frequencies to fit, given as:

            [w1, w2, ..., wn]

        where n must be greater than or equal to parameters. If parameters=0,
        freqs may be None.

    sigma : float or None
        Uncertainty in each data measurement. This function currently accepts
        only a scalar value for sigma. If sigma is None, coefficient
        uncertainties are returned as zeros.

    linear_trend : bool
        If True, include a linear trend term in the fit. If False, only include
        the constant offset and sinusoidal terms.

    Returns
    -------
    hfit : array
        Least squares fit evaluated on the full x grid.

    x_data : array
        Coefficients of the fitted model.

        If linear_trend=False and parameters=2:

            [constant, sin(w1), cos(w1), sin(w2), cos(w2)]

        If linear_trend=True and parameters=2:

            [constant, trend, sin(w1), cos(w1), sin(w2), cos(w2)]

    x_data_sigma : array
        Standard deviation uncertainty estimate for each fitted coefficient.

    L2_norm : float
        L2 norm of the model-data misfit at valid data points.

    """

    # Check if data is a masked array
    assert type(data) == np.ma.core.MaskedArray, "Data is not a masked array"

    # Full x grid for evaluating final fit
    x_full = np.asarray(np.ma.getdata(x), dtype=float).ravel()

    # Build boolean index of valid data points
    mask = np.asarray(data.mask)
    if mask.size > 1:
        valid = ~mask
    else:
        valid = np.ones(data.shape, dtype=bool)

    # Select valid points
    x_n = np.asarray(x_full[valid], dtype=float).ravel()
    data_n = np.asarray(np.ma.getdata(data)[valid], dtype=float).ravel()

    if len(data_n) == 0:
        raise ValueError("No valid unmasked data points available for fitting.")

    # Check frequency input
    if parameters > 0:
        if freqs is None:
            raise ValueError("freqs must be provided when parameters > 0.")
        if len(freqs) < parameters:
            raise ValueError("Number of frequencies must be at least equal to parameters.")

    # ------------------------------------------------
    # Build kernel/design matrix for valid data points
    # ------------------------------------------------
    A_cols = [np.ones(len(data_n))]

    if linear_trend:
        A_cols.append(x_n)

    for n in range(parameters):
        A_cols.append(np.sin(freqs[n] * x_n))
        A_cols.append(np.cos(freqs[n] * x_n))

    A = np.vstack(A_cols).T

    # Solve least squares problem
    x_data = np.linalg.lstsq(A, data_n, rcond=None)[0]

    # -------------------------------------
    # Build design matrix for full x record
    # -------------------------------------
    A_full_cols = [np.ones(len(x_full))]

    if linear_trend:
        A_full_cols.append(x_full)

    for n in range(parameters):
        A_full_cols.append(np.sin(freqs[n] * x_full))
        A_full_cols.append(np.cos(freqs[n] * x_full))

    A_full = np.vstack(A_full_cols).T

    # Evaluate fit on full x grid
    hfit = A_full @ x_data
    hfit = hfit.reshape(np.shape(x))

    # Compute covariance matrix
    if sigma is not None:
        C = sigma**2 * np.linalg.inv(A.T @ A)
    else:
        C = np.zeros((A.shape[1], A.shape[1]))

    # Standard deviation of coefficients
    x_data_sigma = np.sqrt(np.diagonal(C))

    # Misfit and L2 norm
    e = A @ x_data - data_n
    L2_norm = np.sqrt(np.sum(e**2))

    return hfit, x_data, x_data_sigma, L2_norm

# --- Detrend Function --- #
def detrend(data, x, mean=False):
    
    """
    Remove a linear trend from a 1D data record using an unweighted
    least squares fit.

    Parameters
    ----------
    data : masked array
        Data record to detrend.

    x : array
        x-coordinate values associated with data.

    mean : bool, optional
        If False, remove both the linear trend and mean.
        If True, remove the linear trend while retaining the mean.

    Returns
    -------
    data_detrend : masked array
        Detrended data record.
    """

    # Fit constant offset and linear trend
    data_trend, x_data, _, _ = unweighted_lsf(
        data,
        x,
        parameters=0,
        linear_trend=True,
    )

    # Remove linear trend
    if mean:
        data_detrend = data - data_trend + x_data[0]
    else:
        data_detrend = data - data_trend

    return data_detrend








