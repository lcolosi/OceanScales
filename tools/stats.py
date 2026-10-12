# =============================================================================
# Statistics Functions  
# =============================================================================
#
# Description:
#   Functions preforming fundamental statistical analysis on time series.  
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-10-12
# =============================================================================

# Import libraries 
import numpy as np 

# --- 1D Probability Density Function --- #
def compute_pdf_1D(x, edges):
    """
    Compute a one-dimensional histogram and probability density function
    (PDF) estimate from a sample of observations.

    The PDF is estimated by normalizing the histogram counts by the total
    number of observations within the specified bin edges and the width
    of each bin:

        pdf[i] = counts[i] / (N * bin_width[i])

    where N is the total number of observations included in the histogram.
    The resulting PDF satisfies:

        sum(pdf * bin_width) = 1

    Parameters
    ----------
    x : array-like
        One-dimensional array of observations without masked or NaN values.
    edges : array-like
        Monotonically increasing bin edges. Observations outside the
        specified range are excluded.

    Returns
    -------
    counts : ndarray
        Number of observations within each bin.
    pdf : ndarray
        Probability density estimate for each bin.
    bin_centers : ndarray
        Center of each bin.
    """

    # Compute histogram counts
    counts, _ = np.histogram(x, bins=edges)

    # Compute bin widths
    bin_width = np.diff(edges)

    # Compute total number of observations within the histogram
    N = np.sum(counts)

    # Check that the histogram contains observations
    if N == 0:
        raise ValueError("No observations fall within the specified bin edges.")

    # Normalize histogram to obtain probability density
    pdf = counts / (N * bin_width)

    # Compute bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2

    return counts, pdf, bin_centers


