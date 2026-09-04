# =============================================================================
# Ocean Analysis Functions 
# =============================================================================
#
# Description:
#   Functions for calculating common physical ocean parameters include: 
#       (1) Mixed layer depth 
#       (2) Eigenvectors and Eigenvalues of the Rossby modal equation.   
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-25
# =============================================================================

# Import libraries 
import numpy as np
from scipy.linalg import eigh_tridiagonal

# --- Mixed Layer Depth --- #
def compute_mld(
    depth, 
    temp=None, 
    density=None, 
    method='threshold', 
    variable='density',
    dT=0.2, 
    dSigma=0.03, 
    gradT=0.025, 
    gradSigma=0.0005,
    zref=10, 
    phi=100, 
    g=9.81
):
    """
    Compute mixed layer depth (MLD) using threshold, gradient, or potential energy
    anomaly (PEA) criteria. Supports masked arrays for temperature and density.

    Parameters
    ----------
    depth : array_like
        1D array of depths (m, positive downward).
    temp : array_like
        1D array of potential temperature (degrees Celsius).
    density : array_like
        1D array of potential density anomaly (kg/m^3).
    method : {'threshold', 'gradient', 'potential_energy'}, optional
        Criterion to use for MLD definition.
    variable : {'density', 'temperature'}, optional
        Use density (σθ) or temperature to define MLD.
    dT : float, optional
        Temperature threshold (degrees Celsius) for threshold method (default 0.2).
    dSigma : float, optional
        Density threshold (kg/m^3) for threshold method (default 0.03).
    gradT : float, optional
        Temperature gradient threshold (degrees Celsius/m) for gradient method (default 0.025).
    gradSigma : float, optional
        Density gradient threshold (kg/m^4) for gradient method (default 0.0005).
    zref : float, optional
        Reference depth for threshold method (default 10 m).
    phi : float, optional
        Potential energy anomaly threshold (default 100 J/m^2).
    g : float, optional
        Gravitational acceleration (default 9.81 (m/s^2)).

    Returns
    -------
    mld : float
        Mixed layer depth estimate (m). Returns np.nan if no criterion is met.
    """

    # Convert inputs to numpy arrays for safe indexing
    depth = np.asarray(depth)
    if temp is not None: temp = np.ma.masked_invalid(temp)
    if density is not None: density = np.ma.masked_invalid(density)

    # Ensure depth increases with index (surface first, deep last)
    if depth[0] > depth[-1]:
        depth = depth[::-1]
        if temp is not None: temp = temp[::-1]
        if density is not None: density = density[::-1]

    # ------------------------------
    # Helper Function: compress valid values
    # ------------------------------
    def valid_profile(arr):

        """
        Removes masked values from arr variable and depth profile leaving only
        valid (depth_valid, arr_valid) pairs. 
        """

        # Define mask for non-masked data-depth pairs 
        # Sets mask to all False if arr is not a masked array
        mask = ~np.ma.getmaskarray(arr) if np.ma.isMaskedArray(arr) else np.ones_like(arr, dtype=bool)

        # Ensure mask only keeps finite values (no NaN or +/- inf)
        mask &= np.isfinite(np.ma.getdata(arr))

        # Handle case when no valid points remain (all masked or NaNs)
        if not np.any(mask):
            return None, None
        
        return depth[mask], np.ma.getdata(arr)[mask]

    # ------------------------------
    # Threshold method
    # ------------------------------
    if method == 'threshold':

        if variable == 'temperature':

            # Check if temperature variable is not None
            if temp is None:
                raise ValueError("Temperature array required for temperature-based MLD")
            
            # Remove masked values
            depth_valid, temp_valid = valid_profile(temp)
            if depth_valid is None:
                return np.nan

            # Check if the reference depth lies within the valid depth range
            if zref < depth_valid[0] or zref > depth_valid[-1]:
                return np.nan
            
            # Find the index of the reference depth (default = 10 m)
            iref = np.argmin(np.abs(depth_valid - zref))
            
            # Compare each temp value with the reference temp
            ref_val = temp_valid[iref]
            diff = np.ma.abs(temp_valid - ref_val)

            # First depth where difference exceeds threshold dT below the reference depth
            below_ref = depth_valid >= depth_valid[iref]
            idx = np.where(below_ref & (diff >= dT))[0]

        elif variable == 'density':

            # Check if density variable is not None
            if density is None:
                raise ValueError("Density array required for density-based MLD")
            
            # Remove masked values
            depth_valid, dens_valid = valid_profile(density)
            if depth_valid is None:
                return np.nan

            # Check if the reference depth lies within the valid depth range
            if zref < depth_valid[0] or zref > depth_valid[-1]:
                return np.nan
            
            # Find the index of the reference depth (default = 10 m)
            iref = np.argmin(np.abs(depth_valid - zref))

            # Compare each density value with the reference density
            ref_val = dens_valid[iref]
            diff = dens_valid - ref_val

            # First depth where difference exceeds threshold dSigma
            below_ref = depth_valid >= depth_valid[iref]
            idx = np.where(below_ref & (diff >= dSigma))[0]

        else:
            raise ValueError("variable must be 'temperature' or 'density'")

    # ------------------------------
    # Gradient method
    # ------------------------------
    elif method == 'gradient':
        if variable == 'temperature':

            # Check if temperature variable is not None
            if temp is None:
                raise ValueError("Temperature array required for temperature-based MLD")
            
            # Remove masked values
            depth_valid, temp_valid = valid_profile(temp)
            if depth_valid is None or len(depth_valid) < 2:
                return np.nan

            # Compute dT/dz using numpy.gradient
            grad = np.gradient(temp_valid, depth_valid)

            # First depth where gradient exceeds threshold gradT
            idx = np.where(np.ma.abs(grad) >= gradT)[0]

        elif variable == 'density':

            # Check if density variable is not None
            if density is None:
                raise ValueError("Density array required for density-based MLD")
            
            # Remove masked values
            depth_valid, dens_valid = valid_profile(density)
            if depth_valid is None or len(depth_valid) < 2:
                return np.nan

            # Compute dsigma0/dz 
            grad = np.gradient(dens_valid, depth_valid)

            # First depth where gradient exceeds threshold gradSigma
            idx = np.where(grad >= gradSigma)[0]

        else:
            raise ValueError("variable must be 'temperature' or 'density'")
    
    # ------------------------------
    # Potential Energy Anomaly method
    # ------------------------------

    elif method == 'potential_energy':
    
        # Check if density variable is not None
        if density is None:
            raise ValueError("Density array required for potential energy method")
        
        # Remove masked values
        depth_valid, dens_valid = valid_profile(density)

        # Check if the profile has less than 3 data points
        if depth_valid is None or len(depth_valid) < 3:
            return np.nan

        # Rename density variable for clarity
        rho = dens_valid

        # Initialize potential energy anomaly
        pe_anomaly = np.full(len(depth_valid), np.nan)

        # Set PEA at the shallowest level to zero
        pe_anomaly[0] = 0.0

        # --- Compute PEA for progressively deeper candidate mixed layers --- #  

        # Loop through depth 
        for idepth in range(1, len(depth_valid)):

            # Extract the water column from the shallowest valid level
            # to the current candidate mixed-layer depth
            depth_layer = depth_valid[:idepth + 1]
            rho_layer = rho[:idepth + 1]

            # Define vertical position relative to the shallowest valid level
            # so that z = 0 at the top of the candidate layer
            z_layer = depth_layer - depth_layer[0]

            # Candidate mixed-layer thickness
            H = z_layer[-1]

            # Skip degenerate layers
            if H <= 0:
                continue

            # Compute the mean density of the entire candidate layer
            rho_m = np.trapezoid(rho_layer,z_layer) / H

            # Compute the potential energy anomaly required to homogenize
            # the entire candidate layer
            pe_anomaly[idepth] = g * np.trapezoid((rho_layer - rho_m) * z_layer,z_layer)

        # Compute difference between PEA and target threshold
        pea_diff = pe_anomaly - phi

        ###################
        # Note
        # ----
        # Breaking these lines of code down further, the difference between the
        # average density of the water column above each depth and the density at
        # each depth level
        # 
        # rho_m - rho(z) 
        # 
        # tells us how much lighter or denser the local water parcel is compared to
        # the mean. By multiplying by g * z, we convert the density anomaly to a
        # potential energy per unit volume because
        # 
        # PE = rho * g * z (J / m^3)
        # 
        # Because rho_m is the average density of the water column above each depth
        # level, this potential energy represent how much pontential energy is
        # required to homogenize the layer at each depth increment (by layer, we are
        # refering to the layer of fluid between each depth increment).  
        # 
        # Lastly, the mixed layer depth can be defined as the depth where the
        # required mixing energy first exceeds a reference energy phi. 
        # 
        # PE_anomaly(z_mld) = phi
        # 
        # Choosing phi = 100 J/m^2 corresponds to a characteristic energetic
        # "cost" to mix the upper layer that roughly separates weakly stratified
        # surface layers from strongly stratified pycnocline layers. 100 J/m^2 is an
        # empirical standard that has been proven to: 
        # 
        #  (1) Large enough to avoid noise from near-surface micro-stratification
        #  (2) Small enough to capture the physical mixed region. 
        #  
        # Furthermore, the PEA method with phi = 100 J/m^2 have be shown to obtain
        # similar results to the threshold methods. 
        ###################

        # Identify valid PEA estimates
        valid_pea = np.isfinite(pe_anomaly)

        if np.sum(valid_pea) < 2:
            return np.nan

        # Restrict to valid values
        depth_pea = depth_valid[valid_pea]
        pea_diff_valid = pea_diff[valid_pea]

        # Find first interval where PEA reaches/exceeds threshold
        crossing = np.where((pea_diff_valid[:-1] < 0) & (pea_diff_valid[1:] >= 0))[0]

        # No threshold crossing
        if crossing.size == 0:
            return np.nan

        # First threshold crossing
        idx = crossing[0]

        # Linearly interpolate the crossing depth
        mld = (depth_pea[idx] - pea_diff_valid[idx] * (depth_pea[idx + 1] - depth_pea[idx]) / (pea_diff_valid[idx + 1] - pea_diff_valid[idx]))

    else:
        raise ValueError("method must be 'threshold', 'gradient', or 'potential_energy'")

    # ------------------------------
    # Return result
    # ------------------------------
    if method == 'potential_energy': 
        return mld 
    elif method in ('threshold', 'gradient'):
        if idx.size == 0:
            # No depth satisfies the criterion → return NaN
            return np.nan
        else:
            # Return the shallowest depth where criterion is met
            return depth_valid[idx[0]]


# --- Rossby Deformation Radius and Vertical Struction of Displacement --- #
def compute_rossby_modes(
    zin,
    nin,
    lat,
    depth_bottom,
    nmodes=1,
    nz=256,
    grav=9.81,
    omega=7.292115e-5,
    n_floor=1.0e-7,
    return_modes=False,
):
    """
    Compute barotropic and baroclinic gravity-wave modes and the
    corresponding Rossby deformation radii.

    The vertical structure equation is

        d²W/dz² + (N²/c²) W = 0,

    where z is positive downward and c is the modal long-wave
    phase speed.

    A free surface is used at z = 0,

        dW/dz = -(g/c²) W,

    together with a rigid, impermeable bottom,

        W(H) = 0.

    The first eigenmode is the external/barotropic mode. Subsequent
    modes are the baroclinic modes.

    Parameters
    ----------
    zin : array_like
        Vertical coordinates of N [m]. May be either TEOS-10 z
        (negative below the surface) or positive-down depth.
    nin : array_like
        Buoyancy frequency N(z) [s^-1].
    lat : float
        Latitude [degrees].
    depth_bottom : float
        Actual bathymetric water depth H [m].
    nmodes : int, optional
        Number of baroclinic modes to calculate. Default is 1.
    nz : int, optional
        Number of uniformly spaced vertical intervals. Default is 256.
    grav : float, optional
        Gravitational acceleration [m s^-2].
    omega : float, optional
        Earth's rotation rate [s^-1].
    n_floor : float, optional
        Small numerical floor applied to N [s^-1].
    return_modes : bool, optional
        If True, also return the vertical eigenfunctions.

    Returns
    -------
    result : dict
        Dictionary containing modal phase speeds, deformation radii,
        inverse deformation-radius squared, and optionally mode shapes.
    """

    # -----------------------------------------------------------------
    # Prepare input profiles
    # -----------------------------------------------------------------

    z_in = np.asarray(
        np.ma.filled(np.ma.asarray(zin, dtype=float), np.nan),
        dtype=float,
    )

    n_in = np.asarray(
        np.ma.filled(np.ma.asarray(nin, dtype=float), np.nan),
        dtype=float,
    )

    if z_in.ndim != 1 or n_in.ndim != 1 or z_in.size != n_in.size:
        raise ValueError(
            "zin and nin must be 1-D arrays with the same length."
        )

    # Actual water-column depth
    H = float(depth_bottom)

    if not np.isfinite(H) or H <= 0.0:
        raise ValueError(
            "depth_bottom must be a finite positive depth."
        )

    # Accept either negative TEOS-10 z or positive-down depth
    z_in = np.abs(z_in)

    # Remove invalid data
    valid = (
        np.isfinite(z_in)
        & np.isfinite(n_in)
        & (z_in >= 0.0)
        & (z_in < H)
        & (n_in >= 0.0)
    )

    z_in = z_in[valid]
    n_in = n_in[valid]

    if z_in.size < 2:
        raise ValueError(
            "At least two valid N(z) values are required."
        )

    # Sort vertically
    order = np.argsort(z_in)

    z_in = z_in[order]
    n_in = n_in[order]

    # Remove duplicate depths
    z_in, unique = np.unique(z_in, return_index=True)
    n_in = n_in[unique]

    # -----------------------------------------------------------------
    # Coriolis parameter
    # -----------------------------------------------------------------

    f = 2.0 * omega * np.sin(np.deg2rad(lat))

    if np.abs(f) < 1.0e-10:
        raise ValueError(
            "Midlatitude Rossby-radius formula is not valid near "
            "the equator."
        )

    # -----------------------------------------------------------------
    # Uniform vertical grid
    # -----------------------------------------------------------------

    dz = H / nz

    # Bottom point is excluded from the eigenproblem because W(H) = 0
    z = np.arange(nz) * dz

    # -----------------------------------------------------------------
    # Interpolate N² onto uniform grid
    # -----------------------------------------------------------------

    n2_in = n_in**2

    n2 = np.interp(
        z,
        z_in,
        n2_in,
        left=n2_in[0],
        right=n2_in[-1],
    )

    # Numerical floor for essentially unstratified layers
    n2 = np.maximum(n2, n_floor**2)

    # -----------------------------------------------------------------
    # Generalized eigenproblem
    #
    #       K W = mu M W
    #
    # where
    #
    #       mu = 1 / c²
    #
    # -----------------------------------------------------------------

    # Diagonal of -d²/dz²
    k_diag = np.full(nz, 2.0 / dz**2)

    # Free-surface boundary condition changes first diagonal element
    k_diag[0] = 1.0 / dz**2

    # Off-diagonal elements
    k_off = np.full(nz - 1, -1.0 / dz**2)

    # Weight matrix
    m_diag = n2.copy()

    # Free-surface boundary condition
    m_diag[0] = grav / dz

    # -----------------------------------------------------------------
    # Convert generalized problem into standard symmetric problem
    #
    # C y = mu y
    #
    # with
    #
    # C = M^(-1/2) K M^(-1/2)
    # -----------------------------------------------------------------

    c_diag = k_diag / m_diag

    c_off = (
        k_off
        / np.sqrt(m_diag[:-1] * m_diag[1:])
    )

    # Need external mode + requested number of baroclinic modes
    select_range = (0, nmodes)

    # -----------------------------------------------------------------
    # Solve only the modes that are actually needed
    # -----------------------------------------------------------------

    if return_modes:

        mu, eigvecs = eigh_tridiagonal(
            c_diag,
            c_off,
            select="i",
            select_range=select_range,
            check_finite=False,
        )

    else:

        mu = eigh_tridiagonal(
            c_diag,
            c_off,
            eigvals_only=True,
            select="i",
            select_range=select_range,
            check_finite=False,
        )

    if np.any(mu <= 0.0):
        raise RuntimeError(
            "Non-positive vertical-mode eigenvalue encountered."
        )

    # -----------------------------------------------------------------
    # Convert eigenvalues to physical quantities
    # -----------------------------------------------------------------

    # Long-wave phase speed
    phase_speed = 1.0 / np.sqrt(mu)

    # Rossby deformation radius
    radius = phase_speed / np.abs(f)

    # lambda² = 1 / R²
    lambda_sq = 1.0 / radius**2

    # -----------------------------------------------------------------
    # Package results
    # -----------------------------------------------------------------

    result = {
        # External/barotropic mode
        "lambda0_sq": lambda_sq[0],
        "c0": phase_speed[0],
        "Rd0": radius[0],

        # Baroclinic modes
        "lambda_baroclinic_sq": lambda_sq[1:],
        "c_baroclinic": phase_speed[1:],
        "Rd_baroclinic": radius[1:],
    }

    # -----------------------------------------------------------------
    # Optionally recover vertical mode shapes
    # -----------------------------------------------------------------

    if return_modes:

        # Transform eigenvectors back to physical W
        modes = eigvecs / np.sqrt(m_diag)[:, None]

        # Add bottom point, where W(H) = 0
        modes_full = np.zeros((nz + 1, nmodes + 1))

        modes_full[:-1, :] = modes

        # Normalize each mode by maximum absolute amplitude
        for imode in range(nmodes + 1):

            scale = np.max(np.abs(modes_full[:, imode]))

            modes_full[:, imode] /= scale

            # Choose consistent sign
            if modes_full[0, imode] < 0.0:
                modes_full[:, imode] *= -1.0

        result["z"] = np.linspace(0.0, H, nz + 1)

        result["w0"] = modes_full[:, 0]

        result["w_baroclinic"] = modes_full[:, 1:]

    return result

