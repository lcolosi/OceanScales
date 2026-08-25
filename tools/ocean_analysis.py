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

