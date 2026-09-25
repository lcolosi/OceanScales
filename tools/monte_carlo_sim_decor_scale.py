# =============================================================================
# Monte Carlo Simulation for Power-Law Red Spectra (Idealized Sampling)
# =============================================================================
#
# Description:
#   Preforms a Monte Carlo Simulation of the autocorrelation and decorrelation
#   scales of Power-Law red spectra. We will compute an ensemble of autocorrelation
#   for a range of  
# 
#       (1) Spectral slopes 
#       (2) Record durations
# 
#    For each ensemble, we will compute the ensemble mean and standard deviation 
#    and thus recreate data displayed in figure 3 of the main manuscript. This 
#    will be used as a test to verify the analytic results.  
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-09-22
# =============================================================================

# Import python libraries 
import sys
import os
from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm

# Set path to project root directory
ROOT = Path(__file__).resolve().parents[1]

# Set path to project data and tools directory
PATH_data  = ROOT / "data"
PATH_tools = ROOT / "tools"

# Set path to access additional python functions
sys.path.append(str(PATH_tools))

# Import analysis functions 
from spectra import generate_powerlaw_data
from autocorr import compute_autocorr_biased, compute_decor_scale, compute_decor_scale_unc
from lsf import detrend

# -----------------------------------------------------------------------------
# Set data analysis parameters
# -----------------------------------------------------------------------------

# ------------ # 
# --- Note --- # 
# ------------ #
#
# - option_detrend_seg: Specifies whether each segment is detrended or not. 
#                        Options: True or False
# - hours_per_year : Specifies the number of hours in a 365-day year. 
# - dt : The time interval between data points in units of hours. 
# - T : The duration of the record in units of hours. 
# - N : The number of data points of the record of duration T and sampling
#       interval dt. 
# - n_realizations : Specifies the number of realizations in the ensembles. 
# - random_seed : The array of random seeds for the randomization of phases. 
# - alpha : Specifies the array of spectral slopes for which the power-spectra
#           will be computed. 
# - overlap : Fractional amount of overlap between windows for autocorrelation 
#             analysis. 
# - hours_per_month : Specifies the number of hours in a month (give there are
#                     12 months in year and each month has the same number of days)
# segment_duration_months : The array of window durations. 
# segment_duration : The array of window durations in number of data points (which 
#                    where dt = 1 hour is equal to number of hours)
# 
# ------------ # 

# Set processing parameters
option_random_amplitudes = False
option_normalization     = "sample"
option_detrend           = False

# Set time parameters 
hours_per_year = 365 * 24     
dt             = 1   

# Set ensemble parameters
n_realization = 10

# Set power-law parameters
alpha   = np.arange(0,5+0.1,0.1) 
n_alpha = len(alpha)  

# Set window duration parameters
hours_per_month         = hours_per_year / 12
segment_overlap          = 0.0 
segment_duration_months = np.array([1, 2, 3, 4, 6, 8, 12])
segment_duration        = np.round(segment_duration_months * hours_per_month / dt).astype(int)
n_duration              = len(segment_duration)  

# Label segment processing 
seg_proc = "detrend" if option_detrend else "demean"
rand_amp = "random_amplitudes" if option_random_amplitudes else "random_phases"

# -----------------------------------------------------------------------------
# Perform Monte Carlo Simulation  
# -----------------------------------------------------------------------------

# Pick a initial random seed 
rng = np.random.default_rng(42)

# Create a random seed for every instance of the simulation (to make every
# instance statistically independent)
random_seed = rng.integers(
    low=0,
    high=np.iinfo(np.uint32).max,
    size=(n_duration, n_alpha, n_realization),
    dtype=np.uint32,
)

# Set the maximum number of non-negative lags 
nlag_max = np.max(segment_duration)

# Initalize arrays 
autocorr_ens_mean = np.ma.masked_all((n_duration, n_alpha, nlag_max))
autocorr_ens_std  = np.ma.masked_all((n_duration, n_alpha, nlag_max))
autocorr_ens_stdm = np.ma.masked_all((n_duration, n_alpha, nlag_max))
autocorr_discrete = np.ma.masked_all((n_duration, n_alpha, nlag_max))
Lt_mean           = np.ma.masked_all((n_duration, n_alpha))
Lt_std            = np.ma.masked_all((n_duration, n_alpha))
Lt_stdm           = np.ma.masked_all((n_duration, n_alpha))

# Loop through spectral slopes 
for i in tqdm(range(n_alpha), desc="Prerforming Monte Carlo Simulation", unit="slopes"):

    # Obtain ith spectral slope
    ialpha = alpha [i]

    # Loop through segment duration 
    for j in range(n_duration): 

        # Obtain ith window duration
        iT = segment_duration[j]

        # --- Compute the discrete theoretical autocorrelation --- #

        # Set rFFT frequency grid
        freq = np.fft.rfftfreq(iT, d=dt)

        # Set power spectrum
        S = np.zeros_like(freq)
        S[1:] = freq[1:]**(-ialpha)

        # Compute discrete theoretical autocovariance
        #
        # By the discrete Wiener-Khinchin theorem, the autocovariance
        # is the inverse Fourier transform of the power spectrum.
        C_discrete = np.fft.irfft(S, n=iT)

        # Normalize by zero lag to obtain autocorrelation
        rho_discrete = C_discrete / C_discrete[0]

        # Set the number of nonnegative lags
        nlag = len(rho_discrete)

        # Save 
        autocorr_discrete[j, i, :nlag] = rho_discrete

        # Initialize array
        autocorr_realization = np.ma.masked_all((n_realization,2*iT-1))

        # Loop through Monte Carlo Realizations 
        for k in range(n_realization): 

            # Obtain ith random seed (realization)
            iseed = random_seed[j,i,k]

            # Generate simulation data 
            t, x, _, _ = generate_powerlaw_data(N=iT, 
                                                alpha=ialpha, 
                                                random_state=iseed, 
                                                dt=dt, 
                                                random_amplitudes=option_random_amplitudes,
                                                normalization=option_normalization,
                                                target_variance=1.0,
                                                ) 

            # Convert to masked arrays
            x = np.ma.asarray(x)
            t = np.ma.asarray(t)

            # Remove segment-wise mean or linear trend
            if option_detrend: 
                x_dt = detrend(x, t, mean = 0)
            else: 
                x_dt = x - np.ma.mean(x)

            # Compute autocorrelation for the ith realization
            autocorr_realization[k,:], time_lag = compute_autocorr_biased(x_dt, t)

        # Compute the mean autocorrelation over all realizations 
        autocorr_mean = np.ma.mean(autocorr_realization,axis=0)

        # Compute realization-to-realization standard deviation at each lag
        autocorr_std = np.ma.std(autocorr_realization, axis=0, ddof=1)

        # Compute standard error at each lag
        autocorr_stdm = autocorr_std / np.sqrt(n_realization)

        # Compute the decorrelation scale of the mean autocorrelation 
        Lt_mean[j,i], M_lag = compute_decor_scale(autocorr_mean,time_lag)

        # Compute the standard error of the decorrelation scale
        Lt_stdm[j,i], Lt_std[j,i], _ = compute_decor_scale_unc(autocorr_mean, 
                                                                autocorr_realization, 
                                                                M_lag, 
                                                                dt, 
                                                                segment_overlap,
                                                                )

        # Find zero-lag index
        izero = np.argmin(np.abs(time_lag))

        # Extract nonnegative lags and corresponding mean and std autocorrelation
        lag_pos           = time_lag[izero:]
        autocorr_mean_pos = autocorr_mean[izero:]
        autocorr_std_pos  = autocorr_std[izero:]
        autocorr_stdm_pos = autocorr_stdm[izero:]

        # Set the number of nonnegative lags
        nlag = len(lag_pos)

        # Save mean and std autocorrelation 
        autocorr_ens_mean[j,i,:nlag] = autocorr_mean_pos
        autocorr_ens_std[j,i,:nlag]  = autocorr_std_pos
        autocorr_ens_stdm[j,i,:nlag] = autocorr_stdm_pos

# Convert decorrelation scales from hours to days
Lt_mean_days = Lt_mean / 24
Lt_std_days  = Lt_std / 24
Lt_stdm_days = Lt_stdm / 24

# -----------------------------------------------------------------------------
# Save data in a netcdf file
# -----------------------------------------------------------------------------

# --- Define a common lag coordinate --- # 

# Number of lag points in the padded autocorrelation arrays
nlag_max = autocorr_ens_mean.shape[-1]
nlag_max_d = autocorr_discrete.shape[-1]

# Common positive-lag coordinate
lag_positive_c = np.arange(nlag_max) * dt 
lag_positive_cd = np.arange(nlag_max) * dt 

# Convert hours to days
lag_positive_c_days = lag_positive_c / 24
lag_positive_cd_days = lag_positive_cd / 24

# --- Autocorrelation  --- # 
autocorr = xr.DataArray(data=autocorr_ens_mean,
                dims=['duration','alpha','lag'],
                coords=dict(duration=segment_duration_months,alpha=alpha,lag=lag_positive_c_days),
                attrs=dict(
                    description=('Ensemble mean autocorrelation as a ' +
                                 'function of record duration and spectral slope.'),
                    units='unitless'
                    )
)

autocorr_ens_stdm = xr.DataArray(data=autocorr_ens_stdm,
                dims=['duration','alpha','lag'],
                coords=dict(duration=segment_duration_months,alpha=alpha,lag=lag_positive_c_days),
                attrs=dict(
                    description=('Ensemble standard error of the mean autocorrelation as a ' +
                                 'function of record duration and spectral slope.'),
                    units='unitless'
                    )
)

autocorr_ens_std = xr.DataArray(data=autocorr_ens_std,
                dims=['duration','alpha','lag'],
                coords=dict(duration=segment_duration_months,alpha=alpha,lag=lag_positive_c_days),
                attrs=dict(
                    description=('Ensemble standard deviation of the mean autocorrelation as a ' +
                                 'function of record duration and spectral slope.'),
                    units='unitless'
                    )
)

autocorr_disc = xr.DataArray(data=autocorr_discrete,
                dims=['duration','alpha','lag'],
                coords=dict(duration=segment_duration_months,alpha=alpha,lag=lag_positive_c_days),
                attrs=dict(
                    description=('Discrete autocorrelation as a ' +
                                 'function of record duration and spectral slope.'),
                    units='unitless'
                    )
) 

# --- Decorrelation Scale --- # 
decor_scale = xr.DataArray(data=Lt_mean_days,
                        dims=['duration','alpha'],
                        coords=dict(duration=segment_duration_months,alpha=alpha),
                        attrs=dict(
                            description=('Mean decorrelation scale computed from the ensemble mean autocorrelation as a ' + 
                                         'function of record duration and spectral slope.'),
                            units='days'
                        )
)

decor_scale_stdm = xr.DataArray(data=Lt_stdm_days,
                        dims=['duration','alpha'],
                        coords=dict(duration=segment_duration_months,alpha=alpha),
                        attrs=dict(
                            description=('Standard error of the decorrelation scale computed from the ensemble mean autocorrelation as a ' + 
                                         'function of record duration and spectral slope.'),
                            units='days'
                        )
)

decor_scale_std = xr.DataArray(data=Lt_std_days,
                        dims=['duration','alpha'],
                        coords=dict(duration=segment_duration_months,alpha=alpha),
                        attrs=dict(
                            description=('Standard deviation of the decorrelation scale computed from the ensemble mean autocorrelation as a ' + 
                                         'function of record duration and spectral slope.'),
                            units='days'
                        )
)

# Create data set from data arrays 
data = xr.Dataset({'autocorr':autocorr,'autocorr_ens_stdm':autocorr_ens_stdm,'autocorr_ens_std':autocorr_ens_std, 'autocorr_disc':autocorr_disc, 'decor_scale':decor_scale,'decor_scale_stdm':decor_scale_stdm, 'decor_scale_std':decor_scale_std})

# Set global variables to document the processing parameters used 
data.attrs.update({
    "segment_overlap": segment_overlap,
    "segment_processing": seg_proc,
    "sampling_interval_hours": dt,
    "number_of_realizations": n_realization,
    "random_amplitudes": rand_amp,
    "normalization": option_normalization,
})

# Set file path for saving the netcdf file
file_path = PATH_data / "analytic" / f"monte_carlo_sim_decor_scale_ideal_{rand_amp}_norm_{option_normalization}_realizations_{n_realization}_{seg_proc}.nc"

# Check if file exists, then delete it
if os.path.exists(file_path):
    os.remove(file_path)

# Create netcdf file
data.to_netcdf(file_path,mode='w')





