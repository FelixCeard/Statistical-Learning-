# coding: utf-8
"""
1D Wavelet Denoising for Discrete Integer Signals

Modified from skimage's wavelet denoising implementation:
https://github.com/scikit-image/scikit-image/blob/f0d48db4c246989182aa01c837d04903bc2330ae/skimage/restoration/_denoise.py

Simplified for 1D signals of discrete integers.
"""

import warnings
import numpy as np
import scipy.stats
import pywt


def _bayes_thresh(details, var):
    """
    BayesShrink threshold for a zero-mean details coefficient array.
    
    Parameters
    ----------
    details : ndarray
        Detail coefficients from wavelet decomposition.
    var : float
        Estimated noise variance.
        
    Returns
    -------
    thresh : float
        The computed threshold value.
    """
    # For zero-mean array: dvar = np.var(details) == np.mean(details**2)
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def _universal_thresh(signal, sigma):
    """
    Universal threshold used by the VisuShrink method.
    
    Parameters
    ----------
    signal : ndarray
        The input signal.
    sigma : float
        Estimated noise standard deviation.
        
    Returns
    -------
    thresh : float
        The universal threshold value.
    """
    return sigma * np.sqrt(2 * np.log(signal.size))


def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """
    Calculate the robust median estimator of the noise standard deviation.
    
    Uses the Median Absolute Deviation (MAD) estimator from the finest
    scale wavelet coefficients.
    
    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients from the discrete wavelet transform.
    distribution : str
        The underlying noise distribution. Only 'Gaussian' is supported.
        
    Returns
    -------
    sigma : float
        The estimated noise standard deviation.
        
    References
    ----------
    D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
    by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
    DOI:10.1093/biomet/81.3.425
    """
    # Exclude zero coefficients (treat as masked)
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    
    if len(detail_coeffs) == 0:
        return 0.0
    
    if distribution.lower() == 'gaussian':
        # 75th quantile of the standard normal distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently supported")
    
    return sigma


def _wavelet_threshold_1d(signal, wavelet, method=None, threshold=None,
                          sigma=None, mode='soft', wavelet_levels=None):
    """
    Perform wavelet thresholding on a 1D signal.
    
    Parameters
    ----------
    signal : ndarray
        1D input signal to be denoised.
    wavelet : string
        The type of wavelet to use. Can be any of the options from
        pywt.wavelist(). Examples: 'db1', 'db2', 'db4', 'haar', 'sym4'.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to use. If None, a user-specified threshold
        must be provided.
    threshold : float or list, optional
        The thresholding value(s) to apply. If None, uses the selected
        method to estimate appropriate threshold(s).
    sigma : float, optional
        The noise standard deviation. If None, it is estimated from the
        finest scale detail coefficients.
    mode : {'soft', 'hard'}, optional
        Type of thresholding. 'soft' (default) typically works better
        for additive noise.
    wavelet_levels : int, optional
        Number of wavelet decomposition levels. Default is max levels - 3.
        
    Returns
    -------
    denoised : ndarray
        Denoised signal.
    """
    wavelet_obj = pywt.Wavelet(wavelet)
    original_length = len(signal)
    
    # Determine decomposition levels
    if wavelet_levels is None:
        dlen = wavelet_obj.dec_len
        max_level = pywt.dwt_max_level(len(signal), dlen)
        # Skip coarsest scales (similar to original)
        wavelet_levels = max(max_level - 3, 1)
    
    # Perform wavelet decomposition
    # wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=wavelet_levels)
    
    # Approximation coefficients and detail coefficients
    approx = coeffs[0]
    details = coeffs[1:]  # List of detail coefficient arrays
    
    # Estimate sigma from finest scale details if not provided
    if sigma is None:
        sigma = _sigma_est_dwt(details[-1], distribution='Gaussian')
    
    if method is not None and threshold is not None:
        warnings.warn(
            f"Thresholding method {method} selected. "
            "The user-specified threshold will be ignored."
        )
    
    # Compute threshold(s)
    if threshold is None:
        var = sigma ** 2
        if method is None:
            raise ValueError("If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # Compute adaptive threshold for each decomposition level
            threshold = [_bayes_thresh(d, var) for d in details]
        elif method == "VisuShrink":
            # Single universal threshold
            threshold = _universal_thresh(signal, sigma)
        else:
            raise ValueError(f"Unrecognized method: {method}")
    
    # Handle threshold as list (extend if needed)
    if isinstance(threshold, (list, tuple)):
        threshold = list(threshold)
        # Pad threshold list if shorter than number of detail levels
        if len(threshold) < len(details):
            threshold = threshold + [threshold[-1]] * (len(details) - len(threshold))
    
    # Apply thresholding to detail coefficients
    if np.isscalar(threshold):
        denoised_details = [
            pywt.threshold(d, value=threshold, mode=mode) for d in details
        ]
    else:
        denoised_details = [
            pywt.threshold(d, value=thresh, mode=mode)
            for d, thresh in zip(details, threshold)
        ]
    
    # Reconstruct signal
    denoised_coeffs = [approx] + denoised_details
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    
    # Handle potential length mismatch from reconstruction
    return denoised[:original_length]


def denoise_wavelet_1d(signal, sigma=None, wavelet='db1', mode='soft',
                       wavelet_levels=None, method='BayesShrink',
                       threshold=None, return_int=True):
    """
    Perform wavelet denoising on a 1D signal of discrete integers.
    
    Parameters
    ----------
    signal : array-like
        1D input signal to be denoised. Can be integers or floats.
    sigma : float, optional
        The noise standard deviation. If None (default), it is estimated
        from the wavelet detail coefficients using the MAD estimator.
    wavelet : string, optional
        The type of wavelet to use. Default is 'db1' (Haar wavelet).
        Can be any wavelet from pywt.wavelist(), e.g., 'db2', 'db4',
        'sym4', 'coif1', etc.
    mode : {'soft', 'hard'}, optional
        Type of thresholding. 'soft' (default) typically produces better
        results for additive Gaussian noise.
    wavelet_levels : int, optional
        Number of wavelet decomposition levels. Default is automatically
        determined based on signal length.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method. Default is 'BayesShrink'.
        - 'BayesShrink': Adaptive thresholding with level-dependent
          thresholds. Generally preserves more detail.
        - 'VisuShrink': Universal threshold that removes most noise
          but may over-smooth the signal.
    threshold : float or list, optional
        Custom threshold value(s). If provided with method=None, uses
        this value directly. If a list, specifies threshold for each
        decomposition level.
    return_int : bool, optional
        If True (default), round the result to integers. Set to False
        to get floating-point output.
        
    Returns
    -------
    denoised : ndarray
        Denoised signal. Integer array if return_int=True, float otherwise.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create a noisy signal
    >>> np.random.seed(42)
    >>> clean = np.array([0, 0, 5, 10, 15, 20, 20, 20, 15, 10, 5, 0, 0])
    >>> noisy = clean + np.random.randint(-3, 4, size=clean.shape)
    >>> denoised = denoise_wavelet_1d(noisy)
    >>> print(denoised)
    
    >>> # Using VisuShrink method
    >>> denoised = denoise_wavelet_1d(noisy, method='VisuShrink')
    
    >>> # Using different wavelets
    >>> denoised = denoise_wavelet_1d(noisy, wavelet='sym4')
    
    >>> # Custom threshold
    >>> denoised = denoise_wavelet_1d(noisy, method=None, threshold=2.0)
    
    References
    ----------
    [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
        thresholding for image denoising and compression." Image Processing,
        IEEE Transactions on 9.9 (2000): 1532-1546.
        DOI: 10.1109/83.862633
        
    [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
        by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
        DOI: 10.1093/biomet/81.3.425
    """
    if method not in ["BayesShrink", "VisuShrink", None]:
        raise ValueError(
            f'Invalid method: {method}. Supported methods are '
            '"BayesShrink", "VisuShrink", or None (with custom threshold).'
        )
    
    # Convert to numpy array and float for processing
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.ndim}D array")
    
    original_dtype = signal.dtype
    signal_float = signal.astype(np.float64)
    
    # Perform wavelet denoising
    denoised = _wavelet_threshold_1d(
        signal_float,
        wavelet=wavelet,
        method=method,
        sigma=sigma,
        mode=mode,
        wavelet_levels=wavelet_levels,
        threshold=threshold
    )
    
    if return_int:
        # Round to nearest integer
        denoised = np.round(denoised).astype(original_dtype)
    
    return denoised


def estimate_sigma_1d(signal):
    """
    Robust wavelet-based estimator of the noise standard deviation.
    
    Uses the Median Absolute Deviation (MAD) of the finest scale
    wavelet detail coefficients.
    
    Parameters
    ----------
    signal : array-like
        1D signal for which to estimate the noise standard deviation.
        
    Returns
    -------
    sigma : float
        Estimated noise standard deviation.
        
    References
    ----------
    D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
    by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
    DOI: 10.1093/biomet/81.3.425
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.ndim}D array")
    
    # Single-level DWT to get detail coefficients
    _, detail_coeffs = pywt.dwt(signal, wavelet='db2')
    
    return _sigma_est_dwt(detail_coeffs, distribution='Gaussian')


# Convenience function for quick denoising
def denoise(signal, method='BayesShrink', wavelet='db4', **kwargs):
    """
    Quick convenience function for denoising a 1D integer signal.
    
    Parameters
    ----------
    signal : array-like
        1D input signal.
    method : str, optional
        'BayesShrink' (default) or 'VisuShrink'.
    wavelet : str, optional
        Wavelet type. Default is 'db4'.
    **kwargs
        Additional arguments passed to denoise_wavelet_1d.
        
    Returns
    -------
    denoised : ndarray
        Denoised signal (integer array).
    """
    return denoise_wavelet_1d(signal, method=method, wavelet=wavelet, **kwargs)


if __name__ == "__main__":
    # Demo / test
    import matplotlib.pyplot as plt
    
    # Create test signal: step function with noise
    np.random.seed(42)
    n = 200
    
    # Clean signal: piecewise constant
    clean = np.zeros(n, dtype=int)
    clean[20:50] = 10
    clean[50:80] = 25
    clean[80:120] = 15
    clean[120:160] = 30
    clean[160:180] = 5
    
    # Add discrete integer noise
    noise_level = 5
    noisy = clean + np.random.randint(-noise_level, noise_level + 1, size=n)
    
    # Denoise using different methods
    denoised_bayes = denoise_wavelet_1d(noisy, method='BayesShrink', wavelet='db4')
    denoised_visu = denoise_wavelet_1d(noisy, method='VisuShrink', wavelet='db4')
    
    # Estimate noise
    estimated_sigma = estimate_sigma_1d(noisy)
    print(f"True noise std (approx): {noise_level / np.sqrt(3):.2f}")
    print(f"Estimated noise std: {estimated_sigma:.2f}")
    
    # Compute MSE
    mse_noisy = np.mean((noisy - clean) ** 2)
    mse_bayes = np.mean((denoised_bayes - clean) ** 2)
    mse_visu = np.mean((denoised_visu - clean) ** 2)
    
    print(f"\nMSE (noisy): {mse_noisy:.2f}")
    print(f"MSE (BayesShrink): {mse_bayes:.2f}")
    print(f"MSE (VisuShrink): {mse_visu:.2f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(clean, 'g-', linewidth=2, label='Clean')
    axes[0, 0].set_title('Original Clean Signal')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    
    axes[0, 1].plot(noisy, 'b-', alpha=0.7, label='Noisy')
    axes[0, 1].plot(clean, 'g--', linewidth=1, alpha=0.5, label='Clean')
    axes[0, 1].set_title(f'Noisy Signal (MSE: {mse_noisy:.2f})')
    axes[0, 1].legend()
    
    axes[1, 0].plot(denoised_bayes, 'r-', linewidth=2, label='BayesShrink')
    axes[1, 0].plot(clean, 'g--', linewidth=1, alpha=0.5, label='Clean')
    axes[1, 0].set_title(f'BayesShrink Denoised (MSE: {mse_bayes:.2f})')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    
    axes[1, 1].plot(denoised_visu, 'm-', linewidth=2, label='VisuShrink')
    axes[1, 1].plot(clean, 'g--', linewidth=1, alpha=0.5, label='Clean')
    axes[1, 1].set_title(f'VisuShrink Denoised (MSE: {mse_visu:.2f})')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('denoising_demo.png', dpi=150)
    plt.show()

