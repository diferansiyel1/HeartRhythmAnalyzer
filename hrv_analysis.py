import numpy as np
from scipy import signal, interpolate
from scipy.fftpack import fft
import pandas as pd

def validate_rr_data(rr_intervals):
    """Validate RR interval data."""
    if len(rr_intervals) < 2:
        return False, "Not enough RR intervals"
    if any(rr <= 0 for rr in rr_intervals):
        return False, "Invalid RR intervals (negative or zero values)"
    if any(rr > 2500 for rr in rr_intervals):
        return False, "Invalid RR intervals (too large values > 2500ms)"
    return True, "Data is valid"

def calculate_mean_heart_rate(rr_intervals):
    """Calculate mean heart rate in beats per minute."""
    mean_rr_seconds = np.mean(rr_intervals) / 1000.0  # Convert to seconds
    mean_hr = 60.0 / mean_rr_seconds  # Convert to BPM
    return round(mean_hr, 1)

def calculate_time_domain_parameters(rr_intervals):
    """Calculate time domain HRV parameters."""
    rr = np.array(rr_intervals)

    # Calculate basic statistics
    mean_rr = np.mean(rr)
    sdnn = np.std(rr)
    mean_hr = calculate_mean_heart_rate(rr)

    # Calculate RMSSD
    rr_diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))

    # Calculate pNN50
    nn50 = sum(abs(rr_diff) > 50)
    pnn50 = (nn50 / len(rr_diff)) * 100

    return {
        'Mean HR (bpm)': mean_hr,
        'Mean RR (ms)': round(mean_rr, 2),
        'SDNN (ms)': round(sdnn, 2),
        'RMSSD (ms)': round(rmssd, 2),
        'pNN50 (%)': round(pnn50, 2)
    }

def calculate_dfa(rr_intervals, scale_min=4, scale_max=None):
    """Calculate Detrended Fluctuation Analysis."""
    rr = np.array(rr_intervals)

    # Integrate the signal
    y = np.cumsum(rr - np.mean(rr))

    # Set maximum scale if not provided
    if scale_max is None:
        scale_max = len(rr) // 4

    # Create scales array
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 20, dtype=int)

    # Initialize fluctuation array
    fluct = np.zeros(len(scales))

    # Calculate fluctuation for each scale
    for i, scale in enumerate(scales):
        # Calculate number of segments
        segments = len(y) // scale

        if segments > 0:
            # Reshape data into segments
            segments_data = y[:segments * scale].reshape((segments, scale))

            # Calculate local trend for each segment
            x = np.arange(scale)
            local_trends = np.array([np.polyfit(x, segment, 1) for segment in segments_data])
            trends = np.array([np.polyval(p, x) for p in local_trends])

            # Calculate fluctuation
            fluct[i] = np.sqrt(np.mean((segments_data - trends) ** 2))

    # Calculate alpha values using linear regression
    scales_log = np.log10(scales)
    fluct_log = np.log10(fluct)

    # Split into short-term and long-term components
    idx_short = (scales <= 16)
    idx_long = (scales > 16)

    # Calculate alpha1 (short-term)
    if np.sum(idx_short) > 1:
        alpha1 = np.polyfit(scales_log[idx_short], fluct_log[idx_short], 1)[0]
    else:
        alpha1 = np.nan

    # Calculate alpha2 (long-term)
    if np.sum(idx_long) > 1:
        alpha2 = np.polyfit(scales_log[idx_long], fluct_log[idx_long], 1)[0]
    else:
        alpha2 = np.nan

    return {
        'alpha1': round(alpha1, 3),
        'alpha2': round(alpha2, 3)
    }, (scales_log, fluct_log)

def calculate_frequency_domain_parameters(rr_intervals, vlf_range=(0.003, 0.04), lf_range=(0.04, 0.15), hf_range=(0.15, 0.4), fs=4.0):
    """Calculate frequency domain HRV parameters with adjustable frequency bands."""
    # Interpolate RR intervals to get evenly sampled signal
    rr = np.array(rr_intervals)
    t = np.cumsum(rr) / 1000.0  # Convert to seconds
    t = t - t[0]  # Start at 0

    # Create regular time axis
    t_regular = np.linspace(t[0], t[-1], int(len(rr) * fs))

    # Interpolate RR intervals
    f = interpolate.interp1d(t, rr, kind='cubic')
    rr_regular = f(t_regular)

    # Remove mean
    rr_detrend = rr_regular - np.mean(rr_regular)

    # Calculate PSD using Welch's method
    frequencies, psd = signal.welch(rr_detrend, fs=fs, nperseg=256)

    # Calculate power in each band
    vlf_power = np.trapz(psd[(frequencies >= vlf_range[0]) & (frequencies < vlf_range[1])], 
                        frequencies[(frequencies >= vlf_range[0]) & (frequencies < vlf_range[1])])
    lf_power = np.trapz(psd[(frequencies >= lf_range[0]) & (frequencies < lf_range[1])], 
                       frequencies[(frequencies >= lf_range[0]) & (frequencies < lf_range[1])])
    hf_power = np.trapz(psd[(frequencies >= hf_range[0]) & (frequencies < hf_range[1])], 
                       frequencies[(frequencies >= hf_range[0]) & (frequencies < hf_range[1])])

    total_power = vlf_power + lf_power + hf_power
    lf_nu = (lf_power / (lf_power + hf_power)) * 100
    hf_nu = (hf_power / (lf_power + hf_power)) * 100
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    return {
        'VLF Power (ms²)': round(vlf_power, 2),
        'LF Power (ms²)': round(lf_power, 2),
        'HF Power (ms²)': round(hf_power, 2),
        'Total Power (ms²)': round(total_power, 2),
        'LF (n.u.)': round(lf_nu, 2),
        'HF (n.u.)': round(hf_nu, 2),
        'LF/HF Ratio': round(lf_hf_ratio, 2)
    }, (frequencies, psd)