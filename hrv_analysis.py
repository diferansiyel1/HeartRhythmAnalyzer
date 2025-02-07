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

def calculate_time_domain_parameters(rr_intervals):
    """Calculate time domain HRV parameters."""
    # Convert to numpy array if not already
    rr = np.array(rr_intervals)
    
    # Calculate basic statistics
    mean_rr = np.mean(rr)
    sdnn = np.std(rr)
    
    # Calculate RMSSD
    rr_diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    
    # Calculate pNN50
    nn50 = sum(abs(rr_diff) > 50)
    pnn50 = (nn50 / len(rr_diff)) * 100
    
    return {
        'Mean RR (ms)': round(mean_rr, 2),
        'SDNN (ms)': round(sdnn, 2),
        'RMSSD (ms)': round(rmssd, 2),
        'pNN50 (%)': round(pnn50, 2)
    }

def calculate_frequency_domain_parameters(rr_intervals, fs=4.0):
    """Calculate frequency domain HRV parameters."""
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
    
    # Define frequency bands
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # Calculate power in each band
    vlf_power = np.trapz(psd[(frequencies >= vlf_band[0]) & (frequencies < vlf_band[1])], 
                        frequencies[(frequencies >= vlf_band[0]) & (frequencies < vlf_band[1])])
    lf_power = np.trapz(psd[(frequencies >= lf_band[0]) & (frequencies < lf_band[1])], 
                       frequencies[(frequencies >= lf_band[0]) & (frequencies < lf_band[1])])
    hf_power = np.trapz(psd[(frequencies >= hf_band[0]) & (frequencies < hf_band[1])], 
                       frequencies[(frequencies >= hf_band[0]) & (frequencies < hf_band[1])])
    
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
