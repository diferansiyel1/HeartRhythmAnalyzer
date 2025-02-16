import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import warnings
import streamlit as st

def validate_rr_data(rr_intervals):
    """RR aralıklarını doğrula."""
    if not isinstance(rr_intervals, (list, np.ndarray)):
        return False, "RR aralıkları liste veya numpy dizisi olmalıdır."
    
    rr_intervals = np.array(rr_intervals)
    
    if len(rr_intervals) < 100:
        return False, "En az 100 RR aralığı gereklidir."
    
    if np.any(rr_intervals <= 0):
        return False, "Tüm RR aralıkları pozitif olmalıdır."
    
    if np.any(rr_intervals > 2000) or np.any(rr_intervals < 300):
        return False, "RR aralıkları 300-2000 ms aralığında olmalıdır."
    
    return True, "Veri doğrulama başarılı."

def calculate_time_domain_parameters(rr_intervals):
    """Zaman alanı parametrelerini hesapla."""
    rr_intervals = np.array(rr_intervals)
    
    # Ortalama kalp hızı
    mean_rr = np.mean(rr_intervals)
    mean_hr = 60000 / mean_rr
    
    # SDNN
    sdnn = np.std(rr_intervals)
    
    # RMSSD
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    
    # pNN50
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    pnn50 = (nn50 / len(rr_intervals)) * 100
    
    # Stress İndeksi (SI) hesaplama
    bins = np.arange(min(rr_intervals), max(rr_intervals) + 7.8125, 7.8125)
    hist, _ = np.histogram(rr_intervals, bins=bins)
    mode_bin_idx = np.argmax(hist)
    mode_rr = (bins[mode_bin_idx] + bins[mode_bin_idx + 1]) / 2
    
    amo = (hist[mode_bin_idx] / len(rr_intervals)) * 100
    mxdmn = max(rr_intervals) - min(rr_intervals)
    si = (amo / (2 * mode_rr * mxdmn)) * 1000000  # 1000000 ile çarparak düzeltme
    
    return {
        'Ortalama KH (atım/dk)': round(mean_hr, 2),
        'SDNN (ms)': round(sdnn, 2),
        'RMSSD (ms)': round(rmssd, 2),
        'pNN50 (%)': round(pnn50, 2),
        'Stress İndeksi': round(si, 2)
    }

def calculate_frequency_domain_parameters(rr_intervals, fs=4.0, vlf_range=(0.003, 0.04), 
                                       lf_range=(0.04, 0.15), hf_range=(0.15, 0.4)):
    """Frekans alanı parametrelerini hesapla."""
    try:
        # RR aralıklarını yeniden örnekle
        rr_intervals = np.array(rr_intervals)
        time = np.cumsum(rr_intervals) / 1000.0  # saniyeye çevir
        
        # Düzenli aralıklı zaman noktaları oluştur
        t_interpol = np.arange(time[0], time[-1], 1/fs)
        
        # RR aralıklarını interpolasyon ile yeniden örnekle
        f = interp1d(time, rr_intervals, kind='cubic')
        rr_interpol = f(t_interpol)
        
        # Trend kaldırma
        rr_detrend = signal.detrend(rr_interpol)
        
        # Güç spektral yoğunluğunu hesapla
        frequencies, psd = signal.welch(rr_detrend, fs=fs, nperseg=len(rr_detrend)//2)
        
        # Frekans bantlarındaki gücü hesapla
        vlf_power = np.trapz(psd[(frequencies >= vlf_range[0]) & (frequencies < vlf_range[1])])
        lf_power = np.trapz(psd[(frequencies >= lf_range[0]) & (frequencies < lf_range[1])])
        hf_power = np.trapz(psd[(frequencies >= hf_range[0]) & (frequencies < hf_range[1])])
        total_power = vlf_power + lf_power + hf_power
        
        # Normalize edilmiş güçleri hesapla
        lf_nu = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        hf_nu = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        lf_hf = lf_power/hf_power if hf_power > 0 else 0
        
        params = {
            'VLF Güç (ms²)': round(vlf_power, 2),
            'LF Güç (ms²)': round(lf_power, 2),
            'HF Güç (ms²)': round(hf_power, 2),
            'Toplam Güç (ms²)': round(total_power, 2),
            'LF/HF Oranı': round(lf_hf, 2),
            'LF (n.u.)': round(lf_nu, 2),
            'HF (n.u.)': round(hf_nu, 2)
        }
        
        return params, (frequencies, psd)
        
    except Exception as e:
        st.error(f"Frekans alanı parametreleri hesaplanırken hata oluştu: {str(e)}")
        # Boş sonuç döndür ama None değil
        return {}, (np.array([]), np.array([]))

def calculate_dfa(rr_intervals, scale_min=4, scale_max=64):
    """Detrended Fluctuation Analysis hesapla."""
    rr_intervals = np.array(rr_intervals)
    
    # Kümülatif toplam
    y = np.cumsum(rr_intervals - np.mean(rr_intervals))
    
    # Ölçek aralıklarını logaritmik olarak oluştur
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 20, dtype=int)
    fluct = np.zeros(len(scales))
    
    # Her ölçek için dalgalanmayı hesapla
    for i, scale in enumerate(scales):
        # Veriyi bölümlere ayır
        n_segments = int(len(y) / scale)
        
        if n_segments > 0:
            # Her bölüm için trend hesapla ve çıkar
            y_segments = np.array_split(y[:n_segments*scale], n_segments)
            t = np.arange(scale)
            
            fluctuations = []
            for segment in y_segments:
                p = np.polyfit(t, segment, 1)
                trend = np.polyval(p, t)
                fluctuations.append(np.sqrt(np.mean((segment - trend) ** 2)))
            
            fluct[i] = np.mean(fluctuations)
    
    # Logaritmik ölçeklerde dalgalanma-ölçek ilişkisini hesapla
    scales_log = np.log10(scales)
    fluct_log = np.log10(fluct)
    
    # Kısa ve uzun vadeli ölçekleri ayır
    idx_short = scales <= 16
    idx_long = scales > 16
    
    # Alpha değerlerini hesapla
    if np.sum(idx_short) > 1:
        alpha1, _ = np.polyfit(scales_log[idx_short], fluct_log[idx_short], 1)
    else:
        alpha1 = np.nan
        
    if np.sum(idx_long) > 1:
        alpha2, _ = np.polyfit(scales_log[idx_long], fluct_log[idx_long], 1)
    else:
        alpha2 = np.nan
    
    params = {
        'Alpha1': round(alpha1, 3) if not np.isnan(alpha1) else 'N/A',
        'Alpha2': round(alpha2, 3) if not np.isnan(alpha2) else 'N/A'
    }
    
    return params, (scales_log, fluct_log)