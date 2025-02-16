import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import streamlit as st

def load_rr_intervals(file):
    """RR aralıklarını dosyadan yükle."""
    try:
        if isinstance(file, str):
            # Dosya yolu verilmişse
            with open(file, 'r') as f:
                lines = f.readlines()
        else:
            # Streamlit file_uploader'dan gelen dosya
            content = file.getvalue().decode('utf-8')
            lines = content.split('\n')
        
        # Boş satırları temizle ve sayısal değerlere dönüştür
        valid_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Boş satırları atla
                try:
                    value = float(line)
                    if value > 0:  # Sadece pozitif değerleri kabul et
                        valid_lines.append(value)
                except ValueError:
                    continue
        
        if not valid_lines:
            st.error("Dosyada geçerli RR aralığı verisi bulunamadı.")
            st.info("Lütfen dosyanızın her satırında bir RR aralığı değeri olduğundan emin olun.")
            return None
        
        return valid_lines
        
    except Exception as e:
        st.error(f"Dosya okuma hatası: {str(e)}")
        st.info("Lütfen dosya formatını kontrol edin ve tekrar deneyin.")
        return None

def create_tachogram(rr_intervals):
    """Create interactive tachogram plot using plotly."""
    time = np.cumsum(rr_intervals) / 1000  # Convert to seconds

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=rr_intervals,
        mode='lines',
        name='RR Intervals',
        line=dict(color='#2E86C1'),
        selectedpoints=[],
        selected=dict(
            marker=dict(color='red')
        ),
        unselected=dict(
            marker=dict(opacity=0.3)
        )
    ))

    # Seçim aracını ekle
    fig.update_layout(
        title='Tachogram (Bölge seçmek için sürükleyin)',
        xaxis_title='Zaman (s)',
        yaxis_title='RR Aralığı (ms)',
        showlegend=True,
        template='plotly_white',
        dragmode='select',
        selectdirection='h',  # yatay seçim
        modebar=dict(
            add=['select2d', 'lasso2d'],
            remove=[],
            bgcolor='rgba(0,0,0,0)',
            color='rgba(0,0,0,0.3)',
            activecolor='rgba(0,0,0,0.8)'
        ),
        hovermode='closest'
    )

    # Seçim için config ekle
    fig.update_xaxes(rangeslider=dict(visible=True))
    
    # Seçim olayını dinle
    fig.update_layout(
        newshape=dict(line_color='red'),
        dragmode='select',
        clickmode='event+select'
    )
    
    return fig

def get_selected_rr_intervals(rr_intervals, start_time, end_time):
    """Get RR intervals within selected time range."""
    time = np.cumsum(rr_intervals) / 1000  # saniyeye çevir
    
    # Seçilen zaman aralığındaki indeksleri bul
    mask = (time >= start_time) & (time <= end_time)
    
    # Seçilen RR aralıklarını döndür
    return np.array(rr_intervals)[mask].tolist()

def create_psd_plot(frequencies, psd, vlf_range=(0.003, 0.04), lf_range=(0.04, 0.15), hf_range=(0.15, 0.4)):
    """Create power spectral density plot with adjustable frequency bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=psd,
        mode='lines',
        name='PSD',
        line=dict(color='#2E86C1')
    ))

    fig.update_layout(
        title='Power Spectral Density',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power (ms²/Hz)',
        showlegend=True,
        template='plotly_white'
    )

    # Add vertical lines and shaded areas for frequency bands
    colors = ['rgba(255,165,0,0.2)', 'rgba(144,238,144,0.2)', 'rgba(173,216,230,0.2)']
    bands = [
        ('VLF', vlf_range[0], vlf_range[1], colors[0]),
        ('LF', lf_range[0], lf_range[1], colors[1]),
        ('HF', hf_range[0], hf_range[1], colors[2])
    ]

    for band_name, start, end, color in bands:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=band_name,
            annotation_position="top left"
        )

    return fig

def create_dfa_plot(scales_log, fluct_log):
    """Create DFA plot with alpha1 and alpha2 regression lines."""
    fig = go.Figure()

    # Plot fluctuation vs. scale
    fig.add_trace(go.Scatter(
        x=scales_log,
        y=fluct_log,
        mode='markers',
        name='DFA',
        marker=dict(color='#2E86C1')
    ))

    # Split into short-term and long-term components
    idx_short = (10**scales_log <= 16)
    idx_long = (10**scales_log > 16)

    # Calculate and plot regression lines
    if np.sum(idx_short) > 1:
        alpha1, intercept1 = np.polyfit(scales_log[idx_short], fluct_log[idx_short], 1)
        y_fit1 = alpha1 * scales_log[idx_short] + intercept1
        fig.add_trace(go.Scatter(
            x=scales_log[idx_short],
            y=y_fit1,
            mode='lines',
            name=f'α1 = {alpha1:.3f}',
            line=dict(color='#28B463', dash='dash')
        ))

    if np.sum(idx_long) > 1:
        alpha2, intercept2 = np.polyfit(scales_log[idx_long], fluct_log[idx_long], 1)
        y_fit2 = alpha2 * scales_log[idx_long] + intercept2
        fig.add_trace(go.Scatter(
            x=scales_log[idx_long],
            y=y_fit2,
            mode='lines',
            name=f'α2 = {alpha2:.3f}',
            line=dict(color='#E74C3C', dash='dash')
        ))

    fig.update_layout(
        title='Detrended Fluctuation Analysis',
        xaxis_title='log₁₀(n)',
        yaxis_title='log₁₀(F(n))',
        showlegend=True,
        template='plotly_white'
    )

    return fig

def process_multiple_files(files, time_unit="milliseconds"):
    """Process multiple RR interval files and return combined results."""
    results = []
    
    for file in files:
        try:
            # Dosyayı oku
            rr_intervals = load_rr_intervals(file)
            
            # Hata kontrolü
            if rr_intervals is None:
                st.warning(f"{file.name}: Dosya okunamadı veya geçerli veri bulunamadı.")
                continue
            
            # Saniye cinsinden değerleri milisaniyeye çevir
            if time_unit == "seconds":
                rr_intervals = [rr * 1000 for rr in rr_intervals]

            # Veriyi doğrula
            is_valid, message = validate_rr_data(rr_intervals)
            if not is_valid:
                st.warning(f"{file.name}: {message}")
                continue

            # Toplam kayıt süresini hesapla
            total_time_ms = sum(rr_intervals)
            total_time_min = total_time_ms / (1000 * 60)

            # Tüm parametreleri hesapla
            time_params = calculate_time_domain_parameters(rr_intervals)
            freq_params, _ = calculate_frequency_domain_parameters(rr_intervals)
            dfa_params, _ = calculate_dfa(rr_intervals)

            # Sonuçları birleştir
            result = {
                'Dosya Adı': file.name,
                'Kayıt Süresi (dk)': round(total_time_min, 2),
                **time_params,
                **freq_params,
                **dfa_params
            }
            results.append(result)
            
        except Exception as e:
            st.warning(f"{file.name}: İşleme hatası - {str(e)}")
            continue

    # Sonuç kontrolü
    if not results:
        return pd.DataFrame()  # Boş DataFrame döndür
        
    return pd.DataFrame(results)

def generate_report(time_params, freq_params, dfa_params=None, total_time_min=None, psd_html=None, dfa_html=None, full_name=None, age=None, gender=None):
    """Generate report as HTML string with modern styling."""
    
    # Kişisel bilgileri kontrol et ve varsayılan değerler ata
    full_name = full_name if full_name else "Belirtilmedi"
    age = age if age else "Belirtilmedi"
    gender = gender if gender else "Belirtilmedi"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 20px;
        background: #f5f5f5;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .header {{
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }}
    .personal-info {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 15px;
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 8px;
    }}
    .section {{
        background: #f8f9fa;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }}
    th, td {{
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    th {{
        background: #f8f9fa;
    }}
    .plot-container {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }}
    .plot {{
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    @media (max-width: 768px) {{
        .plot-container, .personal-info {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h2>HRV Analiz Raporu</h2>
            <div class="personal-info">
                <div><strong>Ad Soyad:</strong><br>{full_name}</div>
                <div><strong>Yaş:</strong><br>{age}</div>
                <div><strong>Cinsiyet:</strong><br>{gender}</div>
            </div>
        </div>
    """

    if total_time_min is not None:
        html += f"""
        <div class="section">
            <h3>Kayıt Bilgileri</h3>
            <p><strong>Toplam Kayıt Süresi:</strong> {total_time_min:.2f} dakika</p>
        </div>
        """

    # Zaman Alanı Parametreleri
    html += """
        <div class="section">
            <h3>Zaman Alanı Parametreleri</h3>
            <table>
                <tr><th>Parametre</th><th>Değer</th></tr>
    """
    for param, value in time_params.items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    html += "</table></div>"

    # Frekans Alanı Parametreleri
    html += """
        <div class="section">
            <h3>Frekans Alanı Parametreleri</h3>
            <table>
                <tr><th>Parametre</th><th>Değer</th></tr>
    """
    for param, value in freq_params.items():
        if param != 'PSD':
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    html += "</table></div>"

    # DFA Parametreleri
    if dfa_params:
        html += """
            <div class="section">
                <h3>Detrended Fluctuation Analysis</h3>
                <table>
                    <tr><th>Parametre</th><th>Değer</th></tr>
        """
        for param, value in dfa_params.items():
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"
        html += "</table></div>"

    # Grafikler
    if psd_html or dfa_html:
        html += """
        <div class="section">
            <h3>Analiz Grafikleri</h3>
            <div class="plot-container">
        """
        if psd_html:
            html += f"""
                <div class="plot">
                    <h4>Güç Spektral Yoğunluğu</h4>
                    {psd_html}
                </div>
            """
        if dfa_html:
            html += f"""
                <div class="plot">
                    <h4>Detrended Fluctuation Analysis</h4>
                    {dfa_html}
                </div>
            """
        html += "</div></div>"

    html += """
    </div>
    </body>
    </html>
    """
    return html

# Placeholder functions -  These need to be implemented separately based on your HRV analysis requirements.
def validate_rr_data(rr_intervals):
    """RR aralıklarını doğrula."""
    try:
        if not rr_intervals or len(rr_intervals) < 100:
            return False, "En az 100 RR aralığı gereklidir."
        
        # RR aralıklarının pozitif olduğunu kontrol et
        if any(rr <= 0 for rr in rr_intervals):
            return False, "Tüm RR aralıkları pozitif olmalıdır."
        
        # Aşırı değerleri kontrol et (örn. 300ms - 2000ms arası)
        if any(rr < 300 or rr > 2000 for rr in rr_intervals):
            return False, "RR aralıkları 300ms ile 2000ms arasında olmalıdır."
        
        return True, "Veri doğrulama başarılı."
        
    except Exception as e:
        return False, f"Veri doğrulama hatası: {str(e)}"

def calculate_time_domain_parameters(rr_intervals):
    """Zaman alanı parametrelerini hesapla."""
    try:
        # RR aralıklarını numpy dizisine çevir
        rr = np.array(rr_intervals)
        
        # Temel istatistikler
        mean_rr = np.mean(rr)
        sdnn = np.std(rr)
        
        # Ardışık farklar
        diff_rr = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff_rr**2))
        
        # pNN50 hesapla
        nn50 = np.sum(np.abs(diff_rr) > 50.0)
        pnn50 = (nn50 / len(diff_rr)) * 100
        
        # Ortalama kalp atış hızı
        mean_hr = 60000 / mean_rr
        
        return {
            'Ortalama RR (ms)': round(mean_rr, 2),
            'SDNN (ms)': round(sdnn, 2),
            'RMSSD (ms)': round(rmssd, 2),
            'pNN50 (%)': round(pnn50, 2),
            'Ortalama HR (bpm)': round(mean_hr, 2)
        }
        
    except Exception as e:
        st.error(f"Zaman alanı parametreleri hesaplanırken hata oluştu: {str(e)}")
        return {}

def calculate_frequency_domain_parameters(rr_intervals):
    """Frekans alanı parametrelerini hesapla."""
    try:
        # Örnek frekans alanı parametreleri
        return {
            'VLF Power (ms²)': 1000,
            'LF Power (ms²)': 2000,
            'HF Power (ms²)': 1500,
            'LF/HF Ratio': 1.33
        }, [np.array([0.1, 0.2, 0.3]), np.array([100, 200, 300])]
        
    except Exception as e:
        st.error(f"Frekans alanı parametreleri hesaplanırken hata oluştu: {str(e)}")
        return {}, [np.array([]), np.array([])]

def calculate_dfa(rr_intervals):
    """DFA parametrelerini hesapla."""
    try:
        # Örnek DFA parametreleri
        return {
            'Alpha1': 1.2,
            'Alpha2': 0.8
        }, [np.array([1, 2, 3]), np.array([0.5, 1.0, 1.5])]
        
    except Exception as e:
        st.error(f"DFA parametreleri hesaplanırken hata oluştu: {str(e)}")
        return {}, [np.array([]), np.array([])]