import streamlit as st
import pandas as pd
import numpy as np
from streamlit.components.v1 import html
from hrv_analysis import (validate_rr_data, calculate_time_domain_parameters,
                         calculate_frequency_domain_parameters, calculate_dfa)
from utils import (load_rr_intervals, create_tachogram, create_psd_plot, 
                  create_dfa_plot, generate_report, process_multiple_files,
                  get_selected_rr_intervals)
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Callback fonksiyonu ekle
def handle_selection(selected_data):
    if selected_data and 'range' in selected_data:
        start_time = selected_data['range']['x'][0]
        end_time = selected_data['range']['x'][1]
        return start_time, end_time
    return None

# Page configuration
st.set_page_config(
    page_title="HRV Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with HFO signature and improved styling
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        .reportview-container {
            background: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        h1, h2, h3 {
            color: #2C3E50;
        }
        .banner {
            background: linear-gradient(90deg, #2C3E50 0%, #3498DB 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            position: relative;
        }
        .banner h1 {
            color: white !important;
            margin: 0;
        }
        .signature {
            position: absolute;
            bottom: 10px;
            right: 20px;
            color: rgba(255,255,255,0.8);
            font-style: italic;
        }
    </style>

    <div class="banner">
        <h1>Heart Rate Variability Analysis</h1>
        <div class="signature">HFO</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
This application analyzes Heart Rate Variability (HRV) from RR interval data.
Upload one or multiple text files containing RR intervals to begin the analysis.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Kişisel Bilgiler")
    full_name = st.text_input("Ad Soyad")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
    with col2:
        age = st.number_input("Yaş", min_value=0, max_value=120, value=30)

    st.header("Analiz Ayarları")
    # Analysis mode selection
    analysis_mode = st.radio("Analiz Modu", ["Tek Dosya", "Çoklu Dosya"])

    # Data format settings
    st.subheader("Veri Formatı")
    time_unit = st.selectbox("Zaman Birimi", ["milisaniye", "saniye"])

    # Frequency bands settings
    st.subheader("Frekans Bantları")
    with st.expander("Frekans Bandı Ayarları", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            vlf_low = st.number_input("VLF Alt (Hz)", value=0.003, format="%.3f", step=0.001)
            lf_low = st.number_input("LF Alt (Hz)", value=0.04, format="%.2f", step=0.01)
            hf_low = st.number_input("HF Alt (Hz)", value=0.15, format="%.2f", step=0.01)
        with col2:
            vlf_high = st.number_input("VLF Üst (Hz)", value=0.04, format="%.2f", step=0.01)
            lf_high = st.number_input("LF Üst (Hz)", value=0.15, format="%.2f", step=0.01)
            hf_high = st.number_input("HF Üst (Hz)", value=0.40, format="%.2f", step=0.01)

    # DFA settings
    st.subheader("DFA Ayarları")
    with st.expander("DFA Pencere Ayarları", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            scale_min = st.number_input("Min Pencere", value=4, min_value=4, step=1)
            alpha1_max = st.number_input("Alpha1 Max", value=16, min_value=8, step=1)
        with col2:
            scale_max = st.number_input("Max Pencere", value=64, min_value=32, step=1)
            alpha2_min = alpha1_max

try:
    if analysis_mode == "Tek Dosya":
        # Session state'i başlat
        if 'selected_range' not in st.session_state:
            st.session_state.selected_range = None
            st.session_state.analyzed_rr = None

        # Single file analysis
        uploaded_file = st.file_uploader("RR aralığı verisi yükleyin (txt dosyası)", type=['txt'], key='single_file')

        if uploaded_file is not None:
            st.info("Dosya işleniyor...")

            # Dosyayı yükle ve RR aralıklarını al
            rr_intervals = load_rr_intervals(uploaded_file)
            
            if rr_intervals is not None:
                # Convert to milliseconds if needed
                if time_unit == "seconds":
                    rr_intervals = [rr * 1000 for rr in rr_intervals]  # Convert to ms

                # Calculate total recording time
                total_time_ms = sum(rr_intervals)
                total_time_min = total_time_ms / (1000 * 60)  # Convert to minutes
                st.info(f"Toplam Kayıt Süresi: {total_time_min:.2f} dakika")

                # Validate the data
                is_valid, message = validate_rr_data(rr_intervals)

                if not is_valid:
                    st.error(message)
                else:
                    # İlk yüklemede analyzed_rr'yi ayarla
                    if st.session_state.analyzed_rr is None:
                        st.session_state.analyzed_rr = rr_intervals

                    # Takoğramı çiz ve seçim aracını göster
                    st.subheader("Analiz için Bölge Seçin")
                    
                    # Plotly grafiğini oluştur
                    fig = create_tachogram(rr_intervals)
                    
                    # Custom events için config ekle
                    config = {
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'modeBarButtonsToAdd': ['select2d', 'lasso2d', 'zoom', 'pan'],
                        'modeBarButtonsToRemove': [],
                    }
                    
                    # Plotly grafiğini göster
                    plot_placeholder = st.empty()
                    plot = plot_placeholder.plotly_chart(fig, use_container_width=True, key="tachogram", config=config)
                    
                    # JavaScript ile seçim olayını dinle
                    st.markdown("""
                    <script>
                        var plot = document.getElementById('tachogram');
                        plot.on('plotly_selected', function(eventData) {
                            if (eventData) {
                                var range = eventData.range;
                                if (range && range.x) {
                                    var start = range.x[0];
                                    var end = range.x[1];
                                    window.parent.postMessage({
                                        type: "streamlit:set_session_state",
                                        data: {
                                            plotly_selected_data: {
                                                xaxis: {range: [start, end]}
                                            }
                                        }
                                    }, "*");
                                }
                            }
                        });
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Manuel seçim için input alanları
                    st.write("Manuel Seçim")
                    time = np.cumsum(rr_intervals) / 1000  # saniyeye çevir
                    total_duration = time[-1]
                    
                    # Session state kontrolü
                    if 'selected_range' not in st.session_state or st.session_state.selected_range is None:
                        st.session_state.selected_range = (0.0, total_duration)
                    
                    # Grafik seçimini kontrol et
                    selected_data = st.session_state.get("plotly_selected_data")
                    if selected_data and 'range' in selected_data.get('xaxis', {}):
                        start = float(selected_data['xaxis']['range'][0])
                        end = float(selected_data['xaxis']['range'][1])
                        st.session_state.selected_range = (start, end)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_time = st.number_input("Başlangıç Zamanı (s)", 
                                                   min_value=float(0), 
                                                   max_value=float(total_duration),
                                                   value=float(st.session_state.selected_range[0]),
                                                   step=float(1.0),
                                                   key="start_time")
                    with col2:
                        end_time = st.number_input("Bitiş Zamanı (s)", 
                                                 min_value=float(0), 
                                                 max_value=float(total_duration),
                                                 value=float(st.session_state.selected_range[1]),
                                                 step=float(1.0),
                                                 key="end_time")
                    
                    if start_time < end_time:
                        # Tek analiz butonu
                        if st.button("Analiz Et", use_container_width=True):
                            # Seçim yapılıp yapılmadığını kontrol et
                            is_selection_made = (start_time > 0 or end_time < total_duration) and start_time < end_time
                            
                            # Analiz edilecek veriyi belirle
                            if is_selection_made:
                                selected_rr = get_selected_rr_intervals(rr_intervals, start_time, end_time)
                                analysis_message = f"Seçilen bölge analiz ediliyor: {start_time:.2f}s - {end_time:.2f}s"
                            else:
                                selected_rr = rr_intervals
                                start_time = 0
                                end_time = total_duration
                                analysis_message = "Tüm sinyal analiz ediliyor"
                            
                            if len(selected_rr) > 0:
                                # Session state'i güncelle
                                st.session_state.analyzed_rr = selected_rr
                                st.session_state.selected_range = (start_time, end_time)
                                
                                # Seçilen bölgenin süresini hesapla
                                duration = end_time - start_time
                                
                                # Seçilen bölge için analiz yap
                                time_params = calculate_time_domain_parameters(selected_rr)
                                freq_params, psd_data = calculate_frequency_domain_parameters(
                                    selected_rr,
                                    vlf_range=(vlf_low, vlf_high),
                                    lf_range=(lf_low, lf_high),
                                    hf_range=(hf_low, hf_high)
                                )
                                dfa_params, dfa_data = calculate_dfa(selected_rr, 
                                                                   scale_min=scale_min, 
                                                                   scale_max=scale_max)
                                
                                # Başarı mesajı göster
                                st.success(f"{analysis_message} (Süre: {duration:.2f}s)")
                                
                                # Seçili bölge bilgisini göster
                                n_intervals = len(selected_rr)
                                st.info(f"""
                                Seçili Bölge:
                                - Başlangıç: {start_time:.2f}s
                                - Bitiş: {end_time:.2f}s
                                - Süre: {duration:.2f}s
                                - RR Aralığı Sayısı: {n_intervals}
                                """)
                                
                                # Analiz sonuçlarını göster
                                st.markdown("---")
                                st.markdown("## Analiz Sonuçları")
                                
                                # Sekmeli görünüm için tab'ları oluştur
                                tab1, tab2, tab3 = st.tabs(["Zaman Alanı Analizi", "Frekans Alanı Analizi", "DFA Analizi"])
                                
                                with tab1:
                                    st.markdown("### Zaman Alanı Parametreleri")
                                    time_df = pd.DataFrame(time_params.items(), columns=['Parametre', 'Değer'])
                                    st.dataframe(time_df, use_container_width=True)
                                
                                with tab2:
                                    st.markdown("### Frekans Alanı Parametreleri")
                                    freq_df = pd.DataFrame(freq_params.items(), columns=['Parametre', 'Değer'])
                                    st.dataframe(freq_df, use_container_width=True)
                                    
                                    st.plotly_chart(create_psd_plot(
                                        psd_data[0], psd_data[1],
                                        vlf_range=(vlf_low, vlf_high),
                                        lf_range=(lf_low, lf_high),
                                        hf_range=(hf_low, hf_high)
                                    ), use_container_width=True)
                                
                                with tab3:
                                    st.markdown("### DFA Parametreleri")
                                    dfa_df = pd.DataFrame(dfa_params.items(), columns=['Parametre', 'Değer'])
                                    st.dataframe(dfa_df, use_container_width=True)
                                    
                                    st.plotly_chart(create_dfa_plot(dfa_data[0], dfa_data[1]), use_container_width=True)
                                
                                # HTML raporu oluştur
                                st.markdown("## Analiz Raporu")
                                selected_time_min = duration / 60  # Convert to minutes
                                
                                # Grafikleri HTML formatına dönüştür
                                psd_plot = create_psd_plot(
                                    psd_data[0], psd_data[1],
                                    vlf_range=(vlf_low, vlf_high),
                                    lf_range=(lf_low, lf_high),
                                    hf_range=(hf_low, hf_high)
                                )
                                dfa_plot = create_dfa_plot(dfa_data[0], dfa_data[1])
                                
                                # Grafikleri HTML'e çevir
                                psd_html = psd_plot.to_html(full_html=False, include_plotlyjs='cdn')
                                dfa_html = dfa_plot.to_html(full_html=False, include_plotlyjs='cdn')
                                
                                # Raporu oluştur
                                report_html = generate_report(
                                    time_params, 
                                    freq_params, 
                                    dfa_params, 
                                    selected_time_min,
                                    psd_html=psd_html,
                                    dfa_html=dfa_html,
                                    full_name=full_name,  # Kişisel bilgileri ekle
                                    age=age,
                                    gender=gender
                                )
                                st.components.v1.html(report_html, height=1200, scrolling=True)  # Yüksekliği artır ve kaydırmayı etkinleştir
                                
                                # Download butonları
                                col1, col2 = st.columns(2)
                                with col1:
                                    # CSV rapor
                                    report_df = pd.DataFrame({
                                        'Parameter': (list(time_params.keys()) + 
                                                    list(freq_params.keys()) + 
                                                    list(dfa_params.keys())),
                                        'Value': (list(time_params.values()) + 
                                                list(freq_params.values()) + 
                                                list(dfa_params.values()))
                                    })
                                    csv = report_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Report (CSV)",
                                        data=csv,
                                        file_name="hrv_analysis_report.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    # HTML rapor
                                    st.download_button(
                                        label="Download Report (HTML)",
                                        data=report_html,
                                        file_name="hrv_analysis_report.html",
                                        mime="text/html"
                                    )
                            else:
                                st.error("Seçilen aralıkta veri bulunamadı!")
                    else:
                        st.error("Başlangıç zamanı bitiş zamanından küçük olmalıdır!")
                    
                    st.markdown("---")
                    
                    # Seçili bölge bilgisini göster
                    if st.session_state.selected_range:
                        start, end = st.session_state.selected_range
                        duration = end - start
                        n_intervals = len(st.session_state.analyzed_rr)
                        st.info(f"""
                        Seçili Bölge:
                        - Başlangıç: {start:.2f}s
                        - Bitiş: {end:.2f}s
                        - Süre: {duration:.2f}s
                        - RR Aralığı Sayısı: {n_intervals}
                        """)

    else:
        # Multiple files analysis
        uploaded_files = st.file_uploader("RR aralığı veri dosyalarını yükleyin", type=['txt'], accept_multiple_files=True)

        if uploaded_files:
            st.info(f"{len(uploaded_files)} dosya işleniyor...")

            try:
                # Tüm dosyaları işle
                results_df = process_multiple_files(uploaded_files, time_unit)

                if not results_df.empty:
                    st.subheader("Birleştirilmiş Analiz Sonuçları")
                    st.dataframe(results_df)

                    # Birleştirilmiş sonuçları indir
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Birleştirilmiş Sonuçları İndir (CSV)",
                        data=csv,
                        file_name="hrv_analiz_sonuclari.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Yüklenen dosyalardan geçerli sonuç üretilemedi. Lütfen dosya formatlarını kontrol edin.")

            except Exception as e:
                st.error(f"Dosya işleme hatası: {str(e)}")
                st.info("Lütfen her dosyanın her satırında bir RR aralığı değeri olduğundan emin olun.")

except Exception as e:
    st.error(f"An application error occurred: {str(e)}")

# Add explanatory text at the bottom
st.markdown("""
---
### Parameter Explanations

**Time Domain Parameters:**
- **Mean HR**: Mean heart rate in beats per minute
- **SDNN**: Standard deviation of NN intervals
- **RMSSD**: Root mean square of successive differences
- **pNN50**: Percentage of successive NN intervals that differ by more than 50ms
- **Stress Index**: Baevsky's Stress Index, indicates the level of cardiovascular system stress (Normal range: 50-150)

**Frequency Domain Parameters:**
- **VLF**: Very low frequency power
- **LF**: Low frequency power
- **HF**: High frequency power
- **LF/HF Ratio**: Ratio between LF and HF power

**Detrended Fluctuation Analysis:**
- **α1**: Short-term scaling exponent (4-16 beats)
- **α2**: Long-term scaling exponent (16-64 beats)
""")

# Rapor oluşturma fonksiyonunu güncelle
def generate_report(time_params, freq_params, dfa_params=None, total_time_min=None, psd_html=None, dfa_html=None, full_name=None, age=None, gender=None):
    """Generate report as HTML string with modern styling."""
    html = """
    <style>
        .report-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            margin: -20px -20px 20px -20px;
            position: relative;
        }
        .header h2 {
            color: white !important;
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }
        .personal-info {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .signature {
            position: absolute;
            bottom: 10px;
            right: 20px;
            color: rgba(255,255,255,0.8);
            font-style: italic;
        }
        .section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            border-left: 5px solid;
            transition: transform 0.2s;
        }
        .section:hover {
            transform: translateX(5px);
        }
        .time-domain { border-left-color: #2ECC71; }
        .frequency-domain { border-left-color: #E74C3C; }
        .dfa { border-left-color: #F39C12; }
        .plots { border-left-color: #3498DB; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2C3E50;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        h2, h3 {
            color: #2C3E50;
            margin-top: 0;
            font-weight: 600;
        }
        .plot-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .plot {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        @media (max-width: 768px) {
            .plot-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
    <div class="report-container">
        <div class="header">
            <h2>HRV Analiz Raporu</h2>
            <div class="personal-info">
                <strong>Ad Soyad:</strong> {full_name}<br>
                <strong>Yaş:</strong> {age}<br>
                <strong>Cinsiyet:</strong> {gender}
            </div>
            <div class="signature">HFO</div>
        </div>
    """.format(full_name=full_name, age=age, gender=gender)

    # ... rest of the report generation code ...