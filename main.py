import streamlit as st
import pandas as pd
import numpy as np
from hrv_analysis import (validate_rr_data, calculate_time_domain_parameters,
                         calculate_frequency_domain_parameters, calculate_dfa)
from utils import (load_rr_intervals, create_tachogram, create_psd_plot, 
                  create_dfa_plot, generate_report, process_multiple_files)

# Page configuration
st.set_page_config(
    page_title="HRV Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
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
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Heart Rate Variability Analysis Tool")
st.markdown("""
This application analyzes Heart Rate Variability (HRV) from RR interval data.
Upload one or multiple text files containing RR intervals (in milliseconds) to begin the analysis.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Analysis Settings")

    # Analysis mode selection
    analysis_mode = st.radio("Analysis Mode", ["Single File", "Multiple Files"])

    # Frequency bands settings
    st.subheader("Frequency Bands")
    col1, col2 = st.columns(2)

    with col1:
        vlf_low = st.number_input("VLF Low (Hz)", value=0.003, format="%.3f", step=0.001)
        lf_low = st.number_input("LF Low (Hz)", value=0.04, format="%.2f", step=0.01)
        hf_low = st.number_input("HF Low (Hz)", value=0.15, format="%.2f", step=0.01)

    with col2:
        vlf_high = st.number_input("VLF High (Hz)", value=0.04, format="%.2f", step=0.01)
        lf_high = st.number_input("LF High (Hz)", value=0.15, format="%.2f", step=0.01)
        hf_high = st.number_input("HF High (Hz)", value=0.40, format="%.2f", step=0.01)

try:
    if analysis_mode == "Single File":
        # Single file analysis
        uploaded_file = st.file_uploader("Upload RR interval data (txt file)", type=['txt'], key='single_file')

        if uploaded_file is not None:
            st.info("Processing file...")

            try:
                # Read file content directly
                file_contents = uploaded_file.read().decode()
                rr_intervals = [float(line.strip()) for line in file_contents.split('\n') if line.strip()]

                # Validate the data
                is_valid, message = validate_rr_data(rr_intervals)

                if not is_valid:
                    st.error(message)
                else:
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "DFA"])

                    with tab1:
                        st.subheader("Time Domain Analysis")
                        time_params = calculate_time_domain_parameters(rr_intervals)
                        st.table(pd.DataFrame(time_params.items(), columns=['Parameter', 'Value']))
                        st.plotly_chart(create_tachogram(rr_intervals), use_container_width=True)

                    with tab2:
                        st.subheader("Frequency Domain Analysis")
                        freq_params, psd_data = calculate_frequency_domain_parameters(
                            rr_intervals,
                            vlf_range=(vlf_low, vlf_high),
                            lf_range=(lf_low, lf_high),
                            hf_range=(hf_low, hf_high)
                        )
                        st.table(pd.DataFrame(freq_params.items(), columns=['Parameter', 'Value']))
                        st.plotly_chart(create_psd_plot(
                            psd_data[0], psd_data[1],
                            vlf_range=(vlf_low, vlf_high),
                            lf_range=(lf_low, lf_high),
                            hf_range=(hf_low, hf_high)
                        ), use_container_width=True)

                    with tab3:
                        st.subheader("Detrended Fluctuation Analysis")
                        dfa_params, dfa_data = calculate_dfa(rr_intervals)
                        st.table(pd.DataFrame(dfa_params.items(), columns=['Parameter', 'Value']))
                        st.plotly_chart(create_dfa_plot(dfa_data[0], dfa_data[1]), use_container_width=True)

                    # Generate and display report
                    st.subheader("Analysis Report")
                    report_html = generate_report(time_params, freq_params, dfa_params)
                    st.markdown(report_html, unsafe_allow_html=True)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        # CSV report
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
                        # HTML report
                        st.download_button(
                            label="Download Report (HTML)",
                            data=report_html,
                            file_name="hrv_analysis_report.html",
                            mime="text/html"
                        )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please make sure your file contains one RR interval value per line.")

    else:
        # Multiple files analysis
        uploaded_files = st.file_uploader("Upload RR interval data files", type=['txt'], accept_multiple_files=True)

        if uploaded_files:
            st.info(f"Processing {len(uploaded_files)} files...")

            try:
                # Process all files
                results_df = process_multiple_files(uploaded_files)

                if not results_df.empty:
                    st.subheader("Combined Analysis Results")
                    st.dataframe(results_df)

                    # Download combined results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Combined Results (CSV)",
                        data=csv,
                        file_name="hrv_analysis_combined_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No valid results were generated from the uploaded files.")

            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

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

**Frequency Domain Parameters:**
- **VLF**: Very low frequency power
- **LF**: Low frequency power
- **HF**: High frequency power
- **LF/HF Ratio**: Ratio between LF and HF power

**Detrended Fluctuation Analysis:**
- **α1**: Short-term scaling exponent (4-16 beats)
- **α2**: Long-term scaling exponent (16-64 beats)
""")