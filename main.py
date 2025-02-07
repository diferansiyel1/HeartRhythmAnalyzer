import streamlit as st
import pandas as pd
import numpy as np
from hrv_analysis import (validate_rr_data, calculate_time_domain_parameters,
                         calculate_frequency_domain_parameters)
from utils import load_rr_intervals, create_tachogram, create_psd_plot, generate_report

# Page configuration
st.set_page_config(page_title="HRV Analysis Tool", layout="wide")

# Title and description
st.title("Heart Rate Variability Analysis Tool")
st.markdown("""
This application analyzes Heart Rate Variability (HRV) from RR interval data.
Upload a text file containing RR intervals (in milliseconds) to begin the analysis.
""")

try:
    # File upload
    uploaded_file = st.file_uploader("Upload RR interval data (txt file)", type=['txt'])

    if uploaded_file is not None:
        # Load and validate data
        rr_intervals = load_rr_intervals(uploaded_file)

        if isinstance(rr_intervals, tuple):  # Error occurred
            st.error(rr_intervals[1])
        else:
            # Validate the data
            is_valid, message = validate_rr_data(rr_intervals)

            if not is_valid:
                st.error(message)
            else:
                # Create two columns for layout
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Time Domain Analysis")
                    time_params = calculate_time_domain_parameters(rr_intervals)
                    st.table(pd.DataFrame(time_params.items(), columns=['Parameter', 'Value']))

                    # Display tachogram
                    st.plotly_chart(create_tachogram(rr_intervals), use_container_width=True)

                with col2:
                    st.subheader("Frequency Domain Analysis")
                    freq_params, psd_data = calculate_frequency_domain_parameters(rr_intervals)
                    st.table(pd.DataFrame(freq_params.items(), columns=['Parameter', 'Value']))

                    # Display PSD plot
                    st.plotly_chart(create_psd_plot(psd_data[0], psd_data[1]), use_container_width=True)

                # Generate report
                st.subheader("Analysis Report")
                report_html = generate_report(time_params, freq_params)
                st.markdown(report_html, unsafe_allow_html=True)

                # Download button for report
                report_df = pd.DataFrame({
                    'Parameter': list(time_params.keys()) + list(freq_params.keys()),
                    'Value': list(time_params.values()) + list(freq_params.values())
                })

                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name="hrv_analysis_report.csv",
                    mime="text/csv"
                )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Add explanatory text at the bottom
st.markdown("""
---
### Parameter Explanations

**Time Domain Parameters:**
- **SDNN**: Standard deviation of NN intervals
- **RMSSD**: Root mean square of successive differences
- **pNN50**: Percentage of successive NN intervals that differ by more than 50ms

**Frequency Domain Parameters:**
- **VLF**: Very low frequency (0.003-0.04 Hz)
- **LF**: Low frequency (0.04-0.15 Hz)
- **HF**: High frequency (0.15-0.4 Hz)
- **LF/HF Ratio**: Ratio between LF and HF power
""")