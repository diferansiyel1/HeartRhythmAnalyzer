import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io

def load_rr_intervals(file):
    """Load RR intervals from text file."""
    try:
        if isinstance(file, str):
            data = pd.read_csv(file, header=None, names=['RR'])
        else:
            content = file.getvalue().decode('utf-8')
            data = pd.read_csv(io.StringIO(content), header=None, names=['RR'])
        return data['RR'].values.tolist()
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def create_tachogram(rr_intervals):
    """Create tachogram plot using plotly."""
    time = np.cumsum(rr_intervals) / 1000  # Convert to seconds

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=rr_intervals,
        mode='lines',
        name='RR Intervals',
        line=dict(color='#2E86C1')
    ))

    fig.update_layout(
        title='Tachogram',
        xaxis_title='Time (s)',
        yaxis_title='RR Interval (ms)',
        showlegend=True,
        template='plotly_white'
    )

    return fig

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
        rr_intervals = load_rr_intervals(file)
        if not isinstance(rr_intervals, tuple):  # No error
            # Convert to milliseconds if needed
            if time_unit == "seconds":
                rr_intervals = [rr * 1000 for rr in rr_intervals]

            is_valid, message = validate_rr_data(rr_intervals)
            if is_valid:
                # Calculate total recording time
                total_time_ms = sum(rr_intervals)
                total_time_min = total_time_ms / (1000 * 60)

                # Calculate all parameters
                time_params = calculate_time_domain_parameters(rr_intervals)
                freq_params, _ = calculate_frequency_domain_parameters(rr_intervals)
                dfa_params, _ = calculate_dfa(rr_intervals)

                # Combine results
                result = {
                    'Filename': file.name,
                    'Recording Duration (min)': round(total_time_min, 2),
                    **time_params,
                    **freq_params,
                    **dfa_params
                }
                results.append(result)

    return pd.DataFrame(results)

def generate_report(time_params, freq_params, dfa_params=None, total_time_min=None):
    """Generate report as HTML string with modern styling."""
    html = """
    <style>
        .report-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(90deg, #2C3E50 0%, #3498DB 100%);
            color: white;
            padding: 20px;
            border-radius: 10px 10px 0 0;
            margin: -20px -20px 20px -20px;
            position: relative;
        }
        .header h2 {
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
        .section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 5px solid;
        }
        .time-domain {
            border-left-color: #2ECC71;
        }
        .frequency-domain {
            border-left-color: #E74C3C;
        }
        .dfa {
            border-left-color: #F39C12;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background: white;
            border-radius: 5px;
            overflow: hidden;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background-color: rgba(0,0,0,0.05);
            font-weight: 600;
            color: #2C3E50;
        }
        h2, h3 {
            color: #2C3E50;
            margin-top: 0;
        }
    </style>
    <div class="report-container">
        <div class="header">
            <h2>HRV Analysis Report</h2>
            <div class="signature">HFO</div>
        </div>
    """

    if total_time_min is not None:
        html += f"""
        <div class="section">
            <h3>Recording Information</h3>
            <table>
                <tr><th>Total Recording Time</th><td>{total_time_min:.2f} minutes</td></tr>
            </table>
        </div>
        """

    html += """
        <div class="section time-domain">
            <h3>Time Domain Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    """

    for param, value in time_params.items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"

    html += """
            </table>
        </div>

        <div class="section frequency-domain">
            <h3>Frequency Domain Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    """

    for param, value in freq_params.items():
        if param != 'PSD':
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"

    if dfa_params:
        html += """
            </table>
        </div>

        <div class="section dfa">
            <h3>Detrended Fluctuation Analysis</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
        """
        for param, value in dfa_params.items():
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"

    html += """
            </table>
        </div>
    </div>
    """

    return html

# Placeholder functions -  These need to be implemented separately based on your HRV analysis requirements.
def validate_rr_data(rr_intervals):
    # Add your validation logic here
    return True, ""

def calculate_time_domain_parameters(rr_intervals):
    # Add your time domain parameter calculations here
    return {}

def calculate_frequency_domain_parameters(rr_intervals):
    # Add your frequency domain parameter calculations here (including PSD)
    return {}, []

def calculate_dfa(rr_intervals):
    # Add your DFA calculation here
    return {}, []