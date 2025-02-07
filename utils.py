import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def load_rr_intervals(file):
    """Load RR intervals from text file."""
    try:
        data = pd.read_csv(file, header=None, names=['RR'])
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
        name='RR Intervals'
    ))
    
    fig.update_layout(
        title='Tachogram',
        xaxis_title='Time (s)',
        yaxis_title='RR Interval (ms)',
        showlegend=True
    )
    
    return fig

def create_psd_plot(frequencies, psd):
    """Create power spectral density plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=psd,
        mode='lines',
        name='PSD'
    ))
    
    fig.update_layout(
        title='Power Spectral Density',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power (ms²/Hz)',
        showlegend=True
    )
    
    # Add vertical lines for frequency bands
    for freq, label in [(0.04, 'VLF/LF'), (0.15, 'LF/HF')]:
        fig.add_vline(x=freq, line_dash="dash", annotation_text=label)
    
    return fig

def generate_report(time_params, freq_params):
    """Generate report as HTML string."""
    html = """
    <h2>HRV Analysis Report</h2>
    
    <h3>Time Domain Parameters</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    for param, value in time_params.items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html += """
    </table>
    
    <h3>Frequency Domain Parameters</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    for param, value in freq_params.items():
        if param != 'PSD':
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html += "</table>"
    
    return html
