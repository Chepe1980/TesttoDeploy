import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import convolve
import pandas as pd
import io

# ==============================================
# Core Functions (unchanged)
# ==============================================
def ricker_wavelet(freq, length, dt):
    """Generate a Ricker wavelet for seismic modeling"""
    t = np.arange(-length/2, length/2, dt)
    return (1 - 2*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)

def calculate_reflectivity(vp, vs, d, e, g, dlt, theta, azimuth):
    """Calculate anisotropic reflectivity coefficients"""
    VP2 = (vp[1] + vp[2])/2
    VS2 = (vs[1] + vs[2])/2
    DEN2 = (d[1] + d[2])/2

    A2 = -0.5 * ((vp[2]-vp[1])/VP2 + (d[2]-d[1])/DEN2)
    
    az_rad = np.radians(azimuth)
    Biso2 = 0.5*((vp[2]-vp[1])/VP2) - 2*(VS2/VP2)**2*(d[2]-d[1])/DEN2 - 4*(VS2/VP2)**2*(vs[2]-vs[1])/VS2
    Baniso2 = 0.5*((dlt[2]-dlt[1]) + 2*(2*VS2/VP2)**2*(g[2]-g[1]))
    Caniso2 = 0.5*((vp[2]-vp[1])/VP2 - (e[2]-e[1])*np.cos(az_rad)**4 + (dlt[2]-dlt[1])*np.sin(az_rad)**2*np.cos(az_rad)**2)
    
    return A2 + (Biso2 + Baniso2*np.cos(az_rad)**2)*np.sin(theta)**2 + Caniso2*np.sin(theta)**2*np.tan(theta)**2

def brown_korringa_substitution(Km, Gm, Ks, Gs, Kf, phi, delta, gamma):
    """Brown-Korringa fluid substitution for anisotropic media"""
    beta = 1 - (Ks/Km)
    K_sat = Ks + (beta**2) / ((phi/Kf) + ((beta - phi)/Km) - (delta*Ks)/(3*Km))
    G_sat = Gs * (1 - (gamma*Ks)/(3*Km))
    
    # Update anisotropy parameters
    delta_sat = delta * (K_sat/Ks)
    gamma_sat = gamma * (G_sat/Gs)
    
    return K_sat, G_sat, delta_sat, gamma_sat

def moduli_to_velocity(K, G, density):
    """Convert bulk and shear moduli to Vp and Vs"""
    Vp = np.sqrt((K + 4/3*G)/density)
    Vs = np.sqrt(G/density)
    return Vp, Vs

def velocity_to_moduli(Vp, Vs, density):
    """Convert Vp and Vs to bulk and shear moduli"""
    G = density * Vs**2
    K = density * Vp**2 - (4/3)*G
    return K, G

def create_3d_plot(x, y, z, vp, colormap='Viridis'):
    """Create interactive 3D velocity surface plot"""
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=vp,
            colorscale=colormap,
            colorbar=dict(title='Velocity (m/s)'),
            opacity=0.9,
            hoverinfo='x+y+z+text',
            text=[f'Vp: {val:.0f} m/s' for val in vp.flatten()]
        )
    ])
    
    fig.update_layout(
        title='3D Velocity Surface (Drag to rotate)',
        scene=dict(
            xaxis_title='X [m/s]',
            yaxis_title='Y [m/s]',
            zaxis_title='Z [m/s]',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    return fig

def morlet_wavelet(t, s=1.0, w=5.0):
    """Morlet wavelet function"""
    return np.pi**(-0.25) * np.exp(1j * w * t) * np.exp(-0.5 * t**2)

def cwt_analysis(signal_data, scales):
    """Perform Continuous Wavelet Transform on signal data"""
    n_samples = len(signal_data)
    n_scales = len(scales)
    
    # Create empty array for CWT results
    cwt_matrix = np.zeros((n_scales, n_samples))
    
    # Perform CWT for each scale
    for i, scale in enumerate(scales):
        # Ensure scale is at least 1
        scale = max(1, scale)
        
        # Create wavelet with proper length
        wavelet_length = min(10 * scale, n_samples)
        if wavelet_length % 2 == 0:
            wavelet_length += 1  # Make it odd for symmetry
        
        t = np.linspace(-scale*3, scale*3, wavelet_length)
        wavelet = morlet_wavelet(t/scale)
        wavelet = wavelet.real  # Take real part for analysis
        
        # Normalize wavelet
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
        
        # Convolution with signal (mode='same' returns same length)
        conv_result = np.convolve(signal_data, wavelet, mode='same')
        
        # Ensure we have the right length
        if len(conv_result) == n_samples:
            cwt_matrix[i, :] = conv_result
        else:
            # Trim or pad to match expected length
            if len(conv_result) > n_samples:
                cwt_matrix[i, :] = conv_result[:n_samples]
            else:
                cwt_matrix[i, :n_samples] = conv_result
    
    return cwt_matrix  # Return magnitude

def plot_horizontal_angles_seismic_plotly(results, seismic_cmap, diff_cmap):
    """Plot seismic gathers with incidence angles arranged horizontally using Plotly"""
    st.header("Seismic Gathers - Horizontal Angle Arrangement")
    st.markdown("All incidence angles (0-50°) arranged horizontally, each showing azimuth 0-360°")
    
    # Get dimensions
    n_angles = len(results['incidence_angles'])
    time_samples = results['seismic_orig'][0].shape[0]
    n_azimuths = results['seismic_orig'][0].shape[1]
    
    # Original Seismic - Horizontal arrangement with reflectivity plots below
    st.subheader("Original Seismic - Horizontal Angle Arrangement with Reflectivity vs Azimuth")
    
    # Create subplot figure with 2 rows for each angle (heatmap on top, reflectivity below)
    fig_orig = make_subplots(
        rows=2, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']] + 
                      [f'Reflectivity at {angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.03,
        row_heights=[0.5, 0.5]
    )
    
    # Set global vmax for consistent color scaling in heatmaps
    vmax_global = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.abs(results['seismic_orig'][angle_idx]).max()
        vmax_global = max(vmax_global, vmax_angle)
    
    # Add heatmaps in top row and reflectivity plots in bottom row
    for angle_idx in range(n_angles):
        seismic_data = results['seismic_orig'][angle_idx]
        reflectivity_data = results['reflectivity_orig'][angle_idx, :]
        
        # Add heatmap in top row
        fig_orig.add_trace(
            go.Heatmap(
                z=seismic_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=seismic_cmap,
                zmin=-vmax_global,
                zmax=vmax_global,
                showscale=angle_idx == n_angles-1,  # Only show colorbar for last plot
                colorbar=dict(title="Amplitude", len=0.4, y=0.75, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Add reflectivity vs azimuth plot in bottom row
        fig_orig.add_trace(
            go.Scatter(
                x=results['azimuths'],
                y=reflectivity_data,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                showlegend=False,
                name=f'Reflectivity {results["incidence_angles"][angle_idx]:.0f}°'
            ),
            row=2, col=angle_idx+1
        )
        
        # Update axes for top row (heatmaps)
        fig_orig.update_xaxes(
            title_text="",
            row=1, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_orig.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_orig.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
        
        # Update axes for bottom row (reflectivity plots)
        fig_orig.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=2, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_orig.update_yaxes(title_text="Reflectivity", row=2, col=1)
        else:
            fig_orig.update_yaxes(showticklabels=False, row=2, col=angle_idx+1)
        
        # Add horizontal line at y=0 for reference in reflectivity plots
        fig_orig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=angle_idx+1)
    
    fig_orig.update_layout(
        height=900,
        title_text="Original Seismic - Heatmaps (top) with Reflectivity vs Azimuth (bottom)",
        showlegend=False
    )
    st.plotly_chart(fig_orig, use_container_width=True)
    
    # Fluid-Substituted Seismic - Horizontal arrangement with reflectivity plots below
    st.subheader("Fluid-Substituted Seismic - Horizontal Angle Arrangement with Reflectivity vs Azimuth")
    
    fig_sub = make_subplots(
        rows=2, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']] + 
                      [f'Reflectivity at {angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.03,
        row_heights=[0.5, 0.5]
    )
    
    # Set global vmax for consistent color scaling
    vmax_global_sub = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.abs(results['seismic_sub'][angle_idx]).max()
        vmax_global_sub = max(vmax_global_sub, vmax_angle)
    
    # Add heatmaps in top row and reflectivity plots in bottom row
    for angle_idx in range(n_angles):
        seismic_data = results['seismic_sub'][angle_idx]
        reflectivity_data = results['reflectivity_sub'][angle_idx, :]
        
        # Add heatmap in top row
        fig_sub.add_trace(
            go.Heatmap(
                z=seismic_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=seismic_cmap,
                zmin=-vmax_global_sub,
                zmax=vmax_global_sub,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="Amplitude", len=0.4, y=0.75, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Add reflectivity vs azimuth plot in bottom row
        fig_sub.add_trace(
            go.Scatter(
                x=results['azimuths'],
                y=reflectivity_data,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                showlegend=False,
                name=f'Reflectivity {results["incidence_angles"][angle_idx]:.0f}°'
            ),
            row=2, col=angle_idx+1
        )
        
        # Update axes for top row (heatmaps)
        fig_sub.update_xaxes(
            title_text="",
            row=1, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_sub.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_sub.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
        
        # Update axes for bottom row (reflectivity plots)
        fig_sub.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=2, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_sub.update_yaxes(title_text="Reflectivity", row=2, col=1)
        else:
            fig_sub.update_yaxes(showticklabels=False, row=2, col=angle_idx+1)
        
        # Add horizontal line at y=0 for reference in reflectivity plots
        fig_sub.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=angle_idx+1)
    
    fig_sub.update_layout(
        height=900,
        title_text="Fluid-Substituted Seismic - Heatmaps (top) with Reflectivity vs Azimuth (bottom)",
        showlegend=False
    )
    st.plotly_chart(fig_sub, use_container_width=True)
    
    # Difference plots - Horizontal arrangement with reflectivity difference below
    st.subheader("Seismic Difference - Horizontal Angle Arrangement with Reflectivity Difference")
    
    fig_diff = make_subplots(
        rows=2, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']] + 
                      [f'Reflectivity Diff at {angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.03,
        row_heights=[0.5, 0.5]
    )
    
    # Set global vmax for consistent color scaling
    vmax_global_diff = 0
    diff_data_all = []
    for angle_idx in range(n_angles):
        diff_data = results['seismic_sub'][angle_idx] - results['seismic_orig'][angle_idx]
        diff_data_all.append(diff_data)
        vmax_angle = np.abs(diff_data).max()
        vmax_global_diff = max(vmax_global_diff, vmax_angle)
    
    # Calculate reflectivity difference
    reflectivity_diff = results['reflectivity_sub'] - results['reflectivity_orig']
    max_reflectivity_diff = np.max(np.abs(reflectivity_diff))
    
    # Add heatmaps in top row and reflectivity difference plots in bottom row
    for angle_idx in range(n_angles):
        diff_data = diff_data_all[angle_idx]
        reflectivity_diff_data = reflectivity_diff[angle_idx, :]
        
        # Add heatmap in top row
        fig_diff.add_trace(
            go.Heatmap(
                z=diff_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=diff_cmap,
                zmin=-vmax_global_diff,
                zmax=vmax_global_diff,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="Amplitude Diff", len=0.4, y=0.75, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Add reflectivity difference vs azimuth plot in bottom row
        fig_diff.add_trace(
            go.Scatter(
                x=results['azimuths'],
                y=reflectivity_diff_data,
                mode='lines+markers',
                line=dict(color='purple', width=2),
                marker=dict(size=4),
                showlegend=False,
                name=f'Reflectivity Diff {results["incidence_angles"][angle_idx]:.0f}°'
            ),
            row=2, col=angle_idx+1
        )
        
        # Update axes for top row (heatmaps)
        fig_diff.update_xaxes(
            title_text="",
            row=1, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_diff.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_diff.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
        
        # Update axes for bottom row (reflectivity difference plots)
        fig_diff.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=2, col=angle_idx+1,
            showticklabels=angle_idx == n_angles-1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_diff.update_yaxes(title_text="Reflectivity Diff", row=2, col=1)
        else:
            fig_diff.update_yaxes(showticklabels=False, row=2, col=angle_idx+1)
        
        # Add horizontal line at y=0 for reference in reflectivity difference plots
        fig_diff.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=angle_idx+1)
    
    fig_diff.update_layout(
        height=900,
        title_text="Seismic Difference - Heatmaps (top) with Reflectivity Difference vs Azimuth (bottom)",
        showlegend=False
    )
    st.plotly_chart(fig_diff, use_container_width=True)

def plot_horizontal_angles_cwt_plotly(results, cwt_scale, cwt_cmap, diff_cmap):
    """Plot CWT analysis with incidence angles arranged horizontally using Plotly"""
    st.header("CWT Analysis - Horizontal Angle Arrangement")
    st.markdown(f"Continuous Wavelet Transform at Scale {cwt_scale} - All incidence angles arranged horizontally")
    
    # Get dimensions
    n_angles = len(results['incidence_angles'])
    time_samples = results['seismic_orig'][0].shape[0]
    n_azimuths = results['seismic_orig'][0].shape[1]
    
    # Perform CWT for all angles and azimuths
    with st.spinner(f"Performing CWT analysis at scale {cwt_scale}..."):
        # Initialize arrays for CWT results
        cwt_orig = []
        cwt_sub = []
        
        # Process each angle
        for angle_idx in range(n_angles):
            # Get seismic data for this angle
            seismic_orig_angle = results['seismic_orig'][angle_idx]
            seismic_sub_angle = results['seismic_sub'][angle_idx]
            
            # Initialize arrays for CWT results at selected scale
            cwt_orig_at_scale = np.zeros_like(seismic_orig_angle)
            cwt_sub_at_scale = np.zeros_like(seismic_sub_angle)
            
            # Perform CWT for each azimuth trace
            for az_idx in range(n_azimuths):
                # Original seismic trace
                trace_orig = seismic_orig_angle[:, az_idx]
                if len(trace_orig) > 0:
                    cwt_result_orig = cwt_analysis(trace_orig, [cwt_scale])
                    cwt_orig_at_scale[:, az_idx] = cwt_result_orig[0, :]
                
                # Fluid-substituted seismic trace
                trace_sub = seismic_sub_angle[:, az_idx]
                if len(trace_sub) > 0:
                    cwt_result_sub = cwt_analysis(trace_sub, [cwt_scale])
                    cwt_sub_at_scale[:, az_idx] = cwt_result_sub[0, :]
            
            # Store results for this angle
            cwt_orig.append(cwt_orig_at_scale)
            cwt_sub.append(cwt_sub_at_scale)
    
    # Original CWT - Horizontal arrangement
    st.subheader(f"Original CWT at Scale {cwt_scale} - Horizontal Angle Arrangement")
    
    fig_cwt_orig = make_subplots(
        rows=1, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Set global vmax for consistent color scaling
    vmax_global = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.max(cwt_orig[angle_idx])
        vmax_global = max(vmax_global, vmax_angle)
    
    # Add each subplot
    for angle_idx in range(n_angles):
        cwt_data = cwt_orig[angle_idx]
        
        fig_cwt_orig.add_trace(
            go.Heatmap(
                z=cwt_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=cwt_cmap,
                zmin=0,
                zmax=vmax_global,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Update axes
        fig_cwt_orig.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=1, col=angle_idx+1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_cwt_orig.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_cwt_orig.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
    
    fig_cwt_orig.update_layout(
        height=500,
        title_text=f"Original CWT at Scale {cwt_scale} - All Incidence Angles (0-50°)",
        showlegend=False
    )
    st.plotly_chart(fig_cwt_orig, use_container_width=True)
    
    # Fluid-Substituted CWT - Horizontal arrangement
    st.subheader(f"Fluid-Substituted CWT at Scale {cwt_scale} - Horizontal Angle Arrangement")
    
    fig_cwt_sub = make_subplots(
        rows=1, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Set global vmax for consistent color scaling
    vmax_global_sub = 0
    for angle_idx in range(n_angles):
        vmax_angle = np.max(cwt_sub[angle_idx])
        vmax_global_sub = max(vmax_global_sub, vmax_angle)
    
    # Add each subplot
    for angle_idx in range(n_angles):
        cwt_data = cwt_sub[angle_idx]
        
        fig_cwt_sub.add_trace(
            go.Heatmap(
                z=cwt_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=cwt_cmap,
                zmin=0,
                zmax=vmax_global_sub,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Update axes
        fig_cwt_sub.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=1, col=angle_idx+1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_cwt_sub.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_cwt_sub.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
    
    fig_cwt_sub.update_layout(
        height=500,
        title_text=f"Fluid-Substituted CWT at Scale {cwt_scale} - All Incidence Angles (0-50°)",
        showlegend=False
    )
    st.plotly_chart(fig_cwt_sub, use_container_width=True)
    
    # CWT Difference plots - Horizontal arrangement
    st.subheader(f"CWT Difference at Scale {cwt_scale} - Horizontal Angle Arrangement")
    
    fig_cwt_diff = make_subplots(
        rows=1, cols=n_angles,
        subplot_titles=[f'{angle:.0f}°' for angle in results['incidence_angles']],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Calculate differences and set global vmax for consistent color scaling
    vmax_global_diff = 0
    cwt_diff = []
    
    for angle_idx in range(n_angles):
        diff_data = cwt_sub[angle_idx] - cwt_orig[angle_idx]
        cwt_diff.append(diff_data)
        vmax_angle = np.max(np.abs(diff_data))
        vmax_global_diff = max(vmax_global_diff, vmax_angle)
    
    # Add each subplot
    for angle_idx in range(n_angles):
        diff_data = cwt_diff[angle_idx]
        
        fig_cwt_diff.add_trace(
            go.Heatmap(
                z=diff_data,
                x=results['azimuths'],
                y=np.arange(time_samples),
                colorscale=diff_cmap,
                zmin=-vmax_global_diff,
                zmax=vmax_global_diff,
                showscale=angle_idx == n_angles-1,
                colorbar=dict(title="CWT Diff", len=0.6, y=0.5, yanchor="middle") if angle_idx == n_angles-1 else None
            ),
            row=1, col=angle_idx+1
        )
        
        # Update axes
        fig_cwt_diff.update_xaxes(
            title_text="Azimuth (deg)" if angle_idx == n_angles//2 else "",
            row=1, col=angle_idx+1,
            tickangle=-45
        )
        
        if angle_idx == 0:
            fig_cwt_diff.update_yaxes(title_text="Time Samples", row=1, col=1)
        else:
            fig_cwt_diff.update_yaxes(showticklabels=False, row=1, col=angle_idx+1)
    
    fig_cwt_diff.update_layout(
        height=500,
        title_text=f"CWT Difference at Scale {cwt_scale} (Fluid-Substituted - Original)",
        showlegend=False
    )
    st.plotly_chart(fig_cwt_diff, use_container_width=True)
    
    # Return CWT results for further analysis
    return {
        'cwt_orig': cwt_orig,
        'cwt_sub': cwt_sub,
        'cwt_diff': cwt_diff
    }

def pwave_anisotropy_section_plotly(epsilon, delta, vp0, plot_cmap):
    """P-Wave Velocity Anisotropy Visualizer using Plotly"""
    st.header("P-Wave Velocity Anisotropy Visualizer")
    st.markdown("Explore how Thomsen parameters (ε, δ) affect P-wave velocity anisotropy.")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        epsilon = st.number_input(
            "ε (Epsilon)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=float(epsilon),
            step=0.01,
            key="epsilon_ani"
        )
        delta = st.number_input(
            "δ (Delta)", 
            min_value=-0.5, 
            max_value=0.5, 
            value=float(delta),
            step=0.01,
            key="delta_ani"
        )
        vp0 = st.number_input(
            "Vp₀ (m/s)", 
            min_value=1000.0,
            max_value=8000.0,
            value=float(vp0),
            key="vp0_ani"
        )
        show_3d = st.checkbox("Show 3D Visualization", True, key="show3d_ani")

    with col2:
        # Calculate Vp for 2D plot
        theta = np.linspace(0, 90, 90) * np.pi / 180
        Vp = vp0 * (1 + delta * (np.sin(theta))**2 * (np.cos(theta))**2 + epsilon * (np.sin(theta))**4)
        
        # Convert to Cartesian coordinates for 2D plot
        Vpx = Vp * np.sin(theta)
        Vpy = Vp * np.cos(theta)
        
        # 2D polar plot using Plotly
        fig_2d = go.Figure()
        
        # Add trace
        fig_2d.add_trace(go.Scatter(
            x=Vpx, y=Vpy,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f"ε={epsilon:.3f}, δ={delta:.3f}"
        ))
        
        # Update layout
        fig_2d.update_layout(
            title="P-Wave Velocity Anisotropy",
            xaxis_title="Vpx [m/s]",
            yaxis_title="Vpy [m/s]",
            height=600,
            width=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(range=[0, 1.5*vp0]),
            yaxis=dict(range=[0, 1.5*vp0]),
            hovermode='closest'
        )
        
        # Add grid
        fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_2d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 3D plot if enabled
        if show_3d:
            st.subheader("3D Velocity Surface")
            # Calculate full 3D velocity field
            theta_3d = np.linspace(0, np.pi, 90)
            phi_3d = np.linspace(0, 2*np.pi, 90)
            theta_grid, phi_grid = np.meshgrid(theta_3d, phi_3d)
            
            Vp_3d = vp0 * (1 + delta * (np.sin(theta_grid))**2 * (np.cos(theta_grid))**2 
                          + epsilon * (np.sin(theta_grid))**4)
            
            # Convert to Cartesian coordinates
            x = Vp_3d * np.sin(theta_grid) * np.cos(phi_grid)
            y = Vp_3d * np.sin(theta_grid) * np.sin(phi_grid)
            z = Vp_3d * np.cos(theta_grid)
            
            fig_3d = create_3d_plot(x, y, z, Vp_3d)
            st.plotly_chart(fig_3d, use_container_width=True)

def process_excel_data(uploaded_file, depth_ranges):
    """Process uploaded Excel file with individual layer depth ranges"""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Ensure required columns exist
        required_cols = ['Depth', 'Vp', 'Vs', 'Density', 'Epsilon', 'Delta', 'Gamma']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Excel file must contain '{col}' column")
                return None
        
        # Get values for each layer based on individual depth ranges
        params = {}
        for i, (min_depth, max_depth) in enumerate(depth_ranges, 1):
            layer_df = df[(df['Depth'] >= min_depth) & (df['Depth'] <= max_depth)]
            if len(layer_df) == 0:
                st.error(f"No data found in Layer {i} depth range ({min_depth}-{max_depth})")
                return None
            
            # Take median values for the layer
            params[f'vp{i}'] = float(layer_df['Vp'].median())
            params[f'vs{i}'] = float(layer_df['Vs'].median())
            params[f'd{i}'] = float(layer_df['Density'].median())
            params[f'e{i}'] = float(layer_df['Epsilon'].median())
            params[f'g{i}'] = float(layer_df['Gamma'].median())
            params[f'dlt{i}'] = float(layer_df['Delta'].median())
        
        return params
    
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return None

def plot_depth_ranges_plotly(depth_ranges, min_depth, max_depth):
    """Visualize the selected depth ranges using Plotly"""
    fig = go.Figure()
    
    # Colors for each layer
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    labels = ['Upper Layer (1)', 'Target Layer (2)', 'Lower Layer (3)']
    
    # Create horizontal bars for each layer
    for i, ((min_d, max_d), color) in enumerate(zip(depth_ranges, colors)):
        fig.add_trace(go.Bar(
            y=[labels[i]],
            x=[max_d - min_d],
            base=[min_d],
            orientation='h',
            marker=dict(color=color),
            name=labels[i],
            text=[f'{min_d:.1f}-{max_d:.1f}'],
            textposition='inside',
            textfont=dict(color='white', size=12, family="Arial Black"),
            hoverinfo='text',
            hovertext=f'{labels[i]}<br>Depth: {min_d:.1f} - {max_d:.1f} m'
        ))
    
    fig.update_layout(
        title='Selected Depth Ranges',
        xaxis_title='Depth (m)',
        yaxis_title='Layer',
        height=300,
        barmode='overlay',
        showlegend=True,
        legend=dict(x=1.02, y=1),
        xaxis=dict(range=[min_depth, max_depth]),
        yaxis=dict(showticklabels=False),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_modeling(params, enable_fluid_sub, seismic_cmap, selected_angle, azimuth_step, freq):
    """Run the modeling for ALL angles (0-50°) and azimuths (0-360°)"""
    with st.spinner("Computing models..."):
        # Original properties
        vp_orig = [params['vp1'], params['vp2'], params['vp3']]
        vs_orig = [params['vs1'], params['vs2'], params['vs3']]
        d_orig = [params['d1'], params['d2'], params['d3']]
        e_orig = [params['e1'], params['e2'], params['e3']]
        g_orig = [params['g1'], params['g2'], params['g3']]
        dlt_orig = [params['dlt1'], params['dlt2'], params['dlt3']]
        
        # Fluid substituted properties
        vp_sub = vp_orig.copy()
        vs_sub = vs_orig.copy()
        d_sub = d_orig.copy()
        e_sub = e_orig.copy()
        g_sub = g_orig.copy()
        dlt_sub = dlt_orig.copy()
        
        if enable_fluid_sub:
            K_orig, G_orig = velocity_to_moduli(params['vp2'], params['vs2'], params['d2'])
            K_sat, G_sat, delta_sat, gamma_sat = brown_korringa_substitution(
                params['Km']*1e9, params['Gm']*1e9, 
                K_orig, G_orig,
                params['Kf']*1e9, 
                params['phi'], 
                params['dlt2'], params['g2']
            )
            new_density = params['d2'] + params['phi']*(params['new_fluid_density'] - 1.0)
            Vp_new, Vs_new = moduli_to_velocity(K_sat, G_sat, new_density)
            
            vp_sub[1] = Vp_new
            vs_sub[1] = Vs_new
            d_sub[1] = new_density
            dlt_sub[1] = delta_sat
            g_sub[1] = gamma_sat
        
        # Compute for ALL angles (0-50°) and azimuths (0-360°)
        incidence_angles = np.linspace(0, 50, 11)  # 11 steps from 0-50°
        azimuths = np.arange(0, 361, azimuth_step)
        
        # Compute reflectivity (2D array: angles × azimuths)
        reflectivity_orig = np.zeros((len(incidence_angles), len(azimuths)))
        reflectivity_sub = np.zeros((len(incidence_angles), len(azimuths)))
        
        for i, theta in enumerate(incidence_angles):
            theta_rad = np.radians(theta)
            for j, az in enumerate(azimuths):
                reflectivity_orig[i,j] = calculate_reflectivity(
                    vp_orig, vs_orig, d_orig, e_orig, g_orig, dlt_orig, theta_rad, az
                )
                reflectivity_sub[i,j] = calculate_reflectivity(
                    vp_sub, vs_sub, d_sub, e_sub, g_sub, dlt_sub, theta_rad, az
                )
        
        # Generate synthetic seismic for all angles
        n_samples = 150
        wavelet = ricker_wavelet(freq, 0.08, 0.001)
        center_sample = n_samples//2 + len(wavelet)//2
        
        seismic_orig = []
        seismic_sub = []
        
        for i in range(len(incidence_angles)):
            R = np.zeros((n_samples, len(azimuths)))
            R[n_samples//2, :] = reflectivity_orig[i,:]
            seismic = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
            seismic_orig.append(seismic[center_sample-75:center_sample+75, :])
            
            R = np.zeros((n_samples, len(azimuths)))
            R[n_samples//2, :] = reflectivity_sub[i,:]
            seismic = np.array([convolve(R[:,az], wavelet, mode='full') for az in range(len(azimuths))]).T
            seismic_sub.append(seismic[center_sample-75:center_sample+75, :])
        
        return {
            'reflectivity_orig': reflectivity_orig,
            'reflectivity_sub': reflectivity_sub,
            'seismic_orig': seismic_orig,
            'seismic_sub': seismic_sub,
            'incidence_angles': incidence_angles,
            'azimuths': azimuths,
            'vp_orig': vp_orig,
            'vp_sub': vp_sub,
            'vs_orig': vs_orig,
            'vs_sub': vs_sub,
            'd_orig': d_orig,
            'd_sub': d_sub,
            'e_orig': e_orig,
            'e_sub': e_sub,
            'g_orig': g_orig,
            'g_sub': g_sub,
            'dlt_orig': dlt_orig,
            'dlt_sub': dlt_sub
        }

def display_results_plotly(results, seismic_cmap, diff_cmap, cwt_cmap, selected_angle):
    """Display modeling results with 0-50° 3D AVAZ comparison using Plotly"""
    # Find nearest angle index within 0-50° range
    angle_idx = np.argmin(np.abs(results['incidence_angles'] - min(selected_angle, 50)))
    actual_angle = results['incidence_angles'][angle_idx]
    
    st.header("3D AVAZ Response Comparison (0-50° Incidence)")
    col1, col2 = st.columns(2)
    
    # Get min/max for consistent color scaling
    zmin = min(np.min(results['reflectivity_orig']), np.min(results['reflectivity_sub']))
    zmax = max(np.max(results['reflectivity_orig']), np.max(results['reflectivity_sub']))
    
    with col1:
        fig_orig = go.Figure(data=[go.Surface(
            z=results['reflectivity_orig'],
            x=results['azimuths'],
            y=results['incidence_angles'],
            colorscale=seismic_cmap,
            cmin=zmin,
            cmax=zmax
        )])
        fig_orig.update_layout(
            scene=dict(
                xaxis_title='Azimuth (deg)',
                yaxis_title='Incidence Angle (deg)',
                zaxis_title='Reflectivity',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            title="Original Response (0-50°)",
            height=500
        )
        st.plotly_chart(fig_orig, use_container_width=True)
    
    with col2:
        fig_sub = go.Figure(data=[go.Surface(
            z=results['reflectivity_sub'],
            x=results['azimuths'],
            y=results['incidence_angles'],
            colorscale=seismic_cmap,
            cmin=zmin,
            cmax=zmax
        )])
        fig_sub.update_layout(
            scene=dict(
                xaxis_title='Azimuth (deg)',
                yaxis_title='Incidence Angle (deg)',
                zaxis_title='Reflectivity',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            title="Fluid-Substituted Response (0-50°)",
            height=500
        )
        st.plotly_chart(fig_sub, use_container_width=True)
    
    # 2. 2D Comparison at nearest angle using Plotly
    st.header(f"2D Comparison at {actual_angle:.1f}° Incidence (Closest to Selected {selected_angle}°)")
    
    fig_2d = go.Figure()
    
    # Add original reflectivity trace
    fig_2d.add_trace(go.Scatter(
        x=results['azimuths'],
        y=results['reflectivity_orig'][angle_idx, :],
        mode='lines',
        name='Original',
        line=dict(color='blue', width=2)
    ))
    
    # Add fluid-substituted reflectivity trace
    fig_2d.add_trace(go.Scatter(
        x=results['azimuths'],
        y=results['reflectivity_sub'][angle_idx, :],
        mode='lines',
        name='Fluid-Substituted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_2d.update_layout(
        title=f'AVAZ Reflectivity at {actual_angle:.1f}° Incidence',
        xaxis_title='Azimuth (degrees)',
        yaxis_title='Reflectivity',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    # Add grid
    fig_2d.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_2d.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig_2d, use_container_width=True)
    
    # 3. Polar View Comparison using Plotly
    st.header(f"Polar View Comparison at {actual_angle:.1f}° Incidence")
    
    fig_polar = go.Figure()
    
    # Add original reflectivity trace
    fig_polar.add_trace(go.Scatterpolar(
        r=results['reflectivity_orig'][angle_idx, :],
        theta=results['azimuths'],
        mode='lines',
        name='Original',
        line=dict(color='blue', width=2)
    ))
    
    # Add fluid-substituted reflectivity trace
    fig_polar.add_trace(go.Scatterpolar(
        r=results['reflectivity_sub'][angle_idx, :],
        theta=results['azimuths'],
        mode='lines',
        name='Fluid-Substituted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_polar.update_layout(
        title=f'Polar AVAZ Response at {actual_angle:.1f}° Incidence',
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction="clockwise")
        ),
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig_polar, use_container_width=True)
    
    # 4. Horizontal Arrangement of Seismic Gathers using Plotly
    plot_horizontal_angles_seismic_plotly(results, seismic_cmap, diff_cmap)
    
    # 5. Horizontal Arrangement of CWT Analysis using Plotly
    # Add CWT scale selection in the main display
    st.header("CWT Analysis Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        cwt_scale = st.slider(
            "CWT Scale (Higher = Lower Frequency)", 
            min_value=1, 
            max_value=20, 
            value=10,
            help="Scale parameter for Continuous Wavelet Transform"
        )
    with col2:
        st.markdown("**Scale Information:**")
        st.markdown("- Scale 1: Highest frequency details")
        st.markdown("- Scale 10: Mid-frequency features")
        st.markdown("- Scale 20: Lowest frequency trends")
    with col3:
        st.markdown("**Current Colormaps:**")
        st.markdown(f"- Seismic/CWT: {seismic_cmap}")
        st.markdown(f"- Difference: {diff_cmap}")
    
    # Call the new horizontal CWT plotting function
    cwt_results = plot_horizontal_angles_cwt_plotly(results, cwt_scale, cwt_cmap, diff_cmap)
    
    # 6. Additional CWT Visualizations using Plotly
    st.header("Additional CWT Visualizations")
    
    # Frequency-Scale Analysis at middle angle
    st.subheader("Frequency-Scale Analysis")
    
    # Analyze at middle azimuth and middle angle
    mid_azimuth_idx = len(results['azimuths']) // 2
    mid_angle_idx = len(results['incidence_angles']) // 2
    mid_angle = results['incidence_angles'][mid_angle_idx]
    
    # Get traces for analysis
    orig_trace = results['seismic_orig'][mid_angle_idx][:, mid_azimuth_idx]
    sub_trace = results['seismic_sub'][mid_angle_idx][:, mid_azimuth_idx]
    
    # Perform full CWT with all scales
    scales_full = np.arange(1, 21, 1)
    cwt_orig_full = cwt_analysis(orig_trace, scales_full)
    cwt_sub_full = cwt_analysis(sub_trace, scales_full)
    
    # Create subplots for scale-time analysis
    fig_scale_time = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Original: CWT Scale-Time at {mid_angle}°', 
                       f'Fluid-Substituted: CWT Scale-Time at {mid_angle}°'],
        horizontal_spacing=0.1
    )
    
    # Original CWT scale-time
    fig_scale_time.add_trace(
        go.Heatmap(
            z=cwt_orig_full,
            x=np.arange(len(orig_trace)),
            y=scales_full,
            colorscale=cwt_cmap,
            showscale=True,
            colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle", x=0.46)
        ),
        row=1, col=1
    )
    
    # Fluid-substituted CWT scale-time
    fig_scale_time.add_trace(
        go.Heatmap(
            z=cwt_sub_full,
            x=np.arange(len(sub_trace)),
            y=scales_full,
            colorscale=cwt_cmap,
            showscale=True,
            colorbar=dict(title="CWT Magnitude", len=0.6, y=0.5, yanchor="middle", x=1.02)
        ),
        row=1, col=2
    )
    
    # Update axes
    fig_scale_time.update_xaxes(title_text="Time Sample", row=1, col=1)
    fig_scale_time.update_xaxes(title_text="Time Sample", row=1, col=2)
    fig_scale_time.update_yaxes(
        title_text="Scale (Lower = Higher Frequency)", 
        autorange='reversed',
        row=1, col=1
    )
    fig_scale_time.update_yaxes(
        title_text="Scale (Lower = Higher Frequency)", 
        autorange='reversed',
        row=1, col=2
    )
    
    fig_scale_time.update_layout(
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_scale_time, use_container_width=True)
    
    # 7. Difference Analysis using Plotly
    st.header("Difference Analysis")
    
    reflectivity_diff = results['reflectivity_sub'] - results['reflectivity_orig']
    max_diff = np.max(np.abs(reflectivity_diff))
    
    # Difference matrix using Plotly
    fig_diff_matrix = go.Figure(data=[
        go.Heatmap(
            z=reflectivity_diff.T,
            x=results['incidence_angles'],
            y=results['azimuths'],
            colorscale=diff_cmap,
            zmin=-max_diff,
            zmax=max_diff,
            colorbar=dict(title="Reflectivity Difference", len=0.8, y=0.5, yanchor="middle")
        )
    ])
    
    fig_diff_matrix.update_layout(
        title='Reflectivity Difference (Fluid-Substituted - Original)',
        xaxis_title='Incidence Angle (degrees)',
        yaxis_title='Azimuth (degrees)',
        height=500
    )
    
    st.plotly_chart(fig_diff_matrix, use_container_width=True)
    
    # 3D Difference plot
    fig_diff_3d = go.Figure(data=[go.Surface(
        z=reflectivity_diff,
        x=results['azimuths'],
        y=results['incidence_angles'],
        colorscale=diff_cmap,
        cmin=-max_diff,
        cmax=max_diff
    )])
    fig_diff_3d.update_layout(
        scene=dict(
            xaxis_title='Azimuth (deg)',
            yaxis_title='Incidence Angle (deg)',
            zaxis_title='Reflectivity Difference',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        title="3D Reflectivity Difference",
        height=600
    )
    st.plotly_chart(fig_diff_3d, use_container_width=True)

def main():
    st.set_page_config(layout="wide", page_title="AVAZ Modeling with Fluid Substitution")
    st.title("AVAZ Modeling with Brown-Korringa Fluid Substitution")
    
    # Initialize session state
    if 'modeling_mode' not in st.session_state:
        st.session_state.modeling_mode = "manual"
    if 'excel_data_processed' not in st.session_state:
        st.session_state.excel_data_processed = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'show_anisotropy_section' not in st.session_state:
        st.session_state.show_anisotropy_section = False
    
    # Modeling mode selection
    modeling_mode = st.sidebar.radio(
        "Modeling Mode",
        ["Manual Input", "Excel Import"],
        index=0 if st.session_state.modeling_mode == "manual" else 1
    )
    
    # Colormap selection in sidebar for all plots
    st.sidebar.header("Colormap Settings")
    
    # Main colormap for seismic and CWT plots - Set RdBu as default
    main_cmap = st.sidebar.selectbox(
        "Main Colormap (Seismic & CWT)",
        options=['RdBu', 'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'rainbow', 
                 'turbo', 'portland', 'blackbody', 'electric', 'bluered'],
        index=0,  # RdBu as default
        help="Colormap for seismic amplitude and CWT magnitude plots - RdBu set as default"
    )
    
    # Difference colormap
    diff_cmap = st.sidebar.selectbox(
        "Difference Colormap",
        options=['RdBu', 'RdYlBu', 'RdYlGn', 'Picnic', 'Portland', 'Earth', 
                 'Electric', 'Viridis', 'Cividis', 'balance', 'delta', 'curl'],
        index=0,
        help="Colormap for difference plots"
    )
    
    if modeling_mode == "Manual Input":
        st.session_state.modeling_mode = "manual"
        st.session_state.excel_data_processed = False
        st.session_state.show_results = False
        
        with st.sidebar:
            st.header("Model Parameters")
            
            # Rock properties for three layers
            layers = ["Upper (1)", "Target (2)", "Lower (3)"]
            params = {}
            
            for i, layer in enumerate(layers, 1):
                st.subheader(f"Layer {layer}")
                params[f'vp{i}'] = st.number_input(f"Vp{i} (m/s)", value=5500 if i!=2 else 4742)
                params[f'vs{i}'] = st.number_input(f"Vs{i} (m/s)", value=3600 if i!=2 else 3292)
                params[f'd{i}'] = st.number_input(f"Density{i} (g/cc)", value=2.6 if i!=2 else 2.4, step=0.1)
                params[f'e{i}'] = st.number_input(f"ε{i}", value=0.1 if i==1 else (-0.01 if i==2 else 0.2), step=0.01)
                params[f'g{i}'] = st.number_input(f"γ{i}", value=0.05 if i==1 else (-0.05 if i==2 else 0.15), step=0.01)
                params[f'dlt{i}'] = st.number_input(f"δ{i}", value=0.0 if i==1 else (-0.13 if i==2 else 0.1), step=0.01)
            
            st.subheader("Acquisition Parameters")
            selected_angle = st.slider(
                "Angle of Incidence (deg)", 
                1, 70, 30, 1,
                help="Model will show results for this angle in 2D views"
            )
            freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 45)
            azimuth_step = st.slider("Azimuth Step (deg)", 1, 30, 10)
            
            st.subheader("Fluid Substitution Parameters")
            enable_fluid_sub = st.checkbox("Enable Fluid Substitution", True)
            if enable_fluid_sub:
                params['phi'] = st.slider("Porosity (ϕ)", 0.01, 0.5, 0.2, 0.01)
                params['Km'] = st.number_input("Mineral Bulk Modulus (GPa)", 10.0, 100.0, 37.0, 1.0)
                params['Gm'] = st.number_input("Mineral Shear Modulus (GPa)", 10.0, 100.0, 44.0, 1.0)
                params['Kf'] = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 5.0, 2.2, 0.1)
                params['new_fluid_density'] = st.number_input("New Fluid Density (g/cc)", 0.1, 1.5, 1.0, 0.1)
            
            # Add button to show P-wave anisotropy section
            show_anisotropy = st.checkbox("Show P-Wave Anisotropy Section", False)
            
            if st.button("Run Modeling"):
                st.session_state.show_results = True
                st.session_state.show_anisotropy_section = show_anisotropy
                st.session_state.model_params = params
                st.session_state.enable_fluid_sub = enable_fluid_sub
                st.session_state.main_cmap = main_cmap
                st.session_state.diff_cmap = diff_cmap
                st.session_state.selected_angle = selected_angle
                st.session_state.azimuth_step = azimuth_step
                st.session_state.freq = freq
        
        # Main workspace content
        if st.session_state.show_results:
            # Show P-wave anisotropy section if checkbox was checked
            if st.session_state.show_anisotropy_section:
                st.markdown("---")
                pwave_anisotropy_section_plotly(
                    st.session_state.model_params['e2'],
                    st.session_state.model_params['dlt2'],
                    st.session_state.model_params['vp2'],
                    st.session_state.main_cmap
                )
                st.markdown("---")
            
            # Run the main modeling
            results = run_modeling(
                st.session_state.model_params,
                st.session_state.enable_fluid_sub,
                st.session_state.main_cmap,
                st.session_state.selected_angle,
                st.session_state.azimuth_step,
                st.session_state.freq
            )
            display_results_plotly(
                results,
                st.session_state.main_cmap,
                st.session_state.diff_cmap,
                st.session_state.main_cmap,
                st.session_state.selected_angle
            )
    
    else:  # Excel Import mode
        st.session_state.modeling_mode = "excel"
        st.session_state.show_results = False
        
        with st.sidebar:
            st.header("Excel Import Settings")
            
            uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
            
            if uploaded_file is not None:
                try:
                    # Read Excel to get full depth range
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                    min_depth = float(df['Depth'].min())
                    max_depth = float(df['Depth'].max())
                    
                    st.subheader("Layer Depth Ranges")
                    
                    # Calculate default ranges (divide into thirds)
                    range_size = (max_depth - min_depth) / 3
                    default_ranges = [
                        (min_depth, min_depth + range_size),
                        (min_depth + range_size, min_depth + 2*range_size),
                        (min_depth + 2*range_size, max_depth)
                    ]
                    
                    depth_ranges = []
                    layers = ["Upper (1)", "Target (2)", "Lower (3)"]
                    
                    for i, layer in enumerate(layers, 1):
                        st.markdown(f"**{layer}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            min_val = st.number_input(
                                f"Min Depth {i}", 
                                min_value=min_depth, 
                                max_value=max_depth,
                                value=default_ranges[i-1][0],
                                key=f"min_depth_{i}"
                            )
                        with col2:
                            max_val = st.number_input(
                                f"Max Depth {i}", 
                                min_value=min_depth, 
                                max_value=max_depth,
                                value=default_ranges[i-1][1],
                                key=f"max_depth_{i}"
                            )
                        # Ensure valid range
                        if min_val >= max_val:
                            st.error(f"Layer {i}: Min must be less than Max")
                            continue
                        depth_ranges.append((min_val, max_val))
                    
                    # Store depth ranges in session state
                    if len(depth_ranges) == 3:
                        st.session_state.depth_ranges = depth_ranges
                        st.session_state.min_depth = min_depth
                        st.session_state.max_depth = max_depth
                    
                    st.subheader("Acquisition Parameters")
                    selected_angle = st.slider(
                        "Angle of Incidence (deg)", 
                        1, 70, 30, 1,
                        help="Model will show results for this angle in 2D views",
                        key="excel_angle"
                    )
                    freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 45, key="excel_freq")
                    azimuth_step = st.slider("Azimuth Step (deg)", 1, 30, 10, key="excel_azimuth_step")
                    
                    st.subheader("Fluid Substitution Parameters")
                    enable_fluid_sub = st.checkbox("Enable Fluid Substitution", True, key="excel_fluid_sub")
                    if enable_fluid_sub:
                        phi = st.slider("Porosity (ϕ)", 0.01, 0.5, 0.2, 0.01, key="excel_phi")
                        Km = st.number_input("Mineral Bulk Modulus (GPa)", 10.0, 100.0, 37.0, 1.0, key="excel_Km")
                        Gm = st.number_input("Mineral Shear Modulus (GPa)", 10.0, 100.0, 44.0, 1.0, key="excel_Gm")
                        Kf = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 5.0, 2.2, 0.1, key="excel_Kf")
                        new_fluid_density = st.number_input("New Fluid Density (g/cc)", 0.1, 1.5, 1.0, 0.1, key="excel_fluid_density")
                    
                    show_anisotropy = st.checkbox("Show P-Wave Anisotropy Section", False, key="excel_show_anisotropy")
                    
                    if st.button("Run Modeling with Excel Data"):
                        st.session_state.excel_data_processed = True
                        st.session_state.show_anisotropy_section = show_anisotropy
                        st.session_state.uploaded_file = uploaded_file
                        st.session_state.enable_fluid_sub = enable_fluid_sub
                        st.session_state.main_cmap = main_cmap
                        st.session_state.diff_cmap = diff_cmap
                        st.session_state.selected_angle = selected_angle
                        st.session_state.azimuth_step = azimuth_step
                        st.session_state.freq = freq
                        
                        if enable_fluid_sub:
                            st.session_state.phi = phi
                            st.session_state.Km = Km
                            st.session_state.Gm = Gm
                            st.session_state.Kf = Kf
                            st.session_state.new_fluid_density = new_fluid_density
                
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
        
        # Main workspace content for Excel mode
        if uploaded_file is not None and hasattr(st.session_state, 'depth_ranges'):
            st.header("Depth Range Visualization")
            plot_depth_ranges_plotly(
                st.session_state.depth_ranges,
                st.session_state.min_depth,
                st.session_state.max_depth
            )
            
            if st.session_state.excel_data_processed:
                try:
                    # Process Excel data with individual layer ranges
                    params = process_excel_data(
                        st.session_state.uploaded_file,
                        st.session_state.depth_ranges
                    )
                    
                    if params is not None:
                        # Add fluid substitution parameters if enabled
                        if st.session_state.enable_fluid_sub:
                            params.update({
                                'phi': st.session_state.phi,
                                'Km': st.session_state.Km,
                                'Gm': st.session_state.Gm,
                                'Kf': st.session_state.Kf,
                                'new_fluid_density': st.session_state.new_fluid_density
                            })
                        
                        # Show P-wave anisotropy section if checkbox was checked
                        if st.session_state.show_anisotropy_section:
                            st.markdown("---")
                            pwave_anisotropy_section_plotly(params['e2'], params['dlt2'], params['vp2'], st.session_state.main_cmap)
                            st.markdown("---")
                        
                        # Run modeling
                        results = run_modeling(
                            params,
                            st.session_state.enable_fluid_sub,
                            st.session_state.main_cmap,
                            st.session_state.selected_angle,
                            st.session_state.azimuth_step,
                            st.session_state.freq
                        )
                        display_results_plotly(
                            results,
                            st.session_state.main_cmap,
                            st.session_state.diff_cmap,
                            st.session_state.main_cmap,
                            st.session_state.selected_angle
                        )
                
                except Exception as e:
                    st.error(f"Modeling error: {str(e)}")

if __name__ == "__main__":
    main()
