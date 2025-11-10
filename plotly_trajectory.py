#!/usr/bin/env python3
"""
Lunar Orbit Trajectory Visualization - Dash/Plotly Implementation
==================================================================
Interactive 3D spacecraft trajectory visualization using Dash web framework.
No external browser popups - runs on localhost:8050

Features:
- Full 3D rotation/zoom/pan (drag to rotate, scroll to zoom)
- Animated trajectory playback with speed controls
- Real-time telemetry display
- Green-to-red gradient trail
- Velocity direction indicator
- Click points for detailed information
- Optimized for server performance with WebGL rendering

Usage:
    python3 plotly_trajectory.py
    Then open browser to: http://localhost:8050
    Or via SSH tunnel: ssh -L 8050:localhost:8050 user@server

Controls:
- Left-click drag: Rotate view
- Right-click drag: Pan
- Scroll wheel: Zoom
- Play/Pause: Use animation controls
- Speed: 0.5x, 1.0x, 1.5x, 2.0x buttons
- Hover over orbit: See coordinates
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import dash

# ============================================================================
# CONFIGURATION
# ============================================================================

MOON_RADIUS_KM = 1737.4  # Moon radius in kilometers
NUM_TRAJECTORY_POINTS = 600  # Higher detail for smooth orbit
TRAIL_GRADIENT_POINTS = 100  # Number of points to show in animated trail

# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_lunar_orbit_trajectory(num_points=600):
    """
    Generate realistic 3D lunar orbit trajectory with spiral pattern.
    
    Orbital parameters:
    - Altitude: 100 km above surface initially, increases to 200 km
    - Period: ~2 hours for 2 complete orbits
    - Inclination: 15 degrees
    - Eccentricity: 0.05 (slightly elliptical)
    
    Returns:
        pandas.DataFrame with X, Y, Z positions, velocities, altitude, and time
    """
    # Time array for 2-hour mission
    time_seconds = np.linspace(0, 7200, num_points)
    t_norm = np.linspace(0, 1, num_points)
    
    # Orbital parameters
    orbit_altitude = 100  # km above surface at start
    orbit_radius = MOON_RADIUS_KM + orbit_altitude
    
    # Complete 2 full orbits
    n_orbits = 2
    theta = 2 * np.pi * n_orbits * t_norm
    
    # Elliptical orbit with slight eccentricity
    eccentricity = 0.05
    r = orbit_radius * (1 - eccentricity * np.cos(theta))
    
    # Add altitude variation to create visible spiral (prevents orbit overlap)
    altitude_variation = 100 * t_norm  # Increases from 0 to 100 km
    altitude_variation += 30 * np.sin(n_orbits * theta)  # Sinusoidal per orbit
    r = r + altitude_variation
    
    # Calculate position in orbital plane
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Apply 15-degree inclination
    inclination = np.radians(15)
    z = r * np.sin(theta) * np.sin(inclination)
    y = r * np.sin(theta) * np.cos(inclination)
    
    # Add small random perturbations (realistic orbital variations)
    x += np.random.normal(0, 0.1, num_points)
    y += np.random.normal(0, 0.1, num_points)
    z += np.random.normal(0, 0.05, num_points)
    
    # Calculate velocity vectors (derivative of position)
    vx = np.gradient(x, time_seconds)
    vy = np.gradient(y, time_seconds)
    vz = np.gradient(z, time_seconds)
    
    # Calculate altitude above moon surface
    altitude = np.sqrt(x**2 + y**2 + z**2) - MOON_RADIUS_KM
    
    # Create DataFrame
    df = pd.DataFrame({
        'X_km': x,
        'Y_km': y,
        'Z_km': z,
        'VX_km_s': vx,
        'VY_km_s': vy,
        'VZ_km_s': vz,
        'Altitude_km': altitude,
        'Time_s': time_seconds
    })
    
    return df

# ============================================================================
# VISUALIZATION CREATION
# ============================================================================

def create_moon_surface():
    """
    Create moon sphere geometry (optimized resolution for WebGL).
    Returns trace for 3D surface plot.
    """
    # Generate sphere coordinates (low poly for performance)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 25)
    
    x = MOON_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = MOON_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = MOON_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    
    moon_trace = go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#808080'], [1, '#808080']],  # Gray color
        showscale=False,
        opacity=0.4,
        name='Moon',
        hoverinfo='skip',
        lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.9, specular=0.1, fresnel=0.2)
    )
    
    return moon_trace

def create_trajectory_trace(df):
    """
    Create the full orbital path with green-to-red gradient.
    Optimized with line rendering instead of scatter for performance.
    """
    # Calculate colors from green (start) to red (end)
    n_points = len(df)
    progress = np.linspace(0, 1, n_points)
    
    # Create RGB colors: green -> yellow -> red
    colors = []
    for p in progress:
        r = int(p * 255)
        g = int((1 - p) * 255)
        colors.append(f'rgb({r},{g},0)')
    
    trajectory_trace = go.Scatter3d(
        x=df['X_km'],
        y=df['Y_km'],
        z=df['Z_km'],
        mode='lines',
        line=dict(
            color=colors[::10],  # Subsample colors for performance
            width=4
        ),
        name='Trajectory',
        hovertemplate='<b>Point %{pointNumber}</b><br>' +
                      'Time: %{customdata[0]:.1f} s<br>' +
                      'Altitude: %{customdata[1]:.1f} km<br>' +
                      'X: %{x:.1f} km<br>' +
                      'Y: %{y:.1f} km<br>' +
                      'Z: %{z:.1f} km<br>' +
                      '<extra></extra>',
        customdata=np.column_stack([df['Time_s'], df['Altitude_km']])
    )
    
    return trajectory_trace

def create_spacecraft_marker(df, frame_idx=0):
    """
    Create animated spacecraft marker (black sphere).
    """
    spacecraft_trace = go.Scatter3d(
        x=[df['X_km'].iloc[frame_idx]],
        y=[df['Y_km'].iloc[frame_idx]],
        z=[df['Z_km'].iloc[frame_idx]],
        mode='markers',
        marker=dict(
            size=12,
            color='black',
            line=dict(color='gray', width=2)
        ),
        name='Spacecraft',
        hoverinfo='skip'
    )
    
    return spacecraft_trace

def create_velocity_arrow(df, frame_idx=0):
    """
    Create velocity direction arrow (shows where spacecraft is heading).
    Uses 3D cone for better visibility.
    """
    # Current position
    x = df['X_km'].iloc[frame_idx]
    y = df['Y_km'].iloc[frame_idx]
    z = df['Z_km'].iloc[frame_idx]
    
    # Velocity vector
    vx = df['VX_km_s'].iloc[frame_idx]
    vy = df['VY_km_s'].iloc[frame_idx]
    vz = df['VZ_km_s'].iloc[frame_idx]
    
    # Normalize and scale for visibility
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    if vel_mag > 0:
        scale = 400  # Arrow length in km
        vx_norm = vx / vel_mag * scale
        vy_norm = vy / vel_mag * scale
        vz_norm = vz / vel_mag * scale
    else:
        vx_norm = vy_norm = vz_norm = 0
    
    arrow_trace = go.Cone(
        x=[x],
        y=[y],
        z=[z],
        u=[vx_norm],
        v=[vy_norm],
        w=[vz_norm],
        colorscale=[[0, '#444444'], [1, '#444444']],
        showscale=False,
        sizemode='absolute',
        sizeref=150,
        name='Velocity',
        hoverinfo='skip'
    )
    
    return arrow_trace

def create_start_end_markers(df):
    """
    Create start (green) and end (red) position markers.
    """
    markers_trace = go.Scatter3d(
        x=[df['X_km'].iloc[0], df['X_km'].iloc[-1]],
        y=[df['Y_km'].iloc[0], df['Y_km'].iloc[-1]],
        z=[df['Z_km'].iloc[0], df['Z_km'].iloc[-1]],
        mode='markers',
        marker=dict(
            size=[10, 10],
            color=['green', 'red'],
            symbol=['circle', 'square']
        ),
        name='Start/End',
        text=['Start', 'End'],
        hovertemplate='<b>%{text}</b><br>X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Z: %{z:.1f} km<extra></extra>'
    )
    
    return markers_trace

def create_animated_trail(df, frame_idx, trail_length=TRAIL_GRADIENT_POINTS):
    """
    Create trailing path behind spacecraft with gradient (most recent = red).
    """
    # Get trail segment
    start_idx = max(0, frame_idx - trail_length)
    trail_df = df.iloc[start_idx:frame_idx+1]
    
    if len(trail_df) < 2:
        return None
    
    # Create gradient for trail (green at back, red at front)
    n = len(trail_df)
    progress = np.linspace(0, 1, n)
    colors = []
    for p in progress:
        r = int(p * 255)
        g = int((1 - p) * 255)
        colors.append(f'rgb({r},{g},0)')
    
    trail_trace = go.Scatter3d(
        x=trail_df['X_km'],
        y=trail_df['Y_km'],
        z=trail_df['Z_km'],
        mode='lines',
        line=dict(
            color=colors,
            width=6
        ),
        name='Trail',
        hoverinfo='skip',
        showlegend=False
    )
    
    return trail_trace

def create_figure(df):
    """
    Create the main Plotly figure with all traces and animation frames.
    Optimized for WebGL rendering and smooth performance.
    """
    # Create all traces
    moon = create_moon_surface()
    trajectory = create_trajectory_trace(df)
    markers = create_start_end_markers(df)
    
    # Initial frame traces
    spacecraft = create_spacecraft_marker(df, 0)
    velocity = create_velocity_arrow(df, 0)
    trail = create_animated_trail(df, 0)
    
    # Create figure
    fig = go.Figure(data=[moon, trajectory, markers, spacecraft, velocity, trail])
    
    # Create animation frames (subsample for performance)
    frames = []
    frame_step = 2  # Update every 2nd point for smoother performance
    
    for i in range(0, len(df), frame_step):
        spacecraft_frame = create_spacecraft_marker(df, i)
        velocity_frame = create_velocity_arrow(df, i)
        trail_frame = create_animated_trail(df, i)
        
        # Telemetry annotation
        telemetry_text = (
            f"<b>TELEMETRY</b><br>"
            f"Time: {df['Time_s'].iloc[i]:.1f} s<br>"
            f"Altitude: {df['Altitude_km'].iloc[i]:.1f} km<br>"
            f"Frame: {i+1}/{len(df)}"
        )
        
        frame_data = [moon, trajectory, markers, spacecraft_frame, velocity_frame]
        if trail_frame:
            frame_data.append(trail_frame)
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                annotations=[dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=telemetry_text,
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='gray',
                    borderwidth=1,
                    font=dict(size=11, family='monospace'),
                    align='left',
                    xanchor='left',
                    yanchor='top'
                )]
            )
        ))
    
    fig.frames = frames
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text='Lunar Orbit Trajectory - Interactive 3D Visualization',
            font=dict(size=20, color='#2c3e50', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True),
            yaxis=dict(title='Y (km)', backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True),
            zaxis=dict(title='Z (km)', backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.85,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10)
        ),
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='#ffffff',
        margin=dict(l=0, r=0, t=60, b=0),
        
        # Animation controls
        updatemenus=[
            # Play/Pause button
            dict(
                type='buttons',
                showactive=True,
                x=0.12, y=0.02,
                xanchor='left', yanchor='bottom',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=20, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ],
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=12)
            ),
            
            # Speed controls
            dict(
                type='buttons',
                showactive=True,
                x=0.25, y=0.02,
                xanchor='left', yanchor='bottom',
                buttons=[
                    dict(label='0.5x', method='animate',
                         args=[None, dict(frame=dict(duration=40, redraw=True), mode='immediate')]),
                    dict(label='1.0x', method='animate',
                         args=[None, dict(frame=dict(duration=20, redraw=True), mode='immediate')]),
                    dict(label='1.5x', method='animate',
                         args=[None, dict(frame=dict(duration=13, redraw=True), mode='immediate')]),
                    dict(label='2.0x', method='animate',
                         args=[None, dict(frame=dict(duration=10, redraw=True), mode='immediate')])
                ],
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=11)
            )
        ],
        
        # Slider for manual frame selection
        sliders=[dict(
            active=0,
            steps=[dict(
                args=[[f.name], dict(
                    frame=dict(duration=0, redraw=True),
                    mode='immediate',
                    transition=dict(duration=0)
                )],
                label=f"{int(f.name)}",
                method='animate'
            ) for f in frames[::20]],  # Subsample slider for performance
            x=0.12, y=0.0,
            len=0.85,
            xanchor='left', yanchor='top',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1,
            ticklen=5,
            font=dict(size=10)
        )]
    )
    
    # Enable WebGL for better performance
    fig.update_traces(selector=dict(type='scatter3d'))
    
    return fig

# ============================================================================
# DASH APPLICATION
# ============================================================================

# Generate trajectory data
print("Generating trajectory data...")
df = generate_lunar_orbit_trajectory(NUM_TRAJECTORY_POINTS)

# Create figure
print("Creating visualization...")
fig = create_figure(df)

# Initialize Dash app
app = Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H3('Lunar Orbit Trajectory Visualization', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P('Drag to rotate | Right-click to pan | Scroll to zoom | Use controls below to animate',
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'marginBottom': '10px'}),
    
    dcc.Graph(
        id='trajectory-plot',
        figure=fig,
        style={'height': '85vh'},
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'lunar_trajectory',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
    ),
    
    html.Div([
        html.P('Server running on http://localhost:8050 | Press Ctrl+C to stop',
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'marginTop': '10px'})
    ])
], style={'backgroundColor': '#f8f9fa', 'height': '100vh', 'margin': '0', 'padding': '0'})

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("LUNAR ORBIT TRAJECTORY VISUALIZATION SERVER")
    print("="*70)
    print(f"Trajectory Points: {NUM_TRAJECTORY_POINTS}")
    print(f"Moon Radius: {MOON_RADIUS_KM} km")
    print(f"Animation Frames: {len(df)//2}")
    print("="*70)
    print("Server: http://localhost:8050")
    print("Remote access: ssh -L 8050:localhost:8050 user@server")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run_server(debug=False, host='0.0.0.0', port=8050)

