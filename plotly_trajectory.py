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
from dash import Dash, dcc, html, Input, Output, State, callback
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

def create_figure(df, frame_idx=0):
    """
    Create the main Plotly figure for the current frame.
    Uses callback-based animation to preserve camera control.
    """
    # Create all traces
    moon = create_moon_surface()
    trajectory = create_trajectory_trace(df)
    markers = create_start_end_markers(df)
    
    # Current frame traces
    spacecraft = create_spacecraft_marker(df, frame_idx)
    velocity = create_velocity_arrow(df, frame_idx)
    trail = create_animated_trail(df, frame_idx)
    
    # Create figure (filter out None values)
    initial_data = [moon, trajectory, markers, spacecraft, velocity]
    if trail is not None:
        initial_data.append(trail)
    fig = go.Figure(data=initial_data)
    
    # Telemetry annotation for current frame
    telemetry_text = (
        f"<b>TELEMETRY</b><br>"
        f"Time: {df['Time_s'].iloc[frame_idx]:.1f} s<br>"
        f"Altitude: {df['Altitude_km'].iloc[frame_idx]:.1f} km<br>"
        f"Frame: {frame_idx+1}/{len(df)}"
    )
    
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
        )],
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='#ffffff',
        margin=dict(l=0, r=0, t=60, b=0),
        uirevision='constant'  # Preserve camera position between updates
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
    
    # Animation controls
    html.Div([
        html.Button('Play', id='play-button', n_clicks=0, 
                   style={'marginRight': '10px', 'padding': '8px 20px', 'fontSize': '14px'}),
        html.Button('Pause', id='pause-button', n_clicks=0,
                   style={'marginRight': '20px', 'padding': '8px 20px', 'fontSize': '14px'}),
        html.Label('Speed: ', style={'marginRight': '10px', 'fontSize': '14px'}),
        html.Button('0.5x', id='speed-05x', n_clicks=0,
                   style={'marginRight': '5px', 'padding': '6px 15px', 'fontSize': '13px'}),
        html.Button('1.0x', id='speed-10x', n_clicks=0,
                   style={'marginRight': '5px', 'padding': '6px 15px', 'fontSize': '13px', 
                          'backgroundColor': '#e0e0e0'}),
        html.Button('1.5x', id='speed-15x', n_clicks=0,
                   style={'marginRight': '5px', 'padding': '6px 15px', 'fontSize': '13px'}),
        html.Button('2.0x', id='speed-20x', n_clicks=0,
                   style={'padding': '6px 15px', 'fontSize': '13px'}),
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#ffffff'}),
    
    # Graph
    dcc.Graph(
        id='trajectory-plot',
        figure=fig,
        style={'height': '75vh'},
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
    
    # Animation interval (hidden)
    dcc.Interval(id='animation-interval', interval=20, disabled=True),
    
    # Store current frame and speed
    dcc.Store(id='current-frame', data=0),
    dcc.Store(id='animation-speed', data=20),
    
    html.Div([
        html.P('Server running on http://localhost:8050 | Press Ctrl+C to stop',
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'marginTop': '10px'})
    ])
], style={'backgroundColor': '#f8f9fa', 'height': '100vh', 'margin': '0', 'padding': '0'})

# ============================================================================
# CALLBACKS FOR ANIMATION
# ============================================================================

# Play/Pause callbacks
@app.callback(
    Output('animation-interval', 'disabled'),
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks')],
    prevent_initial_call=True
)
def control_animation(play_clicks, pause_clicks):
    """Control play/pause state"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return button_id == 'pause-button'

# Speed control callbacks
@app.callback(
    [Output('animation-interval', 'interval'),
     Output('animation-speed', 'data')],
    [Input('speed-05x', 'n_clicks'),
     Input('speed-10x', 'n_clicks'),
     Input('speed-15x', 'n_clicks'),
     Input('speed-20x', 'n_clicks')],
    prevent_initial_call=True
)
def update_speed(c05, c10, c15, c20):
    """Update animation speed"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return 20, 20
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    speed_map = {
        'speed-05x': 40,
        'speed-10x': 20,
        'speed-15x': 13,
        'speed-20x': 10
    }
    interval = speed_map.get(button_id, 20)
    return interval, interval

# Main animation callback
@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('current-frame', 'data')],
    [Input('animation-interval', 'n_intervals')],
    [State('current-frame', 'data'),
     State('trajectory-plot', 'figure')]
)
def update_frame(n_intervals, current_frame, current_fig):
    """Update the plot for the current animation frame"""
    if n_intervals is None:
        return dash.no_update, dash.no_update
    
    # Increment frame (loop back to start)
    next_frame = (current_frame + 1) % len(df)
    
    # Update only the animated traces (spacecraft, velocity, trail)
    # Keep existing camera position from current_fig
    new_spacecraft = create_spacecraft_marker(df, next_frame)
    new_velocity = create_velocity_arrow(df, next_frame)
    new_trail = create_animated_trail(df, next_frame)
    
    # Update figure data (indices: 0=moon, 1=trajectory, 2=markers, 3=spacecraft, 4=velocity, 5=trail)
    current_fig['data'][3] = new_spacecraft
    current_fig['data'][4] = new_velocity
    if new_trail and len(current_fig['data']) > 5:
        current_fig['data'][5] = new_trail
    elif new_trail:
        current_fig['data'].append(new_trail)
    
    # Update telemetry annotation
    telemetry_text = (
        f"<b>TELEMETRY</b><br>"
        f"Time: {df['Time_s'].iloc[next_frame]:.1f} s<br>"
        f"Altitude: {df['Altitude_km'].iloc[next_frame]:.1f} km<br>"
        f"Frame: {next_frame+1}/{len(df)}"
    )
    current_fig['layout']['annotations'][0]['text'] = telemetry_text
    
    return current_fig, next_frame

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
    
    app.run(debug=False, host='0.0.0.0', port=8050)

