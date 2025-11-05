#!/usr/bin/env python3
"""
Orbital Trajectory Animation Tool
Author: Clean implementation based on NASA trajectory visualization requirements
Date: November 2025
Description: 3D spacecraft trajectory animation with lunar orbit
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

NESC_VAR_NAMES = [
    "miPosition_m_X", "miPosition_m_Y", "miPosition_m_Z",
    "miVelocity_m_s_X", "miVelocity_m_s_Y", "miVelocity_m_s_Z"
]

NESC_TIME_VAR = 'j2000UtcTime_s'
MOON_RADIUS_M = 1737400  # Actual moon radius in meters

# ============================================================================
# DATA GENERATION - LUNAR ORBIT
# ============================================================================

def generate_lunar_orbit_trajectory(num_points=500):
    """
    Generate realistic lunar orbit trajectory.
    Simulates spacecraft orbiting the moon.
    """
    time_seconds = np.linspace(0, 7200, num_points)  # 2 hour orbit
    t_norm = np.linspace(0, 1, num_points)
    
    # Orbital parameters
    orbit_altitude = 100000  # 100 km altitude above moon surface
    orbit_radius = MOON_RADIUS_M + orbit_altitude
    
    # Number of orbits to complete
    n_orbits = 2
    theta = 2 * np.pi * n_orbits * t_norm
    
    # Elliptical orbit (slight eccentricity)
    eccentricity = 0.05
    r = orbit_radius * (1 - eccentricity * np.cos(theta))
    
    # Position in orbital plane
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Add inclination (orbit not perfectly in xy-plane)
    inclination = np.radians(15)  # 15 degree inclination
    z = r * np.sin(theta) * np.sin(inclination)
    y = r * np.sin(theta) * np.cos(inclination)
    
    # Add small perturbations (gravitational irregularities)
    x += np.random.normal(0, 100, num_points)
    y += np.random.normal(0, 100, num_points)
    z += np.random.normal(0, 50, num_points)
    
    # Calculate velocity from position derivatives
    vx = np.gradient(x, time_seconds)
    vy = np.gradient(y, time_seconds)
    vz = np.gradient(z, time_seconds)
    
    df = pd.DataFrame({
        NESC_VAR_NAMES[0]: x,
        NESC_VAR_NAMES[1]: y,
        NESC_VAR_NAMES[2]: z,
        NESC_VAR_NAMES[3]: vx,
        NESC_VAR_NAMES[4]: vy,
        NESC_VAR_NAMES[5]: vz,
        NESC_TIME_VAR: time_seconds.astype(str)
    })
    
    return df

def calculate_velocity_direction(vx, vy, vz):
    """
    Calculate unit vector in direction of velocity.
    Used for showing spacecraft heading.
    """
    magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    
    return vx / magnitude, vy / magnitude, vz / magnitude

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_orbital_visualization(df):
    """Create animated orbital trajectory visualization."""
    
    x = df[NESC_VAR_NAMES[0]].values
    y = df[NESC_VAR_NAMES[1]].values
    z = df[NESC_VAR_NAMES[2]].values
    
    vx = df[NESC_VAR_NAMES[3]].values
    vy = df[NESC_VAR_NAMES[4]].values
    vz = df[NESC_VAR_NAMES[5]].values
    
    # Calculate direction vectors for spacecraft heading
    dir_x, dir_y, dir_z = calculate_velocity_direction(vx, vy, vz)
    
    time_normalized = np.linspace(0, 1, len(x))
    
    fig = go.Figure()
    
    # Add the Moon as a sphere
    moon_theta = np.linspace(0, 2*np.pi, 50)
    moon_phi = np.linspace(0, np.pi, 40)
    moon_theta, moon_phi = np.meshgrid(moon_theta, moon_phi)
    
    moon_x = MOON_RADIUS_M * np.sin(moon_phi) * np.cos(moon_theta)
    moon_y = MOON_RADIUS_M * np.sin(moon_phi) * np.sin(moon_theta)
    moon_z = MOON_RADIUS_M * np.cos(moon_phi)
    
    fig.add_trace(go.Surface(
        x=moon_x, y=moon_y, z=moon_z,
        colorscale='gray',
        showscale=False,
        name='Moon',
        hoverinfo='skip',
        opacity=0.8
    ))
    
    # Add trail only (starts at first point, grows during animation)
    # Trail has green-to-red gradient showing progression
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='lines',
        name='Orbital Trail',
        line=dict(color='cyan', width=6),
        hovertemplate='<b>Trail</b><br>X: %{x:.2e} m<br>Y: %{y:.2e} m<br>Z: %{z:.2e} m<extra></extra>'
    ))
    
    # Add spacecraft marker (orange cone)
    fig.add_trace(go.Cone(
        x=[x[0]], y=[y[0]], z=[z[0]],
        u=[dir_x[0] * 200000], 
        v=[dir_y[0] * 200000], 
        w=[dir_z[0] * 200000],
        colorscale=[[0, 'orange'], [1, 'darkorange']],
        showscale=False,
        sizemode='absolute',
        sizeref=300000,
        name='Spacecraft',
        hovertext='Spacecraft Position'
    ))
    
    # Add velocity direction arrow (red line only, no legend)
    arrow_scale = 400000
    fig.add_trace(go.Scatter3d(
        x=[x[0], x[0] + dir_x[0] * arrow_scale],
        y=[y[0], y[0] + dir_y[0] * arrow_scale],
        z=[z[0], z[0] + dir_z[0] * arrow_scale],
        mode='lines',
        line=dict(color='red', width=8),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Create animation frames with trail effect
    frames = []
    for i in range(len(x)):
        # Calculate current telemetry
        current_time = float(df[NESC_TIME_VAR].iloc[i])
        velocity_mag = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
        altitude = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) - MOON_RADIUS_M
        
        frame = go.Frame(
            data=[
                # Moon (stays same)
                go.Surface(
                    x=moon_x, y=moon_y, z=moon_z,
                    colorscale='gray',
                    showscale=False,
                    opacity=0.8
                ),
                # Trail only (builds up as animation plays)
                # Color it with gradient to show time progression
                go.Scatter3d(
                    x=x[:i+1], y=y[:i+1], z=z[:i+1],
                    mode='lines',
                    line=dict(
                        color=time_normalized[:i+1],
                        colorscale='RdYlGn_r',
                        width=6,
                        colorbar=dict(
                            title=dict(text="Time", side="right"),
                            tickvals=[0, 1],
                            ticktext=["Start", "End"],
                            len=0.5
                        )
                    )
                ),
                # Spacecraft cone (orange)
                go.Cone(
                    x=[x[i]], y=[y[i]], z=[z[i]],
                    u=[dir_x[i] * 200000], 
                    v=[dir_y[i] * 200000], 
                    w=[dir_z[i] * 200000],
                    colorscale=[[0, 'orange'], [1, 'darkorange']],
                    showscale=False,
                    sizemode='absolute',
                    sizeref=300000
                ),
                # Velocity arrow (red)
                go.Scatter3d(
                    x=[x[i], x[i] + dir_x[i] * arrow_scale],
                    y=[y[i], y[i] + dir_y[i] * arrow_scale],
                    z=[z[i], z[i] + dir_z[i] * arrow_scale],
                    mode='lines',
                    line=dict(color='red', width=8)
                )
            ],
            name=str(i),
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f'<b>TELEMETRY</b><br>Time: {current_time:.1f} s<br>Altitude: {altitude/1000:.2f} km<br>Frame: {i+1}/{len(x)}',
                        xref='paper',
                        yref='paper',
                        x=0.02,
                        y=0.98,
                        xanchor='left',
                        yanchor='top',
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='black',
                        borderwidth=2,
                        font=dict(size=11, family='monospace'),
                        showarrow=False
                    )
                ]
            )
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add playback controls with speed options
    fig.update_layout(
        updatemenus=[
            # Speed selector dropdown
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                dict(
                    label='0.5x',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 200, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                ),
                dict(
                    label='1.0x',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                ),
                dict(
                    label='1.5x',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 67, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                ),
                dict(
                    label='2.0x',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                )
            ],
            direction='left',
            pad={'r': 10, 't': 10},
            x=0.4,
            xanchor='left',
            y=1.08,
            yanchor='top'
        ),
            # Play/Pause controls
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='Reset',
                        method='animate',
                        args=[[frames[0].name], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ],
                direction='left',
                pad={'r': 10, 't': 10},
                x=0.12,
                xanchor='left',
                y=1.08,
                yanchor='top'
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': 0,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Position: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': f"{int(i/len(frames)*100)}%",
                    'method': 'animate'
                }
                for i, f in enumerate(frames)
            ]
        }],
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)',
            bgcolor='white',
            xaxis=dict(
                backgroundcolor='rgb(245, 245, 245)', 
                gridcolor='rgb(200, 200, 200)',
                showbackground=True
            ),
            yaxis=dict(
                backgroundcolor='rgb(245, 245, 245)', 
                gridcolor='rgb(200, 200, 200)',
                showbackground=True
            ),
            zaxis=dict(
                backgroundcolor='rgb(245, 245, 245)', 
                gridcolor='rgb(200, 200, 200)',
                showbackground=True
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.5, y=2.5, z=1.5),  # Better viewing angle
                center=dict(x=0, y=0, z=0)
            )
        ),
        title={
            'text': 'Set A - Position Trajectory (Lunar Orbit)<br><sub>Other views: Under Construction</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.98,
            'yanchor': 'top',
            'font': {'size': 16, 'color': 'black'}
        },
        showlegend=True,
        legend=dict(
            x=0.88,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11)
        ),
        width=1400,
        height=900,
        hovermode='closest',
        paper_bgcolor='white'
    )
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading trajectory data...")
    df = generate_lunar_orbit_trajectory(num_points=500)
    
    print("Generating visualization...")
    fig = create_orbital_visualization(df)
    
    print("Opening in browser...")
    fig.show()

if __name__ == "__main__":
    main()

