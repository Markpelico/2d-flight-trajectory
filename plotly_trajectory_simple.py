#!/usr/bin/env python3
import numpy as np
import plotly.graph_objects as go

# same moon size as other scripts
MOON_RADIUS_KM = 1737.4
NUM_TRAJECTORY_POINTS = 600

def generate_lunar_orbit_trajectory(num_points=600):
    
    # lets say 2 hour mission (in seconds)
    time_seconds = np.linspace(0, 7200, num_points)
    t_norm = np.linspace(0, 1, num_points)
    
    # Orbital parameters
    orbit_altitude = 100  
    orbit_radius = MOON_RADIUS_KM + orbit_altitude
    
    # Complete 2 orbits
    n_orbits = 2
    theta = 2 * np.pi * n_orbits * t_norm
    
    # Elliptical orbit (slight eccentricity)
    eccentricity = 0.05
    r = orbit_radius * (1 - eccentricity * np.cos(theta))
    
    # add altitude variation so orbits dont overlap
    altitude_variation = 100 * t_norm  # gradually increases
    altitude_variation += 30 * np.sin(n_orbits * theta) # adds some wave to it
    r = r + altitude_variation
    
    # calculate the x y coords based off the current r and angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Add inclination (15 degree tilt)
    inclination = np.radians(15)
    z = r * np.sin(theta) * np.sin(inclination)
    y = r * np.sin(theta) * np.cos(inclination)
    
    # add tiny random movements to make it look more realistic
    x += np.random.normal(0, 0.1, num_points)
    y += np.random.normal(0, 0.1, num_points)
    z += np.random.normal(0, 0.05, num_points)
    
    # Calculate velocity
    vx = np.gradient(x, time_seconds)
    vy = np.gradient(y, time_seconds)
    vz = np.gradient(z, time_seconds)
    
    # Calculate altitude above surface for the spacecraft
    altitude = np.sqrt(x**2 + y**2 + z**2) - MOON_RADIUS_KM
    
    return x, y, z, vx, vy, vz, altitude, time_seconds

# generate the data
print("Generating orbit data...")
x, y, z, vx, vy, vz, altitude, time_elapsed = generate_lunar_orbit_trajectory(NUM_TRAJECTORY_POINTS)

# CREATE MOON SPHERE
# lower resolution for better performance
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 25)
moon_x = MOON_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
moon_y = MOON_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
moon_z = MOON_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))

moon = go.Surface(
    x=moon_x, y=moon_y, z=moon_z,
    colorscale=[[0, '#808080'], [1, '#808080']],
    showscale=False,
    opacity=0.4,
    name='Moon',
    hoverinfo='skip'
)

# CREATE START AND END MARKERS
markers = go.Scatter3d(
    x=[x[0], x[-1]],
    y=[y[0], y[-1]],
    z=[z[0], z[-1]],
    mode='markers',
    marker=dict(size=[10, 10], color=['green', 'red']),
    name='Start/End',
    text=['Start', 'End'],
    hovertemplate='<b>%{text}</b><br>X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Z: %{z:.1f} km<extra></extra>'
)

# CREATE ANIMATION FRAMES
# each frame shows spacecraft position and trail
frames = []
frame_step = 3  # only create frames every 3 points to reduce lag

for i in range(0, len(x), frame_step):
    # spacecraft position
    spacecraft = go.Scatter3d(
        x=[x[i]],
        y=[y[i]],
        z=[z[i]],
        mode='markers',
        marker=dict(size=12, color='black', line=dict(color='gray', width=2)),
        name='Spacecraft',
        hoverinfo='skip'
    )
    
    # velocity arrow
    vel_mag = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    if vel_mag > 0:
        arrow_length = 400
        vx_norm = vx[i] / vel_mag * arrow_length
        vy_norm = vy[i] / vel_mag * arrow_length
        vz_norm = vz[i] / vel_mag * arrow_length
        
        velocity_arrow = go.Cone(
            x=[x[i]], y=[y[i]], z=[z[i]],
            u=[vx_norm], v=[vy_norm], w=[vz_norm],
            colorscale=[[0, '#444444'], [1, '#444444']],
            showscale=False,
            sizemode='absolute',
            sizeref=150,
            name='Velocity',
            hoverinfo='skip'
        )
    else:
        velocity_arrow = go.Scatter3d(x=[], y=[], z=[], mode='markers')
    
    # trail (last 100 points with gradient)
    trail_start = max(0, i - 100)
    if i > 0:
        trail_x = x[trail_start:i+1]
        trail_y = y[trail_start:i+1]
        trail_z = z[trail_start:i+1]
        
        # green to red gradient
        n = len(trail_x)
        progress = np.linspace(0, 1, n)
        colors = [f'rgb({int(p*255)},{int((1-p)*255)},0)' for p in progress]
        
        trail = go.Scatter3d(
            x=trail_x, y=trail_y, z=trail_z,
            mode='lines',
            line=dict(color=colors, width=6),
            name='Trail',
            hoverinfo='skip',
            showlegend=False
        )
    else:
        trail = go.Scatter3d(x=[], y=[], z=[], mode='lines')
    
    # telemetry text for this frame
    telemetry = (
        f"<b>TELEMETRY</b><br>"
        f"Time: {time_elapsed[i]:.1f} s<br>"
        f"Altitude: {altitude[i]:.1f} km<br>"
        f"Frame: {i+1}/{len(x)}"
    )
    
    frame = go.Frame(
        data=[moon, markers, spacecraft, velocity_arrow, trail],
        name=str(i),
        layout=go.Layout(
            annotations=[dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=telemetry,
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
    )
    frames.append(frame)

# CREATE FIGURE
print("Creating animation...")
fig = go.Figure(
    data=[moon, markers],
    frames=frames
)

# add play/pause buttons
fig.update_layout(
    title=dict(
        text='Lunar Orbit Trajectory - 3D Animation',
        font=dict(size=20, color='#2c3e50'),
        x=0.5,
        xanchor='center'
    ),
    scene=dict(
        xaxis=dict(title='X (km)', backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True),
        yaxis=dict(title='Y (km)', backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True),
        zaxis=dict(title='Z (km)', backgroundcolor='rgb(240,240,240)', gridcolor='white', showbackground=True),
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            x=0.12, y=0.05,
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                     fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        ),
        dict(
            type='buttons',
            x=0.25, y=0.05,
            buttons=[
                dict(label='0.5x', method='animate',
                     args=[None, dict(frame=dict(duration=100, redraw=True), mode='immediate')]),
                dict(label='1.0x', method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True), mode='immediate')]),
                dict(label='1.5x', method='animate',
                     args=[None, dict(frame=dict(duration=33, redraw=True), mode='immediate')]),
                dict(label='2.0x', method='animate',
                     args=[None, dict(frame=dict(duration=25, redraw=True), mode='immediate')])
            ]
        )
    ],
    sliders=[dict(
        active=0,
        steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                   label=f"{int(f.name)//frame_step}s",
                   method='animate')
               for f in frames[::5]],  # subsample for slider performance
        x=0.12, y=0.0,
        len=0.85
    )],
    showlegend=True,
    legend=dict(x=0.02, y=0.85, bgcolor='rgba(255,255,255,0.9)'),
    paper_bgcolor='#f8f9fa',
    margin=dict(l=0, r=0, t=60, b=40)
)

print("Opening browser...")
fig.show()

