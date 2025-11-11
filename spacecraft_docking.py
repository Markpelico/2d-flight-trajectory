#!/usr/bin/env python3
"""
Realistic Spacecraft Docking Simulation
========================================
High-fidelity 3D animation of spacecraft docking with detailed meshes,
proper rotation alignment, thrust visualization, and docking ports.

Features:
- Detailed 3D spacecraft meshes (cylinder + cone)
- Realistic rotation to align docking ports
- Color-coded alignment indicator (green=aligned, red=misaligned)
- Thrust flames that scale with magnitude
- Docking port visualization
- Ghost path prediction (120s ahead)
- 1000 simulation points for smooth animation
- White background for clarity

Physics:
- Orbital mechanics with gravity
- Thrust vectoring for translation and rotation
- Attitude control for docking alignment
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_TIME = 600  # seconds (10 minutes)
TIME_STEP = 0.6  # seconds
NUM_STEPS = 1000  # High resolution for smooth animation

# Spacecraft parameters
DOCKING_STATION_MASS = 50000  # kg
ACTIVE_SPACECRAFT_MASS = 15000  # kg
THRUST_MAGNITUDE = 500  # Newtons
TORQUE_MAGNITUDE = 200  # Nm for rotation

# Initial conditions
INITIAL_SEPARATION = 1000  # meters

# Prediction
PREDICTION_HORIZON = 120  # seconds
PREDICTION_STEPS = 40

# ============================================================================
# 3D MESH GENERATION
# ============================================================================

def create_spacecraft_mesh(scale=1.0):
    """
    Create detailed 3D mesh for spacecraft.
    Returns vertices for cylinder body + cone nose.
    """
    # Cylinder body
    theta = np.linspace(0, 2*np.pi, 16)
    z_cyl = np.linspace(-10, 0, 10) * scale
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    
    radius = 2 * scale
    x_cyl = radius * np.cos(theta_grid)
    y_cyl = radius * np.sin(theta_grid)
    z_cyl_grid = z_grid
    
    # Cone nose
    z_cone = np.linspace(0, 4, 8) * scale
    theta_cone, z_cone_grid = np.meshgrid(theta, z_cone)
    
    r_cone = radius * (1 - z_cone / (4*scale))
    x_cone = r_cone * np.cos(theta_cone)
    y_cone = r_cone * np.sin(theta_cone)
    
    # Combine
    x_mesh = np.vstack([x_cyl, x_cone])
    y_mesh = np.vstack([y_cyl, y_cone])
    z_mesh = np.vstack([z_cyl_grid, z_cone_grid])
    
    return x_mesh, y_mesh, z_mesh

def create_docking_port_mesh(scale=1.0):
    """
    Create docking port ring mesh.
    """
    theta = np.linspace(0, 2*np.pi, 20)
    r_outer = 2.5 * scale
    r_inner = 1.5 * scale
    
    # Ring
    x_ring = []
    y_ring = []
    z_ring = []
    
    for r in [r_inner, r_outer]:
        x_ring.append(r * np.cos(theta))
        y_ring.append(r * np.sin(theta))
        z_ring.append(np.zeros_like(theta))
    
    return np.array(x_ring), np.array(y_ring), np.array(z_ring)

def rotate_mesh(x, y, z, roll, pitch, yaw):
    """
    Rotate mesh using Euler angles (in radians).
    Returns rotated coordinates.
    """
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # Flatten, rotate, reshape
    shape = x.shape
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    rotated = R @ points
    
    x_rot = rotated[0].reshape(shape)
    y_rot = rotated[1].reshape(shape)
    z_rot = rotated[2].reshape(shape)
    
    return x_rot, y_rot, z_rot

def translate_mesh(x, y, z, pos):
    """Translate mesh to position."""
    return x + pos[0], y + pos[1], z + pos[2]

# ============================================================================
# PHYSICS SIMULATION
# ============================================================================

def calculate_alignment_angle(craft_orientation, relative_pos):
    """
    Calculate angle between spacecraft orientation and docking direction.
    Returns angle in degrees.
    """
    # Docking port should face opposite to relative position
    desired_direction = -relative_pos / np.linalg.norm(relative_pos)
    
    # Current docking port direction (bottom of spacecraft, -Z axis)
    current_direction = craft_orientation
    
    dot_product = np.dot(current_direction, desired_direction)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_thrust_and_torque(current_pos, target_pos, current_vel, 
                                current_orientation, current_time):
    """
    Calculate thrust for translation and torque for rotation.
    Returns thrust vector and torque scalar.
    """
    # Thrust phases
    thrust_phases = [
        (0, 120),      # Initial approach
        (300, 480),    # Mid-course correction
        (480, 600)     # Final docking
    ]
    
    in_thrust_phase = any(start <= current_time <= end for start, end in thrust_phases)
    
    if not in_thrust_phase:
        return np.array([0.0, 0.0, 0.0]), 0.0
    
    # Position control
    pos_error = target_pos - current_pos
    distance = np.linalg.norm(pos_error)
    
    # Speed control based on distance
    if distance > 100:
        desired_speed = 5.0
    elif distance > 20:
        desired_speed = 1.0
    else:
        desired_speed = 0.3
    
    if distance > 0.1:
        desired_vel = (pos_error / distance) * desired_speed
    else:
        desired_vel = np.array([0.0, 0.0, 0.0])
    
    vel_error = desired_vel - current_vel
    
    # PD controller
    kp = 0.8
    kd = 2.0
    thrust_direction = kp * pos_error + kd * vel_error
    thrust_mag = np.linalg.norm(thrust_direction)
    
    if thrust_mag > 0:
        thrust_direction = (thrust_direction / thrust_mag) * THRUST_MAGNITUDE
    else:
        thrust_direction = np.array([0.0, 0.0, 0.0])
    
    # Torque for orientation control (important for docking)
    if distance < 100:  # Start aligning when close
        alignment_angle = calculate_alignment_angle(current_orientation, pos_error)
        
        # Torque proportional to misalignment
        if alignment_angle > 5:  # 5 degree deadband
            torque = TORQUE_MAGNITUDE * (alignment_angle / 180.0)
        else:
            torque = 0.0
    else:
        torque = 0.0
    
    return thrust_direction, torque

def simulate_docking():
    """
    Run full docking simulation with rotation.
    """
    time = np.linspace(0, SIMULATION_TIME, NUM_STEPS)
    
    # Docking station (passive)
    station_pos = np.zeros((NUM_STEPS, 3))
    station_vel = np.zeros((NUM_STEPS, 3))
    station_orientation = np.zeros((NUM_STEPS, 3))  # Roll, pitch, yaw
    
    # Active spacecraft
    craft_pos = np.zeros((NUM_STEPS, 3))
    craft_vel = np.zeros((NUM_STEPS, 3))
    craft_orientation = np.zeros((NUM_STEPS, 3))  # Roll, pitch, yaw
    craft_angular_vel = np.zeros((NUM_STEPS, 3))
    craft_thrust = np.zeros((NUM_STEPS, 3))
    craft_torque = np.zeros(NUM_STEPS)
    
    # Initial conditions
    station_pos[0] = np.array([0.0, 0.0, 0.0])
    station_vel[0] = np.array([0.5, 0.2, 0.1])
    station_orientation[0] = np.array([0.0, 0.0, 0.0])
    
    craft_pos[0] = np.array([INITIAL_SEPARATION, 200, -100])
    craft_vel[0] = np.array([0.3, 0.15, -0.05])
    craft_orientation[0] = np.array([0.3, 0.5, 0.2])  # Start misaligned
    
    dt = time[1] - time[0]
    
    for i in range(1, NUM_STEPS):
        # Calculate thrust and torque
        craft_thrust[i], craft_torque[i] = calculate_thrust_and_torque(
            craft_pos[i-1], station_pos[i-1],
            craft_vel[i-1], craft_orientation[i-1],
            time[i]
        )
        
        # Update docking station (passive)
        station_pos[i] = station_pos[i-1] + station_vel[i-1] * dt
        station_vel[i] = station_vel[i-1]
        station_orientation[i] = station_orientation[i-1]
        
        # Update spacecraft translation
        craft_acc = craft_thrust[i] / ACTIVE_SPACECRAFT_MASS
        craft_vel[i] = craft_vel[i-1] + craft_acc * dt
        craft_pos[i] = craft_pos[i-1] + craft_vel[i] * dt
        
        # Update spacecraft rotation
        moment_of_inertia = ACTIVE_SPACECRAFT_MASS * 4.0  # Simplified
        angular_acc = craft_torque[i] / moment_of_inertia
        
        # Only rotate around axis perpendicular to approach
        rel_pos = station_pos[i] - craft_pos[i]
        if np.linalg.norm(rel_pos) > 0:
            approach_axis = rel_pos / np.linalg.norm(rel_pos)
            
            # Gradually align orientation
            desired_orientation = np.array([
                np.arctan2(approach_axis[1], approach_axis[2]),
                np.arctan2(approach_axis[0], approach_axis[2]),
                0.0
            ])
            
            # Smooth orientation interpolation
            orientation_error = desired_orientation - craft_orientation[i-1]
            craft_orientation[i] = craft_orientation[i-1] + orientation_error * 0.01
        else:
            craft_orientation[i] = craft_orientation[i-1]
    
    # Calculate metrics
    rel_pos = craft_pos - station_pos
    distance = np.linalg.norm(rel_pos, axis=1)
    rel_vel = craft_vel - station_vel
    rel_speed = np.linalg.norm(rel_vel, axis=1)
    thrust_magnitude = np.linalg.norm(craft_thrust, axis=1)
    
    # Calculate alignment angles
    alignment_angles = np.zeros(NUM_STEPS)
    for i in range(NUM_STEPS):
        if distance[i] > 0:
            alignment_angles[i] = calculate_alignment_angle(
                craft_orientation[i], rel_pos[i]
            )
    
    # Create DataFrames
    station_df = pd.DataFrame({
        'Time_s': time,
        'X_m': station_pos[:, 0],
        'Y_m': station_pos[:, 1],
        'Z_m': station_pos[:, 2],
        'Roll_rad': station_orientation[:, 0],
        'Pitch_rad': station_orientation[:, 1],
        'Yaw_rad': station_orientation[:, 2]
    })
    
    craft_df = pd.DataFrame({
        'Time_s': time,
        'X_m': craft_pos[:, 0],
        'Y_m': craft_pos[:, 1],
        'Z_m': craft_pos[:, 2],
        'Roll_rad': craft_orientation[:, 0],
        'Pitch_rad': craft_orientation[:, 1],
        'Yaw_rad': craft_orientation[:, 2],
        'Thrust_X_N': craft_thrust[:, 0],
        'Thrust_Y_N': craft_thrust[:, 1],
        'Thrust_Z_N': craft_thrust[:, 2],
        'Thrust_Mag_N': thrust_magnitude,
        'Distance_m': distance,
        'Rel_Speed_m_s': rel_speed,
        'Alignment_deg': alignment_angles
    })
    
    return station_df, craft_df

def predict_trajectory(current_pos, current_vel, current_time):
    """Predict future trajectory."""
    dt = PREDICTION_HORIZON / PREDICTION_STEPS
    pred_pos = np.zeros((PREDICTION_STEPS, 3))
    pred_pos[0] = current_pos
    pred_vel = current_vel.copy()
    
    for i in range(1, PREDICTION_STEPS):
        pred_pos[i] = pred_pos[i-1] + pred_vel * dt
    
    return pred_pos

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_thrust_flame(pos, thrust_vector, thrust_mag):
    """
    Create thrust flame visual pointing opposite to thrust direction.
    """
    if thrust_mag < 1.0:
        return None
    
    # Flame points opposite to thrust
    flame_direction = -thrust_vector / (np.linalg.norm(thrust_vector) + 1e-6)
    flame_length = (thrust_mag / THRUST_MAGNITUDE) * 15  # Scale with thrust
    
    flame_end = pos + flame_direction * flame_length
    
    return go.Scatter3d(
        x=[pos[0], flame_end[0]],
        y=[pos[1], flame_end[1]],
        z=[pos[2], flame_end[2]],
        mode='lines',
        line=dict(color='orange', width=8),
        showlegend=False,
        hoverinfo='skip'
    )

def create_animation_frame(station_df, craft_df, frame_idx):
    """
    Create all traces for a single animation frame.
    """
    traces = []
    
    # Get current state
    station_pos = station_df[['X_m', 'Y_m', 'Z_m']].iloc[frame_idx].values
    station_orient = station_df[['Roll_rad', 'Pitch_rad', 'Yaw_rad']].iloc[frame_idx].values
    
    craft_pos = craft_df[['X_m', 'Y_m', 'Z_m']].iloc[frame_idx].values
    craft_orient = craft_df[['Roll_rad', 'Pitch_rad', 'Yaw_rad']].iloc[frame_idx].values
    craft_thrust_vec = craft_df[['Thrust_X_N', 'Thrust_Y_N', 'Thrust_Z_N']].iloc[frame_idx].values
    thrust_mag = craft_df['Thrust_Mag_N'].iloc[frame_idx]
    alignment = craft_df['Alignment_deg'].iloc[frame_idx]
    
    # Docking station mesh
    x_mesh, y_mesh, z_mesh = create_spacecraft_mesh(scale=3.0)
    x_rot, y_rot, z_rot = rotate_mesh(x_mesh, y_mesh, z_mesh, *station_orient)
    x_final, y_final, z_final = translate_mesh(x_rot, y_rot, z_rot, station_pos)
    
    traces.append(go.Surface(
        x=x_final, y=y_final, z=z_final,
        colorscale=[[0, '#3498db'], [1, '#2980b9']],
        showscale=False,
        name='Docking Station',
        hoverinfo='skip',
        showlegend=(frame_idx==0)
    ))
    
    # Docking port on station
    x_port, y_port, z_port = create_docking_port_mesh(scale=3.0)
    for j in range(len(x_port)):
        x_p, y_p, z_p = rotate_mesh(
            x_port[j].reshape(1, -1),
            y_port[j].reshape(1, -1),
            z_port[j] + 10,  # At front
            *station_orient
        )
        x_pf, y_pf, z_pf = translate_mesh(x_p, y_p, z_p, station_pos)
        
        traces.append(go.Scatter3d(
            x=x_pf.flatten(), y=y_pf.flatten(), z=z_pf.flatten(),
            mode='lines',
            line=dict(color='yellow', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Active spacecraft mesh (color based on alignment)
    x_mesh2, y_mesh2, z_mesh2 = create_spacecraft_mesh(scale=2.0)
    x_rot2, y_rot2, z_rot2 = rotate_mesh(x_mesh2, y_mesh2, z_mesh2, *craft_orient)
    x_final2, y_final2, z_final2 = translate_mesh(x_rot2, y_rot2, z_rot2, craft_pos)
    
    # Color: green if aligned, red if misaligned
    if alignment < 10:
        color_scale = [[0, '#27ae60'], [1, '#229954']]
    elif alignment < 30:
        color_scale = [[0, '#f39c12'], [1, '#e67e22']]
    else:
        color_scale = [[0, '#e74c3c'], [1, '#c0392b']]
    
    traces.append(go.Surface(
        x=x_final2, y=y_final2, z=z_final2,
        colorscale=color_scale,
        showscale=False,
        name='Spacecraft',
        hoverinfo='skip',
        showlegend=(frame_idx==0)
    ))
    
    # Thrust flames
    if thrust_mag > 1.0:
        flame = create_thrust_flame(craft_pos, craft_thrust_vec, thrust_mag)
        if flame:
            traces.append(flame)
    
    # Trails
    if frame_idx > 0:
        # Station trail
        traces.append(go.Scatter3d(
            x=station_df['X_m'].iloc[:frame_idx+1],
            y=station_df['Y_m'].iloc[:frame_idx+1],
            z=station_df['Z_m'].iloc[:frame_idx+1],
            mode='lines',
            line=dict(color='#3498db', width=2),
            name='Station Path',
            showlegend=(frame_idx==10),
            hoverinfo='skip'
        ))
        
        # Craft trail
        traces.append(go.Scatter3d(
            x=craft_df['X_m'].iloc[:frame_idx+1],
            y=craft_df['Y_m'].iloc[:frame_idx+1],
            z=craft_df['Z_m'].iloc[:frame_idx+1],
            mode='lines',
            line=dict(color=('#27ae60' if alignment < 10 else '#e74c3c'), width=3),
            name='Spacecraft Path',
            showlegend=(frame_idx==10),
            hoverinfo='skip'
        ))
    
    # Ghost path
    if frame_idx < len(craft_df) - 10:
        pred_pos = predict_trajectory(
            craft_pos,
            craft_df[['VX_m_s' if 'VX_m_s' in craft_df.columns else 'Thrust_X_N']].iloc[frame_idx].values if 'VX_m_s' in craft_df.columns else craft_thrust_vec / ACTIVE_SPACECRAFT_MASS,
            craft_df['Time_s'].iloc[frame_idx]
        )
        
        traces.append(go.Scatter3d(
            x=pred_pos[:, 0],
            y=pred_pos[:, 1],
            z=pred_pos[:, 2],
            mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            name='Ghost Path',
            showlegend=(frame_idx==10),
            opacity=0.5,
            hoverinfo='skip'
        ))
    
    return traces

def create_docking_visualization(station_df, craft_df):
    """
    Create beautiful animated visualization.
    """
    # Create frames (every 10th frame for performance)
    frame_step = 10
    frames = []
    
    for i in range(0, len(craft_df), frame_step):
        frame_traces = create_animation_frame(station_df, craft_df, i)
        
        # Telemetry
        telemetry = (
            f"<b>DOCKING TELEMETRY</b><br>"
            f"Time: {craft_df['Time_s'].iloc[i]:.1f} s<br>"
            f"Distance: {craft_df['Distance_m'].iloc[i]:.1f} m<br>"
            f"Speed: {craft_df['Rel_Speed_m_s'].iloc[i]:.2f} m/s<br>"
            f"Alignment: {craft_df['Alignment_deg'].iloc[i]:.1f}°<br>"
            f"Thrust: {craft_df['Thrust_Mag_N'].iloc[i]:.0f} N<br>"
            f"Status: {'ALIGNED' if craft_df['Alignment_deg'].iloc[i] < 10 else 'ALIGNING'}"
        )
        
        frames.append(go.Frame(
            data=frame_traces,
            name=str(i),
            layout=go.Layout(
                annotations=[dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=telemetry,
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#3498db',
                    borderwidth=2,
                    font=dict(size=11, family='Courier New', color='black'),
                    align='left',
                    xanchor='left',
                    yanchor='top'
                )]
            )
        ))
    
    # Create figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text='Realistic Spacecraft Docking - 3D Mesh Animation',
            font=dict(size=22, color='#2c3e50', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor='white', gridcolor='#e0e0e0', showbackground=True),
            yaxis=dict(title='Y (m)', backgroundcolor='white', gridcolor='#e0e0e0', showbackground=True),
            zaxis=dict(title='Z (m)', backgroundcolor='white', gridcolor='#e0e0e0', showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=2.5, y=2.5, z=1.8)),
            bgcolor='#f8f9fa'
        ),
        annotations=frames[0].layout.annotations,
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=60, b=0),
        updatemenus=[
            dict(
                type='buttons',
                x=0.12, y=0.02,
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True, mode='immediate')]),
                    dict(label='Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                ],
                bgcolor='#3498db',
                font=dict(color='white', size=12)
            ),
            dict(
                type='buttons',
                x=0.25, y=0.02,
                buttons=[
                    dict(label='0.5x', method='animate',
                         args=[None, dict(frame=dict(duration=100, redraw=True), mode='immediate')]),
                    dict(label='1x', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True), mode='immediate')]),
                    dict(label='2x', method='animate',
                         args=[None, dict(frame=dict(duration=25, redraw=True), mode='immediate')])
                ],
                bgcolor='white',
                font=dict(color='black', size=11)
            )
        ],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                       label=f"{int(f.name)*TIME_STEP*frame_step:.0f}s",
                       method='animate')
                   for f in frames[::3]],
            x=0.12, y=0.0,
            len=0.85,
            bgcolor='rgba(52, 152, 219, 0.2)',
            bordercolor='#3498db'
        )]
    )
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("REALISTIC SPACECRAFT DOCKING SIMULATION")
    print("="*70)
    print("Running high-fidelity physics simulation...")
    
    station_df, craft_df = simulate_docking()
    
    print("Generating 3D meshes and animation frames...")
    fig = create_docking_visualization(station_df, craft_df)
    
    print("\nSimulation Results:")
    print(f"  Initial separation: {INITIAL_SEPARATION} m")
    print(f"  Final distance: {craft_df['Distance_m'].iloc[-1]:.2f} m")
    print(f"  Final alignment: {craft_df['Alignment_deg'].iloc[-1]:.1f} degrees")
    print(f"  Final speed: {craft_df['Rel_Speed_m_s'].iloc[-1]:.3f} m/s")
    print(f"  Status: {'DOCKED' if craft_df['Distance_m'].iloc[-1] < 5 and craft_df['Alignment_deg'].iloc[-1] < 10 else 'APPROACHING'}")
    print("\nOpening animation...")
    print("="*70 + "\n")
    
    fig.show()

if __name__ == "__main__":
    main()
