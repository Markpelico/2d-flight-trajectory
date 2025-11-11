#!/usr/bin/env python3
"""
Spacecraft Docking Animation - Predictive Ghost Path
====================================================
Beautiful real-time animation of two spacecraft docking in space.

Features:
- Docking station (blue) - passive drift
- Active spacecraft (red) - thrust maneuvering  
- Ghost path (cyan) - shows predicted trajectory 120 seconds ahead
- Animated trails with gradient colors
- Thrust indicators (flames when firing)
- Real-time distance and velocity display
- Smooth 3D camera control during animation

Controls:
- Drag to rotate view during animation
- Scroll to zoom
- Animation runs automatically
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
SIMULATION_TIME = 600  # seconds (10 minutes)
TIME_STEP = 0.5  # seconds
NUM_STEPS = int(SIMULATION_TIME / TIME_STEP)

# Spacecraft parameters
DOCKING_STATION_MASS = 50000  # kg
ACTIVE_SPACECRAFT_MASS = 15000  # kg
THRUST_MAGNITUDE = 500  # Newtons

# Orbital parameters (in space, not near planet)
GRAVITY_CONSTANT = 0.00001  # Simplified gravity for deep space rendezvous
INITIAL_SEPARATION = 1000  # meters

# Prediction parameters
PREDICTION_HORIZON = 120  # seconds ahead to predict
PREDICTION_STEPS = 60  # Number of prediction points

# ============================================================================
# PHYSICS ENGINE
# ============================================================================

def calculate_gravity_force(pos1, pos2, mass1, mass2):
    """
    Calculate gravitational attraction between two objects.
    Simplified for space rendezvous scenario.
    """
    r_vec = pos2 - pos1
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag < 1.0:  # Avoid singularity
        return np.array([0.0, 0.0, 0.0])
    
    # Gravitational force (simplified)
    force_mag = GRAVITY_CONSTANT * mass1 * mass2 / (r_mag ** 2)
    force_vec = force_mag * (r_vec / r_mag)
    
    return force_vec

def calculate_thrust_vector(current_pos, target_pos, current_vel, current_time):
    """
    Calculate optimal thrust vector for docking approach.
    Uses a simple proportional-derivative (PD) controller.
    
    Thrust phases:
    - 0-60s: Initial alignment and velocity matching
    - 60-300s: Free flight (coasting)
    - 300-450s: Mid-course correction
    - 450-540s: Free flight
    - 540-600s: Final approach and docking
    """
    # Determine if we're in a thrust phase
    thrust_phases = [
        (0, 60),      # Initial alignment
        (300, 450),   # Mid-course correction
        (540, 600)    # Final approach
    ]
    
    in_thrust_phase = any(start <= current_time <= end for start, end in thrust_phases)
    
    if not in_thrust_phase:
        return np.array([0.0, 0.0, 0.0])  # Free flight
    
    # Calculate position error
    pos_error = target_pos - current_pos
    distance = np.linalg.norm(pos_error)
    
    # Calculate desired velocity (proportional to distance)
    if distance > 100:
        desired_speed = 5.0  # m/s for far approach
    elif distance > 20:
        desired_speed = 1.0  # m/s for close approach
    else:
        desired_speed = 0.2  # m/s for final docking
    
    if distance > 0.1:
        desired_vel = (pos_error / distance) * desired_speed
    else:
        desired_vel = np.array([0.0, 0.0, 0.0])
    
    # Calculate velocity error
    vel_error = desired_vel - current_vel
    
    # PD controller gains
    kp = 0.8  # Proportional gain
    kd = 2.0  # Derivative gain
    
    # Thrust direction (normalized)
    thrust_direction = kp * pos_error + kd * vel_error
    thrust_mag = np.linalg.norm(thrust_direction)
    
    if thrust_mag > 0:
        thrust_direction = thrust_direction / thrust_mag
        return thrust_direction * THRUST_MAGNITUDE
    
    return np.array([0.0, 0.0, 0.0])

def simulate_docking():
    """
    Run the full docking simulation for both spacecraft.
    Returns DataFrames for both spacecraft trajectories.
    """
    # Initialize arrays
    time = np.zeros(NUM_STEPS)
    
    # Docking station (passive)
    station_pos = np.zeros((NUM_STEPS, 3))
    station_vel = np.zeros((NUM_STEPS, 3))
    
    # Active spacecraft
    craft_pos = np.zeros((NUM_STEPS, 3))
    craft_vel = np.zeros((NUM_STEPS, 3))
    craft_thrust = np.zeros((NUM_STEPS, 3))
    
    # Initial conditions - docking station at origin
    station_pos[0] = np.array([0.0, 0.0, 0.0])
    station_vel[0] = np.array([0.5, 0.2, 0.1])  # Slow drift
    
    # Active spacecraft starts offset
    craft_pos[0] = np.array([INITIAL_SEPARATION, 200, -100])
    craft_vel[0] = np.array([0.3, 0.15, -0.05])  # Different velocity
    
    # Run simulation
    for i in range(1, NUM_STEPS):
        time[i] = i * TIME_STEP
        
        # Calculate forces
        # Mutual gravity (very weak in space)
        grav_force = calculate_gravity_force(
            craft_pos[i-1], station_pos[i-1],
            ACTIVE_SPACECRAFT_MASS, DOCKING_STATION_MASS
        )
        
        # Calculate thrust for active spacecraft
        craft_thrust[i] = calculate_thrust_vector(
            craft_pos[i-1], station_pos[i-1],
            craft_vel[i-1], time[i]
        )
        
        # Update docking station (passive motion only)
        station_acc = -grav_force / DOCKING_STATION_MASS
        station_vel[i] = station_vel[i-1] + station_acc * TIME_STEP
        station_pos[i] = station_pos[i-1] + station_vel[i] * TIME_STEP
        
        # Update active spacecraft (thrust + gravity)
        craft_acc = grav_force / ACTIVE_SPACECRAFT_MASS + craft_thrust[i] / ACTIVE_SPACECRAFT_MASS
        craft_vel[i] = craft_vel[i-1] + craft_acc * TIME_STEP
        craft_pos[i] = craft_pos[i-1] + craft_vel[i] * TIME_STEP
    
    # Calculate relative metrics
    rel_pos = craft_pos - station_pos
    distance = np.linalg.norm(rel_pos, axis=1)
    rel_vel = craft_vel - station_vel
    rel_speed = np.linalg.norm(rel_vel, axis=1)
    thrust_magnitude = np.linalg.norm(craft_thrust, axis=1)
    
    # Create DataFrames
    station_df = pd.DataFrame({
        'Time_s': time,
        'X_m': station_pos[:, 0],
        'Y_m': station_pos[:, 1],
        'Z_m': station_pos[:, 2],
        'VX_m_s': station_vel[:, 0],
        'VY_m_s': station_vel[:, 1],
        'VZ_m_s': station_vel[:, 2]
    })
    
    craft_df = pd.DataFrame({
        'Time_s': time,
        'X_m': craft_pos[:, 0],
        'Y_m': craft_pos[:, 1],
        'Z_m': craft_pos[:, 2],
        'VX_m_s': craft_vel[:, 0],
        'VY_m_s': craft_vel[:, 1],
        'VZ_m_s': craft_vel[:, 2],
        'Thrust_X_N': craft_thrust[:, 0],
        'Thrust_Y_N': craft_thrust[:, 1],
        'Thrust_Z_N': craft_thrust[:, 2],
        'Thrust_Mag_N': thrust_magnitude,
        'Distance_m': distance,
        'Rel_Speed_m_s': rel_speed
    })
    
    return station_df, craft_df

def predict_trajectory(current_pos, current_vel, current_time, horizon=PREDICTION_HORIZON):
    """
    Predict future trajectory of spacecraft (ghost path).
    Assumes no thrust (free flight prediction).
    """
    steps = PREDICTION_STEPS
    dt = horizon / steps
    
    pred_pos = np.zeros((steps, 3))
    pred_pos[0] = current_pos
    pred_vel = current_vel.copy()
    
    for i in range(1, steps):
        # Simple ballistic prediction (no thrust)
        pred_pos[i] = pred_pos[i-1] + pred_vel * dt
    
    return pred_pos

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_docking_visualization(station_df, craft_df):
    """
    Create beautiful animated docking visualization with ghost path.
    """
    # Create animated frames (sample every 10 steps for smooth animation)
    frame_step = 10
    frames = []
    
    for i in range(0, len(craft_df), frame_step):
        # Docking station position at this frame
        station_x = station_df['X_m'].iloc[:i+1]
        station_y = station_df['Y_m'].iloc[:i+1]
        station_z = station_df['Z_m'].iloc[:i+1]
        
        # Active spacecraft position at this frame
        craft_x = craft_df['X_m'].iloc[:i+1]
        craft_y = craft_df['Y_m'].iloc[:i+1]
        craft_z = craft_df['Z_m'].iloc[:i+1]
        
        # Predict future trajectory (ghost path)
        if i < len(craft_df) - 1:
            pred_pos = predict_trajectory(
                craft_df[['X_m', 'Y_m', 'Z_m']].iloc[i].values,
                craft_df[['VX_m_s', 'VY_m_s', 'VZ_m_s']].iloc[i].values,
                craft_df['Time_s'].iloc[i]
            )
        else:
            pred_pos = np.array([[craft_df['X_m'].iloc[i], 
                                 craft_df['Y_m'].iloc[i], 
                                 craft_df['Z_m'].iloc[i]]])
        
        # Check if thrusting
        is_thrusting = craft_df['Thrust_Mag_N'].iloc[i] > 1.0
        
        # Telemetry text
        telemetry = (
            f"<b>DOCKING TELEMETRY</b><br>"
            f"Time: {craft_df['Time_s'].iloc[i]:.1f} s<br>"
            f"Distance: {craft_df['Distance_m'].iloc[i]:.1f} m<br>"
            f"Rel Speed: {craft_df['Rel_Speed_m_s'].iloc[i]:.2f} m/s<br>"
            f"Thrust: {'ACTIVE' if is_thrusting else 'FREE FLIGHT'}"
        )
        
        frame_data = [
            # Docking station trail (blue)
            go.Scatter3d(
                x=station_x,
                y=station_y,
                z=station_z,
                mode='lines',
                line=dict(color='#2E86DE', width=3),
                name='Station Path',
                showlegend=(i==0),
                hoverinfo='skip'
            ),
            # Docking station marker
            go.Scatter3d(
                x=[station_df['X_m'].iloc[i]],
                y=[station_df['Y_m'].iloc[i]],
                z=[station_df['Z_m'].iloc[i]],
                mode='markers',
                marker=dict(size=15, color='#2E86DE', symbol='square',
                           line=dict(width=2, color='white')),
                name='Docking Station',
                showlegend=(i==0),
                hoverinfo='skip'
            ),
            # Active spacecraft trail with gradient
            go.Scatter3d(
                x=craft_x,
                y=craft_y,
                z=craft_z,
                mode='lines',
                line=dict(color=('#EE5A24' if is_thrusting else '#10AC84'), width=4),
                name='Spacecraft Path',
                showlegend=(i==0),
                hoverinfo='skip'
            ),
            # Active spacecraft marker
            go.Scatter3d(
                x=[craft_df['X_m'].iloc[i]],
                y=[craft_df['Y_m'].iloc[i]],
                z=[craft_df['Z_m'].iloc[i]],
                mode='markers',
                marker=dict(size=12, color=('#EE5A24' if is_thrusting else '#10AC84'),
                           line=dict(width=2, color='white')),
                name='Active Spacecraft',
                showlegend=(i==0),
                hoverinfo='skip'
            ),
            # Ghost path (predictive trajectory)
            go.Scatter3d(
                x=pred_pos[:, 0],
                y=pred_pos[:, 1],
                z=pred_pos[:, 2],
                mode='lines',
                line=dict(color='#00D2D3', width=2, dash='dash'),
                name='Ghost Path',
                showlegend=(i==0),
                opacity=0.6,
                hoverinfo='skip'
            )
        ]
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                annotations=[dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=telemetry,
                    showarrow=False,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='cyan',
                    borderwidth=2,
                    font=dict(size=12, family='Courier New', color='white'),
                    align='left',
                    xanchor='left',
                    yanchor='top'
                )]
            )
        ))
    
    # Initial figure with first frame data
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text='Spacecraft Docking - Predictive Ghost Path Animation',
            font=dict(size=24, color='white', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
            yaxis=dict(title='Y (m)', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
            zaxis=dict(title='Z (m)', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='#000000'
        ),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.85,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='cyan',
            borderwidth=1,
            font=dict(size=11, color='white')
        ),
        annotations=frames[0].layout.annotations,
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        margin=dict(l=0, r=0, t=60, b=0),
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                x=0.15, y=0.02,
                xanchor='left', yanchor='bottom',
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, dict(frame=dict(duration=30, redraw=True),
                                         fromcurrent=True, mode='immediate')]),
                    dict(label='Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')])
                ],
                bgcolor='cyan',
                font=dict(color='black', size=12),
                bordercolor='white',
                borderwidth=2
            ),
            dict(
                type='buttons',
                showactive=True,
                x=0.30, y=0.02,
                xanchor='left', yanchor='bottom',
                buttons=[
                    dict(label='0.5x', method='animate',
                         args=[None, dict(frame=dict(duration=60, redraw=True), mode='immediate')]),
                    dict(label='1x', method='animate',
                         args=[None, dict(frame=dict(duration=30, redraw=True), mode='immediate')]),
                    dict(label='2x', method='animate',
                         args=[None, dict(frame=dict(duration=15, redraw=True), mode='immediate')])
                ],
                bgcolor='white',
                font=dict(color='black', size=11),
                bordercolor='cyan',
                borderwidth=2
            )
        ],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                       label=f"{int(f.name)*TIME_STEP*frame_step:.0f}s",
                       method='animate')
                   for f in frames[::5]],
            x=0.15, y=0.0,
            len=0.83,
            xanchor='left', yanchor='top',
            bgcolor='rgba(0,255,255,0.3)',
            bordercolor='cyan',
            borderwidth=2,
            font=dict(color='white')
        )]
    )
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("SPACECRAFT DOCKING ANIMATION")
    print("="*70)
    print("Running physics simulation...")
    
    # Run simulation
    station_df, craft_df = simulate_docking()
    
    print("Creating animated visualization...")
    fig = create_docking_visualization(station_df, craft_df)
    
    # Display results
    print("\nSimulation complete!")
    print(f"Initial separation: {INITIAL_SEPARATION:.0f} m")
    print(f"Final distance: {craft_df['Distance_m'].iloc[-1]:.2f} m")
    print(f"Final relative speed: {craft_df['Rel_Speed_m_s'].iloc[-1]:.3f} m/s")
    print(f"Docking: {'SUCCESS' if craft_df['Distance_m'].iloc[-1] < 10 else 'IN PROGRESS'}")
    print("\nOpening animation in browser...")
    print("Drag to rotate | Scroll to zoom | Use controls to play animation")
    print("="*70 + "\n")
    
    # Show animation
    fig.show()

if __name__ == "__main__":
    main()

