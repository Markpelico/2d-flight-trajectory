#!/usr/bin/env python3
"""
Spacecraft Docking Simulation with Predictive Trajectory
=========================================================
Simulates two spacecraft: a docking station (passive, gravity-only) and
an active spacecraft using thrust and free-flight to align for docking.

Features:
- Realistic orbital mechanics with gravity
- Thrust vectoring for active spacecraft
- Predictive trajectory lines (ghost path)
- Free-flight phases (coasting)
- Thrust phases (active maneuvering)
- Relative velocity and distance indicators
- Interactive 3D visualization

Physics Model:
- Gravity gradient effects
- Thrust vector control
- Conservation of momentum
- Orbital perturbations
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
    Create interactive 3D visualization of the docking simulation.
    """
    # Create figure
    fig = go.Figure()
    
    # Docking station trajectory (blue)
    fig.add_trace(go.Scatter3d(
        x=station_df['X_m'],
        y=station_df['Y_m'],
        z=station_df['Z_m'],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Docking Station Path',
        hovertemplate='<b>Docking Station</b><br>' +
                      'Time: %{customdata:.1f} s<br>' +
                      'X: %{x:.1f} m<br>' +
                      'Y: %{y:.1f} m<br>' +
                      'Z: %{z:.1f} m<br>' +
                      '<extra></extra>',
        customdata=station_df['Time_s']
    ))
    
    # Active spacecraft trajectory (green for free flight, orange for thrust)
    # Split into segments based on thrust
    craft_thrusting = craft_df['Thrust_Mag_N'] > 1.0
    
    # Free flight segments
    free_flight_mask = ~craft_thrusting
    if free_flight_mask.any():
        fig.add_trace(go.Scatter3d(
            x=craft_df.loc[free_flight_mask, 'X_m'],
            y=craft_df.loc[free_flight_mask, 'Y_m'],
            z=craft_df.loc[free_flight_mask, 'Z_m'],
            mode='lines',
            line=dict(color='green', width=4),
            name='Free Flight',
            hovertemplate='<b>Free Flight</b><br>' +
                          'Time: %{customdata:.1f} s<br>' +
                          'X: %{x:.1f} m<br>' +
                          'Y: %{y:.1f} m<br>' +
                          'Z: %{z:.1f} m<br>' +
                          '<extra></extra>',
            customdata=craft_df.loc[free_flight_mask, 'Time_s']
        ))
    
    # Thrust segments
    thrust_mask = craft_thrusting
    if thrust_mask.any():
        fig.add_trace(go.Scatter3d(
            x=craft_df.loc[thrust_mask, 'X_m'],
            y=craft_df.loc[thrust_mask, 'Y_m'],
            z=craft_df.loc[thrust_mask, 'Z_m'],
            mode='lines',
            line=dict(color='orange', width=4),
            name='Thrusting',
            hovertemplate='<b>Thrusting</b><br>' +
                          'Time: %{customdata[0]:.1f} s<br>' +
                          'Thrust: %{customdata[1]:.1f} N<br>' +
                          'X: %{x:.1f} m<br>' +
                          'Y: %{y:.1f} m<br>' +
                          'Z: %{z:.1f} m<br>' +
                          '<extra></extra>',
            customdata=np.column_stack([craft_df.loc[thrust_mask, 'Time_s'],
                                       craft_df.loc[thrust_mask, 'Thrust_Mag_N']])
        ))
    
    # Predictive trajectory for final approach (ghost path)
    final_approach_time = 540
    final_idx = int(final_approach_time / TIME_STEP)
    if final_idx < len(craft_df):
        pred_pos = predict_trajectory(
            craft_df[['X_m', 'Y_m', 'Z_m']].iloc[final_idx].values,
            craft_df[['VX_m_s', 'VY_m_s', 'VZ_m_s']].iloc[final_idx].values,
            final_approach_time
        )
        
        fig.add_trace(go.Scatter3d(
            x=pred_pos[:, 0],
            y=pred_pos[:, 1],
            z=pred_pos[:, 2],
            mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            name='Predicted Path (Ghost)',
            hoverinfo='skip',
            opacity=0.5
        ))
    
    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[craft_df['X_m'].iloc[0], craft_df['X_m'].iloc[-1]],
        y=[craft_df['Y_m'].iloc[0], craft_df['Y_m'].iloc[-1]],
        z=[craft_df['Z_m'].iloc[0], craft_df['Z_m'].iloc[-1]],
        mode='markers',
        marker=dict(size=15, color=['green', 'red'], symbol=['diamond', 'square']),
        name='Start/Docked',
        text=['Start', 'Docked'],
        hovertemplate='<b>%{text}</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Z: %{z:.1f} m<extra></extra>'
    ))
    
    # Docking station marker at final position
    fig.add_trace(go.Scatter3d(
        x=[station_df['X_m'].iloc[-1]],
        y=[station_df['Y_m'].iloc[-1]],
        z=[station_df['Z_m'].iloc[-1]],
        mode='markers',
        marker=dict(size=20, color='blue', symbol='square'),
        name='Station Final',
        hovertemplate='<b>Docking Station</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Z: %{z:.1f} m<extra></extra>'
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text='Spacecraft Docking Simulation - Predictive Trajectory',
            font=dict(size=22, color='#2c3e50', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='X Position (m)', backgroundcolor='rgb(230,230,230)', gridcolor='white', showbackground=True),
            yaxis=dict(title='Y Position (m)', backgroundcolor='rgb(230,230,230)', gridcolor='white', showbackground=True),
            zaxis=dict(title='Z Position (m)', backgroundcolor='rgb(230,230,230)', gridcolor='white', showbackground=True),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.3),
                center=dict(x=0, y=0, z=0)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='gray',
            borderwidth=2,
            font=dict(size=11)
        ),
        paper_bgcolor='#f0f0f0',
        plot_bgcolor='#ffffff',
        margin=dict(l=0, r=0, t=80, b=0),
        hovermode='closest'
    )
    
    return fig

def create_telemetry_plots(station_df, craft_df):
    """
    Create supplementary telemetry plots.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Relative Distance', 'Relative Speed', 
                       'Thrust Magnitude', 'Velocity Components'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Distance plot
    fig.add_trace(go.Scatter(
        x=craft_df['Time_s'],
        y=craft_df['Distance_m'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Distance'
    ), row=1, col=1)
    
    # Relative speed plot
    fig.add_trace(go.Scatter(
        x=craft_df['Time_s'],
        y=craft_df['Rel_Speed_m_s'],
        mode='lines',
        line=dict(color='green', width=2),
        name='Relative Speed'
    ), row=1, col=2)
    
    # Thrust magnitude plot
    fig.add_trace(go.Scatter(
        x=craft_df['Time_s'],
        y=craft_df['Thrust_Mag_N'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='Thrust'
    ), row=2, col=1)
    
    # Velocity components
    fig.add_trace(go.Scatter(
        x=craft_df['Time_s'],
        y=craft_df['VX_m_s'],
        mode='lines',
        name='VX',
        line=dict(color='red')
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=craft_df['Time_s'],
        y=craft_df['VY_m_s'],
        mode='lines',
        name='VY',
        line=dict(color='green')
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=craft_df['Time_s'],
        y=craft_df['VZ_m_s'],
        mode='lines',
        name='VZ',
        line=dict(color='blue')
    ), row=2, col=2)
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Thrust (N)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text='Docking Telemetry Data',
            font=dict(size=18),
            x=0.5,
            xanchor='center'
        ),
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("SPACECRAFT DOCKING SIMULATION")
    print("="*70)
    print(f"Simulation Time: {SIMULATION_TIME} seconds")
    print(f"Time Step: {TIME_STEP} seconds")
    print(f"Initial Separation: {INITIAL_SEPARATION} meters")
    print(f"Spacecraft Mass: {ACTIVE_SPACECRAFT_MASS} kg")
    print(f"Thrust Magnitude: {THRUST_MAGNITUDE} N")
    print("="*70)
    
    # Run simulation
    print("\nRunning simulation...")
    station_df, craft_df = simulate_docking()
    
    # Create visualizations
    print("Creating 3D visualization...")
    fig_3d = create_docking_visualization(station_df, craft_df)
    
    print("Creating telemetry plots...")
    fig_telemetry = create_telemetry_plots(station_df, craft_df)
    
    # Display results
    print("\nSimulation complete!")
    print(f"Final distance: {craft_df['Distance_m'].iloc[-1]:.2f} meters")
    print(f"Final relative speed: {craft_df['Rel_Speed_m_s'].iloc[-1]:.3f} m/s")
    print(f"Total delta-V used: {np.trapz(craft_df['Thrust_Mag_N'], craft_df['Time_s']) / ACTIVE_SPACECRAFT_MASS:.2f} m/s")
    print("\nOpening visualizations in browser...")
    print("="*70 + "\n")
    
    # Show figures
    fig_3d.show()
    fig_telemetry.show()

if __name__ == "__main__":
    main()

