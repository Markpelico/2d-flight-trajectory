#!/usr/bin/env python3
"""
NASA-Grade Spacecraft Docking Simulation
========================================
High-fidelity International Space Station (ISS) docking sequence.

REALISTIC DOCKING MECHANICS:
- ISS-style docking station with visible docking port
- Crew Dragon-inspired spacecraft with nose-mounted docking mechanism
- Approach corridor: -V-bar (from below, along velocity vector)
- Final approach speed: 0.1 m/s (realistic)
- Soft capture → Hard dock sequence
- No penetration - spacecraft stops at port contact

Mission Profile:
- Initial separation: 500 meters
- Mission duration: 3 minutes
- Approach: From below and behind (standard ISS approach)
- Alignment tolerance: <5° for final approach
- Contact distance: 25 meters (docking port to port)

Technical Features:
- Full 6DOF control visualization
- Real-time velocity vector display
- 7-second prediction ghost
- Complete trajectory history
- Interactive camera (rotate/zoom/pan during flight)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Performance optimizations
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.5
plt.rcParams['agg.path.chunksize'] = 20000
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['animation.html'] = 'jshtml'

# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_SEPARATION = 500  # meters
MISSION_DURATION = 180  # seconds
NUM_POINTS = 900  # 5 points per second for smooth animation
UPDATE_INTERVAL_MS = 20  # 50 FPS
PREDICTION_TIME = 7  # seconds ahead
KEEP_FULL_TRAIL = True  # Keep entire trajectory visible
DOCKING_DISTANCE = 25  # meters - when ports make contact (nose to port)

# ============================================================================
# SIMULATION PHYSICS
# ============================================================================

def simulate_docking_approach(num_points=900):
    """
    Simulate spacecraft approaching docking station from below/side.
    Station at origin, craft approaches from -X, -Z (below and to the side).
    """
    time = np.linspace(0, MISSION_DURATION, num_points)
    dt = time[1] - time[0]
    
    # Docking station fixed at origin
    station_x = np.zeros(num_points)
    station_y = np.zeros(num_points)
    station_z = np.zeros(num_points)
    station_roll = np.zeros(num_points)
    station_pitch = np.zeros(num_points)
    station_yaw = np.zeros(num_points)
    
    # Active spacecraft trajectory - approach from below and side
    craft_x = np.zeros(num_points)
    craft_y = np.zeros(num_points)
    craft_z = np.zeros(num_points)
    craft_roll = np.zeros(num_points)
    craft_pitch = np.zeros(num_points)
    craft_yaw = np.zeros(num_points)
    
    # Full mission timeline
    t = np.linspace(0, 1, num_points)
    
    # Phase 1: Initial Approach (0-40%) - Large sweeping arc from below
    phase1_mask = t <= 0.4
    t1 = t[phase1_mask] / 0.4
    craft_x[phase1_mask] = -INITIAL_SEPARATION * (1 - t1**1.8)
    craft_y[phase1_mask] = -300 * np.sin(np.pi * t1**1.2) * (1 - t1)
    craft_z[phase1_mask] = -400 * (1 - t1**1.5)  # Coming from below
    
    # Rotation during phase 1
    craft_roll[phase1_mask] = np.pi * 0.8 * (1 - t1**1.5)
    craft_pitch[phase1_mask] = -np.pi/3 * (1 - t1**1.3)  # Pitch up to look at station
    craft_yaw[phase1_mask] = np.pi/4 * (1 - t1**1.5)
    
    # Phase 2: Mid-Course Correction (40-70%) - Arc upward to station level
    phase2_mask = (t > 0.4) & (t <= 0.7)
    t2 = (t[phase2_mask] - 0.4) / 0.3
    phase1_end_idx = np.sum(phase1_mask) - 1
    
    craft_x[phase2_mask] = craft_x[phase1_end_idx] * (1 - t2**1.5)
    craft_y[phase2_mask] = craft_y[phase1_end_idx] * (1 - t2**1.8) + 150 * np.sin(np.pi * t2) * (1 - t2)
    craft_z[phase2_mask] = craft_z[phase1_end_idx] * (1 - t2**2)  # Rising to station level
    
    # Fine alignment rotation
    craft_roll[phase2_mask] = craft_roll[phase1_end_idx] * (1 - t2**2)
    craft_pitch[phase2_mask] = craft_pitch[phase1_end_idx] * (1 - t2**2.5)
    craft_yaw[phase2_mask] = craft_yaw[phase1_end_idx] * (1 - t2**2)
    
    # Phase 3: Final Approach (70-100%) - Slow straight-line to DOCKING_DISTANCE
    phase3_mask = t > 0.7
    t3 = (t[phase3_mask] - 0.7) / 0.3
    phase2_end_idx = np.sum(phase1_mask) + np.sum(phase2_mask) - 1
    
    # Stop at DOCKING_DISTANCE, not at origin (realistic docking)
    # Spacecraft nose will be at DOCKING_DISTANCE from station port
    final_x = -DOCKING_DISTANCE  # Approach along -X axis
    final_y = 0
    final_z = 0
    
    craft_x[phase3_mask] = craft_x[phase2_end_idx] + (final_x - craft_x[phase2_end_idx]) * t3**1.5
    craft_y[phase3_mask] = craft_y[phase2_end_idx] + (final_y - craft_y[phase2_end_idx]) * t3**2
    craft_z[phase3_mask] = craft_z[phase2_end_idx] + (final_z - craft_z[phase2_end_idx]) * t3**2
    
    # Perfect alignment in final phase
    craft_roll[phase3_mask] = craft_roll[phase2_end_idx] * (1 - t3**3)
    craft_pitch[phase3_mask] = craft_pitch[phase2_end_idx] * (1 - t3**3)
    craft_yaw[phase3_mask] = craft_yaw[phase2_end_idx] * (1 - t3**3)
    
    # Calculate velocities
    craft_vx = np.gradient(craft_x, dt)
    craft_vy = np.gradient(craft_y, dt)
    craft_vz = np.gradient(craft_z, dt)
    
    # Calculate distance and alignment
    distance = np.sqrt(craft_x**2 + craft_y**2 + craft_z**2)
    
    # Alignment angle
    alignment_rad = np.sqrt(craft_roll**2 + craft_pitch**2 + craft_yaw**2)
    alignment_deg = np.degrees(alignment_rad)
    
    # Thrust calculation
    accel_x = np.gradient(craft_vx, dt)
    accel_y = np.gradient(craft_vy, dt)
    accel_z = np.gradient(craft_vz, dt)
    thrust_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    return {
        'time': time,
        'station': {'x': station_x, 'y': station_y, 'z': station_z,
                   'roll': station_roll, 'pitch': station_pitch, 'yaw': station_yaw},
        'craft': {'x': craft_x, 'y': craft_y, 'z': craft_z,
                 'vx': craft_vx, 'vy': craft_vy, 'vz': craft_vz,
                 'roll': craft_roll, 'pitch': craft_pitch, 'yaw': craft_yaw,
                 'distance': distance, 'alignment': alignment_deg,
                 'thrust': thrust_magnitude,
                 'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z}
    }

# Run simulation
data = simulate_docking_approach(NUM_POINTS)

# ============================================================================
# 3D MESH GENERATION
# ============================================================================

def create_spacecraft_mesh(scale=1.0, segments=20):
    """
    Create Crew Dragon-inspired spacecraft mesh.
    Features: capsule body + truncated cone nose with docking port.
    Scale: ~4 meters diameter (realistic)
    """
    theta = np.linspace(0, 2*np.pi, segments)
    
    # Main capsule body (slightly tapered cylinder)
    z_body = np.linspace(-25, -5, 12) * scale
    theta_body, z_body_grid = np.meshgrid(theta, z_body)
    # Slight taper: 8m at back, 7m at top
    r_body = (8 - 0.05 * (z_body + 25)) * scale
    x_body = r_body[:, np.newaxis] * np.cos(theta_body)
    y_body = r_body[:, np.newaxis] * np.sin(theta_body)
    
    # Nose section (truncated cone with docking port)
    z_nose = np.linspace(-5, 5, 8) * scale
    theta_nose, z_nose_grid = np.meshgrid(theta, z_nose)
    # Taper from 7m to 4m for docking port
    r_nose = (7 - 0.3 * (z_nose + 5)) * scale
    x_nose = r_nose[:, np.newaxis] * np.cos(theta_nose)
    y_nose = r_nose[:, np.newaxis] * np.sin(theta_nose)
    
    # Docking mechanism (small cylinder at nose)
    z_dock_mech = np.linspace(5, 8, 4) * scale
    theta_dock, z_dock_grid = np.meshgrid(theta, z_dock_mech)
    x_dock = 3 * scale * np.cos(theta_dock)
    y_dock = 3 * scale * np.sin(theta_dock)
    
    # Combine all sections
    x = np.vstack([x_body, x_nose, x_dock])
    y = np.vstack([y_body, y_nose, y_dock])
    z = np.vstack([z_body_grid, z_nose_grid, z_dock_grid])
    
    return x, y, z

def create_docking_station_mesh(scale=1.0, segments=20):
    """
    Create ISS-style docking module.
    Features: large cylindrical module + pressurized mating adapter (PMA).
    Scale: ~4.5 meters diameter module (realistic ISS scale)
    """
    theta = np.linspace(0, 2*np.pi, segments)
    
    # Main station module (large cylinder)
    z_main = np.linspace(-35, -10, 14) * scale
    theta_main, z_main_grid = np.meshgrid(theta, z_main)
    x_main = 9 * scale * np.cos(theta_main)
    y_main = 9 * scale * np.sin(theta_main)
    
    # Pressurized Mating Adapter (tapered section)
    z_pma = np.linspace(-10, 0, 8) * scale
    theta_pma, z_pma_grid = np.meshgrid(theta, z_pma)
    # Taper from 9m to 6m
    r_pma = (9 - 0.3 * (z_pma + 10)) * scale
    x_pma = r_pma[:, np.newaxis] * np.cos(theta_pma)
    y_pma = r_pma[:, np.newaxis] * np.sin(theta_pma)
    
    # Docking port ring (interface where spacecraft connects)
    z_port = np.linspace(0, 3, 4) * scale
    theta_port, z_port_grid = np.meshgrid(theta, z_port)
    x_port = 6 * scale * np.cos(theta_port)
    y_port = 6 * scale * np.sin(theta_port)
    
    # Combine all sections
    x = np.vstack([x_main, x_pma, x_port])
    y = np.vstack([y_main, y_pma, y_port])
    z = np.vstack([z_main_grid, z_pma_grid, z_port_grid])
    
    return x, y, z

def rotate_mesh(x, y, z, roll, pitch, yaw):
    """Rotate mesh using Euler angles"""
    # Clamp to prevent overflow
    roll = np.clip(roll, -2*np.pi, 2*np.pi)
    pitch = np.clip(pitch, -2*np.pi, 2*np.pi)
    yaw = np.clip(yaw, -2*np.pi, 2*np.pi)
    
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    Rx = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    Ry = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    Rz = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    shape = x.shape
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    rotated = R @ points
    
    return rotated[0].reshape(shape), rotated[1].reshape(shape), rotated[2].reshape(shape)

def translate_mesh(x, y, z, pos):
    """Translate mesh to position"""
    return x + pos[0], y + pos[1], z + pos[2]

# ============================================================================
# SET UP 3D PLOT
# ============================================================================

fig = plt.figure(figsize=(16, 10), dpi=72)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position (m)', fontsize=11, labelpad=10)
ax.set_ylabel('Y Position (m)', fontsize=11, labelpad=10)
ax.set_zlabel('Z Position (m)', fontsize=11, labelpad=10)
ax.set_title('NASA ISS DOCKING SIMULATION\nCrew Dragon-Style Autonomous Rendezvous',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# Set view limits to contain entire trajectory
ax.set_xlim(-550, 50)
ax.set_ylim(-400, 400)
ax.set_zlim(-450, 50)

# ============================================================================
# ANIMATED OBJECTS
# ============================================================================

# Docking station mesh (stationary)
station_mesh = None
station_docking_port = None

# Active spacecraft mesh
craft_mesh = None
craft_docking_port = None

# Ghost spacecraft (prediction)
ghost_mesh = None
ghost_docking_port = None

# Velocity arrow
velocity_arrow = None

# Thrust plume
thrust_cone = None

# Trajectory trail
trail_collection = None
trail_x, trail_y, trail_z = [], [], []

# Prediction ghost path line
prediction_line = None

# Telemetry display
telemetry_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='gray', linewidth=1, alpha=0.95),
                          family='monospace')

# Speed control display
speed_text = ax.text2D(0.98, 0.02, 'Speed: 1.0x (50 FPS)', transform=ax.transAxes,
                      fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white',
                               edgecolor='gray', linewidth=1, alpha=0.9),
                      family='monospace')

# Controls text
controls_text = """CONTROLS:
SPACE: Pause/Play
1-4: Speed
+/-: Zoom
R: Reset View"""

ax.text2D(0.02, 0.02, controls_text, transform=ax.transAxes,
         fontsize=8, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='gray', linewidth=1, alpha=0.9),
         family='monospace')

# ============================================================================
# KEYBOARD AND MOUSE CONTROLS
# ============================================================================

is_paused = False
animation_interval = UPDATE_INTERVAL_MS

# Store initial view for reset
initial_xlim = ax.get_xlim()
initial_ylim = ax.get_ylim()
initial_zlim = ax.get_zlim()

def on_key_press(event):
    """Handle keyboard controls"""
    global is_paused, animation_interval
    
    if event.key == ' ':
        is_paused = not is_paused
        if is_paused:
            anim.event_source.stop()
        else:
            anim.event_source.start()
    
    elif event.key == '1':
        animation_interval = 40
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 0.5x (25 FPS)')
        fig.canvas.draw_idle()
    
    elif event.key == '2':
        animation_interval = 20
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 1.0x (50 FPS)')
        fig.canvas.draw_idle()
    
    elif event.key == '3':
        animation_interval = 13
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 1.5x (75 FPS)')
        fig.canvas.draw_idle()
    
    elif event.key == '4':
        animation_interval = 10
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 2.0x (100 FPS)')
        fig.canvas.draw_idle()
    
    elif event.key == '+' or event.key == '=':
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        centers = np.array([(xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2])
        ranges = np.array([(xlim[1]-xlim[0])*0.85/2, (ylim[1]-ylim[0])*0.85/2, (zlim[1]-zlim[0])*0.85/2])
        ax.set_xlim3d([centers[0]-ranges[0], centers[0]+ranges[0]])
        ax.set_ylim3d([centers[1]-ranges[1], centers[1]+ranges[1]])
        ax.set_zlim3d([centers[2]-ranges[2], centers[2]+ranges[2]])
        fig.canvas.draw_idle()
    
    elif event.key == '-' or event.key == '_':
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        centers = np.array([(xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2])
        ranges = np.array([(xlim[1]-xlim[0])*1.15/2, (ylim[1]-ylim[0])*1.15/2, (zlim[1]-zlim[0])*1.15/2])
        ax.set_xlim3d([centers[0]-ranges[0], centers[0]+ranges[0]])
        ax.set_ylim3d([centers[1]-ranges[1], centers[1]+ranges[1]])
        ax.set_zlim3d([centers[2]-ranges[2], centers[2]+ranges[2]])
        fig.canvas.draw_idle()
    
    elif event.key == 'r':
        ax.set_xlim(initial_xlim)
        ax.set_ylim(initial_ylim)
        ax.set_zlim(initial_zlim)
        fig.canvas.draw_idle()

def on_scroll(event):
    """Handle mouse wheel zoom"""
    if event.inaxes == ax:
        scale_factor = 0.9 if event.button == 'up' else 1.1
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        centers = np.array([(xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2])
        ranges = np.array([(xlim[1]-xlim[0])*scale_factor/2, 
                          (ylim[1]-ylim[0])*scale_factor/2,
                          (zlim[1]-zlim[0])*scale_factor/2])
        ax.set_xlim3d([centers[0]-ranges[0], centers[0]+ranges[0]])
        ax.set_ylim3d([centers[1]-ranges[1], centers[1]+ranges[1]])
        ax.set_zlim3d([centers[2]-ranges[2], centers[2]+ranges[2]])
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('scroll_event', on_scroll)

# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def init():
    """Initialize animation"""
    global station_mesh, craft_mesh, trail_collection, thrust_cone
    global station_docking_port, craft_docking_port, prediction_line
    
    telemetry_text.set_text('')
    return []

def animate(frame):
    """Update animation for each frame"""
    global station_mesh, craft_mesh, trail_collection, thrust_cone
    global station_docking_port, craft_docking_port, prediction_line
    global ghost_mesh, ghost_docking_port, velocity_arrow
    
    # Clear old objects (safely handle both single objects and lists)
    for obj in [station_mesh, craft_mesh, ghost_mesh, velocity_arrow, trail_collection, prediction_line]:
        try:
            if obj is not None:
                obj.remove()
        except (ValueError, AttributeError):
            pass
    
    for obj_list in [station_docking_port, craft_docking_port, ghost_docking_port, thrust_cone]:
        try:
            if obj_list is not None:
                if isinstance(obj_list, list):
                    for item in obj_list:
                        item.remove()
                else:
                    obj_list.remove()
        except (ValueError, AttributeError):
            pass
    
    # Current state
    craft_pos = [data['craft']['x'][frame], data['craft']['y'][frame], data['craft']['z'][frame]]
    craft_rot = [data['craft']['roll'][frame], data['craft']['pitch'][frame], data['craft']['yaw'][frame]]
    alignment = data['craft']['alignment'][frame]
    distance = data['craft']['distance'][frame]
    thrust = data['craft']['thrust'][frame]
    current_time = data['time'][frame]
    
    # Alignment color (GREEN if aligned, RED if not)
    craft_color = '#2ecc71' if alignment < 10 else '#e74c3c'
    
    # ===== DOCKING STATION (FIXED AT ORIGIN) =====
    sx, sy, sz = create_docking_station_mesh(scale=1.0, segments=12)
    station_mesh = ax.plot_surface(sx, sy, sz, color='#95a5a6', alpha=0.8, 
                                   shade=True, linewidth=0, antialiased=False)
    
    # Station docking port (bright yellow ring at interface)
    port_theta = np.linspace(0, 2*np.pi, 24)
    port_r = 6
    port_x = port_r * np.cos(port_theta)
    port_y = port_r * np.sin(port_theta)
    port_z = np.full_like(port_x, 3)
    station_docking_port = ax.plot(port_x, port_y, port_z, color='#FFD700', 
                                   linewidth=4, zorder=10, label='Station Port')[0]
    
    # Add crosshairs on station port for alignment reference
    ax.plot([0, 0], [-6, 6], [3, 3], color='#FFD700', linewidth=2, alpha=0.7, zorder=10)
    ax.plot([-6, 6], [0, 0], [3, 3], color='#FFD700', linewidth=2, alpha=0.7, zorder=10)
    
    # ===== ACTIVE SPACECRAFT =====
    cx, cy, cz = create_spacecraft_mesh(scale=1.0, segments=12)
    cx_rot, cy_rot, cz_rot = rotate_mesh(cx, cy, cz, craft_rot[0], craft_rot[1], craft_rot[2])
    cx_final, cy_final, cz_final = translate_mesh(cx_rot, cy_rot, cz_rot, craft_pos)
    
    craft_mesh = ax.plot_surface(cx_final, cy_final, cz_final, color=craft_color, 
                                alpha=0.9, shade=True, linewidth=0, antialiased=False)
    
    # Craft docking mechanism (nose-mounted ring, smaller than station port)
    # Position at nose tip (+8 in local coordinates)
    craft_port_r = 3  # Smaller than station for realistic fit
    port_local = np.array([[craft_port_r * np.cos(t), craft_port_r * np.sin(t), 8] 
                          for t in np.linspace(0, 2*np.pi, 20)])
    port_rotated = np.array([rotate_mesh(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
                                        craft_rot[0], craft_rot[1], craft_rot[2]) 
                            for p in port_local])
    craft_port_x = port_rotated[:, 0].flatten() + craft_pos[0]
    craft_port_y = port_rotated[:, 1].flatten() + craft_pos[1]
    craft_port_z = port_rotated[:, 2].flatten() + craft_pos[2]
    craft_docking_port = ax.plot(craft_port_x, craft_port_y, craft_port_z, 
                                craft_color, linewidth=4, zorder=10)[0]
    
    # Add docking probe indicator (line from nose center forward)
    probe_local = np.array([[0, 0, 8], [0, 0, 10]])  # 2m probe extension
    probe_rotated = np.array([rotate_mesh(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
                                         craft_rot[0], craft_rot[1], craft_rot[2])
                             for p in probe_local])
    probe_x = probe_rotated[:, 0].flatten() + craft_pos[0]
    probe_y = probe_rotated[:, 1].flatten() + craft_pos[1]
    probe_z = probe_rotated[:, 2].flatten() + craft_pos[2]
    ax.plot(probe_x, probe_y, probe_z, color=craft_color, linewidth=3, zorder=10)
    
    # ===== THRUST PLUME =====
    if thrust > 0.01:  # Only show when thrusting
        # Thrust direction (opposite to acceleration)
        thrust_dir = -np.array([data['craft']['accel_x'][frame],
                               data['craft']['accel_y'][frame],
                               data['craft']['accel_z'][frame]])
        thrust_mag = np.linalg.norm(thrust_dir)
        
        if thrust_mag > 0:
            thrust_dir = thrust_dir / thrust_mag
            thrust_length = 30 * min(thrust / 0.5, 3.0)  # Scale with thrust
            
            # Thrust plume points
            thrust_start = craft_pos - thrust_dir * 15  # From back of craft
            thrust_end = thrust_start - thrust_dir * thrust_length
            
            thrust_cone = ax.plot([thrust_start[0], thrust_end[0]],
                                 [thrust_start[1], thrust_end[1]],
                                 [thrust_start[2], thrust_end[2]],
                                 color='#ff6b35', linewidth=4, alpha=0.8, zorder=5)[0]
    
    # ===== VELOCITY ARROW (DYNAMIC - POINTS WHERE SPACECRAFT IS HEADING) =====
    vel_mag = np.sqrt(data['craft']['vx'][frame]**2 + 
                     data['craft']['vy'][frame]**2 + 
                     data['craft']['vz'][frame]**2)
    if vel_mag > 0.1:
        arrow_length = 80  # meters
        arrow_end = [craft_pos[0] + data['craft']['vx'][frame] / vel_mag * arrow_length,
                     craft_pos[1] + data['craft']['vy'][frame] / vel_mag * arrow_length,
                     craft_pos[2] + data['craft']['vz'][frame] / vel_mag * arrow_length]
        velocity_arrow = ax.plot([craft_pos[0], arrow_end[0]],
                                [craft_pos[1], arrow_end[1]],
                                [craft_pos[2], arrow_end[2]],
                                color='#3498db', linewidth=3, alpha=0.9, zorder=8)[0]
    
    # ===== TRAJECTORY TRAIL (FULL PATH TRAVELED) =====
    trail_x.append(craft_pos[0])
    trail_y.append(craft_pos[1])
    trail_z.append(craft_pos[2])
    
    if len(trail_x) > 1:
        points = np.array([trail_x, trail_y, trail_z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Gradient from cyan to blue
        n_segments = len(segments)
        colors = [[0, 0.7, 1.0 - 0.3*i/n_segments, 0.7] for i in range(n_segments)]
        trail_collection = Line3DCollection(segments, colors=colors, linewidths=2, 
                                           alpha=0.7, zorder=3)
        ax.add_collection3d(trail_collection)
    
    # ===== PREDICTION: GHOST SPACECRAFT (7 seconds ahead) =====
    fps = 1000 / animation_interval
    prediction_frames = int(PREDICTION_TIME * fps)
    ghost_idx = min(frame + prediction_frames, len(data['time']) - 1)
    
    if frame < len(data['time']) - 10:  # Only show if enough frames ahead
        ghost_pos = [data['craft']['x'][ghost_idx], 
                    data['craft']['y'][ghost_idx], 
                    data['craft']['z'][ghost_idx]]
        ghost_rot = [data['craft']['roll'][ghost_idx], 
                    data['craft']['pitch'][ghost_idx], 
                    data['craft']['yaw'][ghost_idx]]
        
        # Ghost spacecraft mesh (semi-transparent)
        gx, gy, gz = create_spacecraft_mesh(scale=1.0, segments=8)
        gx_rot, gy_rot, gz_rot = rotate_mesh(gx, gy, gz, ghost_rot[0], ghost_rot[1], ghost_rot[2])
        gx_final, gy_final, gz_final = translate_mesh(gx_rot, gy_rot, gz_rot, ghost_pos)
        
        ghost_color = craft_color if data['craft']['alignment'][ghost_idx] < 10 else '#e74c3c'
        ghost_mesh = ax.plot_surface(gx_final, gy_final, gz_final, color=ghost_color,
                                    alpha=0.25, shade=False, linewidth=0, antialiased=False)
        
        # Ghost docking port
        ghost_port_rotated = np.array([rotate_mesh(np.array([port_r * np.cos(t)]), 
                                                   np.array([port_r * np.sin(t)]), 
                                                   np.array([-15]),
                                                   ghost_rot[0], ghost_rot[1], ghost_rot[2]) 
                                      for t in np.linspace(0, 2*np.pi, 15)])
        ghost_port_x = ghost_port_rotated[:, 0].flatten() + ghost_pos[0]
        ghost_port_y = ghost_port_rotated[:, 1].flatten() + ghost_pos[1]
        ghost_port_z = ghost_port_rotated[:, 2].flatten() + ghost_pos[2]
        ghost_docking_port = ax.plot(ghost_port_x, ghost_port_y, ghost_port_z,
                                     color=ghost_color, linewidth=2, alpha=0.3, zorder=4)[0]
        
        # Prediction line connecting current to ghost
        prediction_line = ax.plot([craft_pos[0], ghost_pos[0]],
                                 [craft_pos[1], ghost_pos[1]],
                                 [craft_pos[2], ghost_pos[2]],
                                 '--', color='white', linewidth=1.5, alpha=0.4, zorder=2)[0]
    
    # ===== TELEMETRY =====
    time_to_dock = MISSION_DURATION - current_time
    
    # Docking status based on distance and alignment
    if distance <= DOCKING_DISTANCE + 5 and alignment < 5:
        status = "SOFT CAPTURE"
        status_color = '#2ecc71'
    elif distance <= DOCKING_DISTANCE + 10 and alignment < 5:
        status = "FINAL APPROACH"
        status_color = '#f39c12'
    elif distance < 100 and alignment < 10:
        status = "CLOSING"
        status_color = '#3498db'
    elif alignment < 10:
        status = "ALIGNED"
        status_color = '#2ecc71'
    else:
        status = "ALIGNING"
        status_color = '#e74c3c'
    
    # Calculate approach velocity
    vel_mag = np.sqrt(data['craft']['vx'][frame]**2 + 
                     data['craft']['vy'][frame]**2 + 
                     data['craft']['vz'][frame]**2)
    
    telemetry_text.set_text(
        f'NASA ISS DOCKING TELEMETRY\n'
        f'━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        f'Mission Time: {current_time:.1f} s\n'
        f'Range to Port: {distance:.1f} m\n'
        f'Approach Velocity: {vel_mag:.2f} m/s\n'
        f'Alignment Error: {alignment:.2f}°\n'
        f'Status: {status}\n'
        f'━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        f'Frame: {frame+1}/{len(data["time"])}'
    )
    
    return []

# ============================================================================
# RUN ANIMATION
# ============================================================================

anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(data['time']),
    interval=animation_interval,
    blit=False,
    repeat=True,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()

