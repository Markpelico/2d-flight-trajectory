#!/usr/bin/env python3
"""
Spacecraft Docking Simulation - Matplotlib Version
==================================================
Real-time 3D docking sequence with full interactive control.

Mission Profile:
- Initial separation: 500 meters
- Mission duration: 3 minutes (180 seconds)
- Docking station: Fixed at origin
- Active spacecraft: Autonomous approach with rotation alignment
- Prediction: 7-second future trajectory ghost

Features:
- Full camera control during animation (rotate, zoom, pan)
- Color-coded alignment (GREEN: aligned <10°, RED: misaligned >10°)
- Thrust plume visualization
- Real-time telemetry
- Smooth 50 FPS performance
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
MAX_TRAIL_LENGTH = 150  # trajectory trail points

# ============================================================================
# SIMULATION PHYSICS
# ============================================================================

def simulate_docking_approach(num_points=900):
    """
    Simulate spacecraft approaching docking station with rotation alignment.
    Uses waypoints for reliable visualization with physics-based velocity.
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
    
    # Active spacecraft trajectory - strategic waypoints
    craft_x = np.zeros(num_points)
    craft_y = np.zeros(num_points)
    craft_z = np.zeros(num_points)
    craft_roll = np.zeros(num_points)
    craft_pitch = np.zeros(num_points)
    craft_yaw = np.zeros(num_points)
    
    # Phase 1: Initial Approach (0-60s) - Large arc from starting position
    phase1_end = int(num_points * 60 / MISSION_DURATION)
    t1 = np.linspace(0, 1, phase1_end)
    craft_x[:phase1_end] = -INITIAL_SEPARATION * (1 - t1**2)
    craft_y[:phase1_end] = 200 * np.sin(np.pi * t1) * (1 - t1)
    craft_z[:phase1_end] = 150 * np.cos(2 * np.pi * t1) * (1 - t1)
    
    # Rotation during phase 1 - gradually align
    craft_roll[:phase1_end] = np.pi * (1 - t1)
    craft_pitch[:phase1_end] = np.pi/4 * (1 - t1**1.5)
    craft_yaw[:phase1_end] = np.pi/2 * (1 - t1)
    
    # Phase 2: Mid-Course (60-120s) - Refined approach with minor corrections
    phase2_start = phase1_end
    phase2_end = int(num_points * 120 / MISSION_DURATION)
    phase2_len = phase2_end - phase2_start
    t2 = np.linspace(0, 1, phase2_len)
    
    craft_x[phase2_start:phase2_end] = craft_x[phase2_start-1] * (1 - t2**1.5)
    craft_y[phase2_start:phase2_end] = craft_y[phase2_start-1] * (1 - t2**2)
    craft_z[phase2_start:phase2_end] = craft_z[phase2_start-1] * (1 - t2**2)
    
    # Fine rotation alignment
    craft_roll[phase2_start:phase2_end] = craft_roll[phase2_start-1] * (1 - t2**2)
    craft_pitch[phase2_start:phase2_end] = craft_pitch[phase2_start-1] * (1 - t2**2)
    craft_yaw[phase2_start:phase2_end] = craft_yaw[phase2_start-1] * (1 - t2**2)
    
    # Phase 3: Final Approach (120-180s) - Slow straight-line docking
    phase3_start = phase2_end
    t3 = np.linspace(0, 1, num_points - phase3_start)
    
    craft_x[phase3_start:] = craft_x[phase3_start-1] * (1 - t3**1.2)
    craft_y[phase3_start:] = craft_y[phase3_start-1] * (1 - t3**1.5)
    craft_z[phase3_start:] = craft_z[phase3_start-1] * (1 - t3**1.5)
    
    # Maintain alignment in final phase
    craft_roll[phase3_start:] = 0
    craft_pitch[phase3_start:] = 0
    craft_yaw[phase3_start:] = 0
    
    # Calculate velocities
    craft_vx = np.gradient(craft_x, dt)
    craft_vy = np.gradient(craft_y, dt)
    craft_vz = np.gradient(craft_z, dt)
    
    # Calculate distance and alignment
    distance = np.sqrt(craft_x**2 + craft_y**2 + craft_z**2)
    
    # Alignment angle (simplified - angle from ideal docking orientation)
    alignment_rad = np.sqrt(craft_roll**2 + craft_pitch**2 + craft_yaw**2)
    alignment_deg = np.degrees(alignment_rad)
    
    # Thrust calculation (non-zero when accelerating)
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

def create_spacecraft_mesh(scale=1.0, segments=16):
    """Create detailed spacecraft mesh (cylinder + cone nose)"""
    theta = np.linspace(0, 2*np.pi, segments)
    
    # Main body (cylinder)
    z_body = np.linspace(-15, 0, 10) * scale
    theta_body, z_body_grid = np.meshgrid(theta, z_body)
    x_body = 3 * scale * np.cos(theta_body)
    y_body = 3 * scale * np.sin(theta_body)
    
    # Nose cone
    z_nose = np.linspace(0, 6, 8) * scale
    theta_nose, z_nose_grid = np.meshgrid(theta, z_nose)
    r_nose = 3 * scale * (1 - z_nose / (6 * scale))
    x_nose = r_nose[:, np.newaxis] * np.cos(theta_nose)
    y_nose = r_nose[:, np.newaxis] * np.sin(theta_nose)
    
    # Combine
    x = np.vstack([x_body, x_nose])
    y = np.vstack([y_body, y_nose])
    z = np.vstack([z_body_grid, z_nose_grid])
    
    return x, y, z

def create_docking_station_mesh(scale=1.0, segments=16):
    """Create docking station mesh (larger, more modular)"""
    theta = np.linspace(0, 2*np.pi, segments)
    
    # Main cylinder
    z_main = np.linspace(-20, 0, 12) * scale
    theta_main, z_main_grid = np.meshgrid(theta, z_main)
    x_main = 5 * scale * np.cos(theta_main)
    y_main = 5 * scale * np.sin(theta_main)
    
    # Docking port (front)
    z_port = np.linspace(0, 3, 4) * scale
    theta_port, z_port_grid = np.meshgrid(theta, z_port)
    x_port = 6 * scale * np.cos(theta_port)
    y_port = 6 * scale * np.sin(theta_port)
    
    # Combine
    x = np.vstack([x_main, x_port])
    y = np.vstack([y_main, y_port])
    z = np.vstack([z_main_grid, z_port_grid])
    
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
ax.set_title('SPACECRAFT DOCKING SIMULATION\nAutonomous Approach with Alignment Control',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# Set view limits
ax.set_xlim(-600, 100)
ax.set_ylim(-350, 350)
ax.set_zlim(-350, 350)

# ============================================================================
# ANIMATED OBJECTS
# ============================================================================

# Docking station mesh (stationary)
station_mesh = None
station_docking_port = None

# Active spacecraft mesh
craft_mesh = None
craft_docking_port = None

# Thrust plume
thrust_cone = None

# Trajectory trail
trail_collection = None
trail_x, trail_y, trail_z = [], [], []

# Prediction ghost path
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
    
    # Clear old meshes
    if station_mesh is not None:
        station_mesh.remove()
    if craft_mesh is not None:
        craft_mesh.remove()
    if station_docking_port is not None:
        station_docking_port.remove()
    if craft_docking_port is not None:
        craft_docking_port.remove()
    if thrust_cone is not None:
        thrust_cone.remove()
    if trail_collection is not None:
        trail_collection.remove()
    if prediction_line is not None:
        prediction_line.remove()
    
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
    
    # Station docking port (yellow ring)
    port_theta = np.linspace(0, 2*np.pi, 20)
    port_r = 6
    port_x = port_r * np.cos(port_theta)
    port_y = port_r * np.sin(port_theta)
    port_z = np.full_like(port_x, 3)
    station_docking_port = ax.plot(port_x, port_y, port_z, 'yellow', linewidth=3, zorder=10)
    
    # ===== ACTIVE SPACECRAFT =====
    cx, cy, cz = create_spacecraft_mesh(scale=1.0, segments=12)
    cx_rot, cy_rot, cz_rot = rotate_mesh(cx, cy, cz, craft_rot[0], craft_rot[1], craft_rot[2])
    cx_final, cy_final, cz_final = translate_mesh(cx_rot, cy_rot, cz_rot, craft_pos)
    
    craft_mesh = ax.plot_surface(cx_final, cy_final, cz_final, color=craft_color, 
                                alpha=0.9, shade=True, linewidth=0, antialiased=False)
    
    # Craft docking port (colored ring based on alignment)
    port_local = np.array([[port_r * np.cos(t), port_r * np.sin(t), -15] 
                          for t in np.linspace(0, 2*np.pi, 20)])
    port_rotated = np.array([rotate_mesh(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
                                        craft_rot[0], craft_rot[1], craft_rot[2]) 
                            for p in port_local])
    craft_port_x = port_rotated[:, 0].flatten() + craft_pos[0]
    craft_port_y = port_rotated[:, 1].flatten() + craft_pos[1]
    craft_port_z = port_rotated[:, 2].flatten() + craft_pos[2]
    craft_docking_port = ax.plot(craft_port_x, craft_port_y, craft_port_z, 
                                craft_color, linewidth=3, zorder=10)
    
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
                                 color='#ff6b35', linewidth=4, alpha=0.8, zorder=5)
    
    # ===== TRAJECTORY TRAIL =====
    trail_x.append(craft_pos[0])
    trail_y.append(craft_pos[1])
    trail_z.append(craft_pos[2])
    
    if len(trail_x) > MAX_TRAIL_LENGTH:
        trail_x.pop(0)
        trail_y.pop(0)
        trail_z.pop(0)
    
    if len(trail_x) > 1:
        points = np.array([trail_x, trail_y, trail_z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = ['cyan'] * len(segments)
        trail_collection = Line3DCollection(segments, colors=colors, linewidths=2, 
                                           alpha=0.6, zorder=3)
        ax.add_collection3d(trail_collection)
    
    # ===== PREDICTION GHOST PATH (7 seconds ahead) =====
    fps = 1000 / animation_interval
    prediction_frames = int(PREDICTION_TIME * fps)
    end_idx = min(frame + prediction_frames, len(data['time']) - 1)
    
    if frame < len(data['time']) - prediction_frames:
        pred_x = data['craft']['x'][frame:end_idx:5]  # Subsample for performance
        pred_y = data['craft']['y'][frame:end_idx:5]
        pred_z = data['craft']['z'][frame:end_idx:5]
        prediction_line = ax.plot(pred_x, pred_y, pred_z, '--', 
                                 color='white', linewidth=2, alpha=0.4, zorder=2)[0]
    
    # ===== TELEMETRY =====
    time_to_dock = MISSION_DURATION - current_time
    aligned_status = "ALIGNED" if alignment < 10 else "ALIGNING"
    
    telemetry_text.set_text(
        f'TELEMETRY\n'
        f'Time: {current_time:.1f} s\n'
        f'Distance: {distance:.1f} m\n'
        f'Alignment: {alignment:.1f}°\n'
        f'Status: {aligned_status}\n'
        f'Time to Dock: {time_to_dock:.1f} s\n'
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

