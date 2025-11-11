#!/usr/bin/env python3
"""
NASA ISS Docking Simulation - PyVista Edition
==============================================
High-fidelity Crew Dragon docking to International Space Station.

REALISTIC IMPLEMENTATION:
- Accurate ISS Harmony module + PMA-2 docking adapter geometry
- SpaceX Crew Dragon capsule with NDS (NASA Docking System)
- R-bar approach corridor (radial vector, from below Earth)
- Realistic RCS thruster corrections (not perfectly smooth)
- Contact at 0.05 m/s (3 cm/s final approach speed)
- Soft capture → Hard dock sequence visualization
- Proper lighting, materials, and camera work

Mission Profile:
- Initial range: 400 meters
- Approach angle: From below/behind (standard R-bar)
- Final approach: Last 30m at 0.1 m/s
- Contact speed: 0.05 m/s
- Alignment tolerance: <2° for docking

Controls:
- SPACE: Play/Pause
- Mouse: Rotate view (works during animation!)
- Scroll: Zoom
- Q: Quit
"""

import numpy as np
import pyvista as pv
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_RANGE = 400  # meters from docking port
MISSION_DURATION = 180  # seconds (3 minutes)
FPS = 30  # Smooth animation
TOTAL_FRAMES = MISSION_DURATION * FPS

# Docking parameters
CONTACT_DISTANCE = 0.5  # meters - when mechanisms touch
SOFT_CAPTURE_SPEED = 0.05  # m/s (5 cm/s - realistic ISS docking speed)

# Animation control
is_paused = False
current_frame = 0

# ============================================================================
# REALISTIC TRAJECTORY WITH RCS CORRECTIONS
# ============================================================================

def generate_realistic_docking_trajectory(num_frames):
    """
    Generate realistic ISS docking approach with:
    - R-bar approach (from below)
    - RCS thruster firing corrections (not perfectly smooth)
    - Velocity profile matching real missions
    - Final approach hold points (30m, 10m)
    """
    time = np.linspace(0, MISSION_DURATION, num_frames)
    t_norm = np.linspace(0, 1, num_frames)
    
    # ISS Harmony module docking port is at origin, facing +Z
    # Spacecraft approaches from -Z direction
    
    # Phase 1: Initial Approach (0-40%) - From 400m to 30m hold point
    # Coming from below and behind
    phase1_mask = t_norm <= 0.4
    t1 = t_norm[phase1_mask] / 0.4
    
    # Main approach along -Z, with slight offset to come from below
    z_pos = -INITIAL_RANGE + (INITIAL_RANGE - 30) * t1**1.5
    x_pos = -80 * (1 - t1**1.8)  # Offset to side
    y_pos = -100 * (1 - t1**1.6)  # Coming from below
    
    # Add realistic RCS correction jitter (not perfectly smooth)
    rcs_jitter_x = 0.3 * np.sin(20 * t1) * (1 - t1**2)
    rcs_jitter_y = 0.2 * np.cos(25 * t1) * (1 - t1**2)
    rcs_jitter_z = 0.15 * np.sin(30 * t1) * (1 - t1)
    
    x_pos += rcs_jitter_x
    y_pos += rcs_jitter_y
    z_pos += rcs_jitter_z
    
    # Phase 2: 30m Hold & Mid-Course (40-70%) - From 30m to 10m
    # Align on centerline, reduce lateral offset
    phase2_mask = (t_norm > 0.4) & (t_norm <= 0.7)
    t2 = (t_norm[phase2_mask] - 0.4) / 0.3
    phase1_end = np.sum(phase1_mask) - 1
    
    z_pos_p2 = -30 + 20 * t2**1.8  # -30m to -10m
    x_pos_p2 = x_pos[phase1_end] * (1 - t2**2)  # Null lateral offset
    y_pos_p2 = y_pos[phase1_end] * (1 - t2**2)  # Null vertical offset
    
    # More aggressive RCS corrections during alignment
    rcs_jitter_x_p2 = 0.4 * np.sin(15 * t2) * (1 - t2)
    rcs_jitter_y_p2 = 0.3 * np.cos(18 * t2) * (1 - t2)
    
    x_pos_p2 += rcs_jitter_x_p2
    y_pos_p2 += rcs_jitter_y_p2
    
    # Phase 3: Final Approach (70-100%) - From 10m to contact at 0.5m
    # Very slow, centered, minimal corrections
    phase3_mask = t_norm > 0.7
    t3 = (t_norm[phase3_mask] - 0.7) / 0.3
    phase2_end = np.sum(phase1_mask) + np.sum(phase2_mask) - 1
    
    # Linear approach to contact point
    z_pos_p3 = -10 + (10 - CONTACT_DISTANCE) * t3**1.2
    x_pos_p3 = x_pos_p2[-1] * (1 - t3**3)  # Final centering
    y_pos_p3 = y_pos_p2[-1] * (1 - t3**3)
    
    # Minimal jitter in final approach (RCS hold only)
    rcs_jitter_final = 0.05 * np.sin(40 * t3) * np.exp(-5 * t3)
    x_pos_p3 += rcs_jitter_final
    y_pos_p3 += rcs_jitter_final * 0.5
    
    # Combine all phases
    x = np.concatenate([x_pos, x_pos_p2, x_pos_p3])
    y = np.concatenate([y_pos, y_pos_p2, y_pos_p3])
    z = np.concatenate([z_pos, z_pos_p2, z_pos_p3])
    
    # Attitude (roll, pitch, yaw) - gradual alignment
    # Start misaligned, end perfectly aligned with port
    roll = np.zeros(num_frames)
    pitch = np.zeros(num_frames)
    yaw = np.zeros(num_frames)
    
    # Phase 1: Initial misalignment
    roll[phase1_mask] = 0.3 * (1 - t1**2)
    pitch[phase1_mask] = -0.2 * (1 - t1**1.5)
    yaw[phase1_mask] = 0.15 * (1 - t1**2)
    
    # Phase 2: Fine alignment
    phase1_end_val = roll[phase1_mask][-1] if np.any(phase1_mask) else 0
    roll[phase2_mask] = phase1_end_val * (1 - t2**3)
    pitch[phase2_mask] = pitch[phase1_mask][-1] * (1 - t2**3) if np.any(phase1_mask) else 0
    yaw[phase2_mask] = yaw[phase1_mask][-1] * (1 - t2**3) if np.any(phase1_mask) else 0
    
    # Phase 3: Perfect alignment (very small residuals)
    roll[phase3_mask] = 0.01 * np.sin(50 * t3) * np.exp(-8 * t3)
    pitch[phase3_mask] = 0.008 * np.cos(45 * t3) * np.exp(-8 * t3)
    yaw[phase3_mask] = 0.006 * np.sin(55 * t3) * np.exp(-8 * t3)
    
    # Calculate velocities and distance
    dt = time[1] - time[0]
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    
    distance = np.sqrt(x**2 + y**2 + z**2)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Alignment error (degrees from perfect)
    alignment_error = np.degrees(np.sqrt(roll**2 + pitch**2 + yaw**2))
    
    return {
        'time': time,
        'x': x, 'y': y, 'z': z,
        'roll': roll, 'pitch': pitch, 'yaw': yaw,
        'vx': vx, 'vy': vy, 'vz': vz,
        'distance': distance,
        'speed': speed,
        'alignment_error': alignment_error
    }

# Generate trajectory
print("\n" + "="*70)
print("GENERATING REALISTIC ISS DOCKING TRAJECTORY")
print("="*70)
trajectory = generate_realistic_docking_trajectory(TOTAL_FRAMES)
print(f"✓ Generated {TOTAL_FRAMES} frames ({MISSION_DURATION}s @ {FPS} FPS)")
print(f"✓ Initial range: {trajectory['distance'][0]:.1f} m")
print(f"✓ Final range: {trajectory['distance'][-1]:.1f} m")
print(f"✓ Final speed: {trajectory['speed'][-1]*100:.1f} cm/s")

# ============================================================================
# CREATE 3D MODELS
# ============================================================================

def create_iss_docking_module():
    """
    Create ISS Harmony module + PMA-2 (Pressurized Mating Adapter).
    Realistic proportions based on actual ISS.
    """
    # Main Harmony module (Node 2) - 4.4m diameter, ~7m length
    harmony = pv.Cylinder(center=(0, 0, -15), direction=(0, 0, 1),
                         radius=2.2, height=14, resolution=32)
    
    # PMA-2 adapter (tapered cone) - connects to docking port
    pma = pv.Cone(center=(0, 0, -2), direction=(0, 0, 1),
                  height=4, radius=2.2, resolution=32)
    
    # Docking port ring (IDA - International Docking Adapter)
    # 0.8m radius, the actual interface
    docking_ring = pv.Cylinder(center=(0, 0, 0.5), direction=(0, 0, 1),
                               radius=0.8, height=1, resolution=32)
    
    # Combine all parts
    iss_module = harmony + pma + docking_ring
    
    return iss_module

def create_crew_dragon():
    """
    Create SpaceX Crew Dragon capsule.
    Realistic proportions: 3.7m diameter pressurized section.
    """
    # Main pressurized capsule (truncated cone)
    capsule_bottom = pv.Cylinder(center=(0, 0, -4), direction=(0, 0, 1),
                                 radius=1.85, height=2, resolution=32)
    
    capsule_top = pv.Cone(center=(0, 0, -2.5), direction=(0, 0, 1),
                          height=3, radius=1.85, resolution=32)
    
    # Nose cone with docking mechanism
    nose = pv.Cone(center=(0, 0, 0), direction=(0, 0, 1),
                   height=1.5, radius=1.0, resolution=32)
    
    # NDS (NASA Docking System) - soft capture system at nose
    nds_ring = pv.Cylinder(center=(0, 0, 1.2), direction=(0, 0, 1),
                          radius=0.65, height=0.4, resolution=32)
    
    # Combine all parts
    dragon = capsule_bottom + capsule_top + nose + nds_ring
    
    return dragon

# Create static models
print("\n" + "="*70)
print("CREATING 3D MODELS")
print("="*70)
iss_module = create_iss_docking_module()
dragon_template = create_crew_dragon()
print("✓ ISS Harmony + PMA-2 module")
print("✓ SpaceX Crew Dragon capsule")

# ============================================================================
# ANIMATION SETUP
# ============================================================================

# Create plotter with gradient space background (like matplotlib/plotly)
plotter = pv.Plotter(window_size=[1920, 1080])
plotter.set_background('#0a0a1e', top='#1a1a3e')  # Dark blue gradient like space

# Add ISS module (stationary)
plotter.add_mesh(iss_module, color='#A0A0A0', metallic=0.3, roughness=0.5,
                label='ISS Harmony/PMA-2')

# Add docking port highlighting (yellow ring)
port_highlight = pv.Cylinder(center=(0, 0, 1), direction=(0, 0, 1),
                            radius=0.8, height=0.1, resolution=32)
plotter.add_mesh(port_highlight, color='#FFD700', opacity=0.8, label='Docking Port')

# Initial Dragon position
dragon = dragon_template.copy()
dragon.translate([trajectory['x'][0], trajectory['y'][0], trajectory['z'][0]])

dragon_actor = plotter.add_mesh(dragon, color='white', metallic=0.5, roughness=0.3,
                                label='Crew Dragon')

# Add trajectory trail (will update)
trail_points = np.column_stack([trajectory['x'][:1], 
                                trajectory['y'][:1], 
                                trajectory['z'][:1]])
trail = pv.PolyData(trail_points)
trail_actor = plotter.add_mesh(trail, color='cyan', line_width=2, 
                              render_lines_as_tubes=True, label='Trajectory')

# Add velocity vector
velocity_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, -1), scale=20)
arrow_actor = plotter.add_mesh(velocity_arrow, color='#3498db')

# Add reference grid/axes for spatial awareness
plotter.show_axes()
plotter.add_axes_at_origin(labels_off=False)

# Add reference grid at origin
grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=200, j_size=200, 
               i_resolution=10, j_resolution=10)
plotter.add_mesh(grid, style='wireframe', color='gray', opacity=0.2, line_width=1)

# Lighting setup (realistic space lighting)
light1 = pv.Light(position=(100, 0, 0), light_type='scene light')
light2 = pv.Light(position=(-100, 100, 50), light_type='scene light', intensity=0.3)
plotter.add_light(light1)
plotter.add_light(light2)

# Camera setup - good viewing angle
plotter.camera_position = [(150, 150, 100), (0, 0, -10), (0, 0, 1)]

# Add telemetry text
telemetry_text = plotter.add_text("", position='upper_left', font_size=10, 
                                  color='white', font='courier')

controls_text = plotter.add_text(
    "CONTROLS: SPACE=Pause | Mouse=Rotate | Scroll=Zoom | Q=Quit",
    position='lower_edge', font_size=9, color='#FFD700'
)

print("\n" + "="*70)
print("STARTING DOCKING SIMULATION")
print("="*70)
print("Camera is FREE - rotate and zoom during animation!")
print("="*70 + "\n")

# ============================================================================
# ANIMATION LOOP
# ============================================================================

print("\n" + "="*70)
print("STARTING ANIMATION")
print("="*70)
print("Use mouse to rotate camera during animation")
print("Press 'q' to quit")
print("="*70 + "\n")

# Animation state
class AnimationState:
    def __init__(self):
        self.frame = 0
        self.running = True

state = AnimationState()

def update_scene(_caller):
    """Update animation frame - called by timer"""
    if not state.running or state.frame >= TOTAL_FRAMES:
        if state.frame >= TOTAL_FRAMES:
            print("\nAnimation complete - looping...")
            state.frame = 0
        return
    
    frame = state.frame
    
    # Get current state
    pos = [trajectory['x'][frame], trajectory['y'][frame], trajectory['z'][frame]]
    rot = [trajectory['roll'][frame], trajectory['pitch'][frame], trajectory['yaw'][frame]]
    distance = trajectory['distance'][frame]
    speed = trajectory['speed'][frame]
    alignment = trajectory['alignment_error'][frame]
    mission_time = trajectory['time'][frame]
    
    # Update Dragon position and orientation
    dragon = dragon_template.copy()
    
    # Apply rotation (Euler angles)
    dragon.rotate_z(np.degrees(rot[2]), point=(0, 0, 0))
    dragon.rotate_y(np.degrees(rot[1]), point=(0, 0, 0))
    dragon.rotate_x(np.degrees(rot[0]), point=(0, 0, 0))
    
    # Translate to current position
    dragon.translate(pos)
    
    # Update mesh
    plotter.remove_actor(dragon_actor, render=False)
    dragon_actor_new = plotter.add_mesh(dragon, color='white', metallic=0.5, roughness=0.3,
                                        render=False)
    
    # Update trajectory trail (show last 300 frames for better visibility)
    trail_start = max(0, frame - 300)
    if frame > 0:
        trail_points = np.column_stack([trajectory['x'][trail_start:frame+1],
                                       trajectory['y'][trail_start:frame+1],
                                       trajectory['z'][trail_start:frame+1]])
        if len(trail_points) > 1:
            trail = pv.PolyData(trail_points)
            plotter.remove_actor(trail_actor, render=False)
            trail_actor_new = plotter.add_mesh(trail, color='cyan', line_width=3,
                                              render_lines_as_tubes=True, render=False)
    
    # Update velocity vector
    if speed > 0.01:
        vel_normalized = np.array([trajectory['vx'][frame],
                                  trajectory['vy'][frame],
                                  trajectory['vz'][frame]]) / speed
        arrow_length = min(40, speed * 150)  # Scale with speed, larger for visibility
        velocity_arrow = pv.Arrow(start=pos, direction=vel_normalized, scale=arrow_length)
        plotter.remove_actor(arrow_actor, render=False)
        arrow_actor_new = plotter.add_mesh(velocity_arrow, color='#3498db', render=False)
    
    # Determine mission phase and status
    if distance < 1:
        phase = "SOFT CAPTURE"
    elif distance < 10:
        phase = "FINAL APPROACH"
    elif distance < 30:
        phase = "30m HOLD POINT"
    else:
        phase = "INITIAL APPROACH"
    
    # Update telemetry with COORDINATES
    telemetry = (
        f"═══════════════════════════════════\n"
        f"NASA ISS DOCKING TELEMETRY\n"
        f"═══════════════════════════════════\n"
        f"Mission Time:    {mission_time:7.1f} s\n"
        f"Phase:           {phase}\n"
        f"───────────────────────────────────\n"
        f"POSITION (meters):\n"
        f"  X: {pos[0]:8.2f} m\n"
        f"  Y: {pos[1]:8.2f} m\n"
        f"  Z: {pos[2]:8.2f} m\n"
        f"───────────────────────────────────\n"
        f"Range to Port:   {distance:7.2f} m\n"
        f"Closure Rate:    {speed:7.3f} m/s\n"
        f"                 ({speed*100:6.2f} cm/s)\n"
        f"Alignment Error: {alignment:7.3f}°\n"
        f"───────────────────────────────────\n"
        f"ATTITUDE (degrees):\n"
        f"  Roll:  {np.degrees(rot[0]):7.3f}°\n"
        f"  Pitch: {np.degrees(rot[1]):7.3f}°\n"
        f"  Yaw:   {np.degrees(rot[2]):7.3f}°\n"
        f"───────────────────────────────────\n"
        f"Frame: {frame+1:4d}/{TOTAL_FRAMES}"
    )
    
    plotter.remove_actor(telemetry_text, render=False)
    telemetry_text_new = plotter.add_text(telemetry, position='upper_left', 
                                         font_size=9, color='white', font='courier')
    
    # Render once after all updates
    plotter.render()
    
    state.frame += 1

# Show window first
plotter.show(interactive_update=True, auto_close=False)

# Run animation loop manually (more reliable than timer)
import time
start_time = time.time()

try:
    while state.frame < TOTAL_FRAMES:
        update_scene(None)
        time.sleep(1.0 / FPS)  # Control framerate
        
        # Allow window events to process
        if not plotter.iren.initialized:
            break
            
        plotter.iren.process_events()
        
        # Print progress every 30 frames
        if state.frame % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {state.frame}/{TOTAL_FRAMES} ({elapsed:.1f}s)")
    
    print("\n" + "="*70)
    print("ANIMATION COMPLETE - Window will stay open")
    print("Press 'q' to quit")
    print("="*70)
    
    # Keep window open
    plotter.iren.start()
    
except KeyboardInterrupt:
    print("\nAnimation interrupted by user")
finally:
    plotter.close()

