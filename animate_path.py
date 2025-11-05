import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# CONFIGURATION
# ============================================================================

MOON_RADIUS_KM = 1737.4  # Actual moon radius in km

# ============================================================================
# GENERATE LUNAR ORBITAL TRAJECTORY DATA
# ============================================================================

def generate_lunar_orbit_trajectory(num_points=500):
    """
    Generate realistic 3D lunar orbit trajectory.
    
    Orbital mechanics:
    - Orbit altitude: 100 km above moon surface
    - Orbit period: 2 hours
    - Number of orbits: 2
    - Inclination: 15 degrees
    """
    # Time array - 2 hour mission
    time_seconds = np.linspace(0, 7200, num_points)
    t_norm = np.linspace(0, 1, num_points)
    
    # Orbital parameters
    orbit_altitude = 100  # km above surface
    orbit_radius = MOON_RADIUS_KM + orbit_altitude
    
    # Complete 2 orbits
    n_orbits = 2
    theta = 2 * np.pi * n_orbits * t_norm
    
    # Elliptical orbit (slight eccentricity)
    eccentricity = 0.05
    r = orbit_radius * (1 - eccentricity * np.cos(theta))
    
    # Position in orbital plane
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Add inclination (15 degree tilt)
    inclination = np.radians(15)
    z = r * np.sin(theta) * np.sin(inclination)
    y = r * np.sin(theta) * np.cos(inclination)
    
    # Add small perturbations
    x += np.random.normal(0, 0.1, num_points)
    y += np.random.normal(0, 0.1, num_points)
    z += np.random.normal(0, 0.05, num_points)
    
    # Calculate velocity
    vx = np.gradient(x, time_seconds)
    vy = np.gradient(y, time_seconds)
    vz = np.gradient(z, time_seconds)
    
    # Calculate altitude above surface
    altitude = np.sqrt(x**2 + y**2 + z**2) - MOON_RADIUS_KM
    
    return x, y, z, vx, vy, vz, altitude, time_seconds

# Generate the trajectory data
x, y, z, vx, vy, vz, altitude, time_elapsed = generate_lunar_orbit_trajectory(500)

# ============================================================================
# SET UP 3D PLOT
# ============================================================================

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position (km)', fontsize=11)
ax.set_ylabel('Y Position (km)', fontsize=11)
ax.set_zlabel('Z Position (km)', fontsize=11)
ax.set_title('Lunar Orbit Trajectory Animation\n[Press SPACE: Pause/Play | 1-4 Keys: Speed Control]', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Set equal aspect ratio for realistic view
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ============================================================================
# ADD MOON SPHERE
# ============================================================================

# Create Moon sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 40)
moon_x = MOON_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
moon_y = MOON_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
moon_z = MOON_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(moon_x, moon_y, moon_z, color='gray', alpha=0.6, shade=True)

# Add start and end markers
ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start', edgecolors='darkgreen', linewidths=2)
ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End', edgecolors='darkred', linewidths=2)

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# ============================================================================
# CREATE ANIMATED OBJECTS
# ============================================================================

# Spacecraft marker (orange)
spacecraft, = ax.plot([], [], [], 'o', markersize=15, color='orange', 
                     markeredgecolor='black', markeredgewidth=2)

# Trail showing path traveled (cyan with gradient)
trail_line, = ax.plot([], [], [], linewidth=3, color='cyan', alpha=0.9)

# Velocity direction arrow (red)
velocity_arrow, = ax.plot([], [], [], 'r-', linewidth=3)

# Telemetry text box
info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                     family='monospace')

# Store trail data
trail_x = []
trail_y = []
trail_z = []

# Animation speed control
animation_interval = 100  # milliseconds per frame

# ============================================================================
# HOVER FUNCTIONALITY
# ============================================================================

# Create annotation for hover display
annot = ax.text2D(0.5, 0.5, '', transform=ax.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2),
                 fontsize=9, family='monospace', visible=False)

def on_hover(event):
    """Show x,y,z coordinates when hovering near trajectory points"""
    if event.inaxes == ax:
        # Check if mouse is near any trajectory point
        if event.xdata is not None and event.ydata is not None:
            for i in range(len(x)):
                # Calculate 2D distance in plot coordinates
                dx = event.xdata - x[i]
                dy = event.ydata - y[i]
                distance = np.sqrt(dx**2 + dy**2)
                
                # If close to a point (within threshold)
                threshold = max_range * 0.05  # 5% of plot range
                if distance < threshold:
                    # Show annotation with position data
                    text = f"Point {i+1}/{len(x)}\n"
                    text += f"X: {x[i]:.2f} km\n"
                    text += f"Y: {y[i]:.2f} km\n"
                    text += f"Z: {z[i]:.2f} km\n"
                    text += f"Time: {time_elapsed[i]:.1f} s"
                    
                    annot.set_text(text)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        
        # Hide annotation if not near any point
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_hover)

# ============================================================================
# KEYBOARD CONTROLS - SPEED AND PAUSE
# ============================================================================

is_paused = False

def on_key_press(event):
    """Handle keyboard input for pause/play and speed control"""
    global is_paused, animation_interval
    
    if event.key == ' ':
        is_paused = not is_paused
        if is_paused:
            anim.event_source.stop()
        else:
            anim.event_source.start()
    
    elif event.key == '1':
        animation_interval = 200
        anim.event_source.interval = animation_interval
    
    elif event.key == '2':
        animation_interval = 100
        anim.event_source.interval = animation_interval
    
    elif event.key == '3':
        animation_interval = 67
        anim.event_source.interval = animation_interval
    
    elif event.key == '4':
        animation_interval = 50
        anim.event_source.interval = animation_interval

fig.canvas.mpl_connect('key_press_event', on_key_press)

# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def init():
    """Initialize animation"""
    spacecraft.set_data([], [])
    spacecraft.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    velocity_arrow.set_data([], [])
    velocity_arrow.set_3d_properties([])
    info_text.set_text('')
    return spacecraft, trail_line, velocity_arrow, info_text

def animate(frame):
    """Update animation for each frame"""
    # Current position
    current_x = x[frame]
    current_y = y[frame]
    current_z = z[frame]
    
    # Update spacecraft position
    spacecraft.set_data([current_x], [current_y])
    spacecraft.set_3d_properties([current_z])
    
    # Update trail (path traveled so far)
    trail_x.append(current_x)
    trail_y.append(current_y)
    trail_z.append(current_z)
    trail_line.set_data(trail_x, trail_y)
    trail_line.set_3d_properties(trail_z)
    
    # Update velocity direction arrow
    # Arrow extends from spacecraft in direction of velocity
    vel_mag = np.sqrt(vx[frame]**2 + vy[frame]**2 + vz[frame]**2)
    if vel_mag > 0:
        # Normalize and scale
        arrow_length = 300  # km
        dir_x = vx[frame] / vel_mag * arrow_length
        dir_y = vy[frame] / vel_mag * arrow_length
        dir_z = vz[frame] / vel_mag * arrow_length
        
        velocity_arrow.set_data([current_x, current_x + dir_x], 
                               [current_y, current_y + dir_y])
        velocity_arrow.set_3d_properties([current_z, current_z + dir_z])
    
    # Update telemetry display
    progress = (frame + 1) / len(x) * 100
    info_text.set_text(
        f'TELEMETRY\n'
        f'Time: {time_elapsed[frame]:.1f} s\n'
        f'Altitude: {altitude[frame]:.2f} km\n'
        f'Frame: {frame+1}/{len(x)}'
    )
    
    return spacecraft, trail_line, velocity_arrow, info_text

# ============================================================================
# CREATE AND RUN ANIMATION
# ============================================================================

anim = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init,
    frames=len(x),
    interval=animation_interval,
    blit=True,
    repeat=True
)

plt.tight_layout()

print("Loading...")
print("Controls: SPACE=Pause/Play | Keys 1-4=Speed | Mouse=Rotate")

plt.show()

# Optional: Save animation
# anim.save('lunar_orbit.gif', writer='pillow', fps=10)

