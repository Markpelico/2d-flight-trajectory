import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

# ============================================================================
# CONFIGURATION
# ============================================================================

MOON_RADIUS_KM = 1737.4  # Actual moon radius in km
MAX_TRAIL_LENGTH = 150  # Limit trail points for performance

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
    
    # Add significant altitude variation for each orbit (prevents overlap)
    # Create clear spiral effect - altitude increases throughout mission
    altitude_variation = 100 * t_norm  # Increases from 0 to 100 km
    # Also add sinusoidal variation per orbit
    altitude_variation += 30 * np.sin(n_orbits * theta)
    r = r + altitude_variation
    
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
ax.set_title('Set A - Position Trajectory (Lunar Orbit)\nOther views: Under Construction', 
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

# Create Moon sphere (optimized resolution for performance)
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 30)
moon_x = MOON_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
moon_y = MOON_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
moon_z = MOON_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(moon_x, moon_y, moon_z, color='#808080', alpha=0.5, shade=True, 
               edgecolor='none', linewidth=0, antialiased=False, rcount=30, ccount=40)

# Add start and end markers
ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start', edgecolors='darkgreen', linewidths=2)
ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End', edgecolors='darkred', linewidths=2)

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Add controls text on plot
controls_text = """CONTROLS:
SPACE: Pause/Play
1-4: Speed
+/-: Zoom
R: Reset
Click: Info"""

ax.text2D(0.02, 0.02, controls_text, transform=ax.transAxes,
         fontsize=8, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', 
                  edgecolor='gray', linewidth=1, alpha=0.9),
         family='monospace')

# ============================================================================
# CREATE ANIMATED OBJECTS
# ============================================================================

# Spacecraft marker
spacecraft, = ax.plot([], [], [], 'o', markersize=12, color='black', 
                     markeredgecolor='gray', markeredgewidth=1.5, zorder=10)

# Trail with gradient
from mpl_toolkits.mplot3d.art3d import Line3DCollection
trail_collection = None

# Velocity direction arrow
velocity_arrow, = ax.plot([], [], [], '-', linewidth=2, color='#333333', zorder=9)

# Telemetry display
info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', 
                              edgecolor='gray', linewidth=1, alpha=0.9),
                     family='monospace')

# Speed display
speed_text = ax.text2D(0.98, 0.02, 'Speed: 1.0x (20 FPS)', transform=ax.transAxes,
                      fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', 
                               edgecolor='gray', linewidth=1, alpha=0.9),
                      family='monospace')

# Store trail data
trail_x = []
trail_y = []
trail_z = []

# Animation speed control
animation_interval = 50  # milliseconds per frame (higher FPS for smoother animation)

# ============================================================================
# CLICK TO SHOW COORDINATES (RELIABLE METHOD)
# ============================================================================

# Make trajectory points clickable
trajectory_scatter = ax.scatter(x, y, z, s=1, alpha=0.01, picker=True, pickradius=15)

# Annotation for click
click_annot = ax.text2D(0.5, 0.85, '', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', 
                                edgecolor='gray', linewidth=1, alpha=0.95),
                       fontsize=9, family='monospace', visible=False,
                       ha='center', va='top')

def on_pick(event):
    """Show data when clicking on trajectory"""
    if event.ind is not None and len(event.ind) > 0:
        idx = event.ind[0]
        
        text = f"Point {idx+1}/{len(x)}\n"
        text += f"Time: {time_elapsed[idx]:.1f} s\n"
        text += f"Altitude: {altitude[idx]:.1f} km\n"
        text += f"X: {x[idx]:.1f} km\n"
        text += f"Y: {y[idx]:.1f} km\n"
        text += f"Z: {z[idx]:.1f} km"
        
        click_annot.set_text(text)
        click_annot.set_visible(True)
        fig.canvas.draw_idle()
        
        # Auto-hide after 3 seconds
        def hide_annot():
            click_annot.set_visible(False)
            fig.canvas.draw_idle()
        
        fig.canvas.get_tk_widget().after(3000, hide_annot) if hasattr(fig.canvas, 'get_tk_widget') else None

fig.canvas.mpl_connect('pick_event', on_pick)

# ============================================================================
# MOUSE WHEEL ZOOM
# ============================================================================

def on_scroll(event):
    """Zoom in/out with mouse wheel"""
    if event.inaxes == ax:
        # Get current distance
        current_dist = np.linalg.norm(ax.get_position())
        
        # Zoom factor
        if event.button == 'up':
            scale_factor = 0.9  # Zoom in
        elif event.button == 'down':
            scale_factor = 1.1  # Zoom out
        else:
            return
        
        # Apply zoom by scaling axis limits
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * scale_factor / 2
        y_range = (ylim[1] - ylim[0]) * scale_factor / 2
        z_range = (zlim[1] - zlim[0]) * scale_factor / 2
        
        ax.set_xlim3d([x_center - x_range, x_center + x_range])
        ax.set_ylim3d([y_center - y_range, y_center + y_range])
        ax.set_zlim3d([z_center - z_range, z_center + z_range])
        
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

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
        animation_interval = 100
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 0.5x')
        fig.canvas.draw_idle()
    
    elif event.key == '2':
        animation_interval = 50
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 1.0x')
        fig.canvas.draw_idle()
    
    elif event.key == '3':
        animation_interval = 33
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 1.5x')
        fig.canvas.draw_idle()
    
    elif event.key == '4':
        animation_interval = 25
        anim.event_source.interval = animation_interval
        speed_text.set_text('Speed: 2.0x')
        fig.canvas.draw_idle()
    
    elif event.key == '+' or event.key == '=':
        # Zoom in
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * 0.9 / 2
        y_range = (ylim[1] - ylim[0]) * 0.9 / 2
        z_range = (zlim[1] - zlim[0]) * 0.9 / 2
        
        ax.set_xlim3d([x_center - x_range, x_center + x_range])
        ax.set_ylim3d([y_center - y_range, y_center + y_range])
        ax.set_zlim3d([z_center - z_range, z_center + z_range])
        fig.canvas.draw_idle()
    
    elif event.key == '-' or event.key == '_':
        # Zoom out
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * 1.1 / 2
        y_range = (ylim[1] - ylim[0]) * 1.1 / 2
        z_range = (zlim[1] - zlim[0]) * 1.1 / 2
        
        ax.set_xlim3d([x_center - x_range, x_center + x_range])
        ax.set_ylim3d([y_center - y_range, y_center + y_range])
        ax.set_zlim3d([z_center - z_range, z_center + z_range])
        fig.canvas.draw_idle()
    
    elif event.key == 'r':
        # Reset view
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key_press)

# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def init():
    """Initialize animation"""
    global trail_collection
    spacecraft.set_data([], [])
    spacecraft.set_3d_properties([])
    velocity_arrow.set_data([], [])
    velocity_arrow.set_3d_properties([])
    info_text.set_text('')
    if trail_collection is not None:
        trail_collection.remove()
        trail_collection = None
    return spacecraft, velocity_arrow, info_text

def animate(frame):
    """Update animation for each frame"""
    global trail_collection
    
    # Current position
    current_x = x[frame]
    current_y = y[frame]
    current_z = z[frame]
    
    # Update spacecraft position
    spacecraft.set_data([current_x], [current_y])
    spacecraft.set_3d_properties([current_z])
    
    # Update trail (path traveled so far) with length limit
    trail_x.append(current_x)
    trail_y.append(current_y)
    trail_z.append(current_z)
    
    # Limit trail length for better performance
    if len(trail_x) > MAX_TRAIL_LENGTH:
        trail_x.pop(0)
        trail_y.pop(0)
        trail_z.pop(0)
    
    # Remove old trail collection
    if trail_collection is not None:
        trail_collection.remove()
    
    # Create trail with green-to-red gradient (like Plotly)
    if len(trail_x) > 1:
        points = np.array([trail_x, trail_y, trail_z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create gradient from green to red
        n_segments = len(segments)
        colors = []
        for i in range(n_segments):
            progress = i / max(n_segments - 1, 1)
            r = progress
            g = 1 - progress
            colors.append((r, g, 0, 0.9))
        
        trail_collection = Line3DCollection(segments, colors=colors, linewidths=3, zorder=5)
        ax.add_collection3d(trail_collection)
    
    # Update velocity direction arrow (points where spacecraft is heading)
    vel_mag = np.sqrt(vx[frame]**2 + vy[frame]**2 + vz[frame]**2)
    if vel_mag > 0:
        arrow_length = 400  # km (longer for visibility)
        dir_x = vx[frame] / vel_mag * arrow_length
        dir_y = vy[frame] / vel_mag * arrow_length
        dir_z = vz[frame] / vel_mag * arrow_length
        
        velocity_arrow.set_data([current_x, current_x + dir_x], 
                               [current_y, current_y + dir_y])
        velocity_arrow.set_3d_properties([current_z, current_z + dir_z])
    else:
        velocity_arrow.set_data([], [])
        velocity_arrow.set_3d_properties([])
    
    # Update telemetry display (matches Plotly format)
    info_text.set_text(
        f'TELEMETRY\n'
        f'Time: {time_elapsed[frame]:.1f} s\n'
        f'Altitude: {altitude[frame]:.2f} km\n'
        f'Frame: {frame+1}/{len(x)}'
    )
    
    return spacecraft, velocity_arrow, info_text

# ============================================================================
# CREATE AND RUN ANIMATION
# ============================================================================

anim = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init,
    frames=len(x),
    interval=animation_interval,
    blit=False,  # Set to False to prevent play/pause issues with 3D
    repeat=True,
    cache_frame_data=False
)

plt.tight_layout()

print("Loading...")
print("Controls: SPACE=Pause/Play | 1-4=Speed | +/-=Zoom | R=Reset | Click Orbit=Show Data")

plt.show()



