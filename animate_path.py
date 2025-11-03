import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

# ============================================================================
# GENERATE REALISTIC ROCKET FLIGHT TRAJECTORY DATA
# ============================================================================

def generate_rocket_trajectory(num_points=100):
    """
    Generate a realistic 2D rocket flight path with:
    - Launch phase (vertical ascent)
    - Pitch-over maneuver (gradual turn)
    - Gravity turn (arc following orbital mechanics)
    - Coast/orbit insertion
    """
    # Time array for the trajectory
    t = np.linspace(0, 1, num_points)
    
    # Horizontal position (x): starts at 0, increases gradually then faster
    # Simulates gaining horizontal velocity during ascent
    x = 50 * t**2  # Parabolic increase in horizontal distance
    
    # Vertical position (y): rapid initial climb, then levels off
    # First stage: rapid climb, second stage: arc over
    y = 100 * t - 50 * t**2  # Parabolic arc (like projectile motion)
    
    # Add some realistic variation/noise
    noise = np.random.normal(0, 0.5, num_points)
    y = y + noise
    
    # Simulate flight parameters for each point
    # Altitude (km), Velocity (m/s), Time (seconds)
    altitude = y  # altitude in km
    velocity = 500 + 7000 * t  # increasing velocity from 500 to 7500 m/s
    time_elapsed = t * 600  # 0 to 600 seconds (10 minutes)
    
    return x, y, altitude, velocity, time_elapsed

# Generate the trajectory data
x, y, altitude, velocity, time_elapsed = generate_rocket_trajectory(100)

# ============================================================================
# SET UP THE PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(x.min() - 5, x.max() + 5)
ax.set_ylim(-5, y.max() + 10)
ax.grid(True, alpha=0.3)
ax.set_xlabel('Downrange Distance (km)', fontsize=12)
ax.set_ylabel('Altitude (km)', fontsize=12)
ax.set_title('Rocket Flight Trajectory Animation\n[Press SPACE to Pause/Play | Hover over points for info]', 
             fontsize=14, fontweight='bold')

# ============================================================================
# PLOT THE TRAJECTORY WITH COLOR GRADIENT (GREEN -> RED)
# ============================================================================

# Create color array: green (start) to red (end)
# Uses RGB values that transition smoothly
num_segments = len(x) - 1
colors = np.zeros((num_segments, 4))  # RGBA colors
for i in range(num_segments):
    # Progress from 0 to 1
    progress = i / num_segments
    # Green to red transition
    colors[i] = [progress, 1-progress, 0, 0.6]  # Red increases, Green decreases

# Plot trajectory as colored line segments
from matplotlib.collections import LineCollection
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, colors=colors, linewidths=3, zorder=1)
ax.add_collection(lc)

# Plot all waypoints with color gradient
scatter = ax.scatter(x, y, c=np.linspace(0, 1, len(x)), cmap='RdYlGn_r', 
                    s=50, zorder=2, alpha=0.6, edgecolors='black', linewidths=0.5)

# Highlight start and end points clearly
ax.scatter(x[0], y[0], c='lime', s=300, marker='o', zorder=4, 
          label='Launch Point', edgecolors='darkgreen', linewidths=2)
ax.scatter(x[-1], y[-1], c='red', s=300, marker='*', zorder=4, 
          label='Target/End', edgecolors='darkred', linewidths=2)

# Place legend in bottom right corner to avoid covering live data in top left
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

# Add colorbar to show time progression
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Flight Progress (Green=Start, Red=End)', rotation=270, labelpad=20)

# ============================================================================
# CREATE ANIMATED ROCKET OBJECT
# ============================================================================

# The moving rocket indicator
moving_rocket, = ax.plot([], [], 'o', markersize=20, color='orange', 
                         markeredgecolor='black', markeredgewidth=2, zorder=5)

# Trail showing path traveled so far
trail_line, = ax.plot([], [], 'yellow', linewidth=4, alpha=0.8, zorder=3)

# Text annotation showing current flight info
info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Store trail data
trail_x = []
trail_y = []

# ============================================================================
# HOVER FUNCTIONALITY - Show info when mouse hovers over points
# ============================================================================

# Annotation box that appears on hover
annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                   bbox=dict(boxstyle="round", fc="lightyellow", ec="black", lw=2),
                   arrowprops=dict(arrowstyle="->", color="black"),
                   fontsize=10, zorder=10)
annot.set_visible(False)

def on_hover(event):
    """Show information when hovering over a data point"""
    if event.inaxes == ax:
        # Check if mouse is near any data point
        for i in range(len(x)):
            # Calculate distance from mouse to point
            distance = np.sqrt((event.xdata - x[i])**2 + (event.ydata - y[i])**2)
            
            # If close enough to a point (within threshold)
            if distance < 2:  # threshold distance
                # Show annotation with flight data
                annot.xy = (x[i], y[i])
                text = f"Point {i+1}/{len(x)}\n"
                text += f"Time: {time_elapsed[i]:.1f} s\n"
                text += f"Altitude: {altitude[i]:.1f} km\n"
                text += f"Velocity: {velocity[i]:.0f} m/s\n"
                text += f"Downrange: {x[i]:.1f} km"
                annot.set_text(text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        
        # Hide annotation if not near any point
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

# Connect hover event to the plot
fig.canvas.mpl_connect("motion_notify_event", on_hover)

# ============================================================================
# PAUSE/PLAY FUNCTIONALITY
# ============================================================================

# Animation state tracker
is_paused = False

def on_key_press(event):
    """Handle keyboard input for pause/play"""
    global is_paused
    if event.key == ' ':  # Space bar toggles pause/play
        is_paused = not is_paused
        if is_paused:
            anim.event_source.stop()
            print("Animation PAUSED - Press SPACE to resume")
        else:
            anim.event_source.start()
            print("Animation PLAYING - Press SPACE to pause")

# Connect keyboard event to the plot
fig.canvas.mpl_connect('key_press_event', on_key_press)

# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def init():
    """Initialize animation - set empty data for moving elements"""
    moving_rocket.set_data([], [])
    trail_line.set_data([], [])
    info_text.set_text('')
    return moving_rocket, trail_line, info_text

def animate(frame):
    """
    Update function called for each animation frame
    frame: current frame number (0 to len(x)-1)
    """
    # Get current position
    current_x = x[frame]
    current_y = y[frame]
    
    # Update rocket position
    moving_rocket.set_data([current_x], [current_y])
    
    # Update trail (path traveled so far)
    trail_x.append(current_x)
    trail_y.append(current_y)
    trail_line.set_data(trail_x, trail_y)
    
    # Update info text box with current flight data
    progress = (frame + 1) / len(x) * 100
    info_text.set_text(
        f'Flight Progress: {progress:.1f}%\n'
        f'Point: {frame+1}/{len(x)}\n'
        f'Time: {time_elapsed[frame]:.1f} s\n'
        f'Altitude: {altitude[frame]:.1f} km\n'
        f'Velocity: {velocity[frame]:.0f} m/s'
    )
    
    return moving_rocket, trail_line, info_text

# ============================================================================
# CREATE AND RUN ANIMATION
# ============================================================================

# Create the animation
anim = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init,
    frames=len(x),  # One frame per data point
    interval=100,  # 100ms between frames (10 fps)
    blit=True,  # Optimization: only redraw changed elements
    repeat=True  # Loop the animation
)

# Display the animation
plt.tight_layout()
print("\n" + "="*60)
print("CONTROLS:")
print("  - Press SPACE to pause/play the animation")
print("  - Hover mouse over any point to see detailed info")
print("  - Close window to exit")
print("="*60 + "\n")
plt.show()

# Optional: Save the animation
# Uncomment to save as GIF (requires pillow: pip install pillow)
# anim.save('rocket_trajectory.gif', writer='pillow', fps=10)

