# User Guide: Lunar Orbit Animation (Matplotlib)

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Code Structure](#code-structure)
5. [Controls](#controls)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This tool creates a 3D animation of a spacecraft orbiting the Moon using Matplotlib. It runs in a native desktop window (no browser required) and allows full 3D rotation even during playback.

**Key Features:**
- 3D lunar orbit visualization
- Moon sphere with realistic dimensions
- Spacecraft marker (orange dot)
- Velocity direction arrow (red line)
- Growing cyan trail showing path traveled
- Live telemetry display
- Keyboard speed controls (0.5x to 2.0x)

---

## Installation

### Requirements
- Python 3.7 or newer
- NumPy
- Matplotlib

### Install Command

```bash
pip install numpy matplotlib
```

### Verify Installation

```bash
python3 -c "import numpy; import matplotlib; print('Ready!')"
```

---

## Quick Start

```bash
cd /Users/bigboi2/Desktop/2DPython
python3 animate_path.py
```

A window will open showing the Moon with an orbiting spacecraft. Press SPACE to start the animation.

---

## Code Structure

```
animate_path.py
├── Configuration (lines 7-11)
├── generate_lunar_orbit_trajectory() (lines 13-68)
├── 3D Plot Setup (lines 70-90)
├── Moon Sphere Creation (lines 92-109)
├── Animated Objects Setup (lines 111-137)
├── Keyboard Controls (lines 139-178)
├── Animation Functions (lines 180-236)
└── Main Execution (lines 238-281)
```

---

## Controls

### Keyboard Controls

**SPACE** - Pause/Play toggle
- Press once to pause
- Press again to resume

**1 Key** - 0.5x Speed (slow)
- Good for detailed observation
- 200ms per frame

**2 Key** - 1.0x Speed (normal)
- Default speed
- 100ms per frame

**3 Key** - 1.5x Speed (fast)
- 67ms per frame

**4 Key** - 2.0x Speed (very fast)
- Quick overview
- 50ms per frame

### Mouse Controls

**Left Click + Drag** - Rotate view
- Works DURING animation (unique to Matplotlib!)
- Full 3D rotation

**Scroll Wheel** - Zoom in/out

**Right Click + Drag** - Pan view

---

## Code Explanation

### Orbital Trajectory Generation (Lines 17-65)

```python
def generate_lunar_orbit_trajectory(num_points=500):
```

**Creates realistic orbital data:**

1. **Time Array** (Lines 28-29)
   - 7200 seconds = 2 hours
   - 500 evenly-spaced data points

2. **Orbital Parameters** (Lines 31-41)
   - Altitude: 100 km above Moon surface
   - Orbit radius: 1737.4 + 100 = 1837.4 km
   - Eccentricity: 0.05 (slightly elliptical)
   - Number of orbits: 2 complete revolutions

3. **Position Calculation** (Lines 43-50)
   - Uses elliptical orbit equation: r = a(1 - e·cos(θ))
   - Adds 15-degree inclination for 3D effect
   - Converts polar to Cartesian coordinates

4. **Perturbations** (Lines 52-55)
   - Adds small random noise (~100 meters)
   - Simulates gravitational irregularities
   - Makes orbit more realistic

5. **Velocity Calculation** (Lines 57-60)
   - Computed from position derivatives
   - `np.gradient()` calculates rate of change
   - Used for velocity direction arrow

---

### Moon Sphere Creation (Lines 96-103)

```python
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 40)
moon_x = MOON_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
moon_y = MOON_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
moon_z = MOON_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
```

**What it does:**
- Creates spherical mesh grid
- `u`: Longitude (0 to 2π)
- `v`: Latitude (0 to π)
- `np.outer()`: Combines arrays for surface

**Why 50x40:**
- Smooth appearance
- Fast rendering
- Good balance

---

### Animation Objects (Lines 115-137)

**Spacecraft** (Lines 115-117)
```python
spacecraft, = ax.plot([], [], [], 'o', markersize=15, color='orange', 
                     markeredgecolor='black', markeredgewidth=2)
```
- Orange circle with black outline
- Size 15 (clearly visible)
- Updates position each frame

**Trail** (Lines 119-120)
```python
trail_line, = ax.plot([], [], [], linewidth=3, color='cyan', alpha=0.9)
```
- Cyan color (bright and visible)
- Width 3 pixels
- Builds up during animation

**Velocity Arrow** (Lines 122-123)
```python
velocity_arrow, = ax.plot([], [], [], 'r-', linewidth=3)
```
- Red line showing direction
- Extends 300 km from spacecraft
- Updates direction each frame

**Telemetry Box** (Lines 125-129)
```python
info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                     family='monospace')
```
- Top-left corner position
- Monospace font for alignment
- Wheat-colored background

---

### Animation Update Function (Lines 195-236)

**Each frame updates:**

1. **Spacecraft Position** (Lines 198-204)
   - Moves to current orbital position

2. **Trail** (Lines 206-211)
   - Appends current position to trail lists
   - Trail grows from start to current position

3. **Velocity Arrow** (Lines 213-225)
   - Calculates velocity magnitude
   - Normalizes to unit vector
   - Scales to 300 km length
   - Points in direction of motion

4. **Telemetry** (Lines 227-234)
   - Updates time, altitude, frame number
   - Formatted as monospace text

---

### Speed Control Implementation (Lines 158-176)

```python
elif event.key == '1':  # 0.5x speed
    animation_interval = 200
    anim.event_source.interval = animation_interval
```

**How it works:**
- Detects key press (1, 2, 3, or 4)
- Changes `animation_interval` variable
- Updates `anim.event_source.interval` in real-time
- Animation immediately switches to new speed

**No need to restart animation!**

---

## Customization

### Change Orbit Altitude

```python
orbit_altitude = 200  # 200 km instead of 100 km
```

### Change Number of Orbits

```python
n_orbits = 3  # 3 orbits instead of 2
```

### Change Mission Duration

```python
time_seconds = np.linspace(0, 14400, num_points)  # 4 hours instead of 2
```

### Change Data Points (Smoothness)

```python
x, y, z, vx, vy, vz, altitude, time_elapsed = generate_lunar_orbit_trajectory(1000)
```
More points = smoother but slower

### Change Spacecraft Color

```python
spacecraft, = ax.plot([], [], [], 'o', markersize=15, color='red', ...)
```

### Change Trail Color

```python
trail_line, = ax.plot([], [], [], linewidth=3, color='yellow', ...)
```

### Change Velocity Arrow Length

```python
arrow_length = 500  # 500 km instead of 300 km
```

### Make Moon Transparent

```python
ax.plot_surface(moon_x, moon_y, moon_z, color='gray', alpha=0.3, shade=True)
```
Lower alpha = more transparent

### Add Green-to-Red Gradient on Trail

Currently trail is solid cyan. To add gradient like Plotly version:

```python
# In animate function, instead of solid color:
from matplotlib.collections import Line3DCollection

segments = np.array([[trail_x[i:i+2], trail_y[i:i+2], trail_z[i:i+2]] 
                     for i in range(len(trail_x)-1)])
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(segments)))
lc = Line3DCollection(segments, colors=colors, linewidths=3)
ax.add_collection(lc)
```

---

## Troubleshooting

### Problem: Window doesn't open

**Solution:**
- Make sure matplotlib is installed
- Check if running in SSH/headless environment
- Try: `export MPLBACKEND=TkAgg` before running

### Problem: Animation is choppy

**Solutions:**
1. Reduce data points:
   ```python
   generate_lunar_orbit_trajectory(num_points=200)
   ```

2. Increase interval (slower but smoother):
   ```python
   interval=150
   ```

3. Disable blitting:
   ```python
   blit=False
   ```

### Problem: Moon doesn't show

**Check:**
- Make sure using 3D projection: `projection='3d'`
- Verify `from mpl_toolkits.mplot3d import Axes3D` is imported
- Check axis limits include Moon radius

### Problem: Speed keys don't work

**Solution:**
- Click on the plot window to give it focus
- Some backends don't support key events well
- Try different backend: `matplotlib.use('TkAgg')`

### Problem: Can't rotate during animation

**This should work in Matplotlib!**

If it doesn't:
- Click window to ensure focus
- Try different matplotlib backend
- Update matplotlib: `pip install --upgrade matplotlib`

### Problem: Trail doesn't show

**Check:**
- Trail starts empty, builds during animation
- Press SPACE to start animation
- Verify trail_line is being updated in animate()

### Problem: Warnings about divide by zero

**These are harmless** - Matplotlib 3D sphere rendering quirks
- Doesn't affect functionality
- Can be ignored
- Or suppress with: `import warnings; warnings.filterwarnings('ignore')`

---

## Advantages vs Plotly Version

**Matplotlib Pros:**
- Native window (no browser)
- Can rotate DURING playback
- Simpler installation
- Works offline
- Faster startup

**Matplotlib Cons:**
- Less polished visuals
- No clickable buttons
- Can't save as interactive HTML
- Keyboard-only speed control

**When to use Matplotlib:**
- Quick analysis
- Don't want browser
- Need to rotate while animating
- Simpler environment

**When to use Plotly:**
- Sharing with others
- Better visuals
- Web deployment
- Prefer mouse controls

---

## Performance Tips

### For Slow Computers

1. Reduce points:
   ```python
   num_points=250
   ```

2. Simplify Moon:
   ```python
   u = np.linspace(0, 2 * np.pi, 30)
   v = np.linspace(0, np.pi, 20)
   ```

3. Remove velocity arrow:
   - Comment out lines 122-123 and 213-225

4. Thinner trail:
   ```python
   linewidth=1
   ```

---

## Summary

This Matplotlib version provides a fast, interactive 3D orbital visualization with the unique ability to rotate the view during animation playback. It uses standard orbital mechanics equations and realistic Moon dimensions to create an educational and visually accurate representation of lunar orbital flight.

Perfect for local analysis, presentations, and situations where browser-based tools aren't ideal.

