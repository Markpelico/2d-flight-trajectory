# User Guide: Orbital Trajectory Animation (Plotly)

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Code Structure](#code-structure)
5. [Detailed Explanation](#detailed-explanation)
6. [Controls and Features](#controls-and-features)
7. [Customization Guide](#customization-guide)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This tool creates an interactive 3D visualization of a spacecraft orbiting the Moon. It features:
- Realistic lunar orbital mechanics
- Animated spacecraft with direction indicators
- Interactive playback controls with variable speed
- Live telemetry display
- Green-to-red gradient showing time progression

**Key Concepts:**
- **Lunar Orbit**: Spacecraft circles the Moon at 100 km altitude
- **Animation**: Frame-by-frame playback showing orbital motion
- **Direction Indicator**: Red arrow shows spacecraft velocity vector
- **Trail**: Cyan line builds up showing path already traveled

---

## Installation

### Requirements
- Python 3.7 or newer
- pip (package installer)
- Web browser (Chrome, Firefox, Safari, Edge)

### Install Required Packages

```bash
pip3 install plotly pandas numpy
```

**Package purposes:**
- `plotly`: Interactive 3D visualization
- `pandas`: Data organization and management
- `numpy`: Mathematical calculations

### Verify Installation

```bash
python3 -c "import plotly; import pandas; import numpy; print('Success!')"
```

If you see "Success!" you're ready to go.

---

## Quick Start

### Run the Script

```bash
cd /Users/bigboi2/Desktop/2DPython
python3 trajectory_orbital_animation.py
```

### What Happens
1. Script generates orbital trajectory data (500 points)
2. Browser opens automatically with 3D visualization
3. Animation starts paused - click "Play" to begin
4. Spacecraft orbits the Moon with growing trail

---

## Code Structure

The code is organized into these sections:

```
1. Imports and Configuration (lines 1-30)
2. Data Generation Functions (lines 32-80)
3. Visualization Functions (lines 82-430)
4. Main Execution (lines 432-end)
```

### File Organization

```
trajectory_orbital_animation.py
├── Configuration (constants and settings)
├── generate_lunar_orbit_trajectory() - Creates orbital data
├── calculate_velocity_direction() - Computes spacecraft heading
├── create_orbital_visualization() - Builds 3D plot
└── main() - Entry point
```

---

## Detailed Explanation

### Section 1: Configuration (Lines 14-28)

```python
NESC_VAR_NAMES = [
    "miPosition_m_X", "miPosition_m_Y", "miPosition_m_Z",
    "miVelocity_m_s_X", "miVelocity_m_s_Y", "miVelocity_m_s_Z"
]

NESC_TIME_VAR = 'j2000UtcTime_s'
MOON_RADIUS_M = 1737400
```

**What it does:**
- Defines standard NASA variable names for position and velocity
- Sets Moon radius to actual value (1,737.4 km)
- Uses J2000 time standard (seconds since Jan 1, 2000 12:00 UTC)

**Why we need it:**
- Maintains compatibility with NASA data formats
- Using actual Moon radius makes visualization realistic
- Centralized constants make code easy to modify

---

### Section 2: Trajectory Generation (Lines 32-80)

#### Function: `generate_lunar_orbit_trajectory(num_points=500)`

**Purpose:** Generate realistic orbital data for spacecraft circling the Moon

#### Time Array (Lines 51-52)

```python
time_seconds = np.linspace(0, 7200, num_points)
t_norm = np.linspace(0, 1, num_points)
```

**What it does:**
- Creates 500 time points over 7200 seconds (2 hours)
- Normalized time (0 to 1) used for calculations

**Why we need it:**
- 2 hours is realistic for one lunar orbit
- Even spacing ensures smooth animation
- Normalized time simplifies mathematical formulas

---

#### Orbital Parameters (Lines 54-57)

```python
orbit_altitude = 100000  # 100 km altitude above moon surface
orbit_radius = MOON_RADIUS_M + orbit_altitude

n_orbits = 2
theta = 2 * np.pi * n_orbits * t_norm
```

**What it does:**
- Sets orbit height at 100 km (typical for lunar missions)
- Calculates total orbit radius: Moon radius + altitude
- Spacecraft completes 2 full orbits
- `theta` = angle around orbit (0 to 4π radians for 2 orbits)

**Real-world comparison:**
- Apollo missions: ~100-300 km altitude
- Lunar Reconnaissance Orbiter: ~50 km
- Our simulation: 100 km (realistic)

---

#### Elliptical Orbit (Lines 59-65)

```python
eccentricity = 0.05
r = orbit_radius * (1 - eccentricity * np.cos(theta))

x = r * np.cos(theta)
y = r * np.sin(theta)
```

**What it does:**
- Creates slight ellipse (eccentricity = 0.05, not perfect circle)
- Radius varies slightly as spacecraft orbits
- Converts polar coordinates (r, theta) to Cartesian (x, y)

**Physics explanation:**
- Real orbits are elliptical (Kepler's laws)
- Eccentricity = 0: perfect circle
- Eccentricity = 0.05: very slight ellipse (realistic)
- Formula: r = a(1 - e·cos(θ)) is standard orbital mechanics

---

#### Orbital Inclination (Lines 67-70)

```python
inclination = np.radians(15)  # 15 degree inclination
z = r * np.sin(theta) * np.sin(inclination)
y = r * np.sin(theta) * np.cos(inclination)
```

**What it does:**
- Tilts orbit 15 degrees from horizontal plane
- Adds Z-component (vertical variation)
- Adjusts Y-component for 3D geometry

**Why we need it:**
- Real lunar orbits are rarely perfectly equatorial
- 15° inclination is common for lunar missions
- Makes visualization more interesting in 3D

---

#### Perturbations (Lines 72-75)

```python
x += np.random.normal(0, 100, num_points)
y += np.random.normal(0, 100, num_points)
z += np.random.normal(0, 50, num_points)
```

**What it does:**
- Adds small random variations (~100 meters)
- Simulates gravitational irregularities in Moon's mass distribution

**Why we need it:**
- Moon's gravity isn't perfectly uniform (mascons)
- Real spacecraft paths have small perturbations
- Makes trajectory look more realistic

---

#### Velocity Calculation (Lines 77-80)

```python
vx = np.gradient(x, time_seconds)
vy = np.gradient(y, time_seconds)
vz = np.gradient(z, time_seconds)
```

**What it does:**
- Calculates velocity from position changes
- `np.gradient()` computes derivative (rate of change)
- Units: meters/second

**Formula:**
- Velocity = change in position / change in time
- `gradient()` handles this automatically for arrays

---

### Section 3: Visualization (Lines 82-430)

#### Calculate Direction Vectors (Lines 82-95)

```python
def calculate_velocity_direction(vx, vy, vz):
    magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    magnitude[magnitude == 0] = 1
    
    return vx / magnitude, vy / magnitude, vz / magnitude
```

**What it does:**
- Converts velocity to unit vector (length = 1)
- Points in direction spacecraft is traveling
- Avoids division by zero

**Why we need it:**
- Unit vectors show direction without magnitude
- Used to orient spacecraft cone and velocity arrow
- Ensures consistent visual scale

---

#### Moon Sphere Creation (Lines 109-123)

```python
moon_theta = np.linspace(0, 2*np.pi, 50)
moon_phi = np.linspace(0, np.pi, 40)
moon_theta, moon_phi = np.meshgrid(moon_theta, moon_phi)

moon_x = MOON_RADIUS_M * np.sin(moon_phi) * np.cos(moon_theta)
moon_y = MOON_RADIUS_M * np.sin(moon_phi) * np.sin(moon_theta)
moon_z = MOON_RADIUS_M * np.cos(moon_phi)
```

**What it does:**
- Creates spherical mesh grid for Moon
- Uses spherical coordinates (theta, phi)
- Converts to Cartesian coordinates (x, y, z)

**Technical details:**
- `theta`: Longitude (0 to 2π)
- `phi`: Latitude (0 to π)
- `meshgrid`: Creates 2D grid from 1D arrays
- Standard sphere equations from spherical geometry

**Why 50x40 points:**
- Enough detail to look smooth
- Not so many that it slows rendering
- Good balance between quality and performance

---

#### Adding Moon to Plot (Lines 115-122)

```python
fig.add_trace(go.Surface(
    x=moon_x, y=moon_y, z=moon_z,
    colorscale='gray',
    showscale=False,
    name='Moon',
    hoverinfo='skip',
    opacity=0.8
))
```

**What it does:**
- Creates 3D surface object for Moon
- Colors it gray
- Makes it slightly transparent (80% opaque)

**Parameters:**
- `colorscale='gray'`: Uniform gray color
- `showscale=False`: No colorbar (not needed)
- `opacity=0.8`: Slight transparency to see orbit behind it
- `hoverinfo='skip'`: No tooltip when hovering over Moon

---

#### Orbital Trail (Lines 124-138)

```python
fig.add_trace(go.Scatter3d(
    x=[x[0]], y=[y[0]], z=[z[0]],
    mode='lines',
    name='Orbital Trail',
    line=dict(color='cyan', width=6),
    hovertemplate='...'
))
```

**What it does:**
- Creates trail line (starts with just first point)
- Will grow during animation to show path traveled

**Why start with one point:**
- Animation will update this to show progressive trail
- Starts empty so trail builds from beginning
- Creates visual sense of motion and progress

---

#### Spacecraft Cone (Lines 140-152)

```python
fig.add_trace(go.Cone(
    x=[x[0]], y=[y[0]], z=[z[0]],
    u=[dir_x[0] * 200000], 
    v=[dir_y[0] * 200000], 
    w=[dir_z[0] * 200000],
    colorscale=[[0, 'orange'], [1, 'darkorange']],
    showscale=False,
    sizemode='absolute',
    sizeref=300000,
    name='Spacecraft',
    hovertext='Spacecraft Position'
))
```

**What it does:**
- Creates 3D cone shape representing spacecraft
- Points in direction of travel
- Orange color for visibility

**Parameters explained:**
- `x, y, z`: Position of cone base
- `u, v, w`: Direction vector (where cone points)
  - Multiplied by 200000 to set cone length
- `colorscale`: Orange gradient
- `sizemode='absolute'`: Size in data units
- `sizeref=300000`: Controls cone size (300 km visual size)

**Why cone:**
- Clearly shows which way spacecraft is facing
- Automatically rotates with velocity direction
- Better than sphere (directional information)

---

#### Velocity Arrow (Lines 154-164)

```python
arrow_scale = 400000
fig.add_trace(go.Scatter3d(
    x=[x[0], x[0] + dir_x[0] * arrow_scale],
    y=[y[0], y[0] + dir_y[0] * arrow_scale],
    z=[z[0], z[0] + dir_z[0] * arrow_scale],
    mode='lines',
    line=dict(color='red', width=8),
    hoverinfo='skip',
    showlegend=False
))
```

**What it does:**
- Creates red arrow showing velocity direction
- Extends 400 km from spacecraft
- Updates each frame to show current heading

**Why separate from cone:**
- Provides additional visual clarity
- Red color contrasts with orange spacecraft
- Can be made longer without distorting spacecraft

---

#### Animation Frames (Lines 166-240)

**Frame Loop (Lines 167-240):**

For each frame, the code:

1. **Calculates telemetry** (Lines 168-171)
```python
current_time = float(df[NESC_TIME_VAR].iloc[i])
velocity_mag = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
altitude = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) - MOON_RADIUS_M
```

- Gets current simulation time
- Computes velocity magnitude
- Calculates altitude above Moon surface

2. **Updates Moon** (stays same every frame)

3. **Updates Trail** (Lines 183-199)
```python
go.Scatter3d(
    x=x[:i+1], y=y[:i+1], z=z[:i+1],
    mode='lines',
    line=dict(
        color=time_normalized[:i+1],
        colorscale='RdYlGn_r',
        width=6,
        colorbar=dict(...)
    )
)
```

**Key feature:** Trail grows each frame
- Frame 0: Just starting point
- Frame 100: First 100 points
- Frame 500: Complete orbit

**Color gradient:**
- Early points: Green
- Middle points: Yellow
- Late points: Red

4. **Updates Spacecraft** (Lines 200-208)
- Moves to current position
- Rotates to face current direction

5. **Updates Velocity Arrow** (Lines 209-218)
- Points in current direction of travel

6. **Updates Telemetry** (Lines 221-237)
- Shows current time, altitude, frame number

---

### Playback Controls (Lines 244-340)

#### Speed Buttons (Lines 248-299)

```python
dict(
    label='0.5x',
    method='animate',
    args=[None, {
        'frame': {'duration': 200, 'redraw': True},
        'fromcurrent': True,
        'mode': 'immediate',
        'transition': {'duration': 0}
    }]
)
```

**Speed settings:**
- **0.5x**: 200ms per frame (slow)
- **1.0x**: 100ms per frame (normal)
- **1.5x**: 67ms per frame (fast)
- **2.0x**: 50ms per frame (very fast)

**Parameters:**
- `duration`: Milliseconds between frames
- `redraw: True`: Update display each frame
- `fromcurrent: True`: Continue from current position
- `transition: 0`: Instant frame changes (no fade)

---

#### Play/Pause/Reset (Lines 300-340)

**Play Button:**
- Starts animation from current frame
- Uses 1.0x speed (100ms per frame)

**Pause Button:**
- Stops animation immediately
- Maintains current frame position

**Reset Button:**
- Returns to frame 0 (start of orbit)
- Does not auto-play

---

### Timeline Slider (Lines 342-368)

```python
sliders=[{
    'active': 0,
    'currentvalue': {
        'prefix': 'Position: ',
        'visible': True
    },
    'steps': [...]
}]
```

**What it does:**
- Shows progress through animation (0% to 100%)
- Click or drag to jump to any point
- Displays current percentage

**How to use:**
- Drag slider to scrub through orbit manually
- Click anywhere on slider to jump to that point
- Use while paused for precise positioning

---

### Scene Configuration (Lines 370-414)

#### Background and Grid (Lines 370-389)

```python
scene=dict(
    xaxis_title='X Position (m)',
    yaxis_title='Y Position (m)',
    zaxis_title='Z Position (m)',
    bgcolor='white',
    xaxis=dict(
        backgroundcolor='rgb(245, 245, 245)', 
        gridcolor='rgb(200, 200, 200)',
        showbackground=True
    ),
    ...
)
```

**Color scheme:**
- Main background: White
- Axis backgrounds: Light gray (245, 245, 245)
- Grid lines: Medium gray (200, 200, 200)

**Why these colors:**
- High contrast for readability
- Numbers and labels clearly visible
- Professional technical appearance

---

#### Camera Position (Lines 390-394)

```python
camera=dict(
    eye=dict(x=2.5, y=2.5, z=1.5),
    center=dict(x=0, y=0, z=0)
)
```

**What it does:**
- Sets initial viewing angle
- Camera positioned above and to the side
- Looks at origin (Moon center)

**Coordinates:**
- `eye`: Camera location (2.5, 2.5, 1.5) in normalized units
- `center`: Point camera looks at (0, 0, 0)
- Result: Angled view showing full orbit

---

## Controls and Features

### Interactive Controls

**Mouse Controls:**
- **Left Click + Drag**: Rotate view in 3D
- **Scroll Wheel**: Zoom in/out
- **Shift + Drag**: Pan camera (move view)
- **Hover**: View data at any point on trajectory

**Animation Controls:**
- **Play**: Start animation from current position
- **Pause**: Stop animation, maintain current frame
- **Reset**: Return to beginning (frame 0)

**Speed Controls:**
- **0.5x**: Half speed (good for detailed observation)
- **1.0x**: Normal speed (default)
- **1.5x**: 50% faster
- **2.0x**: Double speed (quick overview)

**Timeline Slider:**
- Drag to scrub through animation manually
- Shows percentage progress (0% to 100%)
- Works independently of play/pause

---

### Telemetry Display

Located in **top-left corner**, shows real-time data:

```
TELEMETRY
Time: 1457.3 s
Altitude: 175.88 km
Frame: 102/500
```

**Data explained:**
- **Time**: Mission elapsed time in seconds
- **Altitude**: Height above Moon surface in kilometers
- **Frame**: Current frame / total frames

---

### Visual Elements

**Gray Sphere**: The Moon
- Actual size (1,737 km radius)
- Gray color for realism
- Semi-transparent to see orbit behind

**Cyan Trail**: Path traveled so far
- Builds up during animation
- Green→Yellow→Red gradient (time progression)
- Width: 6 pixels

**Orange Cone**: Spacecraft
- Points in direction of travel
- Automatically rotates with velocity
- Visual size: ~300 km (exaggerated for visibility)

**Red Arrow**: Velocity direction
- Extends 400 km from spacecraft
- Shows heading explicitly
- Width: 8 pixels

**Colorbar**: Right side
- Shows time progression
- "Start" (green) to "End" (red)
- Matches trail gradient

---

## Customization Guide

### Change Number of Data Points

**More points (smoother orbit, slower):**
```python
df = generate_lunar_orbit_trajectory(num_points=1000)
```

**Fewer points (faster, choppier):**
```python
df = generate_lunar_orbit_trajectory(num_points=200)
```

---

### Change Orbit Characteristics

**Higher altitude:**
```python
orbit_altitude = 200000  # 200 km instead of 100 km
```

**More circular (less elliptical):**
```python
eccentricity = 0.01  # Nearly circular
```

**More elliptical:**
```python
eccentricity = 0.15  # More elongated
```

**Different inclination:**
```python
inclination = np.radians(30)  # 30 degree tilt
```

**More orbits:**
```python
n_orbits = 3  # 3 complete orbits instead of 2
```

---

### Change Colors

**Different trail color:**
```python
line=dict(color='magenta', width=6)  # Magenta instead of cyan
```

**Different spacecraft:**
```python
colorscale=[[0, 'blue'], [1, 'navy']]  # Blue instead of orange
```

**Different gradient:**
```python
colorscale='Viridis'  # Purple-green-yellow
colorscale='Plasma'   # Purple-pink-yellow
colorscale='Inferno'  # Black-purple-yellow
```

---

### Adjust Animation Speed

**In the code:**
```python
# Make default speed slower
'frame': {'duration': 150, 'redraw': True}  # Was 100

# Make default speed faster  
'frame': {'duration': 50, 'redraw': True}   # Was 100
```

**Add new speed button:**
```python
dict(
    label='3.0x',
    method='animate',
    args=[None, {
        'frame': {'duration': 33, 'redraw': True},
        'fromcurrent': True,
        'mode': 'immediate',
        'transition': {'duration': 0}
    }]
)
```

---

### Change Spacecraft Size

**Make it bigger:**
```python
sizeref=500000  # Was 300000
```

**Make it smaller:**
```python
sizeref=150000  # Was 300000
```

---

### Change Velocity Arrow Length

**Make it longer:**
```python
arrow_scale = 600000  # Was 400000
```

**Make it shorter:**
```python
arrow_scale = 200000  # Was 400000
```

---

### Modify Telemetry Display

**Add velocity back:**
```python
text=f'<b>TELEMETRY</b><br>Time: {current_time:.1f} s<br>Altitude: {altitude/1000:.2f} km<br>Velocity: {velocity_mag:.2f} m/s<br>Frame: {i+1}/{len(x)}'
```

**Add distance traveled:**
```python
distance = np.sum(np.sqrt(np.diff(x[:i+1])**2 + np.diff(y[:i+1])**2 + np.diff(z[:i+1])**2))
text=f'<b>TELEMETRY</b><br>Time: {current_time:.1f} s<br>Altitude: {altitude/1000:.2f} km<br>Distance: {distance/1000:.1f} km<br>Frame: {i+1}/{len(x)}'
```

**Move telemetry box:**
```python
x=0.98,  # Right side instead of left
xanchor='right',
```

---

### Change Mission Duration

**Longer mission:**
```python
time_seconds = np.linspace(0, 14400, num_points)  # 4 hours instead of 2
```

**Shorter mission:**
```python
time_seconds = np.linspace(0, 3600, num_points)  # 1 hour
```

---

## Troubleshooting

### Problem: Script won't run

**Error**: "No module named 'plotly'"

**Solution:**
```bash
pip3 install plotly pandas numpy
```

---

### Problem: Browser doesn't open automatically

**Solutions:**
1. Manually navigate to `http://127.0.0.1:8050` (or check console for URL)
2. Try different browser
3. Check if firewall is blocking

**Console shows:** `Opening in browser...` but nothing happens?
- Copy the URL from console and paste in browser manually

---

### Problem: Animation is too slow

**Solutions:**
1. Reduce number of points:
   ```python
   df = generate_lunar_orbit_trajectory(num_points=200)
   ```

2. Use 2.0x speed button

3. Increase default speed:
   ```python
   'frame': {'duration': 50, ...}  # Faster default
   ```

---

### Problem: Animation is choppy

**Solutions:**
1. Close other browser tabs
2. Reduce points (fewer frames)
3. Simplify by removing velocity arrow:
   - Comment out lines 154-164 and 209-218

---

### Problem: Can't rotate view while playing

**This is normal** - Plotly locks camera during animation for performance

**Solutions:**
- Click "Pause" first, then rotate
- Use slider to scrub manually while rotating
- This is a Plotly limitation, not a bug

---

### Problem: Spacecraft too small/hard to see

**Solutions:**
```python
sizeref=600000  # Make it 2x bigger (was 300000)
```

Or change color to brighter:
```python
colorscale=[[0, 'yellow'], [1, 'gold']]  # Bright yellow
```

---

### Problem: Trail obscures trajectory

**Solution - Make trail thinner:**
```python
line=dict(color='cyan', width=3)  # Was 6
```

**Solution - Make trail transparent:**
```python
line=dict(color='rgba(0, 255, 255, 0.5)', width=6)  # 50% transparent cyan
```

---

### Problem: Moon blocking view of orbit

**Solutions:**
1. Manually rotate view after opening
2. Change initial camera angle:
   ```python
   eye=dict(x=3, y=3, z=2)  # Further back
   ```
3. Make Moon more transparent:
   ```python
   opacity=0.5  # Was 0.8
   ```

---

### Problem: Numbers on axes hard to read

**Solution - Already using white background for readability**

If still difficult:
```python
scene=dict(
    ...
    xaxis=dict(
        backgroundcolor='white',
        gridcolor='black',  # Darker grid
        ...
    )
)
```

---

### Problem: Want to save animation

**As HTML (interactive):**
```python
fig.write_html("lunar_orbit_animation.html")
```

**As static image (requires kaleido):**
```bash
pip3 install kaleido
```
```python
fig.write_image("lunar_orbit.png")
```

**As video (requires external tools):**
- Use screen recording software
- Or export frames individually and compile

---

## Advanced Modifications

### Add Earth Reference

```python
# After creating Moon, add Earth at distance
earth_distance = 384400000  # 384,400 km to Earth
earth = pv.Sphere(radius=6371000, center=[earth_distance, 0, 0])
fig.add_trace(go.Surface(
    x=earth_sphere_x, y=earth_sphere_y, z=earth_sphere_z,
    colorscale='blues',
    showscale=False,
    name='Earth',
    opacity=0.6
))
```

---

### Add Multiple Spacecraft

```python
# In generate function, create two trajectories
# with different inclinations or altitudes
df_a = generate_lunar_orbit_trajectory(...)
df_b = generate_lunar_orbit_trajectory(...)

# Add both to visualization with different colors
```

---

### Add Grid Lines on Moon Surface

```python
fig.add_trace(go.Surface(
    x=moon_x, y=moon_y, z=moon_z,
    colorscale='gray',
    showscale=False,
    opacity=0.8,
    contours={
        "x": {"show": True, "color": "white"},
        "y": {"show": True, "color": "white"},
        "z": {"show": True, "color": "white"}
    }
))
```

---

### Add Orbital Period Calculation

```python
# After velocity calculation
orbital_period = time_seconds[-1] / n_orbits
print(f"Orbital period: {orbital_period:.1f} seconds ({orbital_period/60:.1f} minutes)")
```

---

## Understanding the Physics

### Orbital Mechanics

**Circular orbit velocity:**
\[ v = \sqrt{\frac{GM}{r}} \]

Where:
- G = gravitational constant
- M = Moon mass
- r = orbital radius

**For 100 km lunar orbit:**
- Velocity ≈ 1,633 m/s
- Period ≈ 118 minutes

**Our simulation:**
- Uses simplified Keplerian orbit
- Adds realistic perturbations
- Close to actual lunar mission profiles

---

### Coordinate System

**ECI (Earth-Centered Inertial) Frame:**
- Origin at Moon center (for lunar orbit)
- X, Y, Z in meters
- Right-handed coordinate system

**Position vector:** (x, y, z)
**Velocity vector:** (vx, vy, vz)

---

### Why Green-to-Red Gradient

**Purpose:** Immediately show time progression

- Green: Fresh/start
- Yellow: Middle
- Red: End/completion

This is standard in:
- Mission control displays
- Scientific visualizations
- Data analysis tools

---

## Performance Tips

### For Large Datasets (>1000 points)

1. **Downsample for visualization:**
```python
df_plot = df.iloc[::10, :]  # Use every 10th point
```

2. **Reduce animation frames:**
```python
frames = frames[::2]  # Skip every other frame
```

3. **Simplify geometry:**
- Lower Moon resolution (30x20 instead of 50x40)
- Remove velocity arrow
- Use simpler colorscale

---

### Optimize Browser Performance

1. Close other tabs
2. Use Chrome (generally fastest for WebGL)
3. Disable browser extensions temporarily
4. Update graphics drivers

---

## Summary

This tool provides a professional, interactive visualization of lunar orbital mechanics:

**Data Generation:**
- Realistic Keplerian orbit equations
- Actual Moon dimensions
- Physical perturbations

**Visualization:**
- Hardware-accelerated 3D rendering
- Intuitive controls and playback
- Live telemetry display
- Direction indicators

**Use Cases:**
- Mission planning demonstrations
- Educational orbital mechanics
- Trajectory comparison baseline
- Technical presentations

The code is structured for easy modification and extension. Experiment with different parameters to create custom orbital scenarios!

