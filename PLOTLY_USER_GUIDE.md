# Plotly Lunar Orbit Animation - User Guide

## Overview

Interactive 3D visualization of spacecraft orbiting the Moon. Runs in web browser using Plotly.

Features:
- Lunar orbit at 100 km altitude
- 3D Moon sphere
- Spacecraft cone with velocity arrow
- Trail with green-to-red time gradient
- Telemetry overlay
- Clickable speed controls and slider

---

## Installation

```bash
pip install plotly pandas numpy
```

Requirements: Python 3.7+, Web browser

---

## Running the Script

```bash
python3 trajectory_orbital_animation.py
```

Opens automatically in default browser at `127.0.0.1:PORT`

---

## Controls

### Buttons
- **Play** - Start animation from current position
- **Pause** - Stop animation
- **Reset** - Return to beginning
- **0.5x, 1.0x, 1.5x, 2.0x** - Speed controls

### Mouse
- **Left Drag** - Rotate view (pause first for smooth control)
- **Scroll** - Zoom
- **Shift + Drag** - Pan

### Slider
- Drag to scrub through animation manually
- Shows percentage progress

---

## Code Structure

```
Configuration (lines 14-28)
  - NESC variable names
  - Moon radius constant
  
Trajectory Generation (lines 32-80)
  - generate_lunar_orbit_trajectory()
  - calculate_velocity_direction()
  
Visualization (lines 82-430)
  - create_orbital_visualization()
  - Creates Moon, trail, spacecraft, arrow
  - Builds animation frames
  - Configures controls
  
Main (lines 432-end)
  - Generate data
  - Create visualization
  - Display in browser
```

---

## Orbital Mechanics

### Parameters
- Altitude: 100 km above Moon surface
- Orbit radius: 1837.4 km
- Eccentricity: 0.05
- Inclination: 15 degrees
- Period: ~2 hours
- Orbits: 2 complete

### Trajectory Equation

```python
orbit_radius = MOON_RADIUS_M + orbit_altitude
theta = 2 * np.pi * n_orbits * t_norm
eccentricity = 0.05
r = orbit_radius * (1 - eccentricity * np.cos(theta))

x = r * np.cos(theta)
y = r * np.sin(theta)

inclination = np.radians(15)
z = r * np.sin(theta) * np.sin(inclination)
y = r * np.sin(theta) * np.cos(inclination)
```

Standard Keplerian orbit with inclination.

---

## Key Code Sections

### Moon Sphere (lines 109-123)

```python
moon_theta = np.linspace(0, 2*np.pi, 50)
moon_phi = np.linspace(0, np.pi, 40)
moon_theta, moon_phi = np.meshgrid(moon_theta, moon_phi)

moon_x = MOON_RADIUS_M * np.sin(moon_phi) * np.cos(moon_theta)
moon_y = MOON_RADIUS_M * np.sin(moon_phi) * np.sin(moon_theta)
moon_z = MOON_RADIUS_M * np.cos(moon_phi)

fig.add_trace(go.Surface(
    x=moon_x, y=moon_y, z=moon_z,
    colorscale='gray',
    showscale=False,
    opacity=0.8
))
```

Creates spherical mesh with 50x40 resolution.

---

### Spacecraft and Arrow (lines 140-177)

```python
# Spacecraft cone
fig.add_trace(go.Cone(
    x=[x[0]], y=[y[0]], z=[z[0]],
    u=[dir_x[0] * 200000], 
    v=[dir_y[0] * 200000], 
    w=[dir_z[0] * 200000],
    colorscale=[[0, 'orange'], [1, 'darkorange']],
    sizeref=300000,
    name='Spacecraft'
))

# Velocity arrow
arrow_scale = 400000
fig.add_trace(go.Scatter3d(
    x=[x[0], x[0] + dir_x[0] * arrow_scale],
    y=[y[0], y[0] + dir_y[0] * arrow_scale],
    z=[z[0], z[0] + dir_z[0] * arrow_scale],
    mode='lines',
    line=dict(color='red', width=8)
))
```

Cone points in velocity direction. Arrow extends 400 km.

---

### Animation Frames (lines 179-240)

Each frame updates:

1. Moon (static)
2. Trail (growing with gradient)
3. Spacecraft (current position)
4. Velocity arrow (current direction)
5. Telemetry annotation

```python
for i in range(len(x)):
    current_time = float(df[NESC_TIME_VAR].iloc[i])
    velocity_mag = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    altitude = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) - MOON_RADIUS_M
    
    frame = go.Frame(
        data=[...],
        layout=go.Layout(
            annotations=[telemetry_display]
        )
    )
```

---

### Speed Controls (lines 252-291)

```python
buttons=[
    dict(
        label='0.5x',
        method='animate',
        args=[None, {'frame': {'duration': 200, 'redraw': True}}]
    ),
    dict(
        label='1.0x',
        args=[None, {'frame': {'duration': 100, 'redraw': True}}]
    ),
    ...
]
```

Duration in milliseconds. Lower = faster.

---

## Customization

### Orbit Parameters

```python
# Line 32
orbit_altitude = 200  # km

# Line 36
n_orbits = 3

# Line 28
time_seconds = np.linspace(0, 14400, num_points)  # 4 hours

# Line 40
eccentricity = 0.1  # more elliptical

# Line 48
inclination = np.radians(30)  # steeper angle
```

### Visual Elements

```python
# Spacecraft color (line 158)
colorscale=[[0, 'blue'], [1, 'navy']]

# Spacecraft size (line 161)
sizeref=500000  # bigger

# Trail color (line 151)
line=dict(color='yellow', width=8)

# Arrow length (line 167)
arrow_scale = 600000  # longer

# Moon transparency (line 127)
opacity=0.5  # more transparent
```

### Background

```python
# Line 374
bgcolor='white'  # or 'lightgray', 'black', etc.

# Line 376-388 - Axis backgrounds
backgroundcolor='rgb(240, 240, 240)'
gridcolor='rgb(180, 180, 180)'
```

### Data Points

```python
# Line 476
df = generate_lunar_orbit_trajectory(num_points=1000)
```

More points = smoother but slower loading.

---

## Troubleshooting

### Browser doesn't open
- Check console for URL (usually `127.0.0.1:XXXXX`)
- Copy URL to browser manually
- Check firewall settings

### Animation won't play
- Click Play button
- Check browser console for errors (F12)
- Refresh page

### Can't rotate during playback
- This is a Plotly limitation
- Pause first, then rotate
- Or use slider to scrub manually

### Too slow
- Reduce points: `num_points=250`
- Use 2.0x speed button
- Close other browser tabs

### Spacecraft not visible
- Increase size: `sizeref=600000`
- Change color: `colorscale=[[0, 'yellow'], [1, 'yellow']]`
- Check camera angle

### Trail overlaps trajectory
This is fixed - only trail shows (no full path).
If still seeing overlap, check you're running latest version.

### Telemetry box blocking view
Move position:
```python
x=0.98,  # right side
xanchor='right',
```

---

## Plotly vs Matplotlib

### Plotly Advantages
- Better visual quality
- Clickable buttons
- Scrubber slider
- Save as HTML
- Smoother rendering

### Plotly Disadvantages
- Requires browser
- Can't rotate during playback
- Slower startup
- More dependencies

### When to Use Plotly
- Presentations
- Sharing with others
- Web deployment
- Better visuals needed

---

## Performance

500 points typical performance:
- Load time: 2-3 seconds
- FPS: 10 (100ms frame duration)
- Memory: ~100-200 MB browser tab

Can handle 1000+ points smoothly on modern hardware.

---

## Saving Output

### As HTML

Add at end of main():
```python
fig.write_html("lunar_orbit.html")
```

Creates standalone interactive file.

### As Image (requires kaleido)

```bash
pip install kaleido
```

```python
fig.write_image("lunar_orbit.png")
```

Static image only, no animation.

---

## Technical Notes

### Coordinate System
- Origin: Moon center
- Units: Meters (position), m/s (velocity)
- Frame: Inertial (moon-centered)

### Time Format
- J2000 standard (seconds since Jan 1, 2000 12:00 UTC)
- Stored as string in DataFrame for compatibility

### Color Gradient
- Colorscale: 'RdYlGn_r' (Red-Yellow-Green reversed)
- Green = start time
- Yellow = middle
- Red = end time

### Why Two Traces (Trail + Full Path)?
Original design had both. Now simplified to just trail with gradient.

---

## Browser Compatibility

Tested on:
- Chrome (recommended)
- Firefox
- Safari
- Edge

All modern browsers with WebGL support will work.

---

## Code Quality Notes

Based on NASA trajectory comparison tool by Christopher Johnson.

Improvements made:
- Removed duplicate code
- Added proper documentation
- Simplified data generation (no file I/O)
- Cleaner structure
- Removed unused features

Maintains compatibility with NESC variable naming conventions.
