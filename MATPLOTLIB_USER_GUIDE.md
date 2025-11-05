# Matplotlib Lunar Orbit Animation - User Guide

## Overview

3D animation of a spacecraft orbiting the Moon. Runs in a desktop window using Matplotlib.

Features:
- Lunar orbit at 100 km altitude
- 3D Moon sphere (1737 km radius)
- Spacecraft marker with velocity arrow
- Trail showing path traveled
- Telemetry display
- Speed controls via keyboard

---

## Installation

```bash
pip install numpy matplotlib
```

Requirements: Python 3.7+

---

## Running the Script

```bash
python3 animate_path.py
```

Press SPACE to start the animation.

---

## Controls

### Keyboard
- **SPACE** - Pause/Play
- **1** - 0.5x speed (200ms/frame)
- **2** - 1.0x speed (100ms/frame)
- **3** - 1.5x speed (67ms/frame)
- **4** - 2.0x speed (50ms/frame)

### Mouse
- **Left Drag** - Rotate view (works during playback)
- **Scroll** - Zoom
- **Right Drag** - Pan

---

## Code Structure

```
Configuration (lines 7-11)
  - MOON_RADIUS_KM constant
  
Trajectory Generation (lines 17-65)
  - generate_lunar_orbit_trajectory()
  - Returns: x, y, z, vx, vy, vz, altitude, time
  
Plot Setup (lines 70-90)
  - Create 3D axes
  - Set axis limits
  
Moon Sphere (lines 96-103)
  - Create mesh grid
  - Plot as surface
  
Animated Objects (lines 115-137)
  - Spacecraft marker
  - Trail line
  - Velocity arrow
  - Telemetry text
  
Controls (lines 145-178)
  - on_key_press() function
  - Handles SPACE and 1-4 keys
  
Animation (lines 184-236)
  - init() - Initialize objects
  - animate() - Update each frame
  
Execution (lines 242-250)
  - Create animation
  - Display window
```

---

## Orbital Mechanics

### Parameters
- Orbit altitude: 100 km
- Orbit radius: 1837.4 km (Moon radius + altitude)
- Eccentricity: 0.05 (slightly elliptical)
- Inclination: 15 degrees
- Period: 2 hours
- Orbits shown: 2 complete

### Equations Used

Position (elliptical orbit):
```
r = a(1 - e·cos(θ))
x = r·cos(θ)
y = r·sin(θ)·cos(i)
z = r·sin(θ)·sin(i)
```

Where:
- a = semi-major axis
- e = eccentricity
- θ = true anomaly
- i = inclination

Velocity:
```
v = dr/dt
```
Calculated using np.gradient()

---

## Key Code Sections

### Trajectory Generation (line 17)

```python
def generate_lunar_orbit_trajectory(num_points=500):
    time_seconds = np.linspace(0, 7200, num_points)
    t_norm = np.linspace(0, 1, num_points)
    
    orbit_altitude = 100
    orbit_radius = MOON_RADIUS_KM + orbit_altitude
    
    n_orbits = 2
    theta = 2 * np.pi * n_orbits * t_norm
    
    eccentricity = 0.05
    r = orbit_radius * (1 - eccentricity * np.cos(theta))
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    inclination = np.radians(15)
    z = r * np.sin(theta) * np.sin(inclination)
    y = r * np.sin(theta) * np.cos(inclination)
```

Generates 500 points over 2 hours for 2 complete orbits.

---

### Animation Update (line 195)

```python
def animate(frame):
    current_x = x[frame]
    current_y = y[frame]
    current_z = z[frame]
    
    # Update spacecraft
    spacecraft.set_data([current_x], [current_y])
    spacecraft.set_3d_properties([current_z])
    
    # Update trail
    trail_x.append(current_x)
    trail_y.append(current_y)
    trail_z.append(current_z)
    trail_line.set_data(trail_x, trail_y)
    trail_line.set_3d_properties(trail_z)
    
    # Update velocity arrow
    vel_mag = np.sqrt(vx[frame]**2 + vy[frame]**2 + vz[frame]**2)
    if vel_mag > 0:
        arrow_length = 300
        dir_x = vx[frame] / vel_mag * arrow_length
        dir_y = vy[frame] / vel_mag * arrow_length
        dir_z = vz[frame] / vel_mag * arrow_length
        
        velocity_arrow.set_data([current_x, current_x + dir_x], 
                               [current_y, current_y + dir_y])
        velocity_arrow.set_3d_properties([current_z, current_z + dir_z])
    
    # Update telemetry
    info_text.set_text(
        f'TELEMETRY\n'
        f'Time: {time_elapsed[frame]:.1f} s\n'
        f'Altitude: {altitude[frame]:.2f} km\n'
        f'Frame: {frame+1}/{len(x)}'
    )
```

Called once per frame. Updates all animated elements.

---

## Customization

### Orbit Parameters

Change altitude:
```python
orbit_altitude = 200  # km
```

Change number of orbits:
```python
n_orbits = 3
```

Change inclination:
```python
inclination = np.radians(30)  # degrees
```

Change eccentricity:
```python
eccentricity = 0.1  # more elliptical
```

### Visual Elements

Spacecraft size:
```python
markersize=20  # bigger
```

Trail thickness:
```python
linewidth=5  # thicker
```

Velocity arrow length:
```python
arrow_length = 500  # longer
```

Moon transparency:
```python
alpha=0.4  # more transparent
```

### Animation Speed

Change data points:
```python
num_points=250  # fewer = faster
```

Change default speed:
```python
interval=150  # slower default
```

Add new speed preset:
```python
elif event.key == '5':
    animation_interval = 25
    anim.event_source.interval = 25
```

---

## Troubleshooting

### Window doesn't open
- Install matplotlib: `pip install matplotlib`
- Check for headless environment
- Try: `export MPLBACKEND=TkAgg`

### Animation is slow
- Reduce points: `num_points=250`
- Simplify Moon: Use 30x20 grid instead of 50x40
- Remove velocity arrow (comment out lines 122-123, 213-225)

### Speed keys don't work
- Click window to give it focus
- Check backend supports key events
- Try different backend

### Can't rotate during playback
This should work. If not:
- Update matplotlib: `pip install --upgrade matplotlib`
- Try different backend

### Divide by zero warnings
- Harmless matplotlib 3D rendering warnings
- Ignore or suppress: `warnings.filterwarnings('ignore')`

---

## Matplotlib vs Plotly

### Use Matplotlib when:
- You want a desktop window
- You need to rotate during playback
- You prefer keyboard controls
- You want simpler setup

### Use Plotly when:
- You want better visuals
- You need to share interactive HTML
- You prefer mouse/button controls
- You want a slider for scrubbing

---

## Performance Notes

500 points with 3D rendering:
- Typical FPS: 10-20
- Memory usage: ~50-100 MB
- Can handle 1000+ points on modern hardware

Optimizations applied:
- `blit=True` for faster redraw
- Efficient gradient calculations
- Minimal object recreation

---

## Technical Details

### Coordinate System
- Origin at Moon center
- Units in kilometers
- ECI (inertial) reference frame

### Time Standard
- J2000 epoch (seconds since Jan 1, 2000 12:00 UTC)
- Variable name: j2000UtcTime_s

### Velocity Calculation
```python
vx = np.gradient(x, time_seconds)
```
Uses numpy gradient (central differences for interior points).

### Altitude Calculation
```python
altitude = np.sqrt(x**2 + y**2 + z**2) - MOON_RADIUS_KM
```
Radial distance from Moon center minus Moon radius.
