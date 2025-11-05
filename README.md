# 2D Flight Trajectory

Interactive trajectory visualization tools for orbital mechanics and spacecraft flight paths.

## Overview

This repository contains two trajectory animation tools:

1. **animate_path.py** - Matplotlib version (native desktop window)
2. **trajectory_orbital_animation.py** - Plotly version (browser-based)

Both scripts simulate a spacecraft orbiting the Moon with realistic orbital mechanics.

## Features

### Common Features (Both Scripts)
- Realistic lunar orbital trajectory (100 km altitude)
- 3D visualization with Moon sphere
- Spacecraft marker with velocity direction indicator
- Growing trail showing path traveled
- Live telemetry display (Time, Altitude, Frame)
- Speed controls (0.5x, 1.0x, 1.5x, 2.0x)
- Pause/Play/Reset functionality

### Matplotlib Version (animate_path.py)
- Native desktop window (no browser)
- Can rotate view WHILE animation is playing
- Keyboard controls (SPACE, 1-4 keys)
- Simple and fast

### Plotly Version (trajectory_orbital_animation.py)
- Opens in web browser
- Better 3D rendering quality
- Clickable buttons for controls
- Interactive slider for scrubbing
- Save as interactive HTML

## Installation

### For Matplotlib Version:
```bash
pip install numpy matplotlib
```

### For Plotly Version:
```bash
pip install numpy pandas plotly
```

## Usage

### Matplotlib Version:
```bash
python animate_path.py
```

**Controls:**
- SPACE: Pause/Play
- 1 Key: 0.5x speed
- 2 Key: 1.0x speed
- 3 Key: 1.5x speed
- 4 Key: 2.0x speed
- Mouse: Drag to rotate (works during playback)

### Plotly Version:
```bash
python trajectory_orbital_animation.py
```

**Controls:**
- Click Play/Pause/Reset buttons
- Click speed buttons (0.5x, 1.0x, 1.5x, 2.0x)
- Drag slider to scrub through animation
- Mouse: Drag to rotate (pause first)

## Mission Profile

Both scripts simulate:
- Orbit altitude: 100 km above Moon surface
- Orbit period: ~2 hours
- Number of orbits: 2 complete orbits
- Inclination: 15 degrees
- Total data points: 500

## Documentation

- **USER_GUIDE.md** - Detailed guide for Matplotlib version
- **PLOTLY_USER_GUIDE.md** - Detailed guide for Plotly version

Both guides include:
- Line-by-line code explanations
- Customization instructions
- Troubleshooting tips
- Physics background

## Technologies

- Python 3.7+
- NumPy: Orbital mechanics calculations
- Matplotlib: Desktop visualization
- Plotly: Web-based interactive visualization
- Pandas: Data organization (Plotly version)

## Author

Markpelico

## License

MIT License

