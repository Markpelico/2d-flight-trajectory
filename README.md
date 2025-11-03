# 2D Flight Trajectory Animation

An interactive matplotlib animation that visualizes a realistic rocket flight trajectory with real-time flight data and controls.

## Features

- 🚀 **Realistic Rocket Trajectory**: Simulates actual rocket flight physics with parabolic arc
- 🎨 **Color Gradient**: Visual progression from green (launch) to red (target/end)
- ⏯️ **Pause/Play Control**: Press SPACE to pause/resume animation
- ℹ️ **Hover Info**: Mouse over any point to see detailed flight data
- 📊 **Live Flight Data**: Real-time display of progress, altitude, velocity, and time
- 🎯 **Flight Parameters**: Altitude (km), velocity (m/s), downrange distance, and time elapsed

## Installation

```bash
pip install numpy matplotlib
```

## Usage

```bash
python animate_path.py
```

## Controls

- **SPACE**: Pause/Play the animation
- **HOVER**: Move mouse over any point for detailed information
- **Zoom/Pan**: Use standard matplotlib controls

## Flight Data

The simulation generates realistic flight parameters:
- **Altitude Range**: 0-50 km
- **Velocity**: 500-7500 m/s (approaching orbital velocity)
- **Flight Time**: 600 seconds (10 minutes)
- **Trajectory**: Parabolic gravity turn typical of orbital launches

## Screenshot

The animation shows:
- Launch point (green marker)
- Target/end point (red star)
- Color-coded trajectory path
- Live flight progress indicator
- Real-time telemetry data

## Technologies

- Python 3.x
- NumPy: Data generation and calculations
- Matplotlib: Plotting and animation
- Matplotlib Animation: Frame-by-frame trajectory animation

## Author

Markpelico

## License

MIT License

