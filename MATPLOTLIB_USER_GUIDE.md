Title: Lunar Orbit Animation Tool - Matplotlib Version

Author: Refactored trajectory visualization tool

Date Created: November 2025

Last Updated: November 2025

Description: 3D spacecraft orbital trajectory animation using Matplotlib. Visualizes lunar orbit with realistic orbital mechanics. Runs in native desktop window with keyboard controls for playback speed. Allows 3D rotation during playback. Intended for trajectory analysis, educational purposes, and mission planning visualization.

##### USER GUIDE #####

Library Dependencies:

	* python3
	* numpy
	* matplotlib

Installation:

	pip install numpy matplotlib

Quick Start:

	1) Run the script: python3 animate_path.py

	2) A window opens showing the Moon (gray sphere) and orbital path

	3) Press SPACE to start animation

	4) Use number keys 1-4 to adjust playback speed

	5) Click and drag to rotate the 3D view (works during playback)

Controls:

	Keyboard:
		SPACE - Pause/Play toggle
		1 - 0.5x speed (slow, 200ms per frame)
		2 - 1.0x speed (normal, 100ms per frame)
		3 - 1.5x speed (fast, 67ms per frame)
		4 - 2.0x speed (very fast, 50ms per frame)

	Mouse:
		Left Click + Drag - Rotate 3D view
		Scroll Wheel - Zoom in/out
		Right Click + Drag - Pan view

Visual Elements:

	* Gray sphere: Moon (radius = 1737.4 km)
	* Orange dot: Spacecraft current position
	* Cyan line: Trail showing path traveled
	* Red line: Velocity direction vector (300 km length)
	* Green dot: Orbit start position
	* Red square: Orbit end position
	* Text box: Telemetry (Time, Altitude, Frame number)

##### CODE STRUCTURE #####

Key Functions:

	generate_lunar_orbit_trajectory(num_points=500)
		- Generates orbital position and velocity data
		- Returns: x, y, z, vx, vy, vz, altitude, time_elapsed
		- Uses Keplerian orbital mechanics
		- Adds realistic perturbations

	init()
		- Initializes animation objects to empty state
		- Called once at animation start

	animate(frame)
		- Updates spacecraft position
		- Grows trail
		- Updates velocity arrow direction
		- Updates telemetry display
		- Called once per frame

	on_key_press(event)
		- Handles keyboard input
		- Controls pause/play and speed adjustment

Orbital Parameters:

	* Altitude: 100 km above Moon surface
	* Orbit radius: 1837.4 km
	* Eccentricity: 0.05 (slightly elliptical)
	* Inclination: 15 degrees
	* Period: ~2 hours
	* Number of orbits: 2 complete revolutions
	* Data points: 500

##### CUSTOMIZATION #####

Modify Orbit Parameters (in generate_lunar_orbit_trajectory function):

	orbit_altitude = 200  # Change to 200 km altitude
	n_orbits = 3  # Show 3 complete orbits
	num_points = 1000  # More points for smoother animation
	eccentricity = 0.1  # More elliptical orbit

Modify Visual Elements:

	Spacecraft color (line 116):
		color='red'
	
	Spacecraft size (line 116):
		markersize=20
	
	Trail color (line 120):
		color='yellow'
	
	Trail thickness (line 120):
		linewidth=5
	
	Velocity arrow length (line 218):
		arrow_length = 500
	
	Moon transparency (line 103):
		alpha=0.4

Modify Animation Speed:

	Default interval (line 247):
		interval=150  # Slower default speed
	
	Add new speed preset (in on_key_press function):
		elif event.key == '5':
			animation_interval = 25
			anim.event_source.interval = 25

##### KNOWN ISSUES #####

Limitations:

	* Divide by zero warnings during rendering - harmless, can be ignored
	* Performance degrades with >1000 points on slower machines
	* Some backends don't support keyboard events well
	* 3D rotation can be slow on older hardware

Troubleshooting:

	Window doesn't open:
		- Verify matplotlib installed correctly
		- Check if running headless
		- Try: export MPLBACKEND=TkAgg

	Animation choppy:
		- Reduce num_points to 250
		- Increase interval value
		- Set blit=False in FuncAnimation

	Speed keys don't work:
		- Click window to focus it
		- Try different matplotlib backend

##### TECHNICAL NOTES #####

Coordinate System:
	* Origin at Moon center
	* Units: kilometers
	* Inertial reference frame

Time Format:
	* J2000 standard (seconds since Jan 1, 2000 12:00 UTC)
	* Variable name: j2000UtcTime_s

Velocity Calculation:
	* Computed using np.gradient() on position data
	* Central differences method for interior points

Altitude Calculation:
	* Radial distance from Moon center minus Moon radius
	* altitude = sqrt(x² + y² + z²) - MOON_RADIUS_KM

##### COMPARISON TO PLOTLY VERSION #####

Matplotlib Advantages:
	* Native desktop window (no browser required)
	* Can rotate view during playback
	* Simpler installation
	* Keyboard-only controls
	* Faster startup

Matplotlib Disadvantages:
	* Less polished visuals
	* No clickable buttons
	* Cannot save as interactive HTML
	* No scrubber slider

Use Matplotlib when:
	* Quick local analysis needed
	* Browser not preferred
	* Need to rotate during animation
	* Simpler environment required
