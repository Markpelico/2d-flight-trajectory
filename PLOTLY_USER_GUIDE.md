Title: Lunar Orbit Animation Tool - Plotly Version

Author: Refactored trajectory visualization tool

Date Created: November 2025

Last Updated: November 2025

Description: Interactive 3D spacecraft orbital trajectory animation using Plotly. Visualizes lunar orbit with realistic orbital mechanics. Runs in web browser with clickable controls, speed adjustment, and scrubber slider. Intended for trajectory analysis, presentations, and web-based mission visualization.

##### USER GUIDE #####

Library Dependencies:

	* python3
	* numpy
	* pandas
	* plotly

Installation:

	pip install numpy pandas plotly

Quick Start:

	1) Run the script: python3 trajectory_orbital_animation.py

	2) Browser opens automatically showing 3D visualization

	3) Click "Play" button to start animation

	4) Use speed buttons (0.5x, 1.0x, 1.5x, 2.0x) to adjust playback

	5) Drag slider to scrub through trajectory manually

	6) Click and drag plot to rotate (pause first for smooth control)

Controls:

	Buttons:
		Play - Start animation from current position
		Pause - Stop animation
		Reset - Return to beginning
		0.5x - Half speed (200ms per frame)
		1.0x - Normal speed (100ms per frame)
		1.5x - Fast speed (67ms per frame)
		2.0x - Very fast (50ms per frame)

	Mouse:
		Left Click + Drag - Rotate 3D view (pause first recommended)
		Scroll - Zoom in/out
		Shift + Drag - Pan view
		Hover - View data at point

	Slider:
		Drag to scrub through animation manually
		Shows percentage progress

Visual Elements:

	* Gray sphere: Moon (radius = 1737.4 km)
	* Orange cone: Spacecraft (points in direction of travel)
	* Cyan trail: Path traveled (green to red gradient showing time)
	* Red line: Velocity direction vector (400 km length)
	* Text box: Telemetry (Time, Altitude, Frame number)
	* Colorbar: Time progression indicator

##### CODE STRUCTURE #####

Key Functions:

	generate_lunar_orbit_trajectory(num_points=500)
		- Generates orbital trajectory data
		- Returns DataFrame with position, velocity, time
		- Uses elliptical orbit equations
		- Adds gravitational perturbations

	calculate_velocity_direction(vx, vy, vz)
		- Normalizes velocity to unit vector
		- Used for spacecraft orientation
		- Returns direction components

	create_orbital_visualization(df)
		- Creates Plotly figure with Moon, trajectory, spacecraft
		- Builds animation frames
		- Configures controls and layout
		- Returns complete figure object

	main()
		- Entry point
		- Generates data
		- Creates visualization
		- Opens browser

Orbital Parameters:

	* Altitude: 100 km above Moon surface
	* Orbit radius: 1837.4 km
	* Eccentricity: 0.05 (slightly elliptical)
	* Inclination: 15 degrees
	* Period: ~2 hours
	* Number of orbits: 2 complete revolutions
	* Data points: 500

Data Format:

	DataFrame columns match NESC naming convention:
		- miPosition_m_X, miPosition_m_Y, miPosition_m_Z (meters)
		- miVelocity_m_s_X, miVelocity_m_s_Y, miVelocity_m_s_Z (m/s)
		- j2000UtcTime_s (seconds)

##### CUSTOMIZATION #####

Modify Orbit Parameters (in generate_lunar_orbit_trajectory):

	orbit_altitude = 200  # 200 km altitude
	n_orbits = 3  # 3 complete orbits
	time_seconds = np.linspace(0, 14400, num_points)  # 4 hour mission
	eccentricity = 0.1  # more elliptical
	inclination = np.radians(30)  # 30 degree tilt

Modify Visual Elements:

	Spacecraft size (line 161):
		sizeref=500000
	
	Spacecraft color (line 158):
		colorscale=[[0, 'blue'], [1, 'navy']]
	
	Trail color (line 151):
		line=dict(color='yellow', width=8)
	
	Velocity arrow length (line 167):
		arrow_scale = 600000
	
	Moon transparency (line 127):
		opacity=0.5

Modify Background (line 374):

	bgcolor='lightgray'  # or 'white', 'black'
	
	# For axis backgrounds (lines 376-388):
	backgroundcolor='rgb(250, 250, 250)'
	gridcolor='rgb(150, 150, 150)'

Modify Data Points (line 476):

	df = generate_lunar_orbit_trajectory(num_points=1000)

Add Velocity to Telemetry (line 224):

	text=f'<b>TELEMETRY</b><br>Time: {current_time:.1f} s<br>Altitude: {altitude/1000:.2f} km<br>Velocity: {velocity_mag:.2f} m/s<br>Frame: {i+1}/{len(x)}'

Save as HTML (add to main function):

	fig.write_html("lunar_orbit.html")

##### KNOWN ISSUES #####

Limitations:

	* Cannot rotate view smoothly during playback - Plotly limitation
	* Requires browser - may not work in headless environments
	* Slower initial load compared to Matplotlib version
	* Performance degrades with >1000 points

Workarounds:

	Cannot rotate during playback:
		- Pause animation first
		- Use slider to scrub manually while rotating
		- This is inherent to Plotly's WebGL rendering

	Browser doesn't open:
		- Check console for URL (127.0.0.1:XXXXX)
		- Copy URL to browser manually
		- Check firewall settings

	Animation slow:
		- Reduce num_points to 250
		- Close other browser tabs
		- Use 2.0x speed button

	Spacecraft not visible:
		- Increase sizeref value
		- Change to brighter color
		- Adjust camera angle manually

##### TECHNICAL NOTES #####

Coordinate System:
	* Origin: Moon center
	* Units: Meters (matches NESC convention)
	* Reference frame: Inertial (moon-centered)

Animation Implementation:
	* Uses Plotly's go.Frame system
	* Each frame contains complete scene state
	* Duration parameter controls playback speed
	* redraw=True ensures proper updates

Trajectory Equations:

	Elliptical orbit:
		r = a(1 - e·cos(θ))
		x = r·cos(θ)
		y = r·sin(θ)·cos(i)
		z = r·sin(θ)·sin(i)

	Where:
		a = semi-major axis (orbit radius)
		e = eccentricity
		θ = angular position
		i = inclination angle

	Velocity:
		v = dr/dt (computed via np.gradient)

Browser Compatibility:

	Tested on Chrome, Firefox, Safari, Edge
	Requires WebGL support
	Works on desktop and mobile browsers

##### COMPARISON TO MATPLOTLIB VERSION #####

Plotly Advantages:
	* Better visual quality (WebGL rendering)
	* Clickable buttons for all controls
	* Scrubber slider for precise positioning
	* Can save as interactive HTML file
	* Smoother rendering

Plotly Disadvantages:
	* Requires browser
	* Cannot rotate smoothly during playback
	* More dependencies (pandas, plotly)
	* Slower initial startup

Use Plotly when:
	* Presenting to others
	* Need to share interactive visualization
	* Want better visual quality
	* Prefer mouse/button controls over keyboard

##### FUTURE IMPROVEMENTS #####

Potential Additions:

	* Multiple spacecraft comparison
	* Earth reference position
	* Orbital elements display (semi-major axis, eccentricity, etc.)
	* Export animation as video
	* Load trajectory data from CSV files
	* Attitude indicators (pitch, yaw, roll)
	* Toggle trail on/off
	* Adjustable camera presets
