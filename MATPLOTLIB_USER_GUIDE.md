Title: Lunar Orbit Animation Tool 

Author: Mark Pelico

Date Created: November 2025


Description: 3D spacecraft orbital trajectory animation using Matplotlib. Visualizes lunar orbit, runs in native desktop window with keyboard controls for playback speed. Allows 3D rotation during playback. Intended for trajectory analysis.

##### USER GUIDE #####

Library Dependencies:

	* python3
	* numpy
	* matplotlib



Quick Start:

	1) Run the script: python3.8 animate_path.py

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


##### KNOWN ISSUES #####

Limitations:

	* Divide by zero warnings during rendering - harmless, can be ignored
	* Performance degrades with >1000 points on slower machines
	* Some backends don't support keyboard events well
	* 3D rotation can be slow on older hardware

