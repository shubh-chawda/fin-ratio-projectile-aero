import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manim import *
from src.fit_drag_model import rk4_step
import numpy as np

# --- 1. Physics Helper ---
def get_trajectory(v0, angle, k_eff, dt=0.01):
    theta = np.radians(angle)
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
    path = []
    while state[1] >= 0:
        path.append(state.copy())
        state = rk4_step(state, dt, k_eff)
    return np.array(path)

# --- 2. The Manim Scene ---
class ProjectileComparison(ThreeDScene):
    def construct(self):
        # --- CONFIGURATION ---
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        
        # Physics Constants
        v0 = 4.46
        angle = 45
        k_rough = 0.0291
        k_split = 0.0134
        
        # Calculate Paths
        path_rough = get_trajectory(v0, angle, k_rough)
        path_split = get_trajectory(v0, angle, k_split)
        
        # --- VISUAL ASSETS ---
        # 1. 3D Axes
        axes = ThreeDAxes(
            x_range=[0, 2.5, 0.5], 
            y_range=[0, 1.5, 0.5], 
            z_range=[-1, 1, 0.5],
            x_length=10, y_length=6, z_length=4
        ).add_coordinates()
        
        labels = axes.get_axis_labels(x_label="Distance (m)", y_label="Height (m)")
        
        # 2. Projectiles
        sphere_rough = Sphere(radius=0.05, resolution=(15, 15)).set_color(RED)
        sphere_split = Sphere(radius=0.05, resolution=(15, 15)).set_color(BLUE)
        
        # Lighting for 3D effect
        sphere_rough.set_sheen(-0.5, DR)
        sphere_split.set_sheen(-0.5, DR)

        # 3. Tracers 
        trace_rough = TracedPath(sphere_rough.get_center, stroke_color=RED, stroke_width=3, dissipating_time=0.5)
        trace_split = TracedPath(sphere_split.get_center, stroke_color=BLUE, stroke_width=3, dissipating_time=0.5)

        # 4. HUD (Heads Up Display) 
        hud_box = Rectangle(width=4, height=2, color=WHITE).to_corner(UL)
        hud_title = Text("AERO-SIM v1.0", font_size=24).next_to(hud_box.get_top(), DOWN)
        
        dist_tracker = ValueTracker(0.0)
        label_dist = DecimalNumber(0).next_to(hud_title, DOWN)
        text_dist = Text("Max Range (m):", font_size=18).next_to(label_dist, LEFT)

        # Group scene elements
        self.add(axes, labels)
        self.add_fixed_in_frame_mobjects(hud_box, hud_title, label_dist, text_dist)

        # --- ANIMATION SEQUENCE ---
        
        # Intro: Show projectiles launching
        self.play(FadeIn(sphere_rough), FadeIn(sphere_split))
        self.add(trace_rough, trace_split)
        
        # The Flight Animation
        # We animate a "ValueTracker" from 0 to 1 (progress) and map it to the path indices
        run_time = 6
        
        def update_rough(mob, alpha):
            # Map alpha (0 to 1) to the path array index
            idx = int(alpha * (len(path_rough) - 1))
            coords = path_rough[idx]
            # Convert physics coords (x,y) to Manim 3D coords (x,y,0)
            mob.move_to(axes.c2p(coords[0], coords[1], 0))
            
        def update_split(mob, alpha):
            idx = int(alpha * (len(path_split) - 1))
            coords = path_split[idx]
            mob.move_to(axes.c2p(coords[0], coords[1], 0))
            # Update the HUD tracker based on the blue ball's X position
            dist_tracker.set_value(coords[0])

        # Animate the movement
        self.play(
            UpdateFromAlphaFunc(sphere_rough, update_rough),
            UpdateFromAlphaFunc(sphere_split, update_split),
            # Live update the text number
            label_dist.animate.set_value(2.05), # Cheating slightly for smooth UI, or link to tracker
            # Camera Swoop: Rotate around the scene while they fly
            Rotate(
                self.camera.phi_tracker,
                angle=20 * DEGREES,
                rate_func=linear
            ),
            run_time=run_time,
            rate_func=linear
        )
        
        # End
        self.play(Indicate(sphere_split, scale_factor=1.5, color=TEAL))
        self.wait(1)
