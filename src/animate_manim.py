import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manim import *
from src.fit_drag_model import rk4_step
import numpy as np

# --- 1. Physics Engine ---
def get_trajectory(v0, angle, k_eff, dt=0.01):
    theta = np.radians(angle)
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
    path = []
    while state[1] >= 0:
        path.append(state.copy())
        state = rk4_step(state, dt, k_eff)
    return np.array(path)

# --- 2. The Cinematic Scene ---
class ProjectileComparison(ThreeDScene):
    def construct(self):
        # --- CONFIGURATION ---
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, zoom=0.8)
        
        # Physics Data
        v0 = 4.46
        angle = 45
        path_rough = get_trajectory(v0, angle, 0.0291) # Roughness
        path_split = get_trajectory(v0, angle, 0.0134) # Splitter (Better)

        # --- VISUAL ASSETS ---
        
        # A. The "Tron" Grid Floor
        axes = ThreeDAxes(
            x_range=[0, 2.5, 0.5], 
            y_range=[0, 1.5, 0.5], 
            z_range=[-1, 1, 0.5],
            x_length=10, y_length=6, z_length=4,
            axis_config={"include_tip": True, "stroke_width": 2}
        )
        # Add faint grid lines for depth
        grid = NumberPlane(
            x_range=[0, 2.5, 0.5], 
            y_range=[-1, 1, 0.5], 
            background_line_style={"stroke_color": TEAL, "stroke_width": 1, "stroke_opacity": 0.3}
        ).rotate(90 * DEGREES, axis=RIGHT).shift(DOWN*0.01) # Floor

        # B. Labels (Bold & Clean)
        x_lbl = axes.get_x_axis_label(Text("Distance (m)").scale(0.5), edge=RIGHT, direction=DOWN)
        y_lbl = axes.get_y_axis_label(Text("Height (m)").scale(0.5).rotate(90*DEGREES, RIGHT), edge=UP, direction=LEFT)

        # C. The Projectiles (Glowing Spheres)
        sphere_rough = Sphere(radius=0.06).set_color(RED_E).set_sheen(-0.5, UP)
        sphere_split = Sphere(radius=0.06).set_color(BLUE_E).set_sheen(-0.5, UP)

        # D. The Trails (Thick & Fading)
        trace_rough = TracedPath(sphere_rough.get_center, stroke_color=RED, stroke_width=5, dissipating_time=0.6)
        trace_split = TracedPath(sphere_split.get_center, stroke_color=BLUE, stroke_width=5, dissipating_time=0.6)

        # E. HUD (Glass Panel Look)
        hud_box = Rectangle(width=4.5, height=1.5, color=WHITE, fill_color=BLACK, fill_opacity=0.7).to_corner(UL)
        hud_title = Text("SIMULATION: v1.0", font_size=20, weight=BOLD).move_to(hud_box.get_top()).shift(DOWN*0.3)
        
        # Dynamic Scores
        score_rough = Text("High Drag: 0.00m", font_size=18, color=RED).next_to(hud_title, DOWN, buff=0.15)
        score_split = Text("Low Drag:  0.00m", font_size=18, color=BLUE).next_to(score_rough, DOWN, buff=0.1)

        # --- ANIMATION SEQUENCE ---
        self.add(axes, grid, x_lbl, y_lbl)
        self.add_fixed_in_frame_mobjects(hud_box, hud_title, score_rough, score_split)
        
        self.play(FadeIn(sphere_rough), FadeIn(sphere_split))
        self.add(trace_rough, trace_split)

        # The Flight Logic
        run_time = 7
        
        def update_rough(mob, alpha):
            idx = int(alpha * (len(path_rough) - 1))
            pos = path_rough[idx]
            mob.move_to(axes.c2p(pos[0], pos[1], 0))
            # Update Red Score Text
            new_text = f"High Drag: {pos[0]:.2f}m"
            score_rough.become(Text(new_text, font_size=18, color=RED).move_to(score_rough.get_center()))

        def update_split(mob, alpha):
            idx = int(alpha * (len(path_split) - 1))
            pos = path_split[idx]
            mob.move_to(axes.c2p(pos[0], pos[1], 0))
            # Update Blue Score Text
            new_text = f"Low Drag:  {pos[0]:.2f}m"
            score_split.become(Text(new_text, font_size=18, color=BLUE).move_to(score_split.get_center()))

        # Run Everything
        self.play(
            UpdateFromAlphaFunc(sphere_rough, update_rough),
            UpdateFromAlphaFunc(sphere_split, update_split),
            # Cinematic Camera Rotation
            Rotate(self.camera.phi_tracker, angle=10 * DEGREES, rate_func=linear),
            Rotate(self.camera.theta_tracker, angle=30 * DEGREES, rate_func=linear),
            run_time=run_time,
            rate_func=linear
        )
        
        # Highlight the Winner
        self.play(Indicate(sphere_split, scale_factor=1.2, color=TEAL))
        self.wait(2)
