import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- FIX: Import the function that actually exists ---
from src.fit_drag_model import rk4_step

# --- Helper function to generate the full path ---
def get_trajectory_path(v0, angle, k_eff, dt=0.005):
    # Convert angle to radians
    theta = np.radians(angle)
    
    # Initial state: [x, y, vx, vy]
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    state = np.array([0.0, 0.0, vx, vy])
    
    trajectory = [state.copy()]
    
    # Loop until the projectile hits the ground (y < 0)
    while state[1] >= 0:
        state = rk4_step(state, dt, k_eff)
        trajectory.append(state.copy())
        
    return np.array(trajectory)

def create_comparison_gif():
    # 1. Setup Simulation Parameters
    print("🚀 Setting up simulation...")
    k_roughness = 0.0291  # L/D = 0.75 (High Drag)
    k_splitter  = 0.0134  # L/D = 1.00 (Low Drag)
    
    v0 = 4.46  # Launch speed (m/s)
    angle = 45 # Launch angle (deg)

    # 2. Run Simulations using our new helper
    traj_rough = get_trajectory_path(v0, angle, k_roughness)
    traj_split = get_trajectory_path(v0, angle, k_splitter)
    print(f"✅ Simulations complete. Points: {len(traj_rough)} (Red) vs {len(traj_split)} (Blue)")

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Aerodynamic Regime Shift: Roughness (0.75) vs Splitter Plate (1.00)")

    # Lines and Dots
    line1, = ax.plot([], [], 'r--', label='L/D = 0.75 (Roughness)', alpha=0.6)
    dot1,  = ax.plot([], [], 'ro', markersize=8)
    
    line2, = ax.plot([], [], 'b-', label='L/D = 1.00 (Splitter Plate)', alpha=0.6)
    dot2,  = ax.plot([], [], 'bo', markersize=8)

    # Text annotations
    text1 = ax.text(0.05, 0.9, '', transform=ax.transAxes, color='red')
    text2 = ax.text(0.05, 0.85, '', transform=ax.transAxes, color='blue')

    ax.legend(loc='upper right')

    # 4. Animation Function
    def update(frame):
        # Speed up x5 for smooth viewing
        idx = frame * 5
        
        # Update Roughness Trajectory (Red)
        if idx < len(traj_rough):
            line1.set_data(traj_rough[:idx, 0], traj_rough[:idx, 1])
            dot1.set_data([traj_rough[idx, 0]], [traj_rough[idx, 1]])
            text1.set_text(f'0.75 Range: {traj_rough[idx, 0]:.2f} m')
        else:
            line1.set_data(traj_rough[:, 0], traj_rough[:, 1])
            dot1.set_data([traj_rough[-1, 0]], [traj_rough[-1, 1]])
            text1.set_text(f'0.75 Range: {traj_rough[-1, 0]:.2f} m (High Drag)')

        # Update Splitter Trajectory (Blue)
        if idx < len(traj_split):
            line2.set_data(traj_split[:idx, 0], traj_split[:idx, 1])
            dot2.set_data([traj_split[idx, 0]], [traj_split[idx, 1]])
            text2.set_text(f'1.00 Range: {traj_split[idx, 0]:.2f} m')
        else:
            line2.set_data(traj_split[:, 0], traj_split[:, 1])
            dot2.set_data([traj_split[-1, 0]], [traj_split[-1, 1]])
            text2.set_text(f'1.00 Range: {traj_split[-1, 0]:.2f} m (Drag Reduced!)')

        return line1, dot1, line2, dot2, text1, text2

    # 5. Save
    output_path = 'outputs/regime_shift_demo.gif'
    print(f"🎥 Generating animation frames...")
    
    # Ensure directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=True)
    ani.save(output_path, writer='pillow', fps=30)
    print(f"✨ Success! Animation saved to: {output_path}")

if __name__ == "__main__":
    create_comparison_gif()
