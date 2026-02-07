import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from src.fit_drag_model import rk4_step
import os

# --- 1. Physics Engine Wrapper ---
def run_simulation(v0, angle, k_eff, dt=0.002):
    theta = np.radians(angle)
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)]) # x, y, vx, vy
    
    trajectory = []
    
    while state[1] >= 0:
        # Save state: x, y, velocity_magnitude, drag_force_proxy (v^2)
        v_mag = np.sqrt(state[2]**2 + state[3]**2)
        drag_force = k_eff * (v_mag**2) # Proportional to Force
        
        trajectory.append([state[0], state[1], v_mag, drag_force])
        state = rk4_step(state, dt, k_eff)
        
    return np.array(trajectory)

def create_pro_dashboard():
    print("🚀 Initializing Physics Engine...")
    
    # Constants
    k_rough = 0.0291  # High Drag
    k_split = 0.0134  # Low Drag
    v0 = 4.46
    angle = 45
    
    # Run Data
    data_rough = run_simulation(v0, angle, k_rough)
    data_split = run_simulation(v0, angle, k_split)
    
    # --- 2. Setup "Dashboard" Layout ---
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) # Top plot taller
    
    # Top Plot: Trajectory
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 2.2)
    ax1.set_ylim(0, 1.0)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylabel("Altitude (m)", fontsize=10)
    ax1.set_title("Real-Time Flight Telemetry: Roughness vs Splitter Plate", fontsize=12, pad=10)

    # Bottom Plot: Velocity Decay
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 2.2)
    ax2.set_ylim(2.0, 5.0)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlabel("Distance Downrange (m)", fontsize=10)
    ax2.set_ylabel("Velocity (m/s)", fontsize=10)
    
    # --- 3. Graphics Objects ---
    # Trajectory Lines
    trail_rough, = ax1.plot([], [], 'r--', linewidth=1.5, alpha=0.5, label='L/D=0.75 (High Drag)')
    head_rough,  = ax1.plot([], [], 'ro', markersize=8, markeredgecolor='black')
    
    trail_split, = ax1.plot([], [], 'b-', linewidth=2.0, alpha=0.8, label='L/D=1.00 (Low Drag)')
    head_split,  = ax1.plot([], [], 'bo', markersize=8, markeredgecolor='black')
    
    # Velocity Lines (Bottom Plot)
    vel_rough, = ax2.plot([], [], 'r--', linewidth=1.5, alpha=0.5)
    vel_split, = ax2.plot([], [], 'b-', linewidth=2.0, alpha=0.8)
    
    # HUD Text (Heads-Up Display)
    hud_template = (
        "LIVE TELEMETRY\n"
        "----------------\n"
        "Config:   {name}\n"
        "Alt:      {alt:.2f} m\n"
        "Vel:      {vel:.2f} m/s\n"
        "Drag F:   {drag:.3f} N"
    )
    hud_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, 
                        fontsize=9, family='monospace', verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    ax1.legend(loc='upper right', framealpha=1.0)

    # --- 4. Animation Update Loop ---
    def update(frame):
        # Speed multiplier
        idx = frame * 6 
        
        # --- Update Red (Roughness) ---
        if idx < len(data_rough):
            # Trajectory
            trail_rough.set_data(data_rough[:idx, 0], data_rough[:idx, 1])
            head_rough.set_data([data_rough[idx, 0]], [data_rough[idx, 1]])
            # Velocity
            vel_rough.set_data(data_rough[:idx, 0], data_rough[:idx, 2])
            
            # Save latest stats for HUD
            r_stats = data_rough[idx]
        else:
            # Freeze at end
            head_rough.set_data([data_rough[-1, 0]], [data_rough[-1, 1]])
            r_stats = data_rough[-1]

        # --- Update Blue (Splitter) ---
        if idx < len(data_split):
            # Trajectory
            trail_split.set_data(data_split[:idx, 0], data_split[:idx, 1])
            head_split.set_data([data_split[idx, 0]], [data_split[idx, 1]])
            # Velocity
            vel_split.set_data(data_split[:idx, 0], data_split[:idx, 2])
            
            # Save stats
            s_stats = data_split[idx]
        else:
            head_split.set_data([data_split[-1, 0]], [data_split[-1, 1]])
            s_stats = data_split[-1]

        # --- Update HUD (Show Blue stats as it's the "Hero") ---
        # If blue is still flying, show blue stats. If finished, show final.
        hud_text.set_text(hud_template.format(
            name="Splitter (L/D=1.0)",
            alt=s_stats[1],
            vel=s_stats[2],
            drag=s_stats[3]
        ))

        return trail_rough, head_rough, trail_split, head_split, vel_rough, vel_split, hud_text

    # --- 5. Render ---
    print("🎥 Rendering Dashboard Animation...")
    ani = animation.FuncAnimation(fig, update, frames=180, interval=20, blit=True)
    
    os.makedirs('outputs', exist_ok=True)
    ani.save('outputs/pro_telemetry.gif', writer='pillow', fps=30)
    print("✅ Saved to outputs/pro_telemetry.gif")

if __name__ == "__main__":
    create_pro_dashboard()
