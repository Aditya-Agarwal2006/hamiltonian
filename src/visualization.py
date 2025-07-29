# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_speed_profile(track_data, optimal_speeds):
    """
    Visualizes the track and the optimal speed profile.
    """
    if optimal_speeds is None:
        print("Cannot plot, optimization failed.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Track layout with speed as color
    points = track_data[:, :2]
    speeds_kmh = optimal_speeds * 3.6
    scatter = ax1.scatter(points[:, 0], points[:, 1], c=speeds_kmh, cmap='plasma', s=40)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Speed (km/h)')
    ax1.set_title('Optimal Racing Line Speed Profile')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)

    # Plot 2: Speed vs. Distance
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0) # Add start point
    ax2.plot(distances, speeds_kmh, marker='o', linestyle='-', markersize=3)
    ax2.set_title('Speed vs. Distance Along Track')
    ax2.set_xlabel('Distance (meters)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()