# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_speed_profile(track_data, optimal_speeds, outer_points=None, inner_points=None, save_path=None):
    """
    Visualizes the track borders and the optimal speed profile along the racing line.
    """
    if optimal_speeds is None:
        print("Cannot plot, optimization failed.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Track layout with speed as color
    racing_line_points = track_data[:, :2]
    speeds_kmh = optimal_speeds * 3.6
    
    # Plot track borders if provided
    if outer_points is not None and inner_points is not None:
        # Plot outer boundary
        ax1.plot(outer_points[:, 0], outer_points[:, 1], 'k-', linewidth=2, label='Outer Border')
        # Plot inner boundary
        ax1.plot(inner_points[:, 0], inner_points[:, 1], 'k-', linewidth=2, label='Inner Border')
        # Fill the track area
        ax1.fill(outer_points[:, 0], outer_points[:, 1], alpha=0.1, color='gray')
        ax1.fill(inner_points[:, 0], inner_points[:, 1], alpha=0.1, color='white')
    
    # Plot racing line with speed as color
    scatter = ax1.scatter(racing_line_points[:, 0], racing_line_points[:, 1], 
                         c=speeds_kmh, cmap='plasma', s=40, zorder=5, label='Racing Line')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Speed (km/h)')
    ax1.set_title('Optimal Racing Line Speed Profile')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Speed vs. Distance
    distances = np.cumsum(np.sqrt(np.sum(np.diff(racing_line_points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0) # Add start point
    ax2.plot(distances, speeds_kmh, marker='o', linestyle='-', markersize=3)
    ax2.set_title('Speed vs. Distance Along Racing Line')
    ax2.set_xlabel('Distance (meters)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.grid(True)

    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()