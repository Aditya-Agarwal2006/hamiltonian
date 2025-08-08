# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_speed_profile(track_data, optimal_speeds, outer_points=None, inner_points=None, save_path=None):
    """
    Visualizes the track borders and the optimal speed profile along the racing line
    with improved coordinate handling and display.
    """
    if optimal_speeds is None:
        print("Cannot plot, optimization failed.")
        return

    # Create figure with better layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Track layout with speed as color
    racing_line_points = track_data[:, :2]
    speeds_kmh = optimal_speeds * 3.6
    
    # Plot track borders if provided
    if outer_points is not None and inner_points is not None:
        # Plot outer boundary
        ax1.plot(outer_points[:, 0], outer_points[:, 1], 'k-', linewidth=2, 
                label='Outer Border', alpha=0.8)
        # Plot inner boundary  
        ax1.plot(inner_points[:, 0], inner_points[:, 1], 'k-', linewidth=2, 
                label='Inner Border', alpha=0.8)
        
        # Fill the track area for better visualization
        ax1.fill(outer_points[:, 0], outer_points[:, 1], alpha=0.1, color='gray', label='Track Surface')
        ax1.fill(inner_points[:, 0], inner_points[:, 1], alpha=0.3, color='white')
        
        # Print coordinate ranges for debugging
        print(f"\nTrack coordinate ranges:")
        print(f"  Outer: X=[{np.min(outer_points[:, 0]):.0f}, {np.max(outer_points[:, 0]):.0f}], "
              f"Y=[{np.min(outer_points[:, 1]):.0f}, {np.max(outer_points[:, 1]):.0f}]")
        print(f"  Inner: X=[{np.min(inner_points[:, 0]):.0f}, {np.max(inner_points[:, 0]):.0f}], "
              f"Y=[{np.min(inner_points[:, 1]):.0f}, {np.max(inner_points[:, 1]):.0f}]")
        print(f"  Racing Line: X=[{np.min(racing_line_points[:, 0]):.0f}, {np.max(racing_line_points[:, 0]):.0f}], "
              f"Y=[{np.min(racing_line_points[:, 1]):.0f}, {np.max(racing_line_points[:, 1]):.0f}]")
    
    # Plot racing line with speed as color - use a line plot with color mapping
    # First, plot the racing line as a continuous line
    ax1.plot(racing_line_points[:, 0], racing_line_points[:, 1], 'w-', linewidth=4, alpha=0.8)
    ax1.plot(racing_line_points[:, 0], racing_line_points[:, 1], 'k-', linewidth=2, alpha=0.6)
    
    # Then overlay with colored points for speed visualization
    scatter = ax1.scatter(racing_line_points[:, 0], racing_line_points[:, 1], 
                         c=speeds_kmh, cmap='plasma', s=25, zorder=10, 
                         edgecolors='white', linewidth=0.5, label='Speed Profile')
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Speed (km/h)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Improve plot appearance
    ax1.set_title('Optimal Racing Line Speed Profile', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Add some statistics text
    stats_text = f"Speed Range: {np.min(speeds_kmh):.0f} - {np.max(speeds_kmh):.0f} km/h\n"
    stats_text += f"Avg Speed: {np.mean(speeds_kmh):.0f} km/h\n"
    stats_text += f"Waypoints: {len(racing_line_points)}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Speed vs. Distance with improvements
    # Calculate distances more accurately
    distances = np.zeros(len(racing_line_points))
    for i in range(1, len(racing_line_points)):
        segment_dist = np.sqrt(np.sum((racing_line_points[i] - racing_line_points[i-1])**2))
        distances[i] = distances[i-1] + segment_dist
    
    # Plot speed profile
    ax2.plot(distances, speeds_kmh, 'b-', linewidth=2, alpha=0.8)
    ax2.scatter(distances, speeds_kmh, c=speeds_kmh, cmap='plasma', s=15, 
                edgecolors='darkblue', linewidth=0.3, zorder=5)
    
    # Add some reference lines
    ax2.axhline(y=np.mean(speeds_kmh), color='r', linestyle='--', alpha=0.5, 
                label=f'Average: {np.mean(speeds_kmh):.0f} km/h')
    ax2.axhline(y=np.max(speeds_kmh), color='g', linestyle='--', alpha=0.5, 
                label=f'Maximum: {np.max(speeds_kmh):.0f} km/h')
    ax2.axhline(y=np.min(speeds_kmh), color='orange', linestyle='--', alpha=0.5, 
                label=f'Minimum: {np.min(speeds_kmh):.0f} km/h')
    
    ax2.set_title('Speed vs. Distance Along Racing Line', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance (meters)', fontsize=12)
    ax2.set_ylabel('Speed (km/h)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    # Set reasonable y-axis limits
    speed_margin = (np.max(speeds_kmh) - np.min(speeds_kmh)) * 0.1
    ax2.set_ylim(np.min(speeds_kmh) - speed_margin, np.max(speeds_kmh) + speed_margin)
    
    # Add total distance info
    total_distance = distances[-1]
    distance_text = f"Track Length: {total_distance:.0f} m ({total_distance/1000:.2f} km)"
    ax2.text(0.02, 0.98, distance_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"High-resolution plot saved to: {save_path}")
    
    plt.show()

def plot_track_analysis(track_data, outer_points=None, inner_points=None, save_path=None):
    """
    Additional function to analyze and visualize track properties like curvature.
    Useful for debugging racing line generation.
    """
    racing_line_points = track_data[:, :2]
    curvatures = track_data[:, 2]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Track with curvature visualization
    if outer_points is not None and inner_points is not None:
        ax1.plot(outer_points[:, 0], outer_points[:, 1], 'k-', linewidth=2, alpha=0.6)
        ax1.plot(inner_points[:, 0], inner_points[:, 1], 'k-', linewidth=2, alpha=0.6)
        ax1.fill(outer_points[:, 0], outer_points[:, 1], alpha=0.1, color='gray')
        ax1.fill(inner_points[:, 0], inner_points[:, 1], alpha=0.3, color='white')
    
    # Color racing line by curvature
    scatter = ax1.scatter(racing_line_points[:, 0], racing_line_points[:, 1], 
                         c=curvatures, cmap='hot', s=30, zorder=10)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Curvature (1/m)')
    
    ax1.set_title('Racing Line Colored by Curvature')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Curvature vs distance
    distances = np.zeros(len(racing_line_points))
    for i in range(1, len(racing_line_points)):
        segment_dist = np.sqrt(np.sum((racing_line_points[i] - racing_line_points[i-1])**2))
        distances[i] = distances[i-1] + segment_dist
    
    ax2.plot(distances, curvatures, 'r-', linewidth=2)
    ax2.set_title('Curvature vs. Distance Along Racing Line')
    ax2.set_xlabel('Distance (meters)')
    ax2.set_ylabel('Curvature (1/m)')
    ax2.grid(True, alpha=0.3)
    
    # Add curvature statistics
    curvature_stats = f"Curvature Stats:\n"
    curvature_stats += f"Max: {np.max(curvatures):.4f} 1/m\n"
    curvature_stats += f"Mean: {np.mean(curvatures):.4f} 1/m\n"
    curvature_stats += f"Std: {np.std(curvatures):.4f} 1/m"
    ax2.text(0.02, 0.98, curvature_stats, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is not None:
        base_path = save_path.replace('.png', '_analysis.png')
        plt.savefig(base_path, dpi=300, bbox_inches='tight')
        print(f"Track analysis saved to: {base_path}")
    
    plt.show()