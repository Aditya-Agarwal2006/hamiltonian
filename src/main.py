# main.py

# --- 1. Imports ---
# Import functions and data from your source modules
from optimalSpeed import calculate_optimal_speed
from vehicle import f1_car_parameters
from visualization import plot_speed_profile
import numpy as np
from scipy.interpolate import splev
import os
from datetime import datetime

def load_silverstone_track():
    """
    Load the existing digitized Silverstone track splines and create a racing line
    that follows the centerline between the inner and outer borders.
    """
    # Load the spline data
    spline_0 = np.load("../trackModels/Silverstone_track_spline_0.npz", allow_pickle=True)
    spline_1 = np.load("../trackModels/Silverstone_track_spline_1.npz", allow_pickle=True)
    
    # The spline data is already in the correct format for splev
    tck_outer = spline_0['tck']  # Use directly - it's already a (3,) array
    tck_inner = spline_1['tck']  # Use directly - it's already a (3,) array
    
    # Generate waypoints along the splines with more points for better resolution
    # Use more points and adaptive sampling for corners
    u = np.linspace(0, 1, 300)  # Increased to 300 points for better resolution
    
    # Get outer boundary points
    x_outer, y_outer = splev(u, tck_outer)
    outer_points = np.column_stack([x_outer, y_outer])
    
    # Get inner boundary points  
    x_inner, y_inner = splev(u, tck_inner)
    inner_points = np.column_stack([x_inner, y_inner])
    
    # Flip the track vertically to correct orientation
    outer_points[:, 1] = -outer_points[:, 1]
    inner_points[:, 1] = -inner_points[:, 1]
    
    # Create a proper racing line that follows the track
    # Use the centerline as the base and then optimize it
    racing_line_points = np.zeros_like(outer_points)
    
    for i in range(len(outer_points)):
        # Calculate centerline point
        centerline_point = (outer_points[i] + inner_points[i]) / 2.0
        
        # Calculate track width at this point
        track_width = np.linalg.norm(outer_points[i] - inner_points[i])
        
        # Calculate curvature at this point using centerline
        if i > 0 and i < len(outer_points) - 1:
            p_prev = centerline_point - (centerline_point - (outer_points[i-1] + inner_points[i-1]) / 2.0)
            p_curr = centerline_point
            p_next = (outer_points[i+1] + inner_points[i+1]) / 2.0 - centerline_point
            
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # Calculate curvature
            cross_mag = abs(v1[0] * v2[1] - v1[1] * v2[0])
            dist1 = np.linalg.norm(v1)
            dist2 = np.linalg.norm(v2)
            
            if dist1 > 0 and dist2 > 0:
                curvature = cross_mag / (dist1 * dist2 * (dist1 + dist2) + 1e-10)
            else:
                curvature = 0
        else:
            curvature = 0
        
        # Create racing line based on curvature
        # For high curvature (tight corners), move toward the inside
        # For low curvature (straights), stay more centered
        curvature_factor = min(abs(curvature) * 50, 0.3)  # Limit the effect
        
        # Calculate direction from inner to outer border
        border_direction = outer_points[i] - inner_points[i]
        border_direction = border_direction / np.linalg.norm(border_direction)
        
        # Adjust racing line position based on curvature
        if abs(curvature) > 0.001:  # If there's significant curvature
            # Move toward the inside of the corner
            racing_line_points[i] = centerline_point - border_direction * track_width * (0.2 + curvature_factor)
        else:
            # For straights, stay more centered
            racing_line_points[i] = centerline_point
    
    # Calculate curvature for the racing line
    curvatures = np.zeros(len(racing_line_points))
    
    # Calculate approximate curvature using finite differences
    for i in range(1, len(racing_line_points) - 1):
        p_prev = racing_line_points[i-1]
        p_curr = racing_line_points[i]
        p_next = racing_line_points[i+1]
        
        # Vectors
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # Cross product magnitude
        cross_mag = abs(v1[0] * v2[1] - v1[1] * v2[0])
        
        # Distance
        dist1 = np.linalg.norm(v1)
        dist2 = np.linalg.norm(v2)
        
        if dist1 > 0 and dist2 > 0:
            # Curvature = cross product / (dist1 * dist2 * (dist1 + dist2))
            curvatures[i] = cross_mag / (dist1 * dist2 * (dist1 + dist2) + 1e-10)
    
    # Create track_data array: [x, y, curvature] for the racing line
    track_data = np.column_stack([racing_line_points, curvatures])
    
    return track_data, outer_points, inner_points

def main():
    """
    Main function to run the racing line optimization.
    """
    print("--- Racing Line Optimization Project ---")

    # --- 2. Load Existing Digitized Track ---
    print("\n[Step 1] Loading existing Silverstone track data...")
    track_data, outer_points, inner_points = load_silverstone_track()
    print(f"Track loaded with {len(track_data)} waypoints.")

    # --- 3. Load Vehicle Parameters ---
    print("\n[Step 2] Loading F1 vehicle parameters...")
    vehicle_params = f1_car_parameters
    print("Vehicle data loaded.")

    # --- 4. Run the Solver ---
    print("\n[Step 3] Running numerical optimizer...")
    optimal_speeds = calculate_optimal_speed(track_data, vehicle_params)

    # --- 5. Visualize the Results ---
    if optimal_speeds is not None:
        print("\n[Step 4] Optimization successful. Visualizing results...")
        # Create an 'outputs' directory if it doesn't exist
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speed_profile_{timestamp}.png"
        filepath = os.path.join(output_dir, filename) 

        plot_speed_profile(track_data, optimal_speeds, outer_points, inner_points, save_path=filepath)
        print(f"Plot saved to {filepath}")
    else:
        print("\n[Step 4] Optimization failed. No results to visualize.")

    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()