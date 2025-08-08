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
    that follows a more intelligent path between the inner and outer borders.
    """
    # Load the spline data
    spline_0 = np.load("../trackModels/Silverstone_track_spline_0.npz", allow_pickle=True)
    spline_1 = np.load("../trackModels/Silverstone_track_spline_1.npz", allow_pickle=True)
    
    # The spline data is already in the correct format for splev
    tck_outer = spline_0['tck']  # Use directly - it's already a (3,) array
    tck_inner = spline_1['tck']  # Use directly - it's already a (3,) array
    
    # Generate waypoints along the splines with more points for better resolution
    u = np.linspace(0, 1, 400)  # Increased resolution
    
    # Get outer boundary points
    x_outer, y_outer = splev(u, tck_outer)
    outer_points = np.column_stack([x_outer, y_outer])
    
    # Get inner boundary points  
    x_inner, y_inner = splev(u, tck_inner)
    inner_points = np.column_stack([x_inner, y_inner])
    
    # Flip the track vertically to correct orientation
    outer_points[:, 1] = -outer_points[:, 1]
    inner_points[:, 1] = -inner_points[:, 1]
    
    # Create a proper racing line using a more sophisticated approach
    racing_line_points = np.zeros_like(outer_points)
    
    # First pass: Calculate track properties at each point
    track_widths = np.zeros(len(outer_points))
    centerline_points = np.zeros_like(outer_points)
    
    for i in range(len(outer_points)):
        # Calculate centerline and track width
        centerline_points[i] = (outer_points[i] + inner_points[i]) / 2.0
        track_widths[i] = np.linalg.norm(outer_points[i] - inner_points[i])
    
    # Second pass: Calculate curvature using centerline
    centerline_curvatures = np.zeros(len(centerline_points))
    
    for i in range(2, len(centerline_points) - 2):
        # Use 5-point stencil for better curvature estimation
        p_minus2 = centerline_points[i-2]
        p_minus1 = centerline_points[i-1]
        p_curr = centerline_points[i]
        p_plus1 = centerline_points[i+1]
        p_plus2 = centerline_points[i+2]
        
        # Calculate first and second derivatives
        dx_dt = (-p_plus2[0] + 8*p_plus1[0] - 8*p_minus1[0] + p_minus2[0]) / 12.0
        dy_dt = (-p_plus2[1] + 8*p_plus1[1] - 8*p_minus1[1] + p_minus2[1]) / 12.0
        
        d2x_dt2 = (-p_plus2[0] + 16*p_plus1[0] - 30*p_curr[0] + 16*p_minus1[0] - p_minus2[0]) / 12.0
        d2y_dt2 = (-p_plus2[1] + 16*p_plus1[1] - 30*p_curr[1] + 16*p_minus1[1] - p_minus2[1]) / 12.0
        
        # Calculate curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2)**(3/2)
        
        if denominator > 1e-10:
            centerline_curvatures[i] = numerator / denominator
        else:
            centerline_curvatures[i] = 0
    
    # Smooth the curvature to avoid noise
    window_size = 5
    smoothed_curvatures = np.zeros_like(centerline_curvatures)
    for i in range(len(centerline_curvatures)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(centerline_curvatures), i + window_size//2 + 1)
        smoothed_curvatures[i] = np.mean(centerline_curvatures[start_idx:end_idx])
    
    centerline_curvatures = smoothed_curvatures
    
    # Third pass: Generate racing line based on racing theory
    # Basic principle: late apex for corners, straight lines on straights
    
    for i in range(len(outer_points)):
        # Start with centerline
        base_point = centerline_points[i]
        
        # Calculate direction from inner to outer (positive means toward outer edge)
        track_direction = outer_points[i] - inner_points[i]
        track_direction_norm = track_direction / (np.linalg.norm(track_direction) + 1e-10)
        
        # Determine racing line offset based on curvature and context
        curvature = centerline_curvatures[i]
        
        if curvature < 0.0005:  # Straight section
            # Stay near centerline on straights
            offset_factor = 0.0
        else:
            # For corners, use a more sophisticated approach
            # Look ahead to see if this is entry, apex, or exit
            look_ahead = 20  # points to look ahead/behind
            
            # Calculate average curvature in window
            start_idx = max(0, i - look_ahead)
            end_idx = min(len(centerline_curvatures), i + look_ahead + 1)
            local_curvatures = centerline_curvatures[start_idx:end_idx]
            avg_curvature = np.mean(local_curvatures)
            max_curvature = np.max(local_curvatures)
            
            # Determine position in corner
            curvature_ratio = curvature / (max_curvature + 1e-10)
            
            if curvature_ratio < 0.7:  # Corner entry/exit
                # Move toward outside for entry, inside for exit
                # Simple heuristic: if curvature is increasing, it's entry (go outside)
                # if decreasing, it's exit (go inside)
                if i > 0 and i < len(centerline_curvatures) - 1:
                    curvature_trend = centerline_curvatures[i+1] - centerline_curvatures[i-1]
                    if curvature_trend > 0:  # Curvature increasing (corner entry)
                        offset_factor = 0.3  # Move toward outside
                    else:  # Curvature decreasing (corner exit)
                        offset_factor = -0.2  # Move toward inside
                else:
                    offset_factor = 0.0
            else:  # Near apex
                # Move toward inside for apex
                offset_factor = -0.4 * curvature_ratio
        
        # Apply the offset
        offset_distance = offset_factor * track_widths[i] * 0.4  # Limit maximum offset
        racing_line_points[i] = base_point + track_direction_norm * offset_distance
        
        # Safety check: ensure racing line stays within track bounds
        # Project racing line point onto the line between inner and outer
        inner_to_outer = outer_points[i] - inner_points[i]
        inner_to_racing = racing_line_points[i] - inner_points[i]
        
        # Calculate projection parameter (0 = inner edge, 1 = outer edge)
        if np.dot(inner_to_outer, inner_to_outer) > 1e-10:
            t = np.dot(inner_to_racing, inner_to_outer) / np.dot(inner_to_outer, inner_to_outer)
            t = np.clip(t, 0.1, 0.9)  # Keep within 10% to 90% of track width
            racing_line_points[i] = inner_points[i] + t * inner_to_outer
    
    # Calculate final curvature for the racing line
    racing_line_curvatures = np.zeros(len(racing_line_points))
    
    for i in range(2, len(racing_line_points) - 2):
        # Use the same 5-point stencil method
        p_minus2 = racing_line_points[i-2]
        p_minus1 = racing_line_points[i-1]
        p_curr = racing_line_points[i]
        p_plus1 = racing_line_points[i+1]
        p_plus2 = racing_line_points[i+2]
        
        # Calculate derivatives
        dx_dt = (-p_plus2[0] + 8*p_plus1[0] - 8*p_minus1[0] + p_minus2[0]) / 12.0
        dy_dt = (-p_plus2[1] + 8*p_plus1[1] - 8*p_minus1[1] + p_minus2[1]) / 12.0
        
        d2x_dt2 = (-p_plus2[0] + 16*p_plus1[0] - 30*p_curr[0] + 16*p_minus1[0] - p_minus2[0]) / 12.0
        d2y_dt2 = (-p_plus2[1] + 16*p_plus1[1] - 30*p_curr[1] + 16*p_minus1[1] - p_minus2[1]) / 12.0
        
        # Calculate curvature
        numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2)**(3/2)
        
        if denominator > 1e-10:
            racing_line_curvatures[i] = numerator / denominator
        else:
            racing_line_curvatures[i] = 0
    
    # Smooth the final racing line curvatures
    window_size = 3
    smoothed_racing_curvatures = np.zeros_like(racing_line_curvatures)
    for i in range(len(racing_line_curvatures)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(racing_line_curvatures), i + window_size//2 + 1)
        smoothed_racing_curvatures[i] = np.mean(racing_line_curvatures[start_idx:end_idx])
    
    racing_line_curvatures = smoothed_racing_curvatures
    
    # Create track_data array: [x, y, curvature] for the racing line
    track_data = np.column_stack([racing_line_points, racing_line_curvatures])
    
    return track_data, outer_points, inner_points

def main():
    """
    Main function to run the racing line optimization.
    """
    start_time = datetime.now()
    print(f"--- Racing Line Optimization Project ---")
    print(f"Program started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 2. Load Existing Digitized Track ---
    print("\n[Step 1] Loading existing Silverstone track data...")
    track_data, outer_points, inner_points = load_silverstone_track()
    print(f"Track loaded with {len(track_data)} waypoints.")
    
    # Print some diagnostics
    curvatures = track_data[:, 2]
    print(f"Curvature range: {np.min(curvatures):.6f} to {np.max(curvatures):.6f}")
    print(f"Number of high-curvature points (>0.001): {np.sum(curvatures > 0.001)}")

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
        
        # Print some speed statistics
        print(f"Speed range: {np.min(optimal_speeds):.1f} - {np.max(optimal_speeds):.1f} m/s")
        print(f"Speed range: {np.min(optimal_speeds)*3.6:.0f} - {np.max(optimal_speeds)*3.6:.0f} km/h")
        
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
    
    end_time = datetime.now()
    print(f"Program finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"Total execution time: {duration}")

if __name__ == '__main__':
    main()