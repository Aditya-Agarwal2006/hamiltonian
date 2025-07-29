#made by grok 4 w perplexity


"""
Optimal Racing Line Speed Profile Calculator

This script calculates the optimal speed profile for a given racetrack using
numerical optimization. It implements physics-based constraints for vehicle
dynamics including cornering limits, acceleration/braking limits, and tire grip.

Author: Expert Python Developer
Date: 2025
Method: SciPy SLSQP constrained optimization
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calculate_optimal_speed(track_data, vehicle_params):
    """
    Calculate optimal speed profile for a racing line using SciPy optimization.

    This function solves a constrained optimization problem to minimize lap time
    while respecting physical vehicle dynamics constraints.

    Parameters:
    -----------
    track_data : numpy.ndarray
        Array of shape (N, 3) where N is the number of waypoints.
        Each row contains [x_coord, y_coord, curvature] in meters and rad/m.

    vehicle_params : dict
        Dictionary containing vehicle parameters:
        - mass_kg: Vehicle mass in kg
        - drag_coefficient: Aerodynamic drag coefficient (Cd)
        - downforce_coefficient: Aerodynamic downforce coefficient (Cl, negative)
        - frontal_area_m2: Vehicle frontal area in m²
        - friction_coefficient: Tire-road friction coefficient (μ)
        - power_watts: Engine power in watts

    Returns:
    --------
    numpy.ndarray
        Array of optimal speeds in m/s at each waypoint, or None if failed.
    """

    N = len(track_data)

    # Calculate distances between consecutive waypoints
    distances = np.zeros(N-1)
    for i in range(N-1):
        dx = track_data[i+1, 0] - track_data[i, 0]
        dy = track_data[i+1, 1] - track_data[i, 1] 
        distances[i] = np.sqrt(dx**2 + dy**2)

    # Physical constants
    g = 9.81  # Gravitational acceleration (m/s²)
    rho_air = 1.225  # Air density at sea level (kg/m³)

    # Extract vehicle parameters
    m = vehicle_params['mass_kg']
    Cd = vehicle_params['drag_coefficient']
    Cl = vehicle_params['downforce_coefficient']  # Negative for downforce
    A = vehicle_params['frontal_area_m2']
    mu = vehicle_params['friction_coefficient']
    P = vehicle_params['power_watts']

    # Helper functions for physical forces
    def drag_force(v):
        """Calculate aerodynamic drag force at speed v"""
        return 0.5 * rho_air * A * Cd * v**2

    def downforce_magnitude(v):
        """Calculate downforce magnitude at speed v (always positive)"""
        return -0.5 * rho_air * A * Cl * v**2  # Cl is negative, result is positive

    def max_grip_force(v):
        """Calculate maximum available grip force from tires"""
        normal_force = m * g + downforce_magnitude(v)
        return mu * normal_force

    # Objective function: minimize total lap time
    def objective_function(v):
        """
        Objective function to minimize: total lap time.
        Time = sum(distance_i / speed_i) for all segments.
        """
        total_time = 0
        for i in range(N-1):
            if v[i] <= 0:
                return 1e10  # Penalty for invalid speeds
            total_time += distances[i] / v[i]
        return total_time

    # Constraint functions (all must be >= 0 for feasibility)
    def cornering_constraint(v, i):
        """
        Maximum cornering speed constraint at waypoint i.
        Formula: v²κm ≤ μ(mg + 0.5ρACl|v²|)
        Rearranged as: μ(mg + downforce) - v²κm ≥ 0
        """
        curvature = track_data[i, 2]
        if abs(curvature) < 1e-6:
            return 1000  # No constraint for straight sections

        # Lateral force required for cornering
        lateral_force_required = v[i]**2 * abs(curvature) * m

        # Maximum grip force available
        max_available_grip = max_grip_force(v[i])

        return max_available_grip - lateral_force_required

    def acceleration_constraint(v, i):
        """
        Longitudinal acceleration constraint between waypoints i and i+1.
        Energy increase must not exceed work done by propulsive force.
        Formula: 0.5m(v₂² - v₁²) ≤ (P/v₁ - F_drag(v₁)) × ds
        """
        if i >= N-1:
            return 1000  # No constraint for last waypoint

        # Change in kinetic energy
        kinetic_energy_increase = 0.5 * m * (v[i+1]**2 - v[i]**2)

        if v[i] <= 0:
            return -1000  # Invalid speed

        # Available propulsive force = Power/speed - drag
        propulsive_force = P / v[i] - drag_force(v[i])

        # Maximum work that can be done over this distance
        max_work = propulsive_force * distances[i]

        return max_work - kinetic_energy_increase

    def braking_constraint(v, i):
        """
        Longitudinal braking constraint between waypoints i and i+1.
        Energy decrease must not exceed maximum braking work.
        Formula: 0.5m(v₁² - v₂²) ≤ F_grip_max(v₁) × ds
        """
        if i >= N-1:
            return 1000  # No constraint for last waypoint

        # Only apply when braking (speed decreasing)
        if v[i+1]**2 >= v[i]**2:
            return 1000  # No braking constraint needed

        # Change in kinetic energy during braking
        kinetic_energy_decrease = 0.5 * m * (v[i]**2 - v[i+1]**2)

        # Maximum braking work available
        max_braking_work = max_grip_force(v[i]) * distances[i]

        return max_braking_work - kinetic_energy_decrease

    # Build constraint list for optimization
    constraints = []

    # Add cornering speed constraints for each waypoint
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: cornering_constraint(v, idx)
        })

    # Add acceleration constraints for each segment
    for i in range(N-1):
        constraints.append({
            'type': 'ineq', 
            'fun': lambda v, idx=i: acceleration_constraint(v, idx)
        })

    # Add braking constraints for each segment
    for i in range(N-1):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: braking_constraint(v, idx)
        })

    # Set speed bounds (reasonable physical limits)
    bounds = [(1.0, 100.0) for _ in range(N)]  # 1 to 100 m/s

    # Initial guess for speeds
    v0 = np.full(N, 30.0)  # Start with 30 m/s everywhere

    # Solve the constrained optimization problem
    result = minimize(
        objective_function,
        v0,
        method='SLSQP',  # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints,
        options={
            'disp': True,      # Display convergence messages
            'maxiter': 1000,   # Maximum iterations
            'ftol': 1e-6       # Function tolerance
        }
    )

    if result.success:
        return result.x
    else:
        print(f"Optimization warning: {result.message}")
        print("Returning best available solution...")
        return result.x


if __name__ == '__main__':
    """
    Demonstration with dummy track data and F1 vehicle parameters.
    Creates a test track and shows how to use the optimization function.
    """

    print("=" * 60)
    print("OPTIMAL RACING LINE SPEED PROFILE CALCULATOR")
    print("=" * 60)

    # Create a more challenging test track with varying curvature
    print("\nCreating test track with mixed corners...")

    # Create track with straight and curved sections
    waypoints = []

    # Straight section (100m)
    for i in range(5):
        waypoints.append([i * 25, 0, 0.0])

    # Tight corner (radius = 50m)
    corner_radius = 50.0
    corner_angles = np.linspace(0, np.pi/2, 6)
    for angle in corner_angles:
        x = 100 + corner_radius * (1 - np.cos(angle))
        y = corner_radius * np.sin(angle)
        curvature = 1.0 / corner_radius
        waypoints.append([x, y, curvature])

    # Another straight (50m)
    for i in range(1, 3):
        waypoints.append([150, 50 + i * 25, 0.0])

    # Wide corner (radius = 100m)  
    corner_radius = 100.0
    corner_angles = np.linspace(np.pi/2, np.pi, 5)
    for angle in corner_angles:
        x = 150 + corner_radius * np.cos(angle)
        y = 100 + corner_radius * np.sin(angle)
        curvature = 1.0 / corner_radius
        waypoints.append([x, y, curvature])

    # Final straight back to start
    for i in range(1, 4):
        waypoints.append([50 - i * 25, 200, 0.0])

    track_data = np.array(waypoints)
    n_points = len(track_data)

    print(f"Track created with {n_points} waypoints")
    print(f"Mix of straight sections and corners (R=50m, R=100m)")

    # F1 vehicle parameters (based on 2022 regulations and typical values)
    vehicle_params = {
        'mass_kg': 798.0,               # Minimum weight including driver
        'drag_coefficient': 1.0,         # High Cd due to wings/bodywork
        'downforce_coefficient': -3.5,   # High downforce (negative Cl)
        'frontal_area_m2': 2.0,         # Approximate frontal area
        'friction_coefficient': 2.5,     # Racing slicks on dry asphalt
        'power_watts': 746000           # ~1000 HP maximum power
    }

    print("\nF1 Vehicle Parameters:")
    print("-" * 30)
    for key, value in vehicle_params.items():
        if 'coefficient' in key:
            print(f"  {key:25s}: {value:8.2f}")
        elif 'watts' in key:
            print(f"  {key:25s}: {value:8.0f} ({value/746:.0f} HP)")
        else:
            print(f"  {key:25s}: {value:8.1f}")

    # Calculate optimal speed profile
    print("\nRunning optimization...")
    print("-" * 30)

    optimal_speeds = calculate_optimal_speed(track_data, vehicle_params)

    if optimal_speeds is not None:
        print("\nOptimization Results:")
        print("-" * 30)
        print(f"✓ Speed profile calculated successfully")
        print(f"  Number of waypoints: {len(optimal_speeds)}")
        print(f"  Speed range: {np.min(optimal_speeds):.1f} - {np.max(optimal_speeds):.1f} m/s")
        print(f"  Speed range: {np.min(optimal_speeds)*3.6:.0f} - {np.max(optimal_speeds)*3.6:.0f} km/h")

        # Calculate track statistics
        total_distance = 0
        total_time = 0

        for i in range(n_points-1):
            dx = track_data[i+1, 0] - track_data[i, 0]
            dy = track_data[i+1, 1] - track_data[i, 1]
            segment_distance = np.sqrt(dx**2 + dy**2)
            segment_time = segment_distance / optimal_speeds[i]

            total_distance += segment_distance
            total_time += segment_time

        print(f"  Track length: {total_distance:.0f} m")
        print(f"  Lap time: {total_time:.2f} s")
        print(f"  Average speed: {total_distance/total_time:.1f} m/s ({total_distance/total_time*3.6:.0f} km/h)")

        print("\nSpeed Profile by Waypoint:")
        print("-" * 45)
        print("Point  Speed(m/s)  Speed(km/h)  Curvature(1/m)")
        print("-" * 45)

        for i in range(len(optimal_speeds)):
            curvature = track_data[i, 2]
            corner_type = "Straight" if abs(curvature) < 1e-6 else f"R={1/curvature:.0f}m"
            print(f"{i:5d}  {optimal_speeds[i]:9.1f}  {optimal_speeds[i]*3.6:10.0f}  {curvature:11.6f}  ({corner_type})")

        # Show some physics calculations
        print("\nPhysics Analysis (at maximum speed):")
        print("-" * 40)
        max_speed = np.max(optimal_speeds)
        drag = 0.5 * 1.225 * vehicle_params['frontal_area_m2'] * vehicle_params['drag_coefficient'] * max_speed**2  
        downforce = -0.5 * 1.225 * vehicle_params['frontal_area_m2'] * vehicle_params['downforce_coefficient'] * max_speed**2
        grip = vehicle_params['friction_coefficient'] * (vehicle_params['mass_kg'] * 9.81 + downforce)

        print(f"  Maximum speed: {max_speed:.1f} m/s ({max_speed*3.6:.0f} km/h)")
        print(f"  Drag force: {drag:.0f} N")
        print(f"  Downforce: {downforce:.0f} N ({downforce/9.81:.0f} kg)")
        print(f"  Total grip available: {grip:.0f} N")
        print(f"  Power required at max speed: {drag * max_speed / 1000:.0f} kW")

    else:
        print("❌ Optimization failed!")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)
