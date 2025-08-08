"""
FastF1 Data Pipeline for Racing Line Optimization
Extracts track geometry, telemetry data, and creates training datasets for PINNs

Requirements:
pip install fastf1 pandas numpy matplotlib seaborn
"""

import fastf1
from fastf1 import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
import json

# Suppress FastF1 warnings for cleaner output
warnings.filterwarnings('ignore')

# Create cache directory and enable FastF1 cache for faster subsequent runs
import os
cache_dir = './fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

@dataclass
class TrackGeometry:
    """Container for track geometry data"""
    name: str
    year: int
    track_bounds: Dict[str, np.ndarray]  # inner/outer boundaries
    racing_lines: Dict[str, np.ndarray]  # different drivers' racing lines
    elevation: Optional[np.ndarray] = None
    sectors: Optional[Dict] = None
    metadata: Optional[Dict] = None

@dataclass
class TelemetryData:
    """Container for processed telemetry data"""
    driver: str
    lap_number: int
    lap_time: float
    positions: np.ndarray  # [N, 2] - X, Y coordinates
    speeds: np.ndarray     # [N] - Speed in km/h
    throttle: np.ndarray   # [N] - Throttle 0-100%
    brake: np.ndarray      # [N] - Brake 0-100%
    gear: np.ndarray       # [N] - Gear number
    rpm: np.ndarray        # [N] - Engine RPM
    drs: np.ndarray        # [N] - DRS status
    steering: Optional[np.ndarray] = None  # [N] - Steering angle
    g_forces: Optional[Dict[str, np.ndarray]] = None  # Lateral/Longitudinal G

class FastF1DataExtractor:
    """Main class for extracting F1 data using FastF1 API"""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Available years and tracks
        self.available_years = list(range(2018, 2024))  # FastF1 coverage
        self.track_mapping = {
            'silverstone': 'British Grand Prix',
            'monza': 'Italian Grand Prix', 
            'spa': 'Belgian Grand Prix',
            'monaco': 'Monaco Grand Prix',
            'interlagos': 'São Paulo Grand Prix',
            'suzuka': 'Japanese Grand Prix',
            'austin': 'United States Grand Prix',
            'melbourne': 'Australian Grand Prix'
        }
    
    def get_session(self, year: int, track: str, session_type: str = 'Q'):
        """
        Load F1 session data
        
        Args:
            year: Season year (2018-2023)
            track: Track name or GP name  
            session_type: 'FP1', 'FP2', 'FP3', 'Q', 'SQ', 'R'
        """
        try:
            # Handle track name mapping
            if track.lower() in self.track_mapping:
                track_name = self.track_mapping[track.lower()]
            else:
                track_name = track
            
            print(f"Loading {year} {track_name} {session_type}...")
            session = fastf1.get_session(year, track_name, session_type)
            session.load(telemetry=True, weather=False, messages=False)
            
            print(f"✓ Loaded session with {len(session.laps)} laps")
            return session
            
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            return None
    
    def extract_track_geometry(self, session) -> TrackGeometry:
        """
        Extract track geometry using the fastest lap of the session.
        This is a more robust method using built-in fastf1 features.
        
        Args:
            session: FastF1 session object
        """
        print("Extracting track geometry...")
        
        # Pick the single fastest lap of the entire session
        fastest_lap = session.laps.pick_fastest()
        
        if fastest_lap is None or pd.isna(fastest_lap.LapTime):
            raise ValueError("No valid fastest lap found in the session.")

        driver = fastest_lap['Driver']
        print(f"Using fastest lap from {driver} ({fastest_lap.LapTime}) for geometry.")

        # --- Get the telemetry for the fastest lap FIRST ---
        telemetry = fastest_lap.get_telemetry()
        if 'X' not in telemetry.columns or 'Y' not in telemetry.columns:
            raise ValueError("Telemetry for fastest lap is missing X/Y coordinates.")
        
        # --- 1. Get official track boundaries using the UTILS function ---
        try:
            # Pass the X and Y coordinates to the utility function
            boundaries_coords = utils.get_track_boundaries(telemetry['X'], telemetry['Y'])
            track_bounds = {
                'inner': boundaries_coords[0],
                'outer': boundaries_coords[1]
            }
            print(f"  ✓ Extracted track boundaries.")
        except Exception as e:
            raise RuntimeError(f"Could not extract track boundaries: {e}")

        # --- 2. Get the racing line from the same telemetry data ---
        racing_line = telemetry[['X', 'Y']].values
        racing_lines = {driver: racing_line}
        print(f"  ✓ Extracted racing line for {driver} with {len(racing_line)} points.")
        
        # --- 3. Assemble the Geometry object ---
        weekend = session.event
        metadata = {
            'track_name': weekend['EventName'],
            'location': weekend['Location'],
            'year': session.date.year,
        }
        
        geometry = TrackGeometry(
            name=weekend['EventName'],
            year=session.date.year,
            track_bounds=track_bounds,
            racing_lines=racing_lines,
            metadata=metadata
        )
        
        print(f"✓ Track geometry extracted successfully.")
        return geometry
    
    def extract_telemetry_data(self, session, max_laps_per_driver: int = 5) -> List[TelemetryData]:
        """
        Extract detailed telemetry data from fastest laps
        
        Args:
            session: FastF1 session object
            max_laps_per_driver: Maximum number of laps per driver to extract
        """
        telemetry_data = []
        
        print(f"Extracting telemetry data (max {max_laps_per_driver} laps per driver)...")
        
        for driver in session.drivers:
            driver_laps = session.laps.pick_driver(driver)
            clean_laps = self._filter_clean_laps(driver_laps)
            
            # Sort by lap time and take the fastest ones
            clean_laps = clean_laps.sort_values('LapTime').head(max_laps_per_driver)
            
            for _, lap in clean_laps.iterrows():
                try:
                    car_data = lap.get_telemetry()
                    
                    if len(car_data) < 50:  # Skip laps with insufficient data
                        continue
                    
                    # Extract basic telemetry
                    positions = car_data[['X', 'Y']].values
                    speeds = car_data['Speed'].values
                    throttle = car_data['Throttle'].values
                    brake = car_data['Brake'].values
                    gear = car_data['nGear'].values
                    rpm = car_data['RPM'].values
                    
                    # Handle optional data
                    drs = car_data.get('DRS', np.zeros(len(car_data)))
                    
                    # Clean data (remove NaN values)
                    valid_idx = ~(np.isnan(positions).any(axis=1) | 
                                 np.isnan(speeds) | 
                                 np.isnan(throttle) | 
                                 np.isnan(brake))
                    
                    if np.sum(valid_idx) < 50:  # Not enough valid data
                        continue
                    
                    # Apply filter
                    positions = positions[valid_idx]
                    speeds = speeds[valid_idx]
                    throttle = throttle[valid_idx]
                    brake = brake[valid_idx]
                    gear = gear[valid_idx]
                    rpm = rpm[valid_idx]
                    drs = drs.values[valid_idx] if hasattr(drs, 'values') else drs[valid_idx]
                    
                    # Create telemetry object
                    tel_data = TelemetryData(
                        driver=driver,
                        lap_number=lap['LapNumber'],
                        lap_time=lap['LapTime'].total_seconds(),
                        positions=positions,
                        speeds=speeds,
                        throttle=throttle,
                        brake=brake,
                        gear=gear,
                        rpm=rpm,
                        drs=drs
                    )
                    
                    telemetry_data.append(tel_data)
                    
                except Exception as e:
                    continue  # Skip problematic laps
        
        print(f"✓ Extracted {len(telemetry_data)} valid telemetry datasets")
        return telemetry_data
    
    def _filter_clean_laps(self, laps):
        """Filter laps to get clean, representative data"""
        if len(laps) == 0:
            return laps
        
        # Remove laps with missing times
        clean = laps.dropna(subset=['LapTime'])
        
        if len(clean) == 0:
            return clean
        
        # Remove outliers (laps > 1.5x median time)
        median_time = clean['LapTime'].median()
        time_threshold = median_time * 1.3
        clean = clean[clean['LapTime'] <= time_threshold]
        
        # Remove laps with track limits or yellow flags (if available)
        if 'TrackStatus' in clean.columns:
            clean = clean[clean['TrackStatus'] == 1]  # Green flag
        
        return clean
    
    def _extract_track_boundaries(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract inner and outer track boundaries from position data
        """
        print("  Computing track boundaries...")
        
        # Method 1: Use convex hull for rough outer boundary
        try:
            hull = ConvexHull(positions)
            outer_boundary = positions[hull.vertices]
        except:
            # Fallback: use extreme points
            outer_boundary = self._get_extreme_boundary(positions, 'outer')
        
        # Method 2: Inner boundary from points closest to track center
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        inner_percentile = np.percentile(distances, 20)  # Inner 20% of points
        inner_mask = distances <= inner_percentile
        inner_positions = positions[inner_mask]
        
        # Create smooth inner boundary
        inner_boundary = self._smooth_boundary(inner_positions, center)
        
        return {
            'outer': outer_boundary,
            'inner': inner_boundary,
            'center': center
        }
    
    def _smooth_boundary(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Create smooth boundary from scattered points"""
        # Convert to polar coordinates relative to center
        relative = points - center
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        radii = np.linalg.norm(relative, axis=1)
        
        # Sort by angle
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        radii_sorted = radii[sort_idx]
        
        # Create smooth interpolation
        angle_grid = np.linspace(-np.pi, np.pi, 200)
        
        try:
            # Use spline interpolation with periodic boundary
            spline = UnivariateSpline(angles_sorted, radii_sorted, s=len(points)*0.1)
            radii_smooth = spline(angle_grid)
        except:
            # Fallback: linear interpolation
            interp = interp1d(angles_sorted, radii_sorted, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
            radii_smooth = interp(angle_grid)
        
        # Convert back to cartesian
        smooth_boundary = center + np.column_stack([
            radii_smooth * np.cos(angle_grid),
            radii_smooth * np.sin(angle_grid)
        ])
        
        return smooth_boundary
    
    def _get_extreme_boundary(self, positions: np.ndarray, boundary_type: str) -> np.ndarray:
        """Fallback method for boundary extraction"""
        if boundary_type == 'outer':
            # Find extreme points in different directions
            center = np.mean(positions, axis=0)
            angles = np.linspace(0, 2*np.pi, 32)
            boundary_points = []
            
            for angle in angles:
                direction = np.array([np.cos(angle), np.sin(angle)])
                projections = np.dot(positions - center, direction)
                max_idx = np.argmax(projections)
                boundary_points.append(positions[max_idx])
            
            return np.array(boundary_points)
        
        return positions  # Fallback
    
    def calculate_track_boundaries(self, session: fastf1.Session, num_points=1000) -> Dict[str, np.ndarray]:
        """
        Calculates the inner and outer track boundaries from the telemetry data
        of all laps in the session.

        Args:
            session: A loaded FastF1 session object.
            num_points: The number of points to define the boundary lines with.

        Returns:
            A dictionary with 'inner' and 'outer' boundary lines as numpy arrays.
        """
        print("Calculating track boundaries from all driver telemetry...")

        # 1. Aggregate all position data from all laps
        laps = session.laps.pick_telemetry()
        all_x = np.array([])
        all_y = np.array([])

        for _, lap in laps.iterrows():
            try:
                telemetry = lap.get_telemetry()
                all_x = np.append(all_x, telemetry['X'].values)
                all_y = np.append(all_y, telemetry['Y'].values)
            except Exception:
                continue
                
        if len(all_x) == 0:
            raise ValueError("Could not find any valid telemetry data in the session.")

        all_points = np.vstack((all_x, all_y)).T
        print(f"  Aggregated {len(all_points)} total telemetry points.")

        # 2. Use the fastest lap as a reference centerline
        fastest_lap = session.laps.pick_fastest()
        centerline = fastest_lap.get_telemetry()[['X', 'Y']].values
        
        # 3. For each point on the centerline, find the nearest points from the cloud
        #    and determine the min/max distance to define the boundary.
        inner_boundary = []
        outer_boundary = []

        for i in range(len(centerline) - 1):
            p1 = centerline[i]
            p2 = centerline[i+1]

            # Vector perpendicular to the direction of the track
            direction_vector = p2 - p1
            perp_vector = np.array([-direction_vector[1], direction_vector[0]])
            perp_vector_normalized = perp_vector / np.linalg.norm(perp_vector)

            # Get a segment of all points close to the current centerline point
            distances_to_center = np.linalg.norm(all_points - p1, axis=1)
            # Consider points within a 100m radius to reduce computation
            relevant_points = all_points[distances_to_center < 100]

            if len(relevant_points) < 10:
                continue

            # Project these points onto the perpendicular vector
            projections = np.dot(relevant_points - p1, perp_vector_normalized)

            # The min/max projections define the boundary points for this segment
            # Add a small buffer (e.g., 2 meters) for safety
            inner_dist = np.min(projections) - 2
            outer_dist = np.max(projections) + 2

            inner_boundary.append(p1 + inner_dist * perp_vector_normalized)
            outer_boundary.append(p1 + outer_dist * perp_vector_normalized)
        
        inner_boundary = np.array(inner_boundary)
        outer_boundary = np.array(outer_boundary)

        print(f"  Found {len(inner_boundary)} boundary segments.")

        # 4. Smooth the boundary lines for a cleaner result
        # The window length must be odd and less than the number of points.
        window = min(51, len(inner_boundary) - 2 if len(inner_boundary) % 2 == 0 else len(inner_boundary) - 1)
        
        if window > 3: # Need at least a few points to smooth
            inner_smooth_x = savgol_filter(inner_boundary[:, 0], window, 3)
            inner_smooth_y = savgol_filter(inner_boundary[:, 1], window, 3)
            outer_smooth_x = savgol_filter(outer_boundary[:, 0], window, 3)
            outer_smooth_y = savgol_filter(outer_boundary[:, 1], window, 3)
        else:
            inner_smooth_x, inner_smooth_y = inner_boundary[:, 0], inner_boundary[:, 1]
            outer_smooth_x, outer_smooth_y = outer_boundary[:, 0], outer_boundary[:, 1]

        print("✓ Track boundary calculation complete.")
        return {
            'inner': np.vstack((inner_smooth_x, inner_smooth_y)).T,
            'outer': np.vstack((outer_smooth_x, outer_smooth_y)).T
        }
    
    def save_data(self, data, filename: str):
        """Save extracted data to cache"""
        filepath = self.cache_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Data saved to {filepath}")
    
    def load_data(self, filename: str):
        """Load data from cache"""
        filepath = self.cache_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def create_training_dataset(self, telemetry_data: List[TelemetryData], 
                              resample_points: int = 1000) -> Dict:
        """
        Create normalized training dataset for PINN
        
        Args:
            telemetry_data: List of telemetry data objects
            resample_points: Number of points to resample each lap to
        """
        print(f"Creating training dataset with {resample_points} points per lap...")
        
        # Collect all data
        all_positions = []
        all_speeds = []
        all_throttle = []
        all_brake = []
        lap_labels = []
        driver_labels = []
        
        for i, tel_data in enumerate(telemetry_data):
            # Resample to fixed number of points
            n_original = len(tel_data.positions)
            if n_original < 100:  # Skip very short laps
                continue
                
            # Create distance parameter for interpolation
            distances = np.cumsum([0] + [np.linalg.norm(tel_data.positions[j] - tel_data.positions[j-1]) 
                                       for j in range(1, n_original)])
            distances = distances / distances[-1]  # Normalize to [0, 1]
            
            # Resample all data to fixed grid
            s_grid = np.linspace(0, 1, resample_points)
            
            try:
                # Interpolate positions
                pos_interp_x = interp1d(distances, tel_data.positions[:, 0], kind='cubic')
                pos_interp_y = interp1d(distances, tel_data.positions[:, 1], kind='cubic')
                positions_resampled = np.column_stack([pos_interp_x(s_grid), pos_interp_y(s_grid)])
                
                # Interpolate other variables
                speed_interp = interp1d(distances, tel_data.speeds, kind='linear')
                throttle_interp = interp1d(distances, tel_data.throttle, kind='linear')
                brake_interp = interp1d(distances, tel_data.brake, kind='linear')
                
                speeds_resampled = speed_interp(s_grid)
                throttle_resampled = throttle_interp(s_grid)
                brake_resampled = brake_interp(s_grid)
                
                # Add to dataset
                all_positions.append(positions_resampled)
                all_speeds.append(speeds_resampled)
                all_throttle.append(throttle_resampled)
                all_brake.append(brake_resampled)
                lap_labels.append(i)
                driver_labels.append(tel_data.driver)
                
            except Exception as e:
                print(f"  Skipping lap {i}: {e}")
                continue
        
        if not all_positions:
            raise ValueError("No valid laps could be processed")
        
        # Convert to numpy arrays
        positions = np.array(all_positions)  # [n_laps, n_points, 2]
        speeds = np.array(all_speeds)        # [n_laps, n_points]
        throttle = np.array(all_throttle)    # [n_laps, n_points]
        brake = np.array(all_brake)          # [n_laps, n_points]
        
        # Compute normalization statistics
        pos_mean = np.mean(positions.reshape(-1, 2), axis=0)
        pos_std = np.std(positions.reshape(-1, 2), axis=0)
        speed_mean = np.mean(speeds)
        speed_std = np.std(speeds)
        
        # Create training dataset
        dataset = {
            'positions': positions,
            'speeds': speeds,
            'throttle': throttle,
            'brake': brake,
            'lap_labels': lap_labels,
            'driver_labels': driver_labels,
            'normalization': {
                'pos_mean': pos_mean,
                'pos_std': pos_std,
                'speed_mean': speed_mean,
                'speed_std': speed_std
            },
            'metadata': {
                'n_laps': len(all_positions),
                'points_per_lap': resample_points,
                'drivers': list(set(driver_labels)),
                'lap_times': [tel_data.lap_time for tel_data in telemetry_data if len(tel_data.positions) >= 100]
            }
        }
        
        print(f"✓ Training dataset created: {dataset['metadata']['n_laps']} laps, "
              f"{len(dataset['metadata']['drivers'])} drivers")
        
        return dataset

def visualize_track_data(geometry: TrackGeometry, telemetry_data: List[TelemetryData] = None):
    """
    Create comprehensive visualization of extracted track data
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Track boundaries and racing lines
    ax1 = axes[0, 0]
    
    # Plot track boundaries
    if 'outer' in geometry.track_bounds:
        outer = geometry.track_bounds['outer']
        ax1.plot(outer[:, 0], outer[:, 1], 'k-', linewidth=2, label='Outer Boundary')
    
    if 'inner' in geometry.track_bounds:
        inner = geometry.track_bounds['inner']
        ax1.plot(inner[:, 0], inner[:, 1], 'r-', linewidth=2, label='Inner Boundary')
    
    # Plot racing lines from different drivers
    colors = plt.cm.tab10(np.linspace(0, 1, len(geometry.racing_lines)))
    for (driver, line), color in zip(geometry.racing_lines.items(), colors):
        ax1.plot(line[:, 0], line[:, 1], color=color, alpha=0.7, linewidth=1, label=f'{driver} Racing Line')
    
    ax1.set_title(f'{geometry.name} ({geometry.year}) - Track Layout')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_aspect('equal')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    if telemetry_data:
        # Plot 2: Speed profiles
        ax2 = axes[0, 1]
        for i, tel_data in enumerate(telemetry_data[:10]):  # Plot first 10 laps
            distances = np.cumsum([0] + [np.linalg.norm(tel_data.positions[j] - tel_data.positions[j-1]) 
                                       for j in range(1, len(tel_data.positions))])
            ax2.plot(distances, tel_data.speeds, alpha=0.6, linewidth=1, 
                    label=f'{tel_data.driver} ({tel_data.lap_time:.1f}s)')
        
        ax2.set_title('Speed Profiles')
        ax2.set_xlabel('Distance (meters)')
        ax2.set_ylabel('Speed (km/h)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throttle vs Brake
        ax3 = axes[1, 0]
        fastest_lap = min(telemetry_data, key=lambda x: x.lap_time)
        distances = np.cumsum([0] + [np.linalg.norm(fastest_lap.positions[j] - fastest_lap.positions[j-1]) 
                                   for j in range(1, len(fastest_lap.positions))])
        
        ax3.plot(distances, fastest_lap.throttle, 'g-', label='Throttle %', alpha=0.8)
        ax3.plot(distances, fastest_lap.brake, 'r-', label='Brake %', alpha=0.8)
        ax3.set_title(f'Fastest Lap Controls ({fastest_lap.driver}, {fastest_lap.lap_time:.1f}s)')
        ax3.set_xlabel('Distance (meters)')
        ax3.set_ylabel('Input %')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Lap time distribution
        ax4 = axes[1, 1]
        lap_times = [tel_data.lap_time for tel_data in telemetry_data]
        drivers = [tel_data.driver for tel_data in telemetry_data]
        
        # Create box plot by driver
        driver_times = {}
        for driver, lap_time in zip(drivers, lap_times):
            if driver not in driver_times:
                driver_times[driver] = []
            driver_times[driver].append(lap_time)
        
        ax4.boxplot(driver_times.values(), labels=driver_times.keys())
        ax4.set_title('Lap Time Distribution by Driver')
        ax4.set_ylabel('Lap Time (seconds)')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extractor = FastF1DataExtractor()
    
    print("=" * 60)
    print("FASTF1 DATA EXTRACTION DEMO")
    print("=" * 60)
    
    session = extractor.get_session(2023, 'silverstone', 'Q')
    
    if session:
        # --- NEW: Use the Point Cloud method for boundaries ---
        boundaries = extractor.calculate_track_boundaries(session)
        
        # Get the fastest racing line for visualization and training
        fastest_lap = session.laps.pick_fastest()
        racing_line = fastest_lap.get_telemetry()[['X', 'Y']].values
        
        # Create the geometry object for saving and visualization
        geometry = TrackGeometry(
            name=session.event['EventName'],
            year=session.date.year,
            track_bounds=boundaries,
            racing_lines={fastest_lap['Driver']: racing_line}
        )
        
        extractor.save_data(geometry, 'silverstone_2023_geometry.pkl')

        # The rest of your script can continue as before...
        telemetry_data = extractor.extract_telemetry_data(session, max_laps_per_driver=3)
        extractor.save_data(telemetry_data, 'silverstone_2023_telemetry.pkl')
        
        # Create training dataset
        if telemetry_data:
            dataset = extractor.create_training_dataset(telemetry_data, resample_points=1000)
            extractor.save_data(dataset, 'silverstone_2023_training.pkl')
        
        # Visualize the results
        visualize_track_data(geometry, telemetry_data)
        
        print(f"\n✓ Data extraction complete!")
        print(f"  Track: {geometry.name}")
        print(f"  Racing lines from {len(geometry.racing_lines)} drivers")
        print(f"  Telemetry from {len(telemetry_data)} laps")
        print(f"  Training dataset: {len(telemetry_data)} laps × 1000 points")
    
    print("\n" + "=" * 60)