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
        This method uses FastF1's built-in track boundary utilities.
        
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

        # Get the telemetry for the fastest lap
        telemetry = fastest_lap.get_telemetry()
        if 'X' not in telemetry.columns or 'Y' not in telemetry.columns:
            raise ValueError("Telemetry for fastest lap is missing X/Y coordinates.")
        
        # Get racing line from telemetry
        racing_line = telemetry[['X', 'Y']].values
        racing_lines = {driver: racing_line}
        print(f"  ✓ Extracted racing line for {driver} with {len(racing_line)} points.")
        
        # Extract track boundaries using FastF1's utility function
        try:
            # The utils.get_track_boundaries function expects X and Y as separate arrays
            x_coords = telemetry['X'].values
            y_coords = telemetry['Y'].values
            
            # Get track boundaries - returns tuple of (inner, outer) boundary coordinates
            inner_boundary, outer_boundary = utils.get_track_boundaries(x_coords, y_coords)
            
            track_bounds = {
                'inner': inner_boundary,
                'outer': outer_boundary
            }
            print(f"  ✓ Extracted track boundaries using FastF1 utilities.")
            
        except Exception as e:
            print(f"  ⚠️ FastF1 boundary extraction failed: {e}")
            print("  Falling back to custom boundary calculation...")
            
            # Fallback to custom boundary calculation
            track_bounds = self.calculate_track_boundaries_fallback(session)
        
        # Assemble the Geometry object
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
    
    def calculate_track_boundaries_fallback(self, session, num_drivers: int = 10) -> Dict[str, np.ndarray]:
        """
        Fallback method: Calculate track boundaries by aggregating telemetry data 
        from multiple drivers and laps.
        
        Args:
            session: FastF1 session object
            num_drivers: Maximum number of drivers to use for boundary calculation
        """
        print("Calculating track boundaries from multiple drivers...")
        
        # Collect position data from multiple drivers
        all_positions = []
        drivers_used = 0
        
        # Get fastest lap from each driver
        for driver in session.drivers[:num_drivers]:  # Limit to prevent excessive computation
            try:
                driver_laps = session.laps.pick_driver(driver)
                clean_laps = self._filter_clean_laps(driver_laps)
                
                if len(clean_laps) > 0:
                    # Get fastest clean lap for this driver
                    fastest_lap = clean_laps.pick_fastest()
                    telemetry = fastest_lap.get_telemetry()
                    
                    if len(telemetry) > 0 and 'X' in telemetry.columns and 'Y' in telemetry.columns:
                        positions = telemetry[['X', 'Y']].values
                        positions = positions[~np.isnan(positions).any(axis=1)]  # Remove NaN
                        
                        if len(positions) > 100:  # Ensure reasonable data length
                            all_positions.append(positions)
                            drivers_used += 1
                            print(f"  ✓ Added {driver}: {len(positions)} points")
                            
            except Exception as e:
                print(f"  ❌ Skipped {driver}: {e}")
                continue
        
        if not all_positions:
            raise ValueError("Could not extract position data from any driver")
        
        print(f"  Using telemetry from {drivers_used} drivers")
        
        # Combine all positions
        combined_positions = np.vstack(all_positions)
        print(f"  Total position points: {len(combined_positions)}")
        
        # Use the fastest lap as reference centerline
        fastest_lap = session.laps.pick_fastest()
        centerline = fastest_lap.get_telemetry()[['X', 'Y']].values
        
        # Calculate boundaries using perpendicular projection method
        inner_boundary = []
        outer_boundary = []
        
        for i in range(len(centerline) - 1):
            p1 = centerline[i]
            p2 = centerline[i + 1]
            
            # Calculate perpendicular vector
            direction_vector = p2 - p1
            if np.linalg.norm(direction_vector) == 0:
                continue
                
            perp_vector = np.array([-direction_vector[1], direction_vector[0]])
            perp_vector_normalized = perp_vector / np.linalg.norm(perp_vector)
            
            # Find points near this centerline segment
            distances_to_center = np.linalg.norm(combined_positions - p1, axis=1)
            nearby_mask = distances_to_center < 150  # 150m radius
            nearby_points = combined_positions[nearby_mask]
            
            if len(nearby_points) < 5:
                # Not enough points, use previous boundary points or skip
                if len(inner_boundary) > 0:
                    inner_boundary.append(inner_boundary[-1])
                    outer_boundary.append(outer_boundary[-1])
                continue
            
            # Project points onto perpendicular direction
            projections = np.dot(nearby_points - p1, perp_vector_normalized)
            
            # Find extremes with some buffer
            inner_dist = np.percentile(projections, 5) - 3  # 5th percentile minus 3m buffer
            outer_dist = np.percentile(projections, 95) + 3  # 95th percentile plus 3m buffer
            
            inner_boundary.append(p1 + inner_dist * perp_vector_normalized)
            outer_boundary.append(p1 + outer_dist * perp_vector_normalized)
        
        # Convert to arrays
        inner_boundary = np.array(inner_boundary)
        outer_boundary = np.array(outer_boundary)
        
        # Smooth the boundaries if we have enough points
        if len(inner_boundary) > 10:
            window = min(21, len(inner_boundary) if len(inner_boundary) % 2 == 1 else len(inner_boundary) - 1)
            if window >= 3:
                inner_boundary[:, 0] = savgol_filter(inner_boundary[:, 0], window, 2)
                inner_boundary[:, 1] = savgol_filter(inner_boundary[:, 1], window, 2)
                outer_boundary[:, 0] = savgol_filter(outer_boundary[:, 0], window, 2)
                outer_boundary[:, 1] = savgol_filter(outer_boundary[:, 1], window, 2)
        
        print(f"  ✓ Calculated boundaries: {len(inner_boundary)} points each")
        
        return {
            'inner': inner_boundary,
            'outer': outer_boundary
        }
    
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
        
        # Remove outliers (laps > 1.3x median time)
        median_time = clean['LapTime'].median()
        time_threshold = median_time * 1.3
        clean = clean[clean['LapTime'] <= time_threshold]
        
        return clean
    
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
        # Extract track geometry (uses FastF1 utilities with fallback)
        geometry = extractor.extract_track_geometry(session)
        extractor.save_data(geometry, 'silverstone_2023_geometry.pkl')
        
        # Extract telemetry data  
        telemetry_data = extractor.extract_telemetry_data(session, max_laps_per_driver=3)
        extractor.save_data(telemetry_data, 'silverstone_2023_telemetry.pkl')
        
        # Create training dataset
        if telemetry_data:
            dataset = extractor.create_training_dataset(telemetry_data, resample_points=1000)
            extractor.save_data(dataset, 'silverstone_2023_training.pkl')
        
        # Visualize results
        visualize_track_data(geometry, telemetry_data)
        
        print(f"\n✓ Data extraction complete!")
        print(f"  Track: {geometry.name}")
        print(f"  Racing lines from {len(geometry.racing_lines)} drivers")
        print(f"  Telemetry from {len(telemetry_data)} laps")
        print(f"  Training dataset: {len(telemetry_data)} laps × 1000 points")
    
    print("\n" + "=" * 60)