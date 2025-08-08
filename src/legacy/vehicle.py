# src/vehicle.py

# --- Constants ---
RHO_AIR = 1.225  # Density of air in kg/m^3
GRAVITY = 9.81   # Acceleration due to gravity in m/s^2

# --- F1 Car Parameters ---
f1_car_parameters = {
    # Mass in kilograms (car + driver)
    "mass_kg": 798,

    # Max Power in Watts (~1000 hp)
    "power_watts": 750000,

    # Frontal Area in square meters
    "frontal_area_m2": 1.6,

    # Drag Coefficient (unitless)
    "drag_coefficient": 1.0,

    # Downforce Coefficient (unitless, negative for downforce)
    "downforce_coefficient": -3.5,

    # Peak Tire Friction Coefficient (unitless)
    "friction_coefficient": 1.8,
}
