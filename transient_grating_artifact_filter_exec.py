"""transient_grating_artifact_filter_exec.py

Test script to call transient.grating_artifact_filter.oy

"""
from transient_grating_artifact_filter import transient_grating_artifact_filter
import sys

# Structures to simulate: "croix", "gold_film", "nano_pillars", "rhodamine"
substrate_type: str = "rhodamine"

# Thresholds for filter construction
threshold_ellipse: float = 0.1
threshold_cutout: float = 0.5

# transient_grating_artifact_filter() optional parameters
cross_pass_band_width: int = 0
pass_upper_left_lower_right_quadrants: bool = True

# Define simulation parameters for the selected structure
if substrate_type == "croix":
    # Smooth unstructured gold film
    fname: str = "Croix.mat"
    λ0_pump: float = 670.0
    artifact_extent_λ: float = 26
    artifact_extent_t: float = 0.35

elif substrate_type == "gold_film":
    # Smooth unstructured gold film
    fname: str = "Figure_article_Parallel.mat"
    λ0_pump: float = 600.0
    artifact_extent_λ: float = 26
    artifact_extent_t: float = 0.35

elif substrate_type == "nano_pillars":
    # Nano-pillars
    fname = "Data_ROD_600_long.mat"
    λ0_pump = 600.0
    artifact_extent_λ = 25
    artifact_extent_t = 0.47

elif substrate_type == "rhodamine":
    # Rhodamine solution
    fname = "Data_Rhodamine_570_2.mat"
    λ0_pump = 568.0
    artifact_extent_λ = 22
    artifact_extent_t = 0.55

else:
    raise ValueError("Unknown substrate type!")

# Run the simulation
transient_grating_artifact_filter(
    fname=fname,
    lambda0_pump=λ0_pump,
    artifact_extent_lambda=artifact_extent_λ,
    artifact_extent_t=artifact_extent_t,
    threshold_ellipse=threshold_ellipse,
    threshold_cutout=threshold_cutout,
    cross_pass_band_width=cross_pass_band_width,
    pass_upper_left_lower_right_quadrants=pass_upper_left_lower_right_quadrants,
    interpolate_image_to_power_of_two=True,
)

# If running from the command line, pause for user input to keep figures visible.
# If running in PyCharm debugger, set breakpoint below eto keep figures visible
if sys.gettrace() is not None:
    print("Breakpoint here to keep figures visible in IDE!")
else:
    input("Script paused to display figures, press any key to exit...")
