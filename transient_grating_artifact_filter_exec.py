"""transient_grating_artifact_filter_exec.py

Test script to call transient.grating_artifact_filter.py

"""
import matplotlib.pyplot as plt
from transient_grating_artifact_filter import transient_grating_artifact_filter
import sys

# matplotlib initializations
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "figure.figsize": [10, 5],
        "font.size": 6,
        "lines.linewidth": 0.5,
        "axes.linewidth": 0.5,
        "image.cmap": "coolwarm",
    },
)

# Choice of sample to simulate: "gold_film", "nano_crosses", "nano_rods", or "rhodamine"
substrate_type: str = "nano_crosses"

# Define simulation parameters for the selected sample
if substrate_type == "gold_film":
    # Unstructured (smooth) gold film
    fname: str = "Figure_article_Parallel.mat"
    lambda0_pump: float = 600.0
    artifact_extent_λ: float = 26
    artifact_extent_t: float = 0.35

elif substrate_type == "nano_crosses":
    # Structured gold film (nano-crosses)
    fname: str = "Nano-crosses.mat"
    lambda0_pump: float = 670.0
    artifact_extent_λ: float = 26
    artifact_extent_t: float = 0.35

elif substrate_type == "nano_rods":
    # Structured gold film (nano-rods)
    fname = "Data_ROD_600_long.mat"
    lambda0_pump = 600.0
    artifact_extent_λ = 25
    artifact_extent_t = 0.47

elif substrate_type == "rhodamine":
    # Rhodamine solution atop unstructured gold film
    fname = "Data_Rhodamine_570_2.mat"
    lambda0_pump = 570.0
    artifact_extent_λ = 22
    artifact_extent_t = 0.55
    lambda_time_profile: float = 566.74

else:
    raise ValueError("Unknown substrate type!")

# Thresholds for filter construction
threshold_ellipse: float = 0.1
threshold_cutout: float = 0.5

# Optional parameters
lambda_time_profile: float = 0
cross_pass_band_width: int = 0
pass_upper_left_lower_right_quadrants: bool = True

# Run the simulation
result = transient_grating_artifact_filter(
    fname=fname,
    lambda0_pump=lambda0_pump,
    artifact_extent_lambda=artifact_extent_λ,
    artifact_extent_t=artifact_extent_t,
    threshold_ellipse=threshold_ellipse,
    threshold_cutout=threshold_cutout,
    lambda_time_profile=lambda_time_profile,
    cross_pass_band_width=cross_pass_band_width,
    pass_upper_left_lower_right_quadrants=pass_upper_left_lower_right_quadrants,
)

# If running from the command line, pause for user input to keep figures visible.
# If running in PyCharm debugger, set breakpoint below eto keep figures visible
if sys.gettrace() is not None:
    print("Breakpoint here to keep figures visible in IDE!")
else:
    input("Script paused to display figures, press return to exit...")
