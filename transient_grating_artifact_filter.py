"""transient_grating_artifact_filter.py

Two-dimensional filtering in the Fourier domain of transient grating coherent artifacts
in time-resolved spectroscopy

First separate the input image (time-resolved spectroscopy map) into "smooth" and
"periodic" components to reduce the effect of the "cross" pattern in the Discrete
Fourier transform due to the non-periodic nature of the image (see Moisan, L.
J Math Imaging Vis 39, 161–179, 2011), then filter the artifact from the
periodic component in the Fourier domain using an elliptically shaped stop-band
with a pass-band at the center to preserve the low-frequency content of the baseline
data, and recombine the filtered periodic component with the smooth component
to generate the filtered map. Apply a light Gaussian blur to the result to
remove high frequency noise.

See: M. Vega, J.-F. Bryche, P.-L. Karsenti, P. Gogol, M. Canva, P.G. Charette. Two-dimensional filtering in the Fourier
domain of transient grating coherent artifacts in time-resolved spectroscopy, Analytica Chimica Acta, 2023, 341820.
https://doi.org/10.1016/j.aca.2023.341820.


"""

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from moisan2011 import per
import os
from pathlib import Path
from scipy import ndimage
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator
from skimage.draw import line
import sys
from typing import Tuple

# Script version
__version__: str = "1.0.0"


@dataclass
class ImageSpecs:
    """
    Time-resolved spectroscopy data array specifications

    λs_in (np.ndarray): input array of wavelengths (nm)
    ts_in (np.ndarray): input array of times (ps)
    img_in (np.ndarray): spectroscopy data (arbitrary units)
    interpolate_image_to_power_of_two (bool): interpolate image dimensions to nearest
                                              larger power of two (default: False)
    """

    λs_in: np.ndarray
    ts_in: np.ndarray
    img_in: np.ndarray
    interpolate_image_to_power_of_two: bool = False

    def __post_init__(self):
        # Check that wavelength and time arrays match spectroscopy image data dimensions
        if (
            len(self.λs_in) != self.img_in.shape[1]
            or len(self.ts_in) != self.img_in.shape[0]
        ):
            raise ValueError(
                "Input file array dimensions are inconsistent: mismatch between "
                f"data (nλ = {self.img_in.shape[1]}, nt = {self.img_in.shape[0]}) "
                f"vs wavelength (nλ = {len(self.λs_in)}) "
                f"and/or time (nλ = {len(self.ts_in)}) array dimensions!"
            )

        # Set image dimensions to nearest larger power of 2 for interpolation, if
        # requested, else stick with input image dimensions
        if self.interpolate_image_to_power_of_two:
            self.height, self.width = 2 ** np.ceil(
                np.log2(self.img_in.shape[0])
            ).astype(int), 2 ** np.ceil(np.log2(self.img_in.shape[1])).astype(int)
        else:
            self.height, self.width = self.img_in.shape

        # Image center pixel coordinates
        self.x0, self.y0 = self.width // 2, self.height // 2

        # Resample the input arrays on a regular even-spaced grid to account for any
        # non-uniform time or frequency sampling
        self.λ0, self.λ1 = self.λs_in[0], self.λs_in[-1]
        self.λs: np.ndarray = np.linspace(self.λ0, self.λ1, self.width)
        self.t0, self.t1 = self.ts_in[0], self.ts_in[-1]
        self.ts: np.ndarray = np.linspace(self.t0, self.t1, self.height)
        self.dλ, self.dt = self.λs[1] - self.λs[0], self.ts[1] - self.ts[0]
        interp: RegularGridInterpolator = RegularGridInterpolator(
            (self.ts_in, self.λs_in), self.img_in
        )
        yi, xi = np.meshgrid(self.ts, self.λs, indexing="ij")
        self.img: np.ndarray = interp((yi, xi))


@dataclass
class Artifact:
    """
    Transient grating coherent artifact specifications

    λ0 (float): artifact center wavelength (pump center wavelength, nm)
    extent_λ (float): artifact extent along the wavelength axis (nm)
    extent_t (float): artifact extent along the time axis (ps)
    img_specs (ImageSpecs): image data specifications

    """

    λ0: float
    extent_λ: float
    extent_t: float
    img_specs: ImageSpecs

    def __post_init__(self):
        # Check artifact specifications against image dimensions
        if (
            ((self.λ0 - self.extent_λ / 2) < self.img_specs.λ0)
            or ((self.λ0 + self.extent_λ / 2) > self.img_specs.λ1)
            or (-self.extent_t / 2 < self.img_specs.t0)
            or (self.extent_t / 2 > self.img_specs.t1)
        ):
            raise ValueError(
                "Artifact extent "
                f"([{self.λ0 - self.extent_λ / 2:.1f}:{self.λ0 + self.extent_λ / 2:.1f}] nm x "
                f"[{- self.extent_t / 2: .2f}:{self.extent_t / 2:.2f}] ps) "
                "is outside the input image data range "
                f"([{self.img_specs.λ0:.1f}:{self.img_specs.λ1:.1f}] nm x "
                f"[{self.img_specs.t0:.2f}:{self.img_specs.t1:.2f}] ps)!"
            )

        # Artifact geometry in units of pixels
        self.λ0_pixels: int = np.abs(self.img_specs.λs - self.λ0).argmin()
        self.t0_pixels: int = (
            self.img_specs.height - np.abs(self.img_specs.ts - 0).argmin()
        )
        self.extent_λ_pixels: float = float(self.extent_λ / self.img_specs.dλ)
        self.extent_t_pixels: float = float(self.extent_t / self.img_specs.dt)
        self.α: float = -np.degrees(
            np.arctan(
                (self.extent_t_pixels * self.img_specs.width)
                / (self.extent_λ_pixels * self.img_specs.height)
            )
        )
        self.θ: float = 90 + self.α

        # Coordinate endpoints of line though center of the artifact
        slope: float = self.extent_t_pixels / self.extent_λ_pixels
        intercept: float = self.t0_pixels - slope * self.λ0_pixels
        self.x0_pixels, self.x1_pixels = int(
            self.λ0_pixels - self.extent_λ_pixels // 2
        ), int(self.λ0_pixels + self.extent_λ_pixels // 2)
        self.y0_pixels, self.y1_pixels = int(slope * self.x0_pixels + intercept), int(
            slope * self.x1_pixels + intercept
        )

        # Convert line pixel coordinates to data coordinates
        self.x0, self.x1 = (
            self.img_specs.λs[self.x0_pixels],
            self.img_specs.λs[self.x1_pixels],
        )
        self.y0, self.y1 = (
            self.img_specs.ts[self.img_specs.height - self.y0_pixels],
            self.img_specs.ts[self.img_specs.height - self.y1_pixels],
        )

        # Coordinate endpoints of normal through center of the artifact
        self.normal_y0_pixels, self.normal_y1_pixels = self.y0_pixels, self.y1_pixels
        normal_slope: float = (
            -1 / slope * (self.img_specs.height / self.img_specs.width)
        )
        normal_intercept: float = self.t0_pixels - normal_slope * self.λ0_pixels
        self.normal_x0_pixels: int = int(
            (self.normal_y0_pixels - normal_intercept) / normal_slope
        )
        self.normal_x1_pixels: int = int(
            (self.normal_y1_pixels - normal_intercept) / normal_slope
        )
        self.normal_length_pixels: float = np.sqrt(
            (self.normal_y0_pixels - self.normal_y1_pixels) ** 2
            + (self.normal_x0_pixels - self.normal_x1_pixels) ** 2
        )


@dataclass
class Filter:
    """
    Filter parameters

    img_dft_mag (np.ndarray): DFT magnitude image used to define the filter
                              by thresholding
    threshold_ellipse_stop_band (float): threshold for defining the ellipse-shaped
                              filter stop-band
    threshold_center_pass_band (float): threshold for defining the central pass-band
                              area about the origin in the Fourier domain to pass
                              the low frequency content (baseline) of the
                              spectroscopy data
    img_specs (ImageSpecs): spectroscopy map specifications
    artifact (Artifact): artifact specifications
    cross_pass_band_width (int) = width of cross-shaped pass-band along the horizontal
                              and vertical axes in the Fourier domain in
                              the filter to pass any remaining non-periodic
                              content left over from the smooth/periodic
                              decomposition (disabled if <= 0)
    pass_upper_left_lower_right_quadrants (bool) = pass (do not filter) the upper-left
                              and lower-right quadrants of Fourier space, excluding
                              the horizontal and vertical axes that can be passed with
                              cross_pass_band_width > 0
    gaussian_blur_sigma (float): standard deviation of the Gaussian blur applied
                              to the final result

    """

    img_dft_mag: np.ndarray
    threshold_ellipse_stop_band: float
    threshold_center_pass_band: float
    img_specs: ImageSpecs
    artifact: Artifact
    cross_pass_band_width: int
    pass_upper_left_lower_right_quadrants: bool
    gaussian_blur_sigma: float

    def __post_init__(self):
        # Check thresholds and cross_pass_band_width
        if (
            (
                self.threshold_ellipse_stop_band < 0
                or self.threshold_ellipse_stop_band > 1
            )
            or (
                self.threshold_center_pass_band < 0
                or self.threshold_center_pass_band > 1
            )
            or (
                self.cross_pass_band_width < 0
                or self.cross_pass_band_width > self.img_specs.width
                or self.cross_pass_band_width > self.img_specs.height
            )
        ):
            raise ValueError(
                "One or more of the values supplied for the parameters "
                f"threshold_ellipse_stop_band ({self.threshold_ellipse_stop_band:.2f}),"
                f" threshold_center_pass_band ({self.threshold_center_pass_band:.2f}), "
                f"or cross_pass_band_width ({self.cross_pass_band_width}) are "
                "out of range (thresholds must be in the range [0..1], "
                "the cross pass-band width must be non-negative and less than the image"
                " height/width)!"
            )
        if self.threshold_ellipse_stop_band > self.threshold_center_pass_band:
            raise ValueError(
                f"threshold_ellipse_stop_band ({self.threshold_ellipse_stop_band})"
                " must be less than threshold_center_pass_band "
                f"({self.threshold_center_pass_band})!"
            )

        # Number of decades used in binarizing log scale images, else small pixel
        # values will unduly skew the normalization (4 is a good compromise).
        self.num_decades: int = 4

        # Build the filter
        (
            self.ellipse_long_axis_radius,
            self.ellipse_short_axis_radius,
            self.img_binary_ellipse_rgb,
        ) = self.define_filter_ellipse()
        self.f: np.ndarray = self.build_binary_image_filter()

    def binarize_image(self, img: np.ndarray, threshold: float) -> np.ndarray:
        """
        Binarize a log scale image using a threshold. Limit to self.num_decades decades
        of dynamic range else small pixel values will unduly skew the normalization

        Args:
            img (np.ndarray): log scale image to be converted to binary
            threshold (float): threshold [0, 1]

        Returns: binary image (np.ndarray)

        """

        # Normalize log scale image to [0..1], limit to self.num_decades decades of
        # dynamic range else small pixel values will skew the normalization.
        img[img < np.amax(img) - self.num_decades] = np.amax(img) - self.num_decades
        img_normalized: np.ndarray = cv.normalize(
            src=img, dst=None, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX
        )

        # Binarize image with threshold
        img_binary: np.ndarray = cv.threshold(
            src=img_normalized, thresh=threshold, maxval=1.0, type=cv.THRESH_BINARY
        )[1]

        # Clean up with morphological opening (erosion/dilation), dilate a bit more to
        # round out and slightly increase the thresholded area
        erosion = cv.erode(src=img_binary, kernel=np.ones((3, 3), np.uint8))
        return cv.dilate(src=erosion, kernel=np.ones((5, 5), np.uint8))

    @staticmethod
    def calc_line_length(
        img_binary: np.ndarray, diagonal_pixel_coordinates: np.ndarray
    ) -> float:
        """
        Calculate the length of a line in a binary image along a diagonal

        Args:
            img_binary (np.ndarray): binary image to be analyzed
            diagonal_pixel_coordinates (np.ndarray): array of diagonal pixel coordinates

        Returns: line length (float)

        """

        diagonal_pixel_values: np.ndarray = np.asarray(
            img_binary[diagonal_pixel_coordinates]
        )
        line_pixel_coordinate_indices: np.ndarray = np.asarray(
            np.where(diagonal_pixel_values == 1)
        ).flatten()
        if len(line_pixel_coordinate_indices) > 0:
            line_pixel_coordinates: np.ndarray = np.asarray(
                [
                    diagonal_pixel_coordinates[0][line_pixel_coordinate_indices],
                    diagonal_pixel_coordinates[1][line_pixel_coordinate_indices],
                ]
            )
            return float(
                np.linalg.norm(
                    np.array(line_pixel_coordinates.T[-1] - line_pixel_coordinates.T[0])
                )
            )
        else:
            return 0.0

    def plot_filter_images(self, filter_image: np.ndarray):
        """
        Show images of:
            1) thresholded pixels used to determine the filter ellipse stop-band
            2) outline of the ellipse superimposed on the imaged used for thresholding
            3) resulting binary filter image

        Args:
            filter_image (np.ndarray): binary filter image

        Returns: None

        """

        fig, axs = plt.subplots(3)

        # Draw thresholded ellipse binary image with major/minor axes
        axs[0].set(
            title="Thresholded binary binary image for identifying filter ellipse stop-"
            f"band (threshold = {self.threshold_ellipse_stop_band:.2f})\n"
            f"Long axis radius = {self.ellipse_long_axis_radius} pixels, "
            f"short axis radius = {self.ellipse_short_axis_radius} pixels\n"
            f"NB: data limited to {self.num_decades} decade dynamic range "
            "in binarization"
        )
        axs[0].imshow(self.img_binary_ellipse_rgb, cmap="gray")

        # Draw filter ellipse outline over periodic component DFT
        axs[1].set(title="Filter ellipse outline over periodic component DFT")
        ellipse_image_binary_outline: np.ndarray = np.zeros(
            (self.img_specs.height, self.img_specs.width), dtype=np.uint8
        )
        cv.ellipse(
            img=ellipse_image_binary_outline,
            center=[self.img_specs.x0, self.img_specs.y0],
            axes=[self.ellipse_long_axis_radius, self.ellipse_short_axis_radius],
            angle=-self.artifact.θ,
            startAngle=0.0,
            endAngle=360.0,
            color=[1],
            thickness=1,
        )
        periodic_with_ellipse_outline: np.ndarray = np.copy(self.img_dft_mag)
        periodic_with_ellipse_outline[ellipse_image_binary_outline == 1] = np.max(
            self.img_dft_mag
        )
        axs[1].imshow(periodic_with_ellipse_outline, cmap="gray")

        # Draw binary filter image
        axs[2].set(
            title="Complete filter binary image\n"
            "Stop-band 'ellipse' (pixel values = 0), "
            "pass-band elsewhere (pixel values = 1)"
        )
        axs[2].imshow(filter_image, cmap="gray")

        plt.tight_layout()

        return None

    def define_filter_ellipse(self) -> Tuple[int, int, np.ndarray]:
        """

        Determine the filter ellipse parameters from the periodic component DFT
        magnitude, i.e. the filter "stop band"

        Returns: ellipse_long_axis_radius (int), ellipse_short_axis_radius (int)
                 img_binary_ellipse_rgb (np.ndarray)

        """

        # Binarize the periodic component DFT magnitude image using the
        # "threshold_ellipse_stop_band" parameter to segment the area of
        # the elliptically shaped spectrum of the artifact centered on the origin
        # in Fourier space
        img_binary_ellipse: np.ndarray = self.binarize_image(
            img=self.img_dft_mag, threshold=self.threshold_ellipse_stop_band
        )

        # From the experimental parameters (time and wavelength artifact extent, which
        # yield the angle of the elliptically-shaped spectrum of the artifact in Fourier
        # space), determine the long and short diagonals of this ellipse and the lengths
        # of the above-threshold pixel lines along these diagonals, i.e. the lengths of
        # the long and short axes of the ellipse (actually the radii, half the lengths
        # of the axes).
        l: float = min(self.img_dft_mag.shape) / 2.5
        artifact_long_diagonal_pixel_coordinates: np.ndarray = line(
            r0=int(self.img_specs.y0 - l * np.sin(np.radians(self.artifact.θ))),
            c0=int(self.img_specs.x0 + l * np.cos(np.radians(self.artifact.θ))),
            r1=int(self.img_specs.y0 + l * np.sin(np.radians(self.artifact.θ))),
            c1=int(self.img_specs.x0 - l * np.cos(np.radians(self.artifact.θ))),
        )
        artifact_short_diagonal_pixel_coordinates: np.ndarray = line(
            r0=int(self.img_specs.y0 - l * np.sin(np.radians(self.artifact.θ + 90))),
            c0=int(self.img_specs.x0 + l * np.cos(np.radians(self.artifact.θ + 90))),
            r1=int(self.img_specs.y0 + l * np.sin(np.radians(self.artifact.θ + 90))),
            c1=int(self.img_specs.x0 - l * np.cos(np.radians(self.artifact.θ + 90))),
        )
        ellipse_long_axis_radius: int = int(
            (
                self.calc_line_length(
                    img_binary=img_binary_ellipse,
                    diagonal_pixel_coordinates=artifact_long_diagonal_pixel_coordinates,
                )
            )
            // 2
        )
        ellipse_short_axis_radius: int = int(
            (
                self.calc_line_length(
                    img_binary=img_binary_ellipse,
                    diagonal_pixel_coordinates=artifact_short_diagonal_pixel_coordinates,
                )
            )
            // 2
        )
        if ellipse_long_axis_radius == 0 or ellipse_short_axis_radius == 0:
            raise ValueError(
                "Threshold value for filter ellipse stop-band "
                f"({self.threshold_ellipse_stop_band}) is too high!"
            )

        # For debugging/validation purposes, build an image of the above
        # threshold pixels and the resulting enclosing ellipse (the filter stop band)
        img_binary_ellipse_rgb = np.repeat(
            img_binary_ellipse[:, :, np.newaxis], 3, axis=2
        )
        img_binary_ellipse_rgb[artifact_long_diagonal_pixel_coordinates] = [1, 0, 0]
        img_binary_ellipse_rgb[artifact_short_diagonal_pixel_coordinates] = [1, 0, 0]
        cv.ellipse(
            img=img_binary_ellipse_rgb,
            center=[self.img_specs.x0, self.img_specs.y0],
            axes=[ellipse_long_axis_radius, ellipse_short_axis_radius],
            angle=-self.artifact.θ,
            startAngle=0.0,
            endAngle=360.0,
            color=[1, 0, 0],
            thickness=1,
        )

        return (
            ellipse_long_axis_radius,
            ellipse_short_axis_radius,
            img_binary_ellipse_rgb,
        )

    def build_binary_image_filter(self) -> np.ndarray:
        """
        Build filter image:
            stop-band: ellipse
            pass-bands:
                - central pass-band
                - central cross (optional)
                - upper-left/lower-right quadrants (optional)

        Returns: binary image filter

        """

        # Draw the ellipse in an all-pass binary image (the "stop-band" of the filter)
        ellipse_image_binary: np.ndarray = np.ones(
            (self.img_specs.height, self.img_specs.width), dtype=np.uint8
        )
        cv.ellipse(
            img=ellipse_image_binary,
            center=[self.img_specs.x0, self.img_specs.y0],
            axes=[self.ellipse_long_axis_radius, self.ellipse_short_axis_radius],
            angle=-self.artifact.θ,
            startAngle=0.0,
            endAngle=360.0,
            color=[0],
            thickness=-1,
        )

        # Add center central pass-band (pixels around origin above
        # "threshold_center_pass_band")
        center_pass_band_image: np.ndarray = self.binarize_image(
            img=self.img_dft_mag, threshold=self.threshold_center_pass_band
        )
        if np.count_nonzero(center_pass_band_image > 0) == 0:
            raise ValueError(
                "Threshold value for center pass-band "
                f"({self.threshold_center_pass_band}) is too high!"
            )
        filter_image: np.ndarray = (
            ellipse_image_binary.astype(np.float64) + center_pass_band_image
        )

        # Add cross pass-bands around horizontal and vertical axes, if requested
        if self.cross_pass_band_width > 0:
            filter_image[
                self.img_specs.y0
                - self.cross_pass_band_width // 2 : self.img_specs.y0
                + self.cross_pass_band_width // 2
                + 1,
                :,
            ] = 1
            filter_image[
                :,
                self.img_specs.x0
                - self.cross_pass_band_width // 2 : self.img_specs.x0
                + self.cross_pass_band_width // 2
                + 1,
            ] = 1
            filter_image[filter_image == 2] = 1

        # Add pass-bands for upper-left and lower-right quadrants, if requested
        if self.pass_upper_left_lower_right_quadrants:
            filter_image[
                0 : self.img_specs.height // 2, 0 : self.img_specs.width // 2
            ] = 1
            filter_image[
                self.img_specs.height // 2 + 1 :, self.img_specs.width // 2 + 1 :
            ] = 1
            filter_image[filter_image == 2] = 1

        # For debugging/validation, show filter construction step images
        self.plot_filter_images(filter_image=filter_image)

        # Return filter
        return filter_image


def plot_line_profiles(
    img: np.ndarray,
    img_filtered: np.ndarray,
    img_specs: ImageSpecs,
    artifact: Artifact,
    λ_time_profile: float,
) -> Figure:
    """
    Plot line profiles vertically (time) at wavelength lambda_time_profile,
    horizontally (wavelength) at time mid-point, and perpendicular to the artifact
    though it's center (normal),

    Args:
        img (np.ndarray): original image
        img_filtered (np.ndarray): filtered image
        img_specs (ImageSpecs): image parameters
        artifact (Artifact): artifact parameters
        λ_time_profile (float): wavelength for vertical (time) line profile,
                                if 0, default to artifact.λ0

    Returns: Figure object

    """

    # Define figure and axes
    fig, axs = plt.subplots(3)

    # Determine array index of lambda_time_profile
    if λ_time_profile == 0:
        λ_time_profile_pixels: int = artifact.λ0_pixels
    elif λ_time_profile < img_specs.λ0 or λ_time_profile > img_specs.λ1:
        raise ValueError(
            f"λ_time_profile ({λ_time_profile}) must be between "
            rf"λ$_0$ ({img_specs.λ0:.2f}) and λ$_1$ ({img_specs.λ1:.2f})!"
        )
    else:
        λ_time_profile_pixels: int = np.abs(img_specs.λs - λ_time_profile).argmin()

    # Raw and filtered vertical (time) line profiles at λ0
    λ0_profile: np.ndarray = np.flip(img[:, λ_time_profile_pixels])
    λ0_profile_filtered: np.ndarray = np.flip(img_filtered[:, λ_time_profile_pixels])
    axs[0].plot(img_specs.ts, λ0_profile, label="Raw")
    axs[0].plot(img_specs.ts, λ0_profile_filtered, "r", label="Filtered")
    axs[0].set(
        xlabel="time (s)",
        ylabel=r"$\Delta$R/R",
        title=f"Raw and filtered vertical (time) line profiles at "
        f"{λ_time_profile if λ_time_profile != 0 else artifact.λ0:.2f} nm "
        rf"(λ$_0$ = {artifact.λ0:.2f} nm)",
        xlim=(img_specs.ts[0], img_specs.ts[-1]),
    )

    # Raw and filtered horizontal (wavelength) line profiles at vertical midpoint
    mid_horizontal_profile: np.ndarray = img[img.shape[0] // 2, :]
    mid_horizontal_profile_filtered: np.ndarray = img_filtered[img.shape[0] // 2, :]
    axs[1].plot(img_specs.λs, mid_horizontal_profile, label="Raw")
    axs[1].plot(img_specs.λs, mid_horizontal_profile_filtered, label="Filtered")
    axs[1].set(
        xlabel="λ (pm)",
        ylabel=r"$\Delta$R/R",
        title=f"Raw and filtered horizontal (wavelength) line profiles at vertical "
        f"midpoint ({img_specs.ts[img.shape[0] // 2]:.2f} ps)",
    )

    # Raw and filtered line profiles along the normal to the artifact
    num: int = 1000
    x, y = np.linspace(
        artifact.normal_x0_pixels, artifact.normal_x1_pixels, num
    ), np.linspace(artifact.normal_y0_pixels, artifact.normal_y1_pixels, num)
    normal_profile: np.ndarray = np.flip(
        ndimage.map_coordinates(img, np.vstack((y, x)))
    )
    normal_profile_filtered: np.ndarray = np.flip(
        ndimage.map_coordinates(img_filtered, np.vstack((y, x)))
    )
    x_relative: np.ndarray = np.linspace(
        -artifact.normal_length_pixels / 2,
        artifact.normal_length_pixels / 2,
        normal_profile.size,
    )
    axs[2].plot(x_relative, normal_profile, label="Raw")
    axs[2].plot(x_relative, normal_profile_filtered, "r", label="Filtered")
    axs[2].set(
        xlabel="n (pixels)",
        ylabel=r"$\Delta$R/R",
        title="Raw and filtered line profiles along the normal through the center of "
        "the artifact (bottom/left to top/right)",
    )

    # Plot legends & grids
    for ax in axs:
        ax.grid()
        ax.legend(loc="lower right")
    plt.tight_layout()

    # Return figure object for saving to file
    return fig


def plot_artifact_centerline(ax: Axes, img_specs: ImageSpecs, artifact: Artifact):
    """
    Plot the centerline of the artifact with vertical & horizontal projections

    Args:
        ax (Axes): Axes object to plot on
        img_specs (ImageSpecs): image specifications
        artifact (Artifact): artifact specifications

    Returns: None

    """

    ax.set(ylabel="δt (ps)")
    ax.plot(
        [artifact.x0, artifact.x1],
        [artifact.y0, artifact.y1],
        "k--",
    )
    ax.plot(
        [img_specs.λ0, artifact.x0],
        [artifact.y0, artifact.y0],
        "k--",
    )
    ax.plot(
        [img_specs.λ0, artifact.x1],
        [artifact.y1, artifact.y1],
        "k--",
    )
    ax.plot(
        [artifact.x0, artifact.x0],
        [img_specs.t0, artifact.y0],
        "r--",
    )
    ax.plot(
        [artifact.x1, artifact.x1],
        [img_specs.t0, artifact.y1],
        "r--",
    )


def plot_images_and_dfts(
    images: list,
    dfts: list,
    titles: list,
    img_specs: ImageSpecs,
    artifact: Artifact,
    flt: Filter,
    fname: str,
    image_cmap: str = "seismic",
) -> Figure:
    """

    Plot image & DFT pairs (raw, periodic, smooth, periodic filtered, final)

    Args:
        images (list): images
        dfts (list): DFTs
        titles (list): titles
        img_specs (ImageSpecs): image specifications
        artifact (Artifact): artifact specifications
        flt (Filter): filter specifications
        fname (str): Input image filename
        image_cmap (str): colormap for images (default = "seismic")

    Returns: matplotlib Figure class object

    """

    # Plot the image and DFT pairs
    fig, [top_axes, bot_axes] = plt.subplots(2, len(images), sharey="row")
    plt.suptitle(
        f"Transient gradient artifact filtering with smooth & periodic component "
        f"decomposition for time-resolved spectroscopy map in '{fname}'\n"
        rf"Artifact: λ$_0$ = {artifact.λ0:.1f} nm, "
        rf"$\Delta$λ = {artifact.extent_λ:.1f} nm, "
        rf"$\Delta$t = {artifact.extent_t:.2f} ps, θ = {artifact.θ:.1f}°"
        "\nFilter: "
        f"center pass-band threshold = {flt.threshold_center_pass_band}, "
        f"ellipse stop-band threshold = {flt.threshold_ellipse_stop_band}, "
        f"cross pass-band width = {flt.cross_pass_band_width} pixels, "
        "quadrant pass-band = "
        f"{flt.pass_upper_left_lower_right_quadrants}, "
        f"Gaussian blur σ = {flt.gaussian_blur_sigma:.1f}\n"
        "Top row: images, "
        "bottom row: DFT amplitudes (2X zoom, 5 decade dynamic range)\n"
    )
    dft_x_max: float = 1 / (img_specs.dλ * 2)
    dft_y_max: float = 1 / (img_specs.dt * 2)
    for img, ft, title, top_ax, bot_ax in zip(images, dfts, titles, top_axes, bot_axes):
        # Top row: images with colorbar at the right
        top_im: plt.axes.AxesImage = top_ax.imshow(
            img,
            cmap=image_cmap,
            vmin=-np.max([np.max(img) for img in images]),
            vmax=np.max([np.max(img) for img in images]),
            extent=[img_specs.λ0, img_specs.λ1, img_specs.t0, img_specs.t1],
            aspect=(img_specs.λ1 - img_specs.λ0) / (img_specs.t1 - img_specs.t0),
        )
        top_ax.set(title=f"{title}", xlabel="λ (nm)")
        if top_ax == top_axes[0]:
            top_ax.set(ylabel="δt (ps)")
            plot_artifact_centerline(ax=top_ax, img_specs=img_specs, artifact=artifact)

        # Add colorbar to the right of the last image
        if top_ax == top_axes[-1]:
            plt.colorbar(
                top_im,
                cax=fig.add_axes(
                    [
                        top_ax.get_position().x1 + 0.01,
                        top_ax.get_position().y0,
                        0.01,
                        top_ax.get_position().height,
                    ]
                ),
            )

        # Bottom row: DFT amplitudes (2X zoom, 5 decade dynamic range)
        ft_zoomed: np.ndarray = ft[
            ft.shape[0] // 4 : -ft.shape[0] // 4, ft.shape[1] // 4 : -ft.shape[1] // 4
        ]
        plt.axes.AxesImage = bot_ax.imshow(
            ft_zoomed,
            cmap="gray",
            vmin=np.max([np.max(dft) for dft in dfts]) - 5,
            vmax=np.max([np.max(dft) for dft in dfts]),
            extent=[-dft_x_max // 2, dft_x_max // 2, -dft_y_max // 2, dft_y_max // 2],
            aspect=dft_x_max / dft_y_max,
        )
        bot_ax.set(xlabel=r"|X(x)| (nm$^{-1}$)")
        if bot_ax == bot_axes[0]:
            bot_ax.set(ylabel=r"|X(y)| (ps$^{-1}$)")

    # Return Figure class object
    return fig


def write_excel_sheet(
    array_data: np.ndarray,
    array_name: str,
    df: pd.DataFrame,
    writer: pd.ExcelWriter,
):
    """
    Write a dataframe to an Excel sheet

    Args:
        array_data (np.ndarray): image data to insert into the output array
        array_name (str): name of the array to give to the sheet
        df (DataFrame): reference dataframe containing the t and λ
        writer (pd.ExcelWriter): Excel writer object

    Returns: None

    """

    sheet_array: np.ndarray = df.iloc[:, :].to_numpy()
    sheet_array[1:, 1:] = array_data
    df_local = pd.DataFrame(sheet_array, index=df.index, columns=df.columns)
    df_local.to_excel(writer, sheet_name=array_name, index=False, header=False)


def write_output_excel_file(
    fname_path: Path,
    df: pd.DataFrame,
    img_specs: ImageSpecs,
    artifact: Artifact,
    flt: Filter,
    periodic: np.ndarray,
    smooth: np.ndarray,
    periodic_filtered: np.ndarray,
    img_filtered: np.ndarray,
    img_filtered_final: np.ndarray,
):
    """
    Write data to an Excel file

    Args:
        fname_path (Path): Output filename
        df (pd.DataFrame): Dataframe with image data
        img_specs (ImageSpecs): Image Class object
        artifact (Artifact): Artifact Class object
        flt (Filter): Filter Class object
        periodic (np.ndarray): Periodic component of the image
        smooth (np.ndarray): Smooth component of the image
        periodic_filtered (np.ndarray): Filtered periodic component of the image
        img_filtered (np.ndarray): Filtered image
        img_filtered_final (np.ndarray): Final filtered image (gaussian blur applied)

    Returns: None

    """

    with pd.ExcelWriter(
        Path("output") / f"{fname_path.stem}_filtering_results.xlsx"
    ) as writer:
        df.iloc[0, 1:] = img_specs.λs
        df.iloc[1:, 0] = img_specs.ts

        write_excel_sheet(
            array_data=img_specs.img,
            array_name="Data",
            df=df,
            writer=writer,
        )
        write_excel_sheet(
            array_data=periodic,
            array_name="Periodic",
            df=df,
            writer=writer,
        )
        write_excel_sheet(
            array_data=smooth,
            array_name="Smooth",
            df=df,
            writer=writer,
        )
        write_excel_sheet(
            array_data=periodic_filtered,
            array_name="periodic_filtered",
            df=df,
            writer=writer,
        )
        write_excel_sheet(
            array_data=img_filtered,
            array_name="Data_filtered",
            df=df,
            writer=writer,
        )
        write_excel_sheet(
            array_data=img_filtered_final,
            array_name="Data_filtered_gaussian_blur",
            df=df,
            writer=writer,
        )
        write_excel_sheet(
            array_data=flt.f,
            array_name="filter_2D",
            df=df,
            writer=writer,
        )
        df_info = pd.DataFrame(
            {
                "lambda0_pump_nm": [artifact.λ0],
                "artifact_extent_wavelength_nm": [artifact.extent_λ],
                "artifact_extent_time_ps": [artifact.extent_t],
                "threshold_ellipse_stop_band": [flt.threshold_ellipse_stop_band],
                "threshold_center_pass_band": [flt.threshold_center_pass_band],
                "cross_pass_band_width": [flt.cross_pass_band_width],
                "pass_upper_left_lower_right_quadrants": [
                    flt.pass_upper_left_lower_right_quadrants
                ],
                "Gaussian blur standard deviation": [flt.gaussian_blur_sigma],
                "filter_ellipse_long_axis_radius_pixels": [
                    flt.ellipse_long_axis_radius
                ],
                "filter_ellipse_short_axis_radius_pixels": [
                    flt.ellipse_short_axis_radius
                ],
            }
        )
        df_info.to_excel(writer, sheet_name="info", index=False)

        return None


def write_output_matlab_file(
    fname_path: Path,
    img_specs: ImageSpecs,
    artifact: Artifact,
    flt: Filter,
    periodic: np.ndarray,
    smooth: np.ndarray,
    periodic_filtered: np.ndarray,
    img_filtered: np.ndarray,
    img_filtered_final: np.ndarray,
):
    """
    Write data to a Matlab file

    Args:
        fname_path (Path): Output filename
        img_specs (ImageSpecs): Image Class object
        artifact (Artifact): Artifact Class object
        flt (Filter): Filter Class object
        periodic (np.ndarray): Periodic component of the image
        smooth (np.ndarray): Smooth component of the image
        periodic_filtered (np.ndarray): Filtered periodic component of the image
        img_filtered (np.ndarray): Filtered image
        img_filtered_final (np.ndarray): Final filtered image (gaussian blur applied)

    Returns: None

    """

    savemat(
        str(Path("output") / f"{fname_path.stem}_filtering_results.mat"),
        {
            "Data": img_specs.img,
            "periodic": periodic,
            "smooth": smooth,
            "periodic_filtered": periodic_filtered,
            "Data_filtered": img_filtered,
            "Data_filtered_gaussian_blur": img_filtered_final,
            "Wavelength": img_specs.λs,
            "Time": img_specs.ts,
            "filter_2D": flt.f,
            "lambda0_pump_nm": artifact.λ0,
            "artifact_extent_wavelength_nm": artifact.extent_λ,
            "artifact_extent_time_ps": artifact.extent_t,
            "threshold_ellipse_stop_band": flt.threshold_ellipse_stop_band,
            "threshold_center_pass_band": flt.threshold_center_pass_band,
            "cross_pass_band_width": flt.cross_pass_band_width,
            "pass_upper_left_lower_right_quadrants": flt.pass_upper_left_lower_right_quadrants,
            "Gaussian blur standard deviation": flt.gaussian_blur_sigma,
            "filter_ellipse_long_axis_radius": flt.ellipse_long_axis_radius,
            "filter_ellipse_short_axis_radius": flt.ellipse_short_axis_radius,
        },
    )


def calc_dft_log_magnitude(img) -> np.ndarray:
    """
    Calculate DFT log magnitude image (add small offset so null DC doesn't break log)

    Args:
        img (np.ndarray): input image

    Returns (np.ndarray): DFT log magnitude image

    """

    return np.log10(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1e-10)


def transient_grating_artifact_filter(
    fname: str,
    lambda0_pump: float,
    artifact_extent_lambda: float,
    artifact_extent_t: float,
    threshold_ellipse_stop_band: float,
    threshold_center_pass_band: float,
    upper_left_lower_right_quadrant_pass_band: bool = True,
    cross_pass_band_width: int = 0,
    gaussian_blur_sigma: float = 3,
    lambda_time_profile: float = 0,
    interpolate_image_to_power_of_two: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Separate an image into smooth and periodic components, filter the periodic component
    in the Fourier domain to remove the artifact, sum the result with the smooth
    component, apply light Gaussian blur to remove high frequency noise,and return
    the final result.

    The input file is assumed to be in the ./data subdirectory.
    The output image files are written to the ./output subdirectory.

    Args:
        fname (str): input filename in the ./data subdirectory (Matlab, Excel, or .csv)
        lambda0_pump (float): Pump central wavelength (nm)
        artifact_extent_lambda (float): Artifact extent in the λ direction (nm)
        artifact_extent_t (float): Artifact extent in the t direction (ps)
        threshold_ellipse_stop_band (float): threshold for filter ellipse stop-band
                                             pixel identification and segmentation
                                             ([0..1])
        threshold_center_pass_band (float): threshold for central pass-band
                                            pixel identification and segmentation
                                            ([0..1])
        upper_left_lower_right_quadrant_pass_band (bool): Pass upper left and
                                             lower right quadrants of the filter
                                             (default = True)
        cross_pass_band_width (int): width of cross pass-band in filter (default = 0)
        gaussian_blur_sigma (float): standard deviation of the Gaussian blur applied
                                             to the final result (default = 3)
        lambda_time_profile (float): wavelength at which the time profile is plotted
        interpolate_image_to_power_of_two (bool): Interpolate image dimensions to
                                             nearest larger power of two
                                             (default = False)

    Returns: re-sampled raw image (np.ndarray)
             periodic image component (np.ndarray)
             smooth image component (np.ndarray)
             filtered image (np.ndarray)
             filtered periodic component (np.ndarray)
             binary filter image (np.ndarray)

    """

    # Show script name & version and python interpreter version on console
    python_version = (
        f"{str(sys.version_info.major)}"
        f".{str(sys.version_info.minor)}"
        f".{str(sys.version_info.micro)}"
    )
    print(
        f"{os.path.basename(__file__)} v{__version__} (running Python {python_version})"
    )

    # Load time-resolved 2D spectroscopy data from the input file
    fname_path: Path = Path(f"{fname}")
    img_in: np.ndarray
    λs_in: np.ndarray
    ts_in: np.ndarray
    df: pd.DataFrame = pd.DataFrame()
    if fname_path.suffix == ".mat":
        matlab_data: dict = loadmat(str(Path("data") / fname_path))
        img_in = matlab_data["Data"]
        λs_in = matlab_data["Wavelength"].flatten()
        ts_in = matlab_data["Time"].flatten()
    elif fname_path.suffix in [".csv", ".xlsx", ".xls"]:
        df = pd.read_excel(str(Path("data") / fname_path), header=None)
        img_in = df.iloc[1:, 1:].to_numpy()
        λs_in = df.iloc[0, 1:].to_numpy()
        ts_in = df.iloc[1:, 0].to_numpy()
    else:
        raise ValueError(f"Input file '{fname}' is not a Matlab, Excel, or .csv file!")

    # Create ImageSpecs class object containing time-resolved spectroscopy data,
    # interpolate the data on a regular grid to account for any inhomogeneous
    # time and/or wavelength sampling
    img_specs: ImageSpecs = ImageSpecs(
        λs_in=λs_in,
        ts_in=ts_in,
        img_in=img_in,
        interpolate_image_to_power_of_two=interpolate_image_to_power_of_two,
    )

    # Create Artifact class object containing artifact specifications
    artifact: Artifact = Artifact(
        λ0=lambda0_pump,
        extent_λ=artifact_extent_lambda,
        extent_t=artifact_extent_t,
        img_specs=img_specs,
    )

    # Extract periodic and smooth component DFTs from the input image ([Moisan, 2010]),
    # then calculate the corresponding component images from their inverse DFTs
    periodic_dft, smooth_dft = per(img_specs.img, inverse_dft=False)
    periodic_dft_mag, smooth_dft_mag = np.log10(
        np.abs(np.fft.fftshift(periodic_dft)) + 1e-10
    ), np.log10(np.abs(np.fft.fftshift(smooth_dft)) + 1e-10)
    periodic, smooth = np.real(np.fft.ifft2(periodic_dft)), np.real(
        np.fft.ifft2(smooth_dft)
    )

    # Create the Filter class object containing the filter specifications
    flt: Filter = Filter(
        img_dft_mag=periodic_dft_mag,
        threshold_ellipse_stop_band=threshold_ellipse_stop_band,
        threshold_center_pass_band=threshold_center_pass_band,
        img_specs=img_specs,
        artifact=artifact,
        cross_pass_band_width=cross_pass_band_width,
        pass_upper_left_lower_right_quadrants=upper_left_lower_right_quadrant_pass_band,
        gaussian_blur_sigma=gaussian_blur_sigma,
    )

    # Filter the artifact from the periodic component, reconstruct the filtered image
    # (smooth component + filtered periodic component), and apply a light Gaussian blur
    periodic_filtered_dft: np.ndarray = periodic_dft * np.fft.fftshift(flt.f)
    periodic_filtered: np.ndarray = np.real(np.fft.ifft2(periodic_filtered_dft))
    img_filtered: np.ndarray = smooth + periodic_filtered
    img_filtered_blur: np.ndarray = ndimage.gaussian_filter(
        input=img_filtered, sigma=flt.gaussian_blur_sigma
    )

    # Calculate various DFT magnitude images for plotting and saving
    img_dft_mag: np.ndarray = calc_dft_log_magnitude(img_specs.img)
    periodic_filtered_dft_mag: np.ndarray = calc_dft_log_magnitude(periodic_filtered)
    img_filtered_dft_mag: np.ndarray = calc_dft_log_magnitude(img_filtered)
    img_filtered_final_dft_mag: np.ndarray = calc_dft_log_magnitude(img_filtered_blur)

    # Plot line profiles in along the time, wavelength axes, and normal to the artifact
    fig_line_profiles: Figure = plot_line_profiles(
        img=img_specs.img,
        img_filtered=img_filtered_blur,
        img_specs=img_specs,
        artifact=artifact,
        λ_time_profile=lambda_time_profile,
    )

    # Display the various image & DFT pairs
    fig_images_and_dfts: Figure = plot_images_and_dfts(
        images=[
            img_specs.img,
            periodic,
            smooth,
            periodic_filtered,
            img_filtered,
            img_filtered_blur,
            img_specs.img - img_filtered,
        ],
        dfts=[
            img_dft_mag,
            periodic_dft_mag,
            smooth_dft_mag,
            periodic_filtered_dft_mag,
            img_filtered_dft_mag,
            img_filtered_final_dft_mag,
            img_dft_mag - img_filtered_dft_mag,
        ],
        titles=[
            "\nRaw image\nu = s + u",
            "\nPeriodic\ncomponent\nu",
            "\nSmooth\ncomponent\ns",
            "Filtered periodic\ncomponent\nuf",
            "Filtered image\nuf = s + uf",
            "Filtered image\n"
            f"+gaussian blur (σ={flt.gaussian_blur_sigma:.1f})\n"
            "Blur(s + uf)",
            "Artifact\nu - uf",
        ],
        img_specs=img_specs,
        artifact=artifact,
        flt=flt,
        fname=fname,
    )

    # Save results to the ./output subdirectory (create it, if it doesn't exist)
    Path("output").mkdir(parents=True, exist_ok=True)
    fig_images_and_dfts.savefig(
        Path("output") / f"{fname_path.stem}_images_and_dfts.png"
    )
    fig_line_profiles.savefig(Path("output") / f"{fname_path.stem}_line_profiles.png")
    if fname_path.suffix == ".mat":
        write_output_matlab_file(
            fname_path=fname_path,
            img_specs=img_specs,
            artifact=artifact,
            flt=flt,
            periodic=periodic,
            smooth=smooth,
            periodic_filtered=periodic_filtered,
            img_filtered=img_filtered,
            img_filtered_final=img_filtered_blur,
        )
    else:
        write_output_excel_file(
            fname_path=fname_path,
            df=df,
            img_specs=img_specs,
            artifact=artifact,
            flt=flt,
            periodic=periodic,
            smooth=smooth,
            periodic_filtered=periodic_filtered,
            img_filtered=img_filtered,
            img_filtered_final=img_filtered_blur,
        )

    # Display figures
    plt.show()

    # Return results
    return img_specs.img, periodic, smooth, img_filtered_blur, periodic_filtered, flt.f
