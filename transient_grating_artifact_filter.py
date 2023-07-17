"""transient_grating_artifact_filter.py

Transient gradient artifact filtering in the Fourier domain from 2D time-resolved
spectroscopy map

First separate the input image (time-resolved spectroscopy map) into "smooth" and
"periodic" components as per [Moisan, 2010] to reduce the effect of the "cross" pattern
in the Discrete Fourier transform due to the non-periodic nature of the image
(see https://github.com/sbrisard/moisan2011), then filter the artifact from the periodic
component in the Fourier domain using an ellipse with a cutout at the center
to preserve the low-frequency content of the baseline data, and finally recombine
the filtered periodic component with the smooth component o generate the filtered map.

"""

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from moisan2011 import per
from pathlib import Path
from scipy import ndimage
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator
from skimage.draw import line
from typing import Tuple

# Script version
__version__: str = "2.9"


@dataclass
class ImageSpecs:
    """
    Spectroscopy image specifications

    λs_in (np.ndarray): input array of wavelengths (nm)
    ts_in (np.ndarray): input array of times (ps)
    height (int): scaled image height (pixels)
    width (int): scaled image width (pixels

    """

    λs_in: np.ndarray
    ts_in: np.ndarray
    img_in: np.ndarray
    interpolate_image_to_power_of_two: bool = False

    def __post_init__(self):
        # Check that input wavelength and time arrays match input image dimensions
        if (
            len(self.λs_in) != self.img_in.shape[1]
            or len(self.ts_in) != self.img_in.shape[0]
        ):
            raise ValueError(
                "Input file array dimensions are inconsistent "
                "(mismatch between Data vs Wavelength and/or Time array dimensions)!"
            )

        # Interpolate image dimensions to nearest larger power of two, if requested
        if self.interpolate_image_to_power_of_two:
            self.height, self.width = int(
                2 ** np.ceil(np.log2(self.img_in.shape[0]))
            ), int(2 ** np.ceil(np.log2(self.img_in.shape[1])))
        else:
            self.height, self.width = self.img_in.shape

        # Image dimensions center pixel coordinates
        self.x0 = self.width // 2
        self.y0 = self.height // 2

        # Resample the input arrays on a regular even-spaced grid to account for any
        # non-uniform time or frequency sampling
        self.λ0: float = self.λs_in[0]
        self.λ1: float = self.λs_in[-1]
        self.λs: np.ndarray = np.linspace(self.λ0, self.λ1, self.width)
        self.dλ: float = self.λs[1] - self.λs[0]
        self.t0: float = self.ts_in[0]
        self.t1: float = self.ts_in[-1]
        self.ts: np.ndarray = np.linspace(self.t0, self.t1, self.height)
        self.dt: float = self.ts[1] - self.ts[0]
        interp = RegularGridInterpolator((self.ts_in, self.λs_in), self.img_in)
        yi, xi = np.meshgrid(self.ts, self.λs, indexing="ij")
        self.img: np.ndarray = interp((yi, xi))


@dataclass
class Artifact:
    """
    Artifact specifications

    λ0 (float): artifact center wavelength (pump center wavelength, nm)
    extent_λ (float): artifact extent along the wavelength axis (nm)
    extent_t (float): artifact extent along the time axis (ps)
    img_specs (ImageSpecs): image specifications

    """

    λ0: float
    extent_λ: float
    extent_t: float
    img_specs: ImageSpecs

    def __post_init__(self):
        # Artifact geometry in units of pixels
        self.λ0_pixels: int = np.abs(self.img_specs.λs - self.λ0).argmin()
        self.t0_pixels: int = (
            self.img_specs.height - np.abs(self.img_specs.ts - 0).argmin()
        )
        self.extent_λ_pixels: float = self.extent_λ / self.img_specs.dλ
        self.extent_t_pixels: float = self.extent_t / self.img_specs.dt
        self.angle: float = 90 - (
            np.degrees(
                np.arctan(
                    (self.extent_t_pixels * self.img_specs.width)
                    / (self.extent_λ_pixels * self.img_specs.height)
                )
            )
        )

        # Equation and coordinate endpoints of line though center of the artifact
        slope: float = self.extent_t_pixels / self.extent_λ_pixels
        intercept: float = self.t0_pixels - slope * self.λ0_pixels
        self.x0_pixels: int = int(self.λ0_pixels - self.extent_λ_pixels // 2)
        self.x1_pixels: int = int(self.λ0_pixels + self.extent_λ_pixels // 2)
        self.y0_pixels: int = int(slope * self.x0_pixels + intercept)
        self.y1_pixels: int = int(slope * self.x1_pixels + intercept)

        # Equation and coordinate endpoints of normal through center of the artifact
        self.normal_y0_pixels: int = self.y0_pixels
        self.normal_y1_pixels: int = self.y1_pixels
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

        # Convert pixel to data coordinates
        self.x0: float = self.img_specs.λs[self.x0_pixels]
        self.x1: float = self.img_specs.λs[self.x1_pixels]
        self.y0: float = self.img_specs.ts[self.img_specs.height - self.y0_pixels]
        self.y1: float = self.img_specs.ts[self.img_specs.height - self.y1_pixels]
        self.normal_x0: float = self.img_specs.λs[self.normal_x0_pixels]
        self.normal_x1: float = self.img_specs.λs[self.normal_x1_pixels]
        self.normal_y0: float = self.img_specs.ts[
            self.img_specs.height - self.normal_y0_pixels
        ]
        self.normal_y1: float = self.img_specs.ts[
            self.img_specs.height - self.normal_y1_pixels
        ]


@dataclass
class Filter:
    """
    Filter parameters

    img_dft_mag (np.ndarray): DFT magnitude image used to define the filter
    threshold_ellipse (float): threshold for defining the ellipse
    threshold_cutout (float): threshold for defining the cutout
    img_specs (ImageSpecs): image specifications
    artifact (Artifact): artifact specifications
    ellipse_padding (float) = extra padding around thresholded pixels for filter ellipse
                             ([0..1], disabled if == 0)
    cross_pass_band_width (int) = width of cross-shaped pass band along horizontal
                                  and vertical axes in the Fourier domain, cutout from
                                  the filter to pass any remaining non-periodic
                                  content left over from the smooth/periodic
                                  decomposition (disabled if < 0)
    pass_upper_left_lower_right_quadrants (bool) = pass (do not filter) the upper-left
                                                   and lower-right quadrants of Fourier
                                                   space, excluding the horizontal and
                                                   vertical axes that can be passed with
                                                   cross_pass_band_width > 0
    gaussian_blur: int = gaussian blur kernel size applied to the fileter to
                         reduce ringing (disabled if < 0)

    """

    img_dft_mag: np.ndarray
    threshold_ellipse: float
    threshold_cutout: float
    img_specs: ImageSpecs
    artifact: Artifact
    ellipse_padding: float
    cross_pass_band_width: int
    pass_upper_left_lower_right_quadrants: bool
    gaussian_blur: int

    def __post_init__(self):
        (
            self.ellipse_long_axis_radius,
            self.ellipse_short_axis_radius,
            self.img_binary_ellipse_rgb,
        ) = self.define_filter_ellipse()
        self.f: np.ndarray = self.build_filter()

    @staticmethod
    def binarize_image(img: np.ndarray, threshold: float) -> np.ndarray:
        """
        Binarize image using a threshold

        Args:
            img (np.ndarray): image to be converted to binary
            threshold (float): threshold [0, 1]

        Returns: binary image (np.ndarray)

        """

        # Normalize image to [0, 1], limit to 4 decades of dynamic range, else small pixel
        # values will skew the normalization since values are on a log scale.
        img[img < np.amax(img) - 4] = np.amax(img) - 4
        img_normalized: np.ndarray = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)

        # Binarize image with threshold, clean up with morphological opening and closing
        img_binary: np.ndarray = cv.threshold(
            img_normalized, threshold, 1, cv.THRESH_BINARY
        )[1]
        img_binary_open: np.ndarray = cv.morphologyEx(
            img_binary, cv.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        img_binary_final: np.ndarray = cv.morphologyEx(
            img_binary_open, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        return img_binary_final

    @staticmethod
    def calc_line_length(
        img_binary: np.ndarray, diagonal_pixel_coordinates: np.ndarray
    ) -> float:
        """
        Calculate the length of a line in a binary image along a diagonal

        Args:
            img_binary (np.ndarray): binary image to be analyzed
            diagonal_pixel_coordinates (np.ndarray): array of pixel coordinates along diagonal

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

    def define_filter_ellipse(self) -> Tuple[int, int, np.ndarray]:
        """

        Determine filter ellipse parameters from periodic component DFT magnitude

        Returns: ellipse_long_axis_radius (int), ellipse_short_axis_radius (int)
                 img_binary_ellipse_rgb (np.ndarray)

        """

        # Determine ellipse long & short axis lengths (lengths of line along the artifact
        # diagonal and normal to the diagonal)
        img_binary_ellipse: np.ndarray = self.binarize_image(
            img=self.img_dft_mag, threshold=self.threshold_ellipse
        )
        l: float = min(self.img_dft_mag.shape) / 2.5
        artifact_long_diagonal_pixel_coordinates: np.ndarray = line(
            r0=int(self.img_specs.y0 - l * np.sin(np.radians(self.artifact.angle))),
            c0=int(self.img_specs.x0 + l * np.cos(np.radians(self.artifact.angle))),
            r1=int(self.img_specs.y0 + l * np.sin(np.radians(self.artifact.angle))),
            c1=int(self.img_specs.x0 - l * np.cos(np.radians(self.artifact.angle))),
        )
        artifact_short_diagonal_pixel_coordinates: np.ndarray = line(
            r0=int(
                self.img_specs.y0 - l * np.sin(np.radians(self.artifact.angle + 90))
            ),
            c0=int(
                self.img_specs.x0 + l * np.cos(np.radians(self.artifact.angle + 90))
            ),
            r1=int(
                self.img_specs.y0 + l * np.sin(np.radians(self.artifact.angle + 90))
            ),
            c1=int(
                self.img_specs.x0 - l * np.cos(np.radians(self.artifact.angle + 90))
            ),
        )
        ellipse_long_axis_radius: int = int(
            (
                self.calc_line_length(
                    img_binary=img_binary_ellipse,
                    diagonal_pixel_coordinates=artifact_long_diagonal_pixel_coordinates,
                )
                * (1.0 + self.ellipse_padding)
            )
            // 2
        )
        ellipse_short_axis_radius: int = int(
            (
                self.calc_line_length(
                    img_binary=img_binary_ellipse,
                    diagonal_pixel_coordinates=artifact_short_diagonal_pixel_coordinates,
                )
                * (1.0 + self.ellipse_padding)
            )
            // 2
        )
        if ellipse_long_axis_radius == 0 or ellipse_short_axis_radius == 0:
            raise ValueError(
                f"Threshold value for ellipse ({self.threshold_ellipse}) is too high!"
            )

        # Build image of ellipse above-threshold pixels and resulting shape
        img_binary_ellipse_rgb = np.repeat(
            img_binary_ellipse[:, :, np.newaxis], 3, axis=2
        )
        img_binary_ellipse_rgb[artifact_long_diagonal_pixel_coordinates] = [1, 0, 0]
        img_binary_ellipse_rgb[artifact_short_diagonal_pixel_coordinates] = [1, 0, 0]
        cv.ellipse(
            img_binary_ellipse_rgb,
            (self.img_specs.x0, self.img_specs.y0),
            (ellipse_long_axis_radius, ellipse_short_axis_radius),
            -self.artifact.angle,
            0,
            360,
            (1, 0, 0),
            1,
        )

        return (
            ellipse_long_axis_radius,
            ellipse_short_axis_radius,
            img_binary_ellipse_rgb,
        )

    def build_filter(self) -> np.ndarray:
        """
        Build filter image

        Returns: binary image filter

        """

        # Draw ellipses (filled, and outline only for debugging)
        ellipse_image_binary: np.ndarray = np.ones(
            (self.img_specs.height, self.img_specs.width), dtype=np.uint8
        )
        ellipse_image_binary_outline: np.ndarray = np.zeros(
            (self.img_specs.height, self.img_specs.width), dtype=np.uint8
        )
        cv.ellipse(
            ellipse_image_binary,
            (self.img_specs.x0, self.img_specs.y0),
            (self.ellipse_long_axis_radius, self.ellipse_short_axis_radius),
            -self.artifact.angle,
            0,
            360,
            0,
            -1,
        )
        cv.ellipse(
            ellipse_image_binary_outline,
            (self.img_specs.x0, self.img_specs.y0),
            (self.ellipse_long_axis_radius, self.ellipse_short_axis_radius),
            -self.artifact.angle,
            0,
            360,
            1,
            1,
        )

        # Remove cutout from center of ellipse (pixels around origin above threshold)
        cutout_image: np.ndarray = self.binarize_image(
            img=self.img_dft_mag, threshold=self.threshold_cutout
        )
        if np.count_nonzero(cutout_image > 0) == 0:
            raise ValueError(
                f"Threshold value for cutout ({self.threshold_cutout}) is too high!"
            )
        filter_image: np.ndarray = (
            ellipse_image_binary.astype(np.float64) + cutout_image
        )

        # Pass (do not filter) bands around horizontal and vertical axes, if requested
        if self.cross_pass_band_width > 0:
            filter_image[
            self.img_specs.y0
            - self.cross_pass_band_width // 2: self.img_specs.y0
                                               + self.cross_pass_band_width // 2
                                               + 1,
                :,
            ] = 1
            filter_image[
                :,
            self.img_specs.x0
            - self.cross_pass_band_width // 2: self.img_specs.x0
                                               + self.cross_pass_band_width // 2
                                               + 1,
            ] = 1
            filter_image[filter_image == 2] = 1

        # Pass (do not filter) upper-left and lower-right quadrants, if requested
        if self.pass_upper_left_lower_right_quadrants:
            filter_image[
                0 : self.img_specs.height // 2, 0 : self.img_specs.width // 2
            ] = 1
            filter_image[
                self.img_specs.height // 2 + 1 :, self.img_specs.width // 2 + 1 :
            ] = 1
            filter_image[filter_image == 2] = 1

        # Gaussian blur the filter shape to reduce ringing, if requested
        if self.gaussian_blur > 0:
            filter_image = cv.GaussianBlur(
                filter_image,
                (self.gaussian_blur, self.gaussian_blur),
                sigmaX=0,
                sigmaY=0,
            )

        # Show images of thresholded pixels and filter
        fig, axs = plt.subplots(3)

        # Draw thresholded ellipse binary image with major/minor axes
        axs[0].set(
            title="Filter ellipse binary image: threshold = "
            f"{self.threshold_ellipse:.2f}, "
            f"long axis radius = {self.ellipse_long_axis_radius} pixels, "
            f"short axis radius = {self.ellipse_short_axis_radius} pixels"
            f" ({self.ellipse_padding * 100:.0f}% ellipse padding)"
        )
        axs[0].imshow(self.img_binary_ellipse_rgb, cmap="gray")

        # Draw binary filter image
        axs[1].set(title="Filter image")
        axs[1].imshow(filter_image, cmap="gray")

        # Draw filter ellipse outline over periodic component DFT
        axs[2].set(
            title="Filter ellipse outline over binarized periodic component DFT"
            "\nNB: data limited to 4 decade dynamic range in binarization"
        )
        periodic_with_ellipse_outline: np.ndarray = np.copy(self.img_dft_mag)
        periodic_with_ellipse_outline[ellipse_image_binary_outline == 1] = np.max(
            self.img_dft_mag
        )
        axs[2].imshow(periodic_with_ellipse_outline, cmap="gray")

        plt.tight_layout()

        # Return filter
        return filter_image


def plot_line_profiles(
    img: np.ndarray,
    img_filtered: np.ndarray,
    img_specs: ImageSpecs,
    artifact: Artifact,
) -> Figure:
    """
    Plot line profiles vertically at λ0, horizontally at mid-point,
    and along the normal to the artifact.

    Args:
        img (np.ndarray): original image
        img_filtered (np.ndarray): filtered image
        img_specs (ImageSpecs): image parameters
        artifact (Artifact): artifact parameters

    Returns: Figure object

    """

    # Define figure and axes
    fig, axs = plt.subplots(3)

    # Raw and filtered vertical line profiles at λ0
    λ0_profile: np.ndarray = np.flip(img[:, artifact.λ0_pixels])
    λ0_profile_filtered: np.ndarray = np.flip(img_filtered[:, artifact.λ0_pixels])
    axs[0].plot(img_specs.ts, λ0_profile, label="Raw")
    axs[0].plot(img_specs.ts, λ0_profile_filtered, "r", label="Filtered")
    axs[0].set(
        xlabel="time (s)",
        ylabel=r"$\Delta$R/R",
        title=f"Raw and filtered vertical line profiles at λ0 ({artifact.λ0:.2f} nm)",
        xlim=(img_specs.ts[0], img_specs.ts[-1]),
    )

    # Raw and filtered horizontal line profiles at vertical midpoint
    mid_horizontal_profile: np.ndarray = img[img.shape[0] // 2, :]
    mid_horizontal_profile_filtered: np.ndarray = img_filtered[img.shape[0] // 2, :]
    axs[1].plot(img_specs.λs, mid_horizontal_profile, label="Raw")
    axs[1].plot(img_specs.λs, mid_horizontal_profile_filtered, label="Filtered")
    axs[1].set(
        xlabel="λ (pm)",
        ylabel=r"$\Delta$R/R",
        title=f"Raw and filtered horizontal line profiles at vertical midpoint "
        f"({img_specs.ts[img.shape[0] // 2]:.2f} ps)",
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
        ax.legend(loc="upper left")
    plt.tight_layout()

    # Return figure object for saving to file
    return fig


def plot_artifact_centerline(ax: Axes, img_specs: ImageSpecs, artifact: Artifact):
    """
    Plot the centerline of the artifact.

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

    Plot image & DFT pairs (original, periodic, smooth, periodic filtered, final)

    Args:
        images (list): images
        dfts (list): DFTs
        titles (list): titles
        img_specs (ImageSpecs): image specifications
        artifact (Artifact): artifact specifications
        flt (Filter): filter specifications
        fname (str): Input image filename
        image_cmap (str): colormap for images

    Returns: matplotlib Figure class object

    """

    # Plot the image and DFT pairs
    fig, [top_axes, bot_axes] = plt.subplots(2, len(images), sharey="row")
    plt.suptitle(
        f"Transient gradient artifact filtering with smooth & periodic component "
        f"decomposition ([Moisan, 2010]) for image in '{fname}'\n"
        "Filtering parameters : "
        f"ellipse long axis radius = {flt.ellipse_long_axis_radius} pixels, "
        f"ellipse short axis radius = {flt.ellipse_short_axis_radius} pixels\n"
        f"Artifact parameters : λ0 = {artifact.λ0:.1f} nm, "
        f"extent λ = {artifact.extent_λ:.1f} nm, "
        f"extent t = {artifact.extent_t:.2f} ps\n"
        "Top row: images\n"
        "Bottom row: DFT amplitudes (2X zoom, 5 decade dynamic range)"
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

    # Show plots, return Figure class object
    plt.show()
    return fig


def write_excel_sheet(
    sheet_array: np.ndarray,
    array_data: np.ndarray,
    array_name: str,
    df: pd.DataFrame,
    writer: pd.ExcelWriter,
):
    sheet_array[1:, 1:] = array_data
    df_local = pd.DataFrame(sheet_array, index=df.index, columns=df.columns)
    df_local.to_excel(writer, sheet_name=array_name, index=False, header=False)


def transient_grating_artifact_filter(
    fname: str,
    lambda0_pump: float,
    artifact_extent_lambda: float,
    artifact_extent_t: float,
    threshold_ellipse: float,
    threshold_cutout: float,
    ellipse_padding=0.20,
    cross_pass_band_width=0,
    pass_upper_left_lower_right_quadrants=False,
    gaussian_blur=0,
    interpolate_image_to_power_of_two: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Separate an image into smooth and periodic components as per [Moisan, 2010], filter
    the periodic component in the Fourier domain to remove the coherent artifact, then
    sum with the smooth component to generate the final result.

    The input file is assumed to be in the ./data subdirectory.
    The output image files are written to the ./output subdirectory.

    Args:
        fname (str): input filename in the ./data subdirectory (Matlab, Excel, or .csv)
        lambda0_pump (float): Pump central wavelength (nm)
        artifact_extent_lambda (float): Artifact extent in the λ direction (nm)
        artifact_extent_t (float): Artifact extent in the t direction (ps)
        threshold_ellipse (float): threshold for filter ellipse identification ([0..1])
        threshold_cutout (float): threshold for filter cutout identification ([0..1])
        ellipse_padding (float): padding for filter ellipse size
                                 (default = 0.20, i.e. +20%)
        cross_pass_band_width (int): width of cross cutout in filter (default = 0)
        pass_upper_left_lower_right_quadrants (bool): Pass upper left and lower right
                                              quadrants of the filter (default = False)
        gaussian_blur (int): width of Gaussian blur kernel in filter (default = 0,
                             i.e. no blur)
        interpolate_image_to_power_of_two (bool): Interpolate image dimensions to
                                                  nearest larger power of two
                                                  (default = False)

    Returns: periodic image component (np.ndarray)
             smooth image component (np.ndarray)
             filtered image (np.ndarray)
             filtered periodic component (np.ndarray)
             binary filter image (np.ndarray)

    """

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
    plt.ion()

    # Check input parameters
    if threshold_ellipse > threshold_cutout:
        raise ValueError(
            f"threshold_ellipse ({threshold_ellipse})"
            f"must be less than threshold_cutout ({threshold_cutout})!"
        )

    # Load 2D spectroscopy measurement data from input file (Matlab, Excel, or .csv)
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

    # Create ImageSpecs class object containing spectroscopy image specifications
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

    # Extract periodic and smooth component DFTs from input image, calculate
    # corresponding component images from the inverse DFTs
    periodic_dft, smooth_dft = per(img_specs.img, inverse_dft=False)
    periodic_dft_mag, smooth_dft_mag = np.log10(
        np.abs(np.fft.fftshift(periodic_dft)) + 1e-10
    ), np.log10(np.abs(np.fft.fftshift(smooth_dft)) + 1e-10)
    periodic, smooth = np.real(np.fft.ifft2(periodic_dft)), np.real(
        np.fft.ifft2(smooth_dft)
    )

    # Design ellipse-shaped filter with cutout at center
    img_dft_mag: np.ndarray = np.log10(
        np.abs(np.fft.fftshift(np.fft.fft2(img_specs.img))) + 1e-10
    )
    flt: Filter = Filter(
        img_dft_mag=periodic_dft_mag,
        threshold_ellipse=threshold_ellipse,
        threshold_cutout=threshold_cutout,
        img_specs=img_specs,
        artifact=artifact,
        ellipse_padding=ellipse_padding,
        cross_pass_band_width=cross_pass_band_width,
        pass_upper_left_lower_right_quadrants=pass_upper_left_lower_right_quadrants,
        gaussian_blur=gaussian_blur,
    )

    # Filter periodic component in the Fourier domain, reconstruct filtered image
    periodic_filtered_dft: np.ndarray = periodic_dft * np.fft.fftshift(flt.f)
    periodic_filtered: np.ndarray = np.real(np.fft.ifft2(periodic_filtered_dft))
    img_filtered: np.ndarray = smooth + periodic_filtered

    # Calculate DFT log magnitude images (add small offset so null DC doesn't break log)
    img_filtered_dft: np.ndarray = np.fft.fft2(img_filtered)
    img_filtered_dft_mag: np.ndarray = np.log10(
        np.abs(np.fft.fftshift(img_filtered_dft)) + 1e-10
    )
    periodic_filtered_dft_mag: np.ndarray = np.log10(
        np.abs(np.fft.fftshift(periodic_filtered_dft)) + 1e-10
    )

    # Plot line profile at λ0 and perpendicular to artifact though it's center (normal)
    fig_normal: Figure = plot_line_profiles(
        img=img_specs.img,
        img_filtered=img_filtered,
        img_specs=img_specs,
        artifact=artifact,
    )

    # Plot the image & DFT pairs (original, periodic, smooth, filtered u, filtered u)
    fig_images_and_dfts: Figure = plot_images_and_dfts(
        images=[
            img_specs.img,
            periodic,
            smooth,
            periodic_filtered,
            img_filtered,
            img_specs.img - img_filtered,
        ],
        dfts=[
            img_dft_mag,
            periodic_dft_mag,
            smooth_dft_mag,
            periodic_filtered_dft_mag,
            img_filtered_dft_mag,
            img_dft_mag - img_filtered_dft_mag,
        ],
        titles=[
            "Original image (u = s + u)",
            "Periodic comp. (u)",
            "Smooth comp. (s)",
            "Filt. periodic comp. (uf)",
            "Filt. image (uf = s + uf)",
            "Artifact (u - uf)",
        ],
        img_specs=img_specs,
        artifact=artifact,
        flt=flt,
        fname=fname,
    )
    plt.show()

    # Save results to the ./output subdirectory
    fig_images_and_dfts.savefig(
        Path("output") / f"{fname_path.stem}_images_and_dfts.png"
    )
    fig_normal.savefig(Path("output") / f"{fname_path.stem}_normal.png")
    if fname_path.suffix == ".mat":
        savemat(
            str(Path("output") / f"{fname_path.stem}_filtering_results.mat"),
            {
                "Data": img_specs.img,
                "periodic": periodic,
                "smooth": smooth,
                "periodic_filtered": periodic_filtered,
                "Data_filtered": img_filtered,
                "Wavelength": img_specs.λs,
                "Time": img_specs.ts,
                "filter_2D": flt.f,
                "lambda0_pump_nm": lambda0_pump,
                "artifact_extent_wavelength_nm": artifact.extent_λ,
                "artifact_extent_time_ps": artifact.extent_t,
                "threshold_ellipse": threshold_ellipse,
                "threshold_cutout": threshold_cutout,
                "ellipse_padding": flt.ellipse_padding,
                "cross_pass_band_width": flt.cross_pass_band_width,
                "gaussian_blur": flt.gaussian_blur,
                "filter_ellipse_long_axis_radius": flt.ellipse_long_axis_radius,
                "filter_ellipse_short_axis_radius": flt.ellipse_short_axis_radius,
            },
        )
    else:
        with pd.ExcelWriter(
            Path("output") / f"{fname_path.stem}_filtering_results.xlsx"
        ) as writer:
            df.iloc[0, 1:] = img_specs.λs
            df.iloc[1:, 0] = img_specs.ts
            sheet_array: np.ndarray = df.iloc[:, :].to_numpy()
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=img_specs.img,
                array_name="Data",
                df=df,
                writer=writer,
            )
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=periodic,
                array_name="Periodic",
                df=df,
                writer=writer,
            )
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=smooth,
                array_name="Smooth",
                df=df,
                writer=writer,
            )
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=periodic_filtered,
                array_name="periodic_filtered",
                df=df,
                writer=writer,
            )
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=img_filtered,
                array_name="Data_filtered",
                df=df,
                writer=writer,
            )
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=flt.f,
                array_name="filter_2D",
                df=df,
                writer=writer,
            )
            df_info = pd.DataFrame(
                {
                    "lambda0_pump_nm": [lambda0_pump],
                    "artifact_extent_wavelength_nm": [artifact.extent_λ],
                    "artifact_extent_time_ps": [artifact.extent_t],
                    "threshold_ellipse": [threshold_ellipse],
                    "threshold_cutout": [threshold_cutout],
                    "ellipse_padding": [flt.ellipse_padding],
                    "cross_pass_band_width": [flt.cross_pass_band_width],
                    "gaussian_blur": [flt.gaussian_blur],
                    "filter_ellipse_long_axis_radius_pixels": [
                        flt.ellipse_long_axis_radius
                    ],
                    "filter_ellipse_short_axis_radius_pixels": [
                        flt.ellipse_short_axis_radius
                    ],
                }
            )
            df_info.to_excel(writer, sheet_name="info", index=False)

    # Return results
    return periodic, smooth, img_filtered, periodic_filtered, flt.f
