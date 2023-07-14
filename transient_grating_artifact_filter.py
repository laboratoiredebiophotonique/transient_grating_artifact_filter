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
from skimage.draw import line
from typing import Tuple

# Script version
__version__: str = "2.2"


@dataclass
class ImageSpecs:
    """
    Spectroscopy image specifications

    λs_unscaled (np.ndarray): unscaled array of wavelengths (nm)
    ts_unscaled (np.ndarray): unscaled array of times (ps)
    height (int): scaled image height (pixels)
    width (int): scaled image width (pixels

    """

    λs_unscaled: np.ndarray
    ts_unscaled: np.ndarray
    height: int
    width: int

    def __post_init__(self):
        self.λ0: float = self.λs_unscaled[0]
        self.λ1: float = self.λs_unscaled[-1]
        self.λs: np.ndarray = np.linspace(self.λ0, self.λ1, self.width)
        self.dλ: float = self.λs[1] - self.λs[0]

        self.t0: float = self.ts_unscaled[0]
        self.t1: float = self.ts_unscaled[-1]
        self.ts: np.ndarray = np.linspace(self.t0, self.t1, self.height)
        self.dt: float = self.ts[1] - self.ts[0]


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
    padding (float): extra padding for the filter ellipse ([0..1])
    img_specs (ImageSpecs): image specifications
    artifact (Artifact): artifact specifications
    fill_ellipse (bool): fill ellipse (False for outline only for debugging, default is True)

    """

    img_dft_mag: np.ndarray
    threshold_ellipse: float
    threshold_cutout: float
    img_specs: ImageSpecs
    artifact: Artifact
    padding: float = 0.20
    fill_ellipse: bool = True

    def __post_init__(self):
        (
            self.ellipse_long_axis_length,
            self.ellipse_short_axis_length,
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

    def define_filter_ellipse(self) -> Tuple[int, int]:
        """

        Determine filter ellipse parameters from periodic component DFT magnitude

        Returns: ellipse_long_axis_length (int), ellipse_short_axis_length (int)

        """

        # Determine ellipse long & short axis lengths (lengths of line along the artifact
        # diagonal and normal to the diagonal)
        img_binary_ellipse: np.ndarray = self.binarize_image(
            img=self.img_dft_mag, threshold=self.threshold_ellipse
        )
        x0, y0 = self.img_dft_mag.shape[1] // 2, self.img_dft_mag.shape[0] // 2
        l: float = min(self.img_dft_mag.shape) / 2.5
        artifact_long_diagonal_pixel_coordinates: np.ndarray = line(
            r0=int(y0 - l * np.sin(np.radians(self.artifact.angle))),
            c0=int(x0 + l * np.cos(np.radians(self.artifact.angle))),
            r1=int(y0 + l * np.sin(np.radians(self.artifact.angle))),
            c1=int(x0 - l * np.cos(np.radians(self.artifact.angle))),
        )
        artifact_short_diagonal_pixel_coordinates: np.ndarray = line(
            r0=int(y0 - l * np.sin(np.radians(self.artifact.angle + 90))),
            c0=int(x0 + l * np.cos(np.radians(self.artifact.angle + 90))),
            r1=int(y0 + l * np.sin(np.radians(self.artifact.angle + 90))),
            c1=int(x0 - l * np.cos(np.radians(self.artifact.angle + 90))),
        )
        ellipse_long_axis_length: int = int(
            self.calc_line_length(
                img_binary=img_binary_ellipse,
                diagonal_pixel_coordinates=artifact_long_diagonal_pixel_coordinates,
            )
            * (1.0 + self.padding)
        )
        ellipse_short_axis_length: int = int(
            self.calc_line_length(
                img_binary=img_binary_ellipse,
                diagonal_pixel_coordinates=artifact_short_diagonal_pixel_coordinates,
            )
            * (1.0 + self.padding)
        )
        if ellipse_long_axis_length == 0 or ellipse_short_axis_length == 0:
            raise ValueError(
                f"Threshold value for ellipse ({self.threshold_ellipse}) is too high!"
            )

        # Draw ellipse above-threshold pixels and resulting shape, for validation
        fig, ax = plt.subplots()
        ax.set(
            title="Filter ellipse binary image: threshold = "
            f"{self.threshold_ellipse:.2f}, "
            f"long axis = {ellipse_long_axis_length} pixels, "
            f"short axis = {ellipse_short_axis_length} pixels"
            f" ({self.padding * 100:.0f}% padding)"
        )
        img_binary_ellipse_rgb = np.repeat(
            img_binary_ellipse[:, :, np.newaxis], 3, axis=2
        )
        img_binary_ellipse_rgb[artifact_long_diagonal_pixel_coordinates] = [1, 0, 0]
        img_binary_ellipse_rgb[artifact_short_diagonal_pixel_coordinates] = [1, 0, 0]
        cv.ellipse(
            img_binary_ellipse_rgb,
            (self.img_specs.width // 2, self.img_specs.height // 2),
            (ellipse_long_axis_length // 2, ellipse_short_axis_length // 2),
            -self.artifact.angle,
            0,
            360,
            (1, 0, 0),
            1,
        )
        ax.imshow(img_binary_ellipse_rgb, cmap="gray")

        return (
            ellipse_long_axis_length,
            ellipse_short_axis_length,
        )

    def build_filter(self) -> np.ndarray:
        """
        Build filter image

        Returns: binary image filter

        """

        # Draw ellipse
        ellipse_image_binary: np.ndarray = np.ones(
            (self.img_specs.height, self.img_specs.width), dtype=np.uint8
        )
        cv.ellipse(
            ellipse_image_binary,
            (self.img_specs.width // 2, self.img_specs.height // 2),
            (self.ellipse_long_axis_length // 2, self.ellipse_short_axis_length // 2),
            -self.artifact.angle,
            0,
            360,
            (0, 0, 0) if self.fill_ellipse else (255, 255, 255),
            -1 if self.fill_ellipse else 1,
        )

        # Remove cutout at the center of ellipse (pixela around origin above threshold)
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

        # Return filter, either filled (with Gaussian blur) or outlined only (debug)
        return (
            cv.GaussianBlur(filter_image, (3, 3), sigmaX=0, sigmaY=0)
            if self.fill_ellipse
            else filter_image
        )


def plot_λ0_and_normal_line_profiles(
    img: np.ndarray,
    img_filtered: np.ndarray,
    img_specs: ImageSpecs,
    artifact: Artifact,
) -> Figure:
    """
    Plot line profiles vertically at λ0 and along the normal to the artifact

    Args:
        img (np.ndarray): original image
        img_filtered (np.ndarray): filtered image
        img_specs (ImageSpecs): image parameters
        artifact (Artifact): artifact parameters

    Returns: Figure object

    """

    # Line profiles (raw & filtered) along the normal to the artifact
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

    # Vertical line profiles (raw & filtered) at λ0
    λ0_profile: np.ndarray = np.flip(img[:, artifact.λ0_pixels])
    λ0_profile_filtered: np.ndarray = np.flip(img_filtered[:, artifact.λ0_pixels])

    # Plot line profiles and their DFTs
    normal_profile_dft: np.ndarray = np.fft.fft(normal_profile)
    x_relative: np.ndarray = np.linspace(
        -artifact.normal_length_pixels / 2,
        artifact.normal_length_pixels / 2,
        normal_profile.size,
    )
    fig, [ax0, ax1, ax2] = plt.subplots(3)

    ax0.plot(img_specs.ts, λ0_profile, label="Raw")
    ax0.plot(img_specs.ts, λ0_profile_filtered, "r", label="Filtered")
    ax0.set(
        xlabel="time (s)",
        ylabel=r"$\Delta$R/R",
        title=f"Raw and filtered time profiles at λ0 ({artifact.λ0} nm)",
        xlim=(img_specs.ts[0], img_specs.ts[-1]),
    )
    ax0.legend()
    ax0.grid()

    ax1.plot(x_relative, normal_profile, label="Raw")
    ax1.plot(x_relative, normal_profile_filtered, "r", label="Filtered")
    ax1.set(
        xlabel="n (pixels)",
        ylabel=r"$\Delta$R/R",
        title="Raw and filtered line profiles along the normal through the center of "
        "the artifact (bottom/left to top/right)",
    )
    ax1.legend()
    ax1.grid()
    normal_profile_dft_filtered: np.ndarray = np.fft.fft(normal_profile_filtered)
    f: np.ndarray = np.linspace(
        0, artifact.normal_length_pixels / 2, normal_profile.size // 2
    )

    ax2.plot(f, np.log10(np.abs(normal_profile_dft))[: num // 2], label="Raw")
    ax2.plot(
        f,
        np.log10(np.abs(normal_profile_dft_filtered))[: num // 2],
        "r",
        label="Filtered",
    )
    ax2.set(
        title="DFTs of raw and filtered line profiles along the normal",
        xlabel="k (samples)",
        ylabel="log10(|DFT|)",
    )
    ax2.set_xlim(left=0)
    ax2.legend()
    ax2.grid()
    plt.tight_layout()

    # return start/and data coordinates (ps, nm) of line through center of artifact
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

    Returns: matplotlib Figure class object

    """

    # Plot the image and DFT pairs
    fig, [top_axes, bot_axes] = plt.subplots(2, len(images), sharey="row")
    plt.suptitle(
        f"Transient gradient artifact filtering with smooth & periodic component "
        f"decomposition ([Moisan, 2010]) for image in '{fname}'\n"
        "Filtering parameters : "
        f"ellipse long axis length = {flt.ellipse_long_axis_length} pixels, "
        f"ellipse short axis length = {flt.ellipse_short_axis_length} pixels\n"
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
            cmap="seismic",
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
    filter_fill_ellipse: bool = True,
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
        filter_fill_ellipse (bool): see Filter Class docstring (default = True)

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

    # Load 2D spectroscopy measurement data from input file (Matlab, Excel, or .csv)
    fname_path: Path = Path(f"{fname}")
    img: np.ndarray
    λs_unscaled: np.ndarray
    ts_unscaled: np.ndarray
    df: pd.DataFrame = pd.DataFrame()
    if fname_path.suffix == ".mat":
        matlab_data: dict = loadmat(str(Path("data") / fname_path))
        img: np.ndarray = matlab_data["Data"]
        λs_unscaled = matlab_data["Wavelength"].flatten()
        ts_unscaled = matlab_data["Time"].flatten()
    elif fname_path.suffix in [".csv", ".xlsx", ".xls"]:
        df = pd.read_excel(str(Path("data") / fname_path), header=None)
        img: np.ndarray = df.iloc[1:, 1:].to_numpy()
        λs_unscaled = df.iloc[0, 1:].to_numpy()
        ts_unscaled = df.iloc[1:, 0].to_numpy()
    else:
        raise ValueError(f"Input file '{fname}' is not a Matlab, Excel, or .csv file!")
    if len(λs_unscaled) != img.shape[1] or len(ts_unscaled) != img.shape[0]:
        raise ValueError(
            f"Input file '{fname}' array dimensions are inconsistent"
            " (mismatch between Data vs Wavelength and/or Time array dimensions)!"
        )

    # Create ImageSpecs class object containing spectroscopy image specifications
    img_specs: ImageSpecs = ImageSpecs(
        λs_unscaled=λs_unscaled,
        ts_unscaled=ts_unscaled,
        height=img.shape[0],
        width=img.shape[1],
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
    periodic_dft, smooth_dft = per(img, inverse_dft=False)
    periodic_dft_mag, smooth_dft_mag = np.log10(
        np.abs(np.fft.fftshift(periodic_dft)) + 1e-10
    ), np.log10(np.abs(np.fft.fftshift(smooth_dft)) + 1e-10)
    periodic, smooth = np.real(np.fft.ifft2(periodic_dft)), np.real(
        np.fft.ifft2(smooth_dft)
    )

    # Design ellipse-shaped filter with cutout at center
    img_dft_mag: np.ndarray = np.log10(
        np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1e-10
    )
    flt: Filter = Filter(
        img_dft_mag=periodic_dft_mag,
        threshold_ellipse=threshold_ellipse,
        threshold_cutout=threshold_cutout,
        fill_ellipse=filter_fill_ellipse,
        img_specs=img_specs,
        artifact=artifact,
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
    fig_normal: Figure = plot_λ0_and_normal_line_profiles(
        img=img,
        img_filtered=img_filtered,
        img_specs=img_specs,
        artifact=artifact,
    )

    # Plot the image & DFT pairs (original, periodic, smooth, filtered u, filtered u)
    fig_images_and_dfts: Figure = plot_images_and_dfts(
        images=[img, periodic, smooth, periodic_filtered, img_filtered],
        dfts=[
            img_dft_mag,
            periodic_dft_mag,
            smooth_dft_mag,
            periodic_filtered_dft_mag,
            img_filtered_dft_mag,
        ],
        titles=[
            "Original image (u = s + u)",
            "Periodic component (u)",
            "Smooth component (s)",
            "Filtered periodic component (uf)",
            "Filtered image (uf = s + uf)",
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
                "Data": img,
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
                "filter_ellipse_long_axis_length_pixels": flt.ellipse_long_axis_length,
                "filter_ellipse_shirt_axis_length_pixels": flt.ellipse_short_axis_length,
            },
        )
    else:
        with pd.ExcelWriter(
            Path("output") / f"{fname_path.stem}_filtering_results.xlsx"
        ) as writer:
            sheet_array: np.ndarray = df.iloc[:, :].to_numpy()
            write_excel_sheet(
                sheet_array=sheet_array,
                array_data=img,
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
                    "filter_ellipse_long_axis_length_pixels": [
                        flt.ellipse_long_axis_length
                    ],
                    "filter_ellipse_shirt_axis_length_pixels": [
                        flt.ellipse_short_axis_length
                    ],
                }
            )
            df_info.to_excel(writer, sheet_name="info", index=False)

    # Return results
    return periodic, smooth, img_filtered, periodic_filtered, flt.f
