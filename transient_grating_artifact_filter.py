"""transient_grating_artifact_filter.py

Separate an image into smooth and periodic components as per [Moisan, 2010] to reduce
the effect of the "cross" pattern in the Discrete Fourier transform due to the
non-periodic nature of the image (see https://github.com/sbrisard/moisan2011), then
filter the coherent artifact from the periodic component in the Fourier domain using
an ellipse with its center removed to preserve the low-frequency content of the
image data.

NB: the moisan2011 package must be installed explicitly

"""

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from moisan2011 import per
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat, savemat
from skimage.draw import line

# Script version
__version__: str = "1.0"


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
            np.degrees(np.arctan(self.extent_t_pixels / self.extent_λ_pixels))
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
        normal_slope: float = -1 / slope
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
    Filter parameters (ellipse, with rectangular exclusion zone at the origin)

    cutout_size_horizontal (int): width of rectangle cutout from the ellipse (pixels)
    cutout_size_vertical (int): height of rectangle cutout from the ellipse (pixels)
    ellipse_long_axis_length (int): long axis of the ellipse filter (pixels)
    ellipse_short_axis_length (int): short axis of the ellipse filter (pixels)
    fill_ellipse (bool): fill ellipse (False for outline only for debugging)
    img_specs (ImageSpecs): image specifications
    artifact (Artifact): artifact specifications

    """

    cutout_size_horizontal: int
    cutout_size_vertical: int
    ellipse_long_axis_length: int
    ellipse_short_axis_length: int
    fill_ellipse: bool
    img_specs: ImageSpecs
    artifact: Artifact

    def __post_init__(self):
        self.f: np.ndarray = self.build_filter()

    def build_filter(self) -> np.ndarray:
        """
        Build filter image

        Returns: binary image filter

        """

        # Blank binary filter image (all ones)
        filter_image: np.ndarray = np.ones(
            (self.img_specs.height, self.img_specs.width), dtype=np.uint8
        )

        # Draw ellipse
        cv.ellipse(
            filter_image,
            (self.img_specs.width // 2, self.img_specs.height // 2),
            (self.ellipse_long_axis_length, self.ellipse_short_axis_length),
            -self.artifact.angle,
            0,
            360,
            (0, 0, 0) if self.fill_ellipse else (255, 255, 255),
            -1 if self.fill_ellipse else 1,
        )

        # Draw rectangle cutout at the center of the ellipse
        filter_image[
            filter_image.shape[1] // 2
            - self.cutout_size_vertical // 2 : filter_image.shape[1] // 2
            + self.cutout_size_vertical // 2,
            filter_image.shape[0] // 2
            - self.cutout_size_horizontal // 2 : filter_image.shape[0] // 2
            + self.cutout_size_horizontal // 2,
        ] = 1
        """
        # Draw cross cutout at the center of the ellipse
        filter_image[
            filter_image.shape[1] // 2
            - self.cutout_size_vertical // 2 : filter_image.shape[1] // 2
            + self.cutout_size_vertical // 2,
            :,
        ] = 1
        filter_image[
            :,
            filter_image.shape[0] // 2
            - self.cutout_size_horizontal // 2 : filter_image.shape[0] // 2
            + self.cutout_size_horizontal // 2,
        ] = 1
        """

        # Return filter, either filled (with Gaussian blur) or outlined only (debug)
        return (
            cv.GaussianBlur(filter_image, (3, 3), sigmaX=0, sigmaY=0).astype(np.float64)
            if self.fill_ellipse
            else filter_image.astype(np.float64)
        )


def plot_λ0_and_normal_line_profiles(
    u: np.ndarray,
    uf: np.ndarray,
    img_specs: ImageSpecs,
    artifact: Artifact,
) -> Figure:
    """
    Plot line profiles vertically at λ0 and along the normal to the artifact

    Args:
        u (np.ndarray): original image
        uf (np.ndarray): filtered image
        img_specs (ImageSpecs): image parameters
        artifact (Artifact): artifact parameters

    Returns: Figure object

    """

    # Line profiles (raw & filtered) along the normal to the artifact
    num: int = 1000
    x, y = np.linspace(
        artifact.normal_x0_pixels, artifact.normal_x1_pixels, num
    ), np.linspace(artifact.normal_y0_pixels, artifact.normal_y1_pixels, num)
    normal_profile: np.ndarray = np.flip(ndimage.map_coordinates(u, np.vstack((y, x))))
    normal_profile_filtered: np.ndarray = np.flip(
        ndimage.map_coordinates(uf, np.vstack((y, x)))
    )

    # Vertical line profiles (raw & filtered) at λ0
    λ0_profile: np.ndarray = np.flip(u[:, artifact.λ0_pixels])
    λ0_profile_filtered: np.ndarray = np.flip(uf[:, artifact.λ0_pixels])

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


def plot_artifact_outline(ax: Axes, img_specs: ImageSpecs, artifact: Artifact):
    """
    Plot the outline of the artifact.

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
        "Filtering parameters : cutout dims (w/h) = "
        "{flt.cutout_size_horizontal}/{flt.cutout_size_vertical} pixels, "
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
            plot_artifact_outline(ax=top_ax, img_specs=img_specs, artifact=artifact)

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


def interpolate_image(u_in: np.ndarray, dim: int) -> np.ndarray:
    """
    Interpolate image to dim x dim

    Args:
        u_in (np.ndarray): original input image
        dim (int): output image dimension in both axes

    Returns: interpolated image

    """

    x: np.ndarray = np.linspace(0, 1, u_in.shape[1])
    y: np.ndarray = np.linspace(0, 1, u_in.shape[0])
    interp: RegularGridInterpolator = RegularGridInterpolator((y, x), u_in)
    xi: np.ndarray = np.linspace(0, 1, dim)
    yi: np.ndarray = np.linspace(0, 1, dim)
    xx, yy = np.meshgrid(xi, yi, indexing="ij")

    return interp((xx, yy))


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


def calc_line_length(
    img_binary: np.ndarray, diagonal_pixel_coordinates: np.ndarray
) -> tuple[int, np.ndarray]:
    """
    Calculate the length of a line in a binary image along a diagonal

    Args:
        img_binary (np.ndarray): binary image to be analyzed
        diagonal_pixel_coordinates (np.ndarray): array of pixel coordinates along diagonal

    Returns: line length (int), line pixel coordinates (np.ndarray)

    """

    diagonal_pixel_values: np.ndarray = np.asarray(
        img_binary[diagonal_pixel_coordinates]
    )
    line_pixel_coordinate_indices: np.ndarray = np.asarray(
        np.where(diagonal_pixel_values == 1)
    ).flatten()
    line_pixel_coordinates: np.ndarray = np.asarray(
        [
            diagonal_pixel_coordinates[0][line_pixel_coordinate_indices],
            diagonal_pixel_coordinates[1][line_pixel_coordinate_indices],
        ]
    )
    line_length: int = int(
        np.linalg.norm(
            np.array(line_pixel_coordinates.T[-1] - line_pixel_coordinates.T[0])
        )
    )
    return line_length, line_pixel_coordinates


def define_filter_parameters(
    img: np.ndarray,
    artifact: Artifact,
    threshold_ellipse: float,
    threshold_cutout: float,
) -> tuple[int, int, int, int]:
    """

    Determine filter parameters from periodic component DFT magnitude
    (ellipse long & short axis, rectangular cutout width & height)

    Args:
        img: image used to calculate filter parameters (magnitude of the periodic
             component DFT)
        artifact: Artifact class object
        threshold_ellipse: threshold value for identifying filter ellipse ([0, 1]
        threshold_cutout: threshold value for identifying filter cutout ([0, 1]

    Returns: ellipse_long_axis_length (int), ellipse_short_axis_length (int),
             cut_out_width (int), cut_out_height (int)

    """

    # Determine cutout height & width (width/height of pixel rectangle around the origin
    # above threshold in binary image)
    img_binary_cutout: np.ndarray = binarize_image(img=img, threshold=threshold_cutout)
    horizontal_axis_pixels_above_threshold_coordinates = np.where(
        img_binary_cutout[img_binary_cutout.shape[0] // 2, :] == 1
    )
    vertical_axis_pixels_above_threshold_coordinates = np.where(
        img_binary_cutout[:, img_binary_cutout.shape[1] // 2] == 1
    )
    min_x, max_x = min(horizontal_axis_pixels_above_threshold_coordinates[0]), max(
        horizontal_axis_pixels_above_threshold_coordinates[0]
    )
    min_y, max_y = min(vertical_axis_pixels_above_threshold_coordinates[0]), max(
        vertical_axis_pixels_above_threshold_coordinates[0]
    )
    cut_out_width: int = max_x - min_x
    cut_out_height: int = max_y - min_y

    # Determine ellipse long & short axis lengths (lengths of line along the artifact
    # diagonal and normal to the diagonal)
    img_binary_ellipse: np.ndarray = binarize_image(
        img=img, threshold=threshold_ellipse
    )
    x0, y0 = img.shape[1] // 2, img.shape[0] // 2
    l: float = img.shape[0] / 4
    artifact_long_diagonal_pixel_coordinates: np.ndarray = line(
        r0=int(y0 - l * np.sin(np.radians(artifact.angle))),
        c0=int(x0 + l * np.cos(np.radians(artifact.angle))),
        r1=int(y0 + l * np.sin(np.radians(artifact.angle))),
        c1=int(x0 - l * np.cos(np.radians(artifact.angle))),
    )
    artifact_short_diagonal_pixel_coordinates: np.ndarray = line(
        r0=int(y0 - l * np.sin(np.radians(artifact.angle + 90))),
        c0=int(x0 + l * np.cos(np.radians(artifact.angle + 90))),
        r1=int(y0 + l * np.sin(np.radians(artifact.angle + 90))),
        c1=int(x0 - l * np.cos(np.radians(artifact.angle + 90))),
    )
    ellipse_long_axis_length, ellipse_long_axis_pixels_coordinates = calc_line_length(
        img_binary=img_binary_ellipse,
        diagonal_pixel_coordinates=artifact_long_diagonal_pixel_coordinates,
    )
    ellipse_short_axis_length, ellipse_short_axis_pixels_coordinates = calc_line_length(
        img_binary=img_binary_ellipse,
        diagonal_pixel_coordinates=artifact_short_diagonal_pixel_coordinates,
    )

    # Draw the binary images for visual validation
    fig, ax = plt.subplots(1, 2)
    ax[0].set(title=f"Cutout (threshold = {threshold_cutout:.2f})")
    ax[1].set(title=f"Ellipse (threshold = {threshold_ellipse:.2f})")
    ax[0].imshow(img_binary_cutout, cmap="gray")
    img_binary_ellipse[artifact_long_diagonal_pixel_coordinates] = 1
    img_binary_ellipse[tuple(ellipse_long_axis_pixels_coordinates)] = 0
    img_binary_ellipse[artifact_short_diagonal_pixel_coordinates] = 1
    img_binary_ellipse[tuple(ellipse_short_axis_pixels_coordinates)] = 0
    ax[1].imshow(img_binary_ellipse, cmap="gray")
    plt.suptitle("Binary images for filter component determination")

    return (
        ellipse_long_axis_length,
        ellipse_short_axis_length,
        cut_out_width,
        cut_out_height,
    )


def transient_grating_artifact_filter(
    fname: str,
    λ0_pump: float,
    artifact_extent_λ: float,
    artifact_extent_t: float,
    threshold_ellipse: float,
    threshold_cutout: float,
    filter_fill_ellipse: bool = True,
):
    """

    Separate an image into smooth and periodic components as per [Moisan, 2010], filter
    the periodic component in the Fourier domain to remove the coherent artifact, then
    sum with the smooth component to generate the final result.

    Args:
        fname (str): Matlab format input image filename in the ./data subdirectory
        λ0_pump (float): Pump central wavelength (nm)
        artifact_extent_λ (float): Artifact extent in the λ direction (nm)
        artifact_extent_t (float): Artifact extent in the t direction (ps)
        threshold_ellipse (float): threshold for filter ellipse identification ([0..1])
        threshold_cutout (float): threshold for filter cutout identification ([0..1])
        filter_fill_ellipse (bool): see Filter Class docstring (default = True)

    Returns: None

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

    # Load 2D spectroscopy image data from matlab file, interpolate to 1024 x 1024
    matlab_data: dict = loadmat(f"data/{fname}")
    u_in: np.ndarray = np.rot90(matlab_data["Data"])
    u: np.ndarray = interpolate_image(u_in=u_in, dim=1024)
    u_dft: np.ndarray = np.fft.fft2(u)

    # Create ImageSpecs class object containing spectroscopy image specifications
    img_specs: ImageSpecs = ImageSpecs(
        λs_unscaled=matlab_data["Wavelength"].flatten(),
        ts_unscaled=matlab_data["Time"].flatten(),
        height=u.shape[0],
        width=u.shape[1],
    )

    # Create Artifact class object containing artifact specifications
    artifact: Artifact = Artifact(
        λ0=λ0_pump,
        extent_λ=artifact_extent_λ,
        extent_t=artifact_extent_t,
        img_specs=img_specs,
    )

    # Extract periodic and smooth component DFTs, calculate images from inverse DFTs
    p_dft, s_dft = per(u, inverse_dft=False)
    p_dft_mag: np.ndarray = np.log10(np.abs(np.fft.fftshift(p_dft)) + 1e-10)
    s_dft_mag: np.ndarray = np.log10(np.abs(np.fft.fftshift(s_dft)) + 1e-10)
    p: np.ndarray = np.real(np.fft.ifft2(p_dft))
    s: np.ndarray = np.real(np.fft.ifft2(s_dft))

    # Define filter parameters (ellipse long & short axis, cutout width & height)
    (
        filter_ellipse_long_axis_length,
        filter_ellipse_short_axis_length,
        cut_out_width,
        cut_out_height,
    ) = define_filter_parameters(
        img=p_dft_mag,
        artifact=artifact,
        threshold_ellipse=threshold_ellipse,
        threshold_cutout=threshold_cutout,
    )

    # Design ellipse-shaped filter with rectangular cutout at center
    flt: Filter = Filter(
        ellipse_long_axis_length=filter_ellipse_long_axis_length,
        ellipse_short_axis_length=filter_ellipse_short_axis_length,
        cutout_size_horizontal=cut_out_width,
        cutout_size_vertical=cut_out_height,
        fill_ellipse=filter_fill_ellipse,
        img_specs=img_specs,
        artifact=artifact,
    )

    # Filter periodic component in the Fourier domain, reconstruct filtered image
    pf_dft: np.ndarray = p_dft * np.fft.fftshift(flt.f)
    pf: np.ndarray = np.real(np.fft.ifft2(pf_dft))
    uf: np.ndarray = s + pf

    # Calculate DFT log magnitude images (add small offset so null DC doesn't break log)
    uf_dft: np.ndarray = np.fft.fft2(uf)
    u_dft_mag: np.ndarray = np.log10(np.abs(np.fft.fftshift(u_dft)) + 1e-10)
    pf_dft_mag: np.ndarray = np.log10(np.abs(np.fft.fftshift(pf_dft)) + 1e-10)
    uf_dft_mag: np.ndarray = np.log10(np.abs(np.fft.fftshift(uf_dft)) + 1e-10)

    # Plot the line profile perpendicular to the artifact though it's center (normal)
    fig_normal: Figure = plot_λ0_and_normal_line_profiles(
        u=u,
        uf=uf,
        img_specs=img_specs,
        artifact=artifact,
    )

    # Plot the image & DFT pairs (original, periodic, smooth, filtered u, filtered u)
    fig_images_and_dfts: Figure = plot_images_and_dfts(
        images=[u, p, s, pf, uf],
        dfts=[u_dft_mag, p_dft_mag, s_dft_mag, pf_dft_mag, uf_dft_mag],
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

    # Save results to the ./output subdirectory
    fig_images_and_dfts.savefig(
        f"./output/{fname.split('.mat')[0]}_images_and_dfts.png"
    )
    fig_normal.savefig(f"./output/{fname.split('.mat')[0]}_normal.png")
    savemat(
        f"output/{fname.split('.mat')[0]}_filtering_results.mat",
        {
            "u": u,
            "p": p,
            "s": s,
            "pf": pf,
            "uf": uf,
            "Wavelength": img_specs.λs,
            "Time": img_specs.ts,
            "f": flt.f,
            "u_dft": u_dft,
            "uf_dft": uf_dft,
            "p_dft": p_dft,
            "pf_dft": pf_dft,
            "s_dft": s_dft,
            "artifact_coordinates_pixels": [
                [artifact.x0_pixels, artifact.x1_pixels],
                [artifact.y0_pixels, artifact.y1_pixels],
            ],
            "lambda0_pump_nm": λ0_pump,
            "artifact_extent_wavelength_nm": artifact.extent_λ,
            "artifact_extent_time_ps": artifact.extent_t,
            "filter_cutout_size_horizontal_pixels": flt.cutout_size_horizontal,
            "filter_cutout_size_vertical_pixels": flt.cutout_size_vertical,
            "filter_ellipse_long_axis_length_pixels": flt.ellipse_long_axis_length,
            "filter_ellipse_shirt_axis_length_pixels": flt.ellipse_short_axis_length,
        },
    )


def main():
    """
    Main function for running the script.

    Returns: None

    """

    # Structures to simulate: "gold_film", "nano_pillars", "rhodamine"
    substrate_type: str = "gold_film"

    # Thresholds for filter construction
    threshold_ellipse: float = 0.3
    threshold_cutout: float = 0.5

    # Filter design & debugging: if False, draw ellipse outline only
    filter_fill_ellipse: bool = True

    # Define simulation parameters for the selected structure
    if substrate_type == "gold_film":
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
        λ0_pump = 570.0
        artifact_extent_λ = 20
        artifact_extent_t = 0.55

    else:
        raise ValueError("Unknown substrate type!")

    # Run the simulation
    transient_grating_artifact_filter(
        fname=fname,
        λ0_pump=λ0_pump,
        artifact_extent_λ=artifact_extent_λ,
        artifact_extent_t=artifact_extent_t,
        threshold_ellipse=threshold_ellipse,
        threshold_cutout=threshold_cutout,
        filter_fill_ellipse=filter_fill_ellipse,
    )

    return None


if __name__ == "__main__":
    main()
    print("Done!")
