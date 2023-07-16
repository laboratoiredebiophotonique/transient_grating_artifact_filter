Transient gradient artifact filtering in the Fourier domain from 2D time-resolved spectroscopy map

First separate the input image (time-resolved spectroscopy map) into "smooth" and "periodic" components as per [Moisan, 2010] to
reduce the effect of the "cross" pattern in the Discrete Fourier transform due to the
non-periodic nature of the image (see https://github.com/sbrisard/moisan2011), then
filter the artifact from the periodic component in the Fourier domain using
an ellipse with a cutout at the center to preserve the low-frequency content of the
baseline data, and finally recombine the filtered periodic component with the smooth component
to generate the filtered map. NB: the moisan2011 python package must be installed explicitly.

Calling the script: *transient_grating_artifact_filter(fname, λ0_pump, artifact_extent_λ, artifact_extent_t, threshold_ellipse, threshold_cutout, filter_fill_ellipse)*

The data are read from a Matlab, Excel, or .csv format file from the *data* subdirectory,
the results are written to the *output* subdirectory.

Function parameters:

- *fname*: input file in the *data* subdirectory containing the following arrays:
  - *Data*: *nλ* x *nt* spectroscopy measurements (arbitrary units)
  - *Wavelength*: *nλ* wavelength samples (nm)
  - *Time*: *nt* time samples (ps)

*Artifact* class object parameters (see class definition for details):
- *λ0*: pump central wavelength (nm)
- *extent_t*: artifact extent in time (ps)
- *extent_λ*: artifact extent in wavelength (nm)

*Filter* class object parameters (see class definition for details):
- *threshold_ellipse*: threshold for filter ellipse identification ([0..1])
- *threshold_cutout*: threshold for filter central cutout identification ([0..1])

Optional parameters:
- *filter_fill_ellipse*: if False, draw the outline of the filter ellipse only for debugging purposes (default = True)

Output:
- Files written to the *output* subdirectory

Testing the script: call *transient_grating_artifact_filter_exec.py*

Debugging:
- The *threshold_ellipse* and *threshold_cutout* input parameters must be adjusted to reach the optimal compromise between removing the artifact and preserving the underlying baseline spectroscopy data.
- In the first displayed figure, the script draws a cross-hair pattern over the elliptical mask identified from the thresholded area with *threshold_ellipse*. The *extent_t* and *extent_λ* input parameters can be fine-tuned to line up the cross-hair with the ellipse axes.
- Class *Filter*, some parameters are input arugments to the *transient_grating_artifact_filter()* method:
  - *padding*: extra padding for the filter ellipse relative the thresholded area with *threshold_ellipse* ([0..1], default is 0.2, i.e. +20%)
  - *cross_width* = width of cross-shaped band along the horizontal and vertical
                        axes in the Fourier domain cutout from the filter to pass
                        any remaining non-periodic content left over from
                        the smooth/periodic decomposition (default is 0 pixels)
  - *gaussian_blur* = gaussian blur kernel size applied to the fileter to
                         reduce ringing(default is 0 pixels)
  - *fill_ellipse*: fill ellipse (False for outline only for debugging, default is True)

