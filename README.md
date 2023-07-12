Transient gradient artifact filtering in the Fourier domain

Separate an image into "smooth" and "periodic" components as per [Moisan, 2010] to
reduce the effect of the "cross" pattern in the Discrete Fourier transform due to the
non-periodic nature of the image (see https://github.com/sbrisard/moisan2011), then
filter the artifact from the periodic component in the Fourier domain using
an ellipse with a rectangular cutout at the center to preserve the low-frequency content of the
image data, recombine the filtered periodic component with the smooth component
to generate the filtered image.

The data image data are read from a matlab format file from the *data* subdirectory,
the results are written to the *output* subdirectory.

NB: the moisan2011 package must be installed explicitly

Calling the script:
transient_grating_artifact_filter(...)

Function parameters:

- *fname*: matlab input file in the *data* subdirectory containing the following arrays:
  - *Data*: *nλ* x *nt* spectroscopy measurements (arbitrary units)
  - *Wavelength*: *nλ* wavelength samples (nm)
  - *Time*: *nt* time samples (ps)

*Artifact* class object parameters (see class definition for details):
- *λ0*: pump central wavelength (nm)
- *extent_t*: artifact extent in time (ps)
- *extent_λ*: artifact extent in wavelength (nm)

*Filter* class object parameters (see class definition for details):
- *threshold_ellipse*: threshold for filter ellipse identification ([0..1])
- *threshold_cutout*: threshold for filter cutoff identification ([0..1])

Output:
- Files written to the *output* subdirectory

Debugging:
- The *threshold_ellipse* and *threshold_cutout* input parameters must be adjusted to reach the optimal compromise between removing the artifact and preserving the underlying baseline spectroscopy data.
- In the first figure, the script draws a cross-hair pattern over the elliptical mask identified by thresholding. The *extent_t* and *extent_λ* input parameters can be fine-tuned to line up the cross-hair with the ellipse axes.
