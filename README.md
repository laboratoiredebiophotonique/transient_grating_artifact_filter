Transient gradient artifact filtering in the Fourier domain

Separate an image into "smooth" and "periodic" components as per [Moisan, 2010] to
reduce the effect of the "cross" pattern in the Discrete Fourier transform due to the
non-periodic nature of the image (see https://github.com/sbrisard/moisan2011), then
filter the artifact from the periodic component in the Fourier domain using
an ellipse with a cutout at the center to preserve the low-frequency content of the
image data, recombine the filtered periodic component with the smooth component
to generate the filtered image.

The data image data are read from a matlab format file from the *data* directory,
the results are written to the the *output* directory.

NB:
- the moisan2011 package must be installed explicitly 
- there doesn't seem to be any difference in the results obtained with
  the different "per" operators (per, rper, ...)

Calling the script:
moisan_smooth_periodic(...)

User-specified parameters:

*Artifact* class object (see class definition for details):
- λ0
- extent_t 
- extent_λ

*Filter* class object (see class definition for details):
- cutout_size
- ellipse_long_axis_radius
- ellipse_short_axis_radius
