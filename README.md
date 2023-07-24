**Transient gradient artifact filtering in the Fourier domain on 2D time-resolved spectroscopy map:**

- Separate the input image (time-resolved spectroscopy map) into "smooth" and
"periodic" components as per [[Moisan, 2010](https://link.springer.com/article/10.1007/s10851-010-0227-1)]
to reduce the effect of the "cross" pattern at the center of the Discrete Fourier transform due to the 
non-periodic nature of the image.

- Filter the artifact from the periodic component in the Fourier domain using
an ellipse with a pass-band cutout at the center to preserve the low-frequency content of the
baseline data. The process to build the filter is described in detail in the *Filter*
Class declaration. Additional pass-band areas can be added to fine-tune the filtering
(see *Optional parameters* below).

- Recombine the filtered periodic component with the smooth component
to generate the filtered image.

- A light gaussian filtering using a 3x3 kernel (blurring) is applied to the filtered result
to remove any remaining high frequency noise.

- The script can accommodate non-uniformly sampled data in time and/or wavelength.

- The moisan2011 python package must be installed explicitly from [GitHub](https://github.com/sbrisard/moisan2011).

**Usage**: *transient_grating_artifact_filter(fname, lambda0_pump, artifact_extent_lambda,
artifact_extent_t, threshold_ellipse, threshold_cutout)*

- **Required function parameters**:

  - *fname* (str): input file in the *data* subdirectory containing the following data (see examples files in the *data*
  subdirectory):
    - *Data*: *nt* (rows) x *nλ* (columns) spectroscopy measurements (arbitrary units)
    - *Wavelength*: *nλ* wavelength samples (nm)
    - *Time*: *nt* time samples (ps)

  - *Artifact* class object parameters (see class definition for details):
    - *lambda0_pump* (float): pump central wavelength (nm)
    - *artifact_extent_t* (float): artifact extent in time (ps)
    - *artifact_extent_lambda* (float): artifact extent in wavelength (nm)

  - *Filter* class object parameters (see class definition for details):
    - *threshold_ellipse* (float): threshold for filter ellipse identification ([0..1])
    - *threshold_cutout* (float): threshold for filter central cutout identification ([0..1])
    - NB: *threshold_cutout* > *threshold_ellipse*

- **Optional function parameters**:
  - *lambda_time_profile* (float): Wavelength at which the time line-profile is 
                    plotted (nm, if 0 default to *lambda0_pump*).
  - *cross_pass_band_width* (int) = width of a cross-shaped pass-band in the filter along the
                    horizontal and vertical axes of the Fourier plane to pass
                    (i.e. not filter) any remaining non-periodic content left over from the
                    smooth/periodic decomposition (default = 0, i.e. no cross pass-band).
  - *pass_upper_left_lower_right_quadrants* (bool): enable/disable filter pass-band
                    areas for upper-left and lower-right quadrants of the Fourier plane
                    (default = True, i.e. pass upper-left and lower-right quadrants).

**Output**: Plot displays and files written to the *output* subdirectory (created automatically if it doesn't exist)

**Examples**: run the script *transient_grating_artifact_filter_exec.py* or the notebook *transient_grating_artifact_filter_exec.ipynb*.

**Debugging**:
- The *threshold_ellipse* and *threshold_cutout* input parameters must be adjusted to
  reach the optimal compromise between removing the artifact and preserving the 
  underlying baseline spectroscopy data.
- The script draws a cross-hair pattern over the elliptical mask identified from the
  thresholded area with *threshold_ellipse*. The *artifact_extent_t* and *artifact_extent_lambda* input
  parameters can be fine-tuned to line up the cross-hair with the ellipse axes.

