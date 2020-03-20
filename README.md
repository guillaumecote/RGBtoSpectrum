# RGBtoSpectrum
Produces a smooth metamer (EM spectrum) for given RGB values. 

It's the inverse problem of asking "What color would this spectral data be to the standard observer?". Since there isn't a finite answer, additional information about the emitting system can be incorporated in a minimizing function, potentially giving insight on what the source might be.

In its current form, it uses scipy's minimize method with an optimization function that limits sharp changes in spectral distribution (by minimizing the sum of its differences across the visible range). As such, results are smooth and better resemble light emitted by large objects, more commonly found in nature. Sharp "emission line"-like spectra can also be produced by minimizing the integral of that spectral data.

I've used this tool to show which potential spectral changes could cause a linear displacement in a given color space. In this case, could a single absorption peak be responsible for a smooth transition along the hue axis of the HSL (or LCH) color space? 

This serves more as an educational tool than anything else, so it prints frames of the spectrum to generate animations like these: 

<img align="center" src="https://github.com/guillaumecote/RGBtoSpectrum/blob/master/animations/HSL-20x4.gif">

In which you'll notice I also display the LMS functions in the background and the RGB color in the square insert for reference.
