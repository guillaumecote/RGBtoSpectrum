# RGBtoSpectrum
Produces a smooth metamer (EM spectrum) for given RGB values. 

It's the inverse problem of asking "What color would this spectral data be to the standard observer?". Since there isn't a finite answer, additional information about the emitting system can be incorporated in a minimizing function, potentially giving insight on what the source might be.

In its current form, it uses scipy's minimize method to with an optimization function that limits sharp changes in spectral distribution (by minimizing the integral of its derivative over the visible range). As such, results are smooth and better resemble light emitted by large objects, more commonly found in nature. Sharp "emission line"-like spectra can also be produced by minimizing the integral of that spectral data.

I've used this tool to show which potential spectral changes could cause a linear displacement in a given color space. In this case, could a single absorption peak be responsible for a smooth transition along the hue axis of the HSL (or LCH) color space? 

This serves more as an educational tool than anything else, so it prints frames of the spectrum to potentially generate the following animation: 
