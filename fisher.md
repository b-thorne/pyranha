# Fisher forecasts for B-modes and foreground removal

We want to develop Fisher forecasts for the constraints on $r$ in the presence of foreground residuals, intsrument noise, and lensing.  This is actually a super-tough question to do in all generality for spatially varying foreground indices, so we will end up assuming spatially constant spectral indices to do a best-case scenario analysis.

## Inputs and outputs

Let's first decide on the inputs and outputs.

- Clearly it will be required to specify the parameters we want to constrain. These should probably be supplied as a list of strings.
- We will need to specify the instrument we are using. This should just be a list of frequencies, Gaussian beam widths, and polarisation noise (assume root(2)*pol_noise for now, and assume white Gaussian.)
- Probably specify and ell_max here for convenience.

## Processing

- We will want to iterate through the parameters and calculate the derivative matrix for each parameter.
- Will need to assemble the Fisher matrix and invert it.
- 
