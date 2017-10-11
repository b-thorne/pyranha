[experiment]
# Frequencies at which to observe. Must be comma-separated.
nus: 40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235, 280, 337, 402
# Polarization sensitivities in uk_amin. Must be comma-separated.
sens: 36.8, 23.6, 19.5, 15.9, 13.3, 11.5, 9., 7.5, 5.8, 6.3, 5.7, 7.5, 13., 19.1, 36.9
# FWHM beams in arcminutes. Must be comma-separated.
beams: 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70
# Sky fraction observed and used in analysis.
fsky: 0.53
lmin: 2
lmax: 1000

[cosmology]
# Include lensing in fisher forecast
lensing: True
# Delens by some factor
delensing: False
# Factor by which to delens the noise lensing spectrum
delensing_factor: 1.
# tensor-to-scalar ratio
r: 0.0

[foreground]
# whether or not to include foreground residuals
include: True
# map space residual level
map_res: 0.0
