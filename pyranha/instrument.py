import numpy as np
from foreground import *

def N_ell(ell, beam, sens):
    """Gaussian white-noise spectrum for a given beam and sensitivity.

    :param numpy.array ell: Multipole range to consider
    :param float beam: fwhm in arcmin of the Gaussian beam
    :param float sens: sensitivity in X per arcmin where X is the unit of temperature
    :return: noise spectrum deconvolved with beam
    :rtype: numpy.array
    """
    b_rad = beam / 60. * np.pi / 180.
    beam =  np.exp(ell * (ell + 1.) * b_rad  ** 2 / (8. * np.log(2)))
    noise_level = sens ** 2 * (1. / 60. * np.pi / 180.) ** 2
    return beam * noise_level


def instrument(nus, p_sens, beams, map_res, lmin=0, lmax=2500):
    """Function to calculate the noise spectrum of a given set of instrument
    specifications.

    :param numpy.array nus: observing freuqencies in GHz.
    :param numpy.array p_sens: sensitivity in uK_amin
    :param numpy.array beams: beam FWHM in arcmin
    :param int lmin: Minimum multipole to consider
    :param int lmax: Maximum multipole to consider
    :return: instrument and foreground polarization noise spectrum
    :rtype: numpy.array
    """
    specs = zip(nus, p_sens, beams)

    """Filter lists to include only CMB channels. The cmb_filter contains
    the definitions of lower and upper channels used for CMB analysis. The
    other channels are assumed to go into foreground removal.
    """
    cmb_filter = lambda (nu, p_sens, beams): (nu > 90.) & (nu < 220.)
    fg_filter = lambda (nu, p_sens, beams): not cmb_filter((nu, p_sens, beams))

    cmb_nu, cmb_p_sens, cmb_beams = zip(*filter(cmb_filter, specs))
    fg_nu, fg_p_sens, fg_beams = zip(*filter(fg_filter, specs))

    N_chan = len(fg_nu)
    nu_S_ref = np.min(fg_nu)
    nu_D_ref = np.max(fg_nu)

    """Compute the original noise in the foreground channels.
    """
    ell = np.arange(lmin, lmax + 1)
    fg_N_ell_p = [N_ell(ell, beam, p_sens) for (beam, p_sens) in zip(fg_beams,
                                                                fg_p_sens)]
    cmb_N_ell_p = [N_ell(ell, beam, p_sens) for (beam, p_sens) in zip(cmb_beams,
                                                                cmb_p_sens)]

    """Compute the foreground residual spectra and template noise
    scaled to the CMB channels.

    Foreground parameters used in calculating the residual fg noise:
    """
    A_S = 6.3e-18
    alpha_S = -3.
    beta_S = -2.6
    nu_S_0 = 30.
    ell_S_0 = 350.

    A_D = 1.3e-13
    alpha_D = 2.2
    beta_D = -2.5
    nu_D_0 = 94.
    ell_D_0 = 10.

    T = 18.
    p_d = 0.15

    cmb_channel_fgnd_res = lambda nu: (dust_cl(nu, ell, p_d, T, A_D, alpha_D,
                                                beta_D, nu_D_0, ell_D_0) + \
        synch_cl(nu, ell, A_S, alpha_S, beta_S, nu_S_0, ell_S_0)) * map_res ** 2 + \
        fg_res_sys(nu, nu_S_ref, alpha_S, nu_D_ref, alpha_D, N_chan, fg_N_ell_p)

    """Now compute the combined foreground residual contribution and systematic
    residual contribution of foreground channels to the cmb channels.
    """
    fg_cmb_noise_spec = [cmb_channel_fgnd_res(nu) for nu in cmb_nu]

    """Now sum the foreground and cmb channel noise contributions to the CMB
    channel noise.
    """
    total_channel_noise_spec_p = [c_p + fg for c_p, fg in zip(cmb_N_ell_p,
                                                            fg_cmb_noise_spec)]

    """Now can compute the final noise spectrum in CMB spectrum.
    This is computed by treating each ell as a Gaussian variate and
    so combining inverse squares.
    """
    N_ell_p = 1. / sum([1. / n_l for n_l in total_channel_noise_spec_p])

    return N_ell_p
