"""
.. module:: pyranha
   :platform: Unix
   :synopsis: Main module containiny `Pyranha` class responsible for configuring and carrying out Fisher matrix calculations.
.. moduleauthor: Ben Thorne <ben.thorne@physics.ox.ac.uk>
"""

import numpy as np
from .instrument import instrument
from classy import Class
import ConfigParser


def read_array_from_csv(csv_str):
    """Function to return an array from a string of float separated by commas.

    Parameters
    ----------
    csv_str: str
        String containing numbers that may be interpreted as floats,
        separated by commas.

    Returns
    -------
    array_like
        1d array of floats.
    """
    return np.array([float(f) for f in csv_str.split(',')])


def calculate_lensing_template(bb_lensed, bb_unlensed):
    """Function to calculate the BB spectrum of lensing from output CLASS
    BB spectra.

    These spectra are the lened and unlensed outputs for a given
    set of cosmological parameters, therefore we simply take the difference
    of the lensed and unlensed BB spectrum. Note that this requires CLASS
    to have been run for l_max >= 2500 in order to achieve reliable accuracy.

    Parameters
    ----------
    bb_lensed: array_like
        BB spectrum containing both primordial and lensing contributions
        for a given run of CLASS.
    bb_unlensed: array_like
        BB spectrum containing only the primordial contribution to the BB
        spectrum.

    Returns
    -------
    array_like
        Template for the lensing spectrum (A_L=1).
    """
    return bb_lensed - bb_unlensed


class Pyranha(object):
    """Class to calculate the fisher information for a given input set of
    cosmological spectra and instrument spectra.

    `Pyranha` reads an input configuration file, which specifies the
    instrument parameters, and cosmology, to be used for a Fisher forecase
    of tensor-to-scalar ratio constraints. Its
    `iterate_instrument_parameter_1d` amd `iterate_instrument_parameter_2d`
    methods then allow for different parameters of the instrument to be
    iterated over in order to investigate how the constraints depend on them.

    Attributes
    ----------
    nus : array_like(`float`, ndim=1)
        Frequencies at which the instrument observes.
    sens : array_like(`float`, ndim=1)
        Sensitivities in uK_amin of the instrument channels.
    beams : array_like(`float`, ndim=1)
        FWHM in arcminutes of the beams for each channel.
    fsky : `float`
        Fraction of sky used in observations and in analysis.
    lmin : `int`
        Minimum multipole to be used.
    lmax : `int`
        Maximum multipole to be used.
    include_foregrounds : `bool`
        Whether to include foreground residuals in noise curve.
    map_res : `float`
        Level of foreground residuals in map-space (expresed as
        a decimal i.e. 0.01 -> 1%. Suggested level is 0.02).
    delensing : `bool`
        Whether or not to include a factor between 0 and 1 to reduce
        the contribution from lensing in BB.
    delensing_factor : `float`
        If `delensing` is used this factor multiplies the lensing BB
        spectrum before adding it to the noise spectrum.
    r : `float`
        Tensor-to-scalar ratio to be used in Fisher forecast. This is usually
        set to zero.
    BB_lens_template : array_like(`float`, ndim=1)
        Template for lensing contribution to BB.
    BB_lensed : array_like(`float`, ndim=1)
        The fully lensed BB spectrum for a given cosmology.
    BB_inst : array_like(`float`, ndim=1)
        The BB noise spectrum for a given instrument design and residual
        foreground level.

    """
    def __init__(self, fpath):
        """Read in a configuration file specified by fpath. This file
        containts all the information about the experimental setup, the
        various foreground models, fitting parameters, and number of runs.

        Parameters
        ----------
        fpath : `str`
            Path to the configuration file used to initalize this object.
        """
        cnfg = ConfigParser.RawConfigParser()
        cnfg.read(fpath)

        # Read in experiment parameters
        self.nus = read_array_from_csv(cnfg.get('experiment', 'nus'))
        self.sens = read_array_from_csv(cnfg.get('experiment', 'sens'))
        # Convert sensitivity from uK to unitless in order to combine with
        # spectra from CLASS.
        self.sens /= 2.7255e6
        self.beams = read_array_from_csv(cnfg.get('experiment', 'beams'))
        self.fsky = cnfg.getfloat('experiment', 'fsky')
        self.lmin = cnfg.getint('experiment', 'lmin')
        self.lmax = cnfg.getint('experiment', 'lmax')

        # Read in foreground parameters
        self.include_foregrounds = cnfg.getboolean('foreground', 'include')
        self.map_res = cnfg.getfloat('foreground', 'map_res')

        # Read in cosmology parameters
        self.delensing = cnfg.get('cosmology', 'delensing')
        if self.delensing:
            self.delensing_factor = cnfg.getfloat('cosmology', 'delensing_factor')
        else:
            self.delensing_factor = 1.
        if cnfg.getboolean('cosmology', 'lensing'):
            self.lensing = 'yes'
        self.r = cnfg.getfloat('cosmology', 'r')
        return


    def _run_class(self, **kwargs):
        """Method to run class and return the lensed and unlensed spectra as
        dictionaries.

        Returns
        -------
        cls_l : `dict`
            Dictionary containing the lensed spectra for a given run of CLASS.
        cls_u : `dict`
            Dictionary containing the unlensed spectra for the same run of
            CLASS.

        """
        cosmo = Class()

        # Set some parameters we want that are not in the default CLASS setting.
        class_pars = {
            'output': 'tCl pCl lCl',
            'modes': 's, t',
            'lensing': self.lensing,
            'r': self.r,
            }

        # Update CLASS run with any kwargs that were passed. This is useful in
        # Pyranha.compute_cosmology in order to compute the r=1 case.
        class_pars.update(kwargs)
        cosmo.set(class_pars)
        cosmo.compute()

        # Get the lensed and unlensed spectra as dictionaries.
        cls_l = cosmo.lensed_cl(2500)
        cls_u = cosmo.raw_cl(2500)

        # Do the memory cleanup.
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_l, cls_u


    def compute_cosmology(self):
        """Method to run CLASS with the current state of the object. This
        defines a few new instance attributes for the lensed BB template,
        and primoridal BB template which are used in the calculation
        of the Fisher matrix.
        """
        # Calculate the lensed and unlensed spectra for the requested cosmology.
        cls_l, cls_u = self._run_class()

        # Calculate the lensing template for this cosmology
        # Note that ALL CLASS SPECTRA ARE COMPUTE FOR ELL = 0 TO 2500. This is
        # because we want the option to iterate over lmin, lmax without
        # recomputing the cosmology. Therefore, the ell slices are implemented
        # in Pyranha.fisher
        self.BB_lens_template = calculate_lensing_template(cls_l['bb'], cls_u['bb'])
        self.BB_lensed = cls_l['bb']

        # Change the tensor-to-scalar ratio to one for the calculation of the
        # primordial B-mode signal template.
        _, cls_u = self._run_class(r=1)
        self.BB_prim_template = cls_u['bb']
        return


    def compute_instrument(self):
        """Method to compute the instrumental and foreground contribution to
        BB spectrum given the current state of the object and a given value of
        the residual foregrounds, sigma_rf. Defines a new instance attribute
        to store this spectrum.
        """
        self.BB_inst = instrument(self.nus, self.sens, self.beams,
                            self.map_res, lmin=self.lmin, lmax=self.lmax)


    def fisher(self):
        """Method to compute the Fisher matrix corresponding to the current
        state of the object (can change during a parameter iteration).

        Returns
        -------
        array_like
            Fisher matrix corresponding to current properties
        """
        # Shape of summand is then multipolesxNparamsxNparams.
        fish = np.zeros((2, 2), dtype=np.float32)
        self.ell = np.arange(self.lmin, self.lmax + 1)
        dCdi = np.zeros((2, 2, self.lmax - self.lmin + 1), dtype=np.float32)
        # Assemble noise spectrum from lensing spectrum and instrument spectrum
        # including foreground residuals.
        # NOTE THAT ALL THE CLASS SPECTRA ARE CALCULATE TO ELL=2500 FIRST IN
        # ORDER TO ALLO ITERATION OF LMIN AND LMAX WITHOUT HAVING TO
        # RECOMPUTE THE SPECTRA EACH TIME. HENCE THE UGLY SLICING BELOW.
        N_ell_BB = self.delensing_factor * self.BB_lensed[self.lmin : self.lmax + 1] + self.BB_inst
        #fill up the summand arrays.
        dCdi[0, 0] = self.BB_prim_template[self.lmin : self.lmax + 1] ** 2 / N_ell_BB ** 2
        dCdi[1, 1] = self.BB_lens_template[self.lmin : self.lmax + 1] ** 2 / N_ell_BB ** 2
        dCdi[0, 1] = self.BB_lens_template[self.lmin : self.lmax + 1] * self.BB_prim_template[self.lmin : self.lmax + 1] / N_ell_BB ** 2
        dCdi[1, 0] = dCdi[0, 1]

        summand =  (2. * self.ell + 1.) / 2. * self.fsky * dCdi
        fish = np.sum(summand[..., self.lmin : self.lmax + 1], axis=2)
        return fish


    def iterate_instrument_parameter_1d(self, par, par_values):
        """Method to iterate over an istrument parameter, holding all the
        cosmological parameter constant. Therefore, we do not need to rerun
        CLASS.

        Parameters
        ----------
        par: str
            Name of the parameter over which to iterate. Must exactly match
            name of a class attribute.
        par_values: array_like
            Set of values to iterate the parameter `par` over.
        """
        # Compute the cosmology, which remains constant across iterations.
        self.compute_cosmology()
        # Intialize list for output.
        fisher = []
        for par_value in par_values:
            # Set the parmaeter value and compute the corresponding
            # instrument noise spectrum.
            setattr(self, par, par_value)
            self.compute_instrument()
            fisher.append(self.fisher())
        return fisher

    def iterate_instrument_parameter_2d(self, par_x, par_y, xarr, yarr):
        """Method to iterate over an istrument parameter, holding all the
        cosmological parameter constant. Therefore, we do not need to rerun
        CLASS.

        Parameters
        ----------
        par_x: str
            name of first parameter to vary. This must match exactly the
            name of the attribute.
        par_y: str
            name of the second parameter to vary.
        xarr: array_like
            values of parameter corresponding to `par_x` to iterate over.
        yarr: array_like
            values of parameter corresponding to `par_y` to iterate over.

        Returns
        -------
        array_like
            Array of Fisher matrices calculated over the set of parameters
            par_x, par_y. Will have shape (len(xarr), len(yarr), 2, 2).
        """
        # Compute the cosmology, which remains constant across iterations.
        self.compute_cosmology()
        # Iteratte over the key-value pairs in the kwargs. This is only
        # designed to accept one item in the dictionary.
        fisher = np.zeros((len(xarr), len(yarr), 2, 2))
        for i, x in enumerate(xarr):
            for j, y in enumerate(yarr):
                setattr(self, par_x, x)
                setattr(self, par_y, y)
                self.compute_instrument()
                fisher[i, j, ...] = self.fisher()
        return fisher
