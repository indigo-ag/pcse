# -*- coding: utf-8 -*-
# Copyright (c) 2004-2022 Wageningen Environmental Research, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), August 2022
import logging
import os

import yaml
from cachetools import Cache, cachedmethod
from copy import deepcopy

from ..base import MultiCropDataProvider
from .. import exceptions as exc
from ..util import version_tuple


_CROP_PARAMETER_CACHE = Cache(maxsize=30)


class YAMLCropDataProvider(MultiCropDataProvider):
    """A crop data provider for reading crop parameter sets stored in the YAML format.

        :param fpath: full path to directory containing YAML files

    This crop data provider can read and store the parameter sets for multiple
    crops which is different from most other crop data providers that only can
    hold data for a single crop. This crop data providers is therefore suitable
    for running crop rotations with different crop types as the data provider
    can switch the active crop.

    YAML parameter files are loaded from your local file system:

        >>> from pcse.fileinput import YAMLCropDataProvider
        >>> p = YAMLCropDataProvider(fpath=r"D:\\UserData\\sources\\WOFOST_crop_parameters")
        >>> print(p)
        YAMLCropDataProvider - crop and variety not set: no activate crop parameter set!

    All crops and varieties have been loaded from the YAML file, however no activate
    crop has been set. Therefore, we need to activate a particular crop and variety:

        >>> p.set_active_crop('wheat', 'Winter_wheat_101')
        >>> print(p)
        YAMLCropDataProvider - current active crop 'wheat' with variety 'Winter_wheat_101'
        Available crop parameters:
         {'DTSMTB': [0.0, 0.0, 30.0, 30.0, 45.0, 30.0], 'NLAI_NPK': 1.0, 'NRESIDLV': 0.004,
         'KCRIT_FR': 1.0, 'RDRLV_NPK': 0.05, 'TCPT': 10, 'DEPNR': 4.5, 'KMAXRT_FR': 0.5,
         ...
         ...
         'TSUM2': 1194, 'TSUM1': 543, 'TSUMEM': 120}

    To increase performance of loading parameters, the YAMLCropDataProvider will cache files
    in memory after the initial read.
    """

    current_crop_name = None
    current_variety_name = None

    # Compatibility of data provider with YAML parameter file version
    compatible_version = "1.0.0"

    def __init__(self, fpath, force_reload=False):
        MultiCropDataProvider.__init__(self)

        if force_reload:
            _CROP_PARAMETER_CACHE.clear()
            self.clear()
            self._store.clear()

        self.read_local_repository(fpath)

    def _check_version(self, parameters, crop_fname):
        """Checks the version of the parameter input with the version supported by this data provider.

        Raises an exception if the parameter set is incompatible.

        :param parameters: The parameter set loaded by YAML
        """
        try:
            v = parameters["Version"]
            if version_tuple(v) != version_tuple(self.compatible_version):
                msg = (
                    "Version supported by %s is %s, while parameter set version is %s!"
                )
                raise exc.PCSEError(
                    msg
                    % (
                        self.__class__.__name__,
                        self.compatible_version,
                        parameters["Version"],
                    )
                )
        except Exception as e:
            msg = f"Version check failed on crop parameter file: {crop_fname}"
            raise exc.PCSEError(msg)

    @cachedmethod(lambda self: _CROP_PARAMETER_CACHE)
    def _read_crop_parameter_yaml_file(self, yaml_fname):
        with open(yaml_fname) as fp:
            parameters = yaml.safe_load(fp)

        self._check_version(parameters, crop_fname=yaml_fname)

        return parameters

    def read_local_repository(self, fpath):
        """Reads the crop YAML files on the local file system

        :param fpath: the location of the YAML files on the filesystem
        """
        yaml_file_names = self._get_yaml_files(fpath)
        for crop_name, yaml_fname in yaml_file_names.items():
            # Deepcopy just to ensure that nothing mutates them
            parameters = deepcopy(self._read_crop_parameter_yaml_file(yaml_fname))
            self._add_crop(crop_name, parameters)

    def _add_crop(self, crop_name, parameters):
        """Store the parameter sets for the different varieties for the given crop."""
        variety_sets = parameters["CropParameters"]["Varieties"]
        self._store[crop_name] = variety_sets

    def _get_yaml_files(self, fpath):
        """Returns all the files ending on *.yaml in the given path."""
        fname = os.path.join(fpath, "crops.yaml")
        if not os.path.exists(fname):
            msg = "Cannot find 'crops.yaml' at {f}".format(f=fname)
            raise exc.PCSEError(msg)
        crop_names = yaml.safe_load(open(fname))["available_crops"]
        crop_yaml_fnames = {
            crop: os.path.join(fpath, crop + ".yaml") for crop in crop_names
        }
        for crop, fname in crop_yaml_fnames.items():
            if not os.path.exists(fname):
                msg = f"Cannot find yaml file for crop '{crop}': {fname}"
                raise RuntimeError(msg)
        return crop_yaml_fnames

    def set_active_crop(self, crop_name, variety_name):
        """Sets the parameters in the internal dict for given crop_name and variety_name

        It first clears the active set of crop parameter sin the internal dict.

        :param crop_name: the name of the crop
        :param variety_name: the variety for the given crop
        """
        self.clear()
        if crop_name not in self._store:
            msg = "Crop name '%s' not available in %s " % (
                crop_name,
                self.__class__.__name__,
            )
            raise exc.PCSEError(msg)
        variety_sets = self._store[crop_name]
        if variety_name not in variety_sets:
            msg = "Variety name '%s' not available for crop '%s' in " "%s " % (
                variety_name,
                crop_name,
                self.__class__.__name__,
            )
            raise exc.PCSEError(msg)

        self.current_crop_name = crop_name
        self.current_variety_name = variety_name

        # Retrieve parameter name/values from input (ignore description and units)
        parameters = {
            k: v[0] for k, v in variety_sets[variety_name].items() if k != "Metadata"
        }
        # update internal dict with parameter values for this variety
        self.update(parameters)

    def get_crops_varieties(self):
        """Return the names of available crops and varieties per crop.

        :return: a dict of type {'crop_name1': ['variety_name1', 'variety_name1', ...],
                                 'crop_name2': [...]}
        """
        return {k: v.keys() for k, v in self._store.items()}

    def print_crops_varieties(self):
        """Gives a printed list of crops and varieties on screen."""
        msg = ""
        for crop, varieties in self.get_crops_varieties().items():
            msg += "crop '%s', available varieties:\n" % crop
            for var in varieties:
                msg += " - '%s'\n" % var
        print(msg)

    def __str__(self):
        if not self:
            msg = (
                "%s - crop and variety not set: no active crop parameter set!\n"
                % self.__class__.__name__
            )
            return msg
        else:
            msg = "%s - current active crop '%s' with variety '%s'\n" % (
                self.__class__.__name__,
                self.current_crop_name,
                self.current_variety_name,
            )
            msg += "Available crop parameters:\n %s" % str(dict.__str__(self))
            return msg

    @property
    def logger(self):
        loggername = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        return logging.getLogger(loggername)
