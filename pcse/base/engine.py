# -*- coding: utf-8 -*-
# Copyright (c) 2004-2018 Alterra, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), April 2014
"""Base classes for creating PCSE simulation units.

In general these classes are not to be used directly, but are to be subclassed
when creating PCSE simulation units.
"""
import logging


from .dispatcher import DispatcherObject
from .simulationobject import SimulationObject


class BaseEngine(DispatcherObject):
    """Base Class for Engine to inherit from"""

    __slots__ = ["_sub_sim_attrs"]

    _sub_sim_attrs: dict

    def __init__(self):
        DispatcherObject.__init__(self)
        self._sub_sim_attrs = {}

    @property
    def logger(self):
        loggername = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        return logging.getLogger(loggername)

    def __setattr__(self, attr, value):
        # Need to safely grab this because we may not have fully
        # initialized our class before setting some variables
        sub_sim_attrs = getattr(self, "_sub_sim_attrs", None)

        if isinstance(value, SimulationObject):
            if sub_sim_attrs is None:
                raise AttributeError(
                    "Class is not yet initialized before receiving a SimulationObject"
                )
            self._sub_sim_attrs[attr] = value
        elif sub_sim_attrs is not None and attr in sub_sim_attrs:
            del self._sub_sim_attrs[attr]

        super().__setattr__(attr, value)

    @property
    def subSimObjects(self):
        """Find SimulationObjects embedded within self."""
        return list(self._sub_sim_attrs.values())

    def get_variable(self, varname):
        """Return the value of the specified state or rate variable.

        :param varname: Name of the variable.

        Note that the `get_variable()` will first search for `varname` exactly
        as specified (case sensitive). If the variable cannot be found, it will
        look for the uppercase name of that variable. This is purely for
        convenience.
        """

        # Check if variable is registered in the kiosk, also check for
        # name in upper case as most variables are defined in upper case.
        # If variable is not registered in the kiosk then return None directly.
        if self.kiosk.variable_exists(varname):
            v = varname
        elif self.kiosk.variable_exists(varname.upper()):
            v = varname.upper()
        else:
            return None

        if v in self.kiosk:
            return self.kiosk[v]

        # Search for variable by traversing the hierarchy
        value = None
        for simobj in self.subSimObjects:
            value = simobj.get_variable(v)
            if value is not None:
                break
        return value

    def zerofy(self):
        """Zerofy the value of all rate variables of any sub-SimulationObjects."""
        # Walk over possible sub-simulation objects.
        for simobj in self.subSimObjects:
            simobj.zerofy()
