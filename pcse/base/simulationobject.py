# -*- coding: utf-8 -*-
# Copyright (c) 2004-2018 Alterra, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), April 2014
import logging
from datetime import date

from .dispatcher import DispatcherObject

from .. import exceptions as exc
from .variablekiosk import VariableKiosk
from .states_rates import StatesTemplate, RatesTemplate, ParamTemplate


class SimulationObject(DispatcherObject):
    """Base class for PCSE simulation objects.

    :param day: start date of the simulation
    :param kiosk: variable kiosk of this PCSE instance

    The day and kiosk are mandatory variables and must be passed when
    instantiating a SimulationObject.
    """

    __slots__ = [
        "states",
        "rates",
        "params",
        "kiosk",
        "_for_finalize",
        "_sub_sim_attrs",
    ]

    # Placeholders for logger, params, states, rates and variable kiosk
    states: StatesTemplate | None
    rates: RatesTemplate | None
    params: ParamTemplate | None
    kiosk: VariableKiosk

    # Placeholder for variables that are to be set during finalizing.
    _for_finalize: dict

    _sub_sim_attrs: dict

    def __init__(self, day, kiosk, *args, **kwargs):
        self._for_finalize = {}
        self._sub_sim_attrs = {}
        self.states = None
        self.rates = None
        self.params = None

        # Check that day variable is specified
        if not isinstance(day, date):
            this = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
            msg = (
                "%s should be instantiated with the simulation start "
                + "day as first argument!"
            )
            raise exc.PCSEError(msg % this)

        # Check that kiosk variable is specified and assign to self
        if not isinstance(kiosk, VariableKiosk):
            this = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
            msg = (
                "%s should be instantiated with the VariableKiosk "
                + "as second argument!"
            )
            raise exc.PCSEError(msg % this)
        self.kiosk = kiosk

        self.initialize(day, kiosk, *args, **kwargs)
        self.logger.debug("Component successfully initialized on %s!" % day)

    def initialize(self, *args, **kwargs):
        msg = "`initialize` method not yet implemented on %s" % self.__class__.__name__
        raise NotImplementedError(msg)

    @property
    def logger(self):
        loggername = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        return logging.getLogger(loggername)

    def integrate(self, *args, **kwargs):
        msg = "`integrate` method not yet implemented on %s" % self.__class__.__name__
        raise NotImplementedError(msg)

    def calc_rates(self, *args, **kwargs):
        msg = "`calc_rates` method not yet implemented on %s" % self.__class__.__name__
        raise NotImplementedError(msg)

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

    def get_variable(self, varname):
        """Return the value of the specified state or rate variable.

        :param varname: Name of the variable.

        Note that the `get_variable()` will searches for `varname` exactly
        as specified (case sensitive).
        """

        # Search for variable in the current object, then traverse the hierarchy
        value = None
        if hasattr(self.states, varname):
            value = getattr(self.states, varname)
        elif hasattr(self.rates, varname):
            value = getattr(self.rates, varname)
        # Query individual sub-SimObject for existence of variable v
        else:
            for simobj in self.subSimObjects:
                value = simobj.get_variable(varname)
                if value is not None:
                    break
        return value

    def set_variable(self, varname, value, incr):
        """Sets the value of the specified state or rate variable.

        :param varname: Name of the variable to be updated (string).
        :param value: Value that it should be updated to (float)
        :param incr: dict that will receive the increments to the updated state
            variables.

        :returns: either the increment of the variable (new - old) or `None`
          if the call was unsuccessful in finding the class method (see below).

        Note that 'setting'  a variable (e.g. updating a model state) is much more
        complex than just `getting` a variable, because often some other
        internal variables (checksums, related state variables) must be updated
        as well. As there is no generic rule to 'set' a variable it is up to
        the model designer to implement the appropriate code to do the update.

        The implementation of `set_variable()` works as follows. First it will
        recursively search for a class method on the simulationobjects with the
        name `_set_variable_<varname>` (case sensitive). If the method is found,
        it will be called by providing the value as input.

        So for updating the crop leaf area index (varname 'LAI') to value '5.0',
        the call will be: `set_variable('LAI', 5.0)`. Internally, this call will
        search for a class method `_set_variable_LAI` which will be executed
        with the value '5.0' as input.
        """
        method_name = "_set_variable_%s" % varname.strip()
        try:
            method_obj = getattr(self, method_name)
            rv = method_obj(value)
            if not isinstance(rv, dict):
                msg = (
                    "Method %s on '%s' should return a dict with the increment of the "
                    + "updated state variables!"
                ) % (method_name, self.__class__.__name__)
                raise exc.PCSEError(msg)
            incr.update(rv)
        except AttributeError:  # method is not present: just continue
            pass
        except TypeError:  # method is present but is not callable: error!
            msg = (
                "Method '%s' on '%s' could not be called by 'set_variable()': "
                + "check your code!"
            ) % (method_name, self.__class__.__name__)
            raise exc.PCSEError(msg)

        for simobj in self.subSimObjects:
            simobj.set_variable(varname, value, incr)

    def _delete(self):
        """Runs the _delete() methods on the states/rates objects and recurses
        trough the list of subSimObjects.
        """
        if self.states is not None:
            self.states._delete()
            self.states = None
        if self.rates is not None:
            self.rates._delete()
            self.rates = None
        for obj in self.subSimObjects:
            obj._delete()

    @property
    def subSimObjects(self):
        """Return SimulationObjects embedded within self."""
        return list(self._sub_sim_attrs.values())

    def finalize(self, day):
        """Run the _finalize call on subsimulation objects"""
        # Update the states object with the values stored in the _for_finalize dictionary
        if self.states is not None:
            self.states.unlock()
            while len(self._for_finalize) > 0:
                k, v = self._for_finalize.popitem()
                setattr(self.states, k, v)
            self.states.lock()
        # Walk over possible sub-simulation objects.
        for simobj in self.subSimObjects:
            simobj.finalize(day)

    def touch(self):
        """'Touch' all state variables of this and any sub-SimulationObjects.

        The name comes from the UNIX `touch` command which does nothing on the
        contents of a file but only updates the file metadata (time, etc).
        Similarly, the `touch` method re-assigns the state of each state
        variable causing any triggers to go off.
        This will guarantee that these state values remain available in the
        VariableKiosk.
        """

        if self.states is not None:
            self.states.touch()
        # Walk over possible sub-simulation objects.
        for simobj in self.subSimObjects:
            simobj.touch()

    def zerofy(self):
        """Zerofy the value of all rate variables of this and any sub-SimulationObjects."""

        if self.rates is not None:
            self.rates.zerofy()

        # Walk over possible sub-simulation objects.
        for simobj in self.subSimObjects:
            simobj.zerofy()


class AncillaryObject(DispatcherObject):
    """Base class for PCSE ancillary objects.

    Ancillary objects do not carry out simulation, but often are useful for
    wrapper objects. Still to have some aspects in common with SimulationObjects
    such as the existence of self.logger and self.kiosk, the locked
    behaviour requiring you to define the class attributes and the possibility
    to send/receive signals.
    """

    __slots__ = ["kiosk", "params"]

    # Placeholders for logger, variable kiosk and parameters
    kiosk: VariableKiosk
    params: ParamTemplate

    def __init__(self, kiosk, *args, **kwargs):
        # Check that kiosk variable is specified and assign to self
        if not isinstance(kiosk, VariableKiosk):
            this = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
            msg = (
                "%s should be instantiated with the VariableKiosk "
                "as second argument!"
            )
            raise RuntimeError(msg % this)

        self.kiosk = kiosk
        self.initialize(kiosk, *args, **kwargs)
        self.logger.debug("Component successfully initialized!")

    @property
    def logger(self):
        loggername = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        return logging.getLogger(loggername)
