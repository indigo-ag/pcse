# -*- coding: utf-8 -*-
# Copyright (c) 2004-2018 Alterra, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), April 2014
import logging

from ..util import Afgen
from .. import exceptions as exc
from .variablekiosk import VariableKiosk


def _is_private(name: str) -> bool:
    # Note that we just check the first character for underscore
    # since we know these are strings with length > 0 and it's much
    # faster than `startswith`
    return name[0] == "_"


class ParamTemplate:
    """Template for storing parameter values.

    This is meant to be subclassed by the actual class where the parameters
    are defined.

    example::

        >>> import pcse
        >>> from pcse.base import ParamTemplate
        >>>
        >>>
        >>> class Parameters(ParamTemplate):
        ...     A: float
        ...     B: float
        ...     C: float
        ...
        >>> parvalues = {"A" :1., "B" :-99, "C":2.45}
        >>> params = Parameters(parvalues)
        >>> params.A
        1.0
        >>> params.A; params.B; params.C
        1.0
        -99.0
        2.4500000000000002
        >>> parvalues = {"A" :1., "B" :-99}
        >>> params = Parameters(parvalues)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "pcse/base.py", line 205, in __init__
            raise exc.ParameterError(msg)
        pcse.exceptions.ParameterError: Value for parameter C missing.
    """

    __slots__ = []

    def __init__(self, parvalues):
        for parname in self.__slots__:
            # check if the parname is available in the dictionary of parvalues
            if parname not in parvalues:
                msg = "Value for parameter %s missing." % parname
                raise exc.ParameterError(msg)
            try:
                type_ = self.__annotations__[parname]
            except KeyError as err:
                raise RuntimeError(
                    f"Could not determine type for {parname}. All variables must have a type annotation"
                ) from err

            value = parvalues[parname]
            if type_ == Afgen:
                # AFGEN table parameter
                setattr(self, parname, Afgen.create(value))
            else:
                # Single value parameter
                setattr(self, parname, value)


def check_publish(publish):
    """Convert the list of published variables to a set with unique elements."""

    if isinstance(publish, (list, tuple)):
        return set(publish)
    if isinstance(publish, str):
        return {publish}
    if publish is None:
        return set()

    msg = "The publish keyword should specify a string or a list of strings"
    raise RuntimeError(msg)


class StatesRatesCommon:
    __slots__ = [
        "_kiosk",
        "_valid_vars",
        "_locked",
        "_published_attrs",
        "_publish_enabled",
    ]

    _kiosk: VariableKiosk
    _valid_vars: set[str]
    _locked: bool
    _published_attrs: set[str]
    _publish_enabled: bool

    def __init__(self, kiosk=None, publish=None):
        """Set up the common stuff for the states and rates template
        including variables that have to be published in the kiosk
        """
        # Make sure that the variable kiosk is provided
        if not isinstance(kiosk, VariableKiosk):
            msg = (
                "Variable Kiosk must be provided when instantiating rate "
                + "or state variables."
            )
            raise RuntimeError(msg)

        self._kiosk = kiosk
        self._locked = False
        self._published_attrs = set()
        self._publish_enabled = True

        # Check publish variable for correct usage
        publish = check_publish(publish)

        # Determine the rate/state attributes defined by the user
        self._valid_vars = self._find_valid_variables()

        # Register all variables with the kiosk and optionally publish them.
        self._register_with_kiosk(publish)

    def _find_valid_variables(self):
        """Returns a set with the valid state/rate variables names. Valid rate
        variables have names not starting with '_'.
        """
        return {a for a in self.__slots__ if not _is_private(a)}

    def _register_with_kiosk(self, publish):
        """Register the variable with the variable kiosk.

        Here several operations are carried out:
         1. Register the  variable with the kiosk, if rates/states are
            registered twice an error will be raised, this ensures
            uniqueness of rate/state variables across the entire model.
         2 If the  variable name is included in the list set by publish
           keyword then set a trigger on that variable to update its value
           in the kiosk.

         Note that self._vartype determines if the variables is registered
         as a state variable (_vartype=="S") or rate variable (_vartype=="R")
        """

        for attr in self._valid_vars:
            if attr in publish:
                publish.remove(attr)
                self._kiosk.register_variable(
                    id(self), attr, type=self._vartype, publish=True
                )
                self._published_attrs.add(attr)
            else:
                self._kiosk.register_variable(
                    id(self), attr, type=self._vartype, publish=False
                )
        # Check if the set of published variables is exhausted, otherwise
        # raise an error.
        if len(publish) > 0:
            msg = (
                "Unknown variable(s) specified with the publish " + "keyword: %s"
            ) % publish
            raise exc.PCSEError(msg)

    def __setattr__(self, name, value):
        if (
            not _is_private(name)
            and self._publish_enabled
            and name in self._published_attrs
        ):
            self._kiosk.set_variable(id(self), name, value)
        super().__setattr__(name, value)

    def unlock(self):
        "Unlocks the attributes of this class."
        self._locked = False

    def lock(self):
        "Locks the attributes of this class."
        self._locked = True

    def _delete(self):
        """Deregister the variables from the kiosk before garbage
        collecting.

        This method is coded as _delete() and must by explicitly called
        because of precarious handling of __del__() in python.
        """
        for attr in self._valid_vars:
            self._kiosk.deregister_variable(id(self), attr)

    @property
    def logger(self):
        loggername = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        return logging.getLogger(loggername)


class StatesTemplate(StatesRatesCommon):
    """Takes care of assigning initial values to state variables, registering
    variables in the kiosk and monitoring assignments to variables that are
    published.

    :param kiosk: Instance of the VariableKiosk class. All state variables
        will be registered in the kiosk in order to enforce that variable names
        are unique across the model. Moreover, the value of variables that
        are published will be available through the VariableKiosk.
    :param publish: Lists the variables whose values need to be published
        in the VariableKiosk. Can be omitted if no variables need to be
        published.

    Initial values for state variables can be specified as keyword when instantiating
    a States class.

    example::

        >>> import pcse
        >>> from pcse.base import VariableKiosk, StatesTemplate
        >>> from datetime import date
        >>>
        >>> k = VariableKiosk()
        >>> class StateVariables(StatesTemplate):
        ...     StateA: float
        ...     StateB: int
        ...     StateC: date
        ...
        >>> s1 = StateVariables(k, StateA=0., StateB=78, StateC=date(2003,7,3),
        ...                     publish="StateC")
        >>> print s1.StateA, s1.StateB, s1.StateC
        0.0 78 2003-07-03
        >>> print k
        Contents of VariableKiosk:
         * Registered state variables: 3
         * Published state variables: 1 with values:
          - variable StateC, value: 2003-07-03
         * Registered rate variables: 0
         * Published rate variables: 0 with values:

        >>>
        >>> s2 = StateVariables(k, StateA=200., StateB=1240)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "pcse/base.py", line 396, in __init__
            raise exc.PCSEError(msg)
        pcse.exceptions.PCSEError: Initial value for state StateC missing.

    """

    __slots__ = ["_vartype"]

    _vartype: str

    def __init__(self, kiosk=None, publish=None, **kwargs):
        self._vartype = "S"
        StatesRatesCommon.__init__(self, kiosk, publish)

        # set initial state value
        for attr in self._valid_vars:
            if attr in kwargs:
                value = kwargs.pop(attr)
                setattr(self, attr, value)
            else:
                msg = "Initial value for state %s missing." % attr
                raise exc.PCSEError(msg)

        # Check if kwargs is empty, otherwise issue a warning
        if len(kwargs) > 0:
            msg = (
                "Initial value given for unknown state variable(s): " + "%s"
            ) % kwargs.keys()
            logging.warning(msg)

        # Lock the object to prevent further changes at this stage.
        self._locked = True

    def touch(self):
        """Re-assigns the value of each state variable, thereby updating its
        value in the variablekiosk if the variable is published."""

        self.unlock()
        for name in self._valid_vars:
            value = getattr(self, name)
            setattr(self, name, value)
        self.lock()


class StatesWithImplicitRatesTemplate(StatesTemplate):
    """Container class for state variables that have an associated rate.

    The rates will be generated upon initialization having the same name as their states,
    prefixed by a lowercase character 'r'.
    After initialization no more attributes can be implicitly added.
    Call integrate() to integrate all states with their current rates; the rates are reset to 0.0.

    States are all attributes descending from Float and not prefixed by an underscore.
    """

    rates = {}
    __initialized = False

    def __setattr__(self, name, value):
        if name in self.rates:
            # known attribute: set value:
            self.rates[name] = value
        elif not self.__initialized:
            # new attribute: allow whe not yet initialized:
            object.__setattr__(self, name, value)
        else:
            # new attribute: disallow according ancestorial rules:
            super(StatesWithImplicitRatesTemplate, self).__setattr__(name, value)

    def __getattr__(self, name):
        if name in self.rates:
            return self.rates[name]
        else:
            object.__getattribute__(self, name)

    def initialize_rates(self):
        self.rates = {}
        self.__initialized = True

        for s in self.__class__.listIntegratedStates():
            self.rates["r" + s] = 0.0

    def integrate(self, delta):
        # integrate all:
        for s in self.listIntegratedStates():
            rate = getattr(self, "r" + s)
            state = getattr(self, s)
            newvalue = state + delta * rate
            setattr(self, s, newvalue)

        # reset all rates
        for r in self.rates:
            self.rates[r] = 0.0

    @classmethod
    def listIntegratedStates(cls):
        return sorted(
            [
                a
                for a in cls.__dict__
                if not _is_private(a) and isinstance(getattr(cls, a), float)
            ]
        )

    @classmethod
    def initialValues(cls):
        return dict(
            (a, 0.0)
            for a in cls.__dict__
            if not not _is_private(a) and isinstance(getattr(cls, a), float)
        )


class RatesTemplate(StatesRatesCommon):
    """Takes care of registering variables in the kiosk and monitoring
    assignments to variables that are published.

    :param kiosk: Instance of the VariableKiosk class. All rate variables
        will be registered in the kiosk in order to enforce that variable names
        are unique across the model. Moreover, the value of variables that
        are published will be available through the VariableKiosk.
    :param publish: Lists the variables whose values need to be published
        in the VariableKiosk. Can be omitted if no variables need to be
        published.

    For an example see the `StatesTemplate`. The only difference is that the
    initial value of rate variables does not need to be specified because
    the value will be set to zero (Int, Float variables) or False (Boolean
    variables).
    """

    __slots__ = ["_vartype", "_rate_vars_zero"]

    _rate_vars_zero: dict

    def __init__(self, kiosk=None, publish=None):
        """Set up the RatesTemplate and set monitoring on variables that
        have to be published.
        """
        self._vartype = "R"
        StatesRatesCommon.__init__(self, kiosk, publish)

        # Determine the zero value for all rate variable if possible
        self._rate_vars_zero = self._find_rate_zero_values()

        # We want to disable publishing while we zero out the parameters
        # initially. This matches behavior in mainline PCSE where the traits
        # were updated via `self._trait_values.update` which bypassed the publish
        # callback
        self._publish_enabled = False
        # Initialize all rate variables to zero or False
        self.zerofy()
        self._publish_enabled = True

        # Lock the object to prevent further changes at this stage.
        self._locked = True

    def _find_rate_zero_values(self):
        """Returns a dict with the names with the valid rate variables names as keys and
        the values are the zero values used by the zerofy() method. This means 0 for int,
        0.0 for float and False for bool.
        """

        d = {}
        for attr in self._valid_vars:
            try:
                type_ = self.__annotations__[attr]
            except KeyError as err:
                raise RuntimeError(
                    f"Could not determine type for {attr}. All variables must have a type annotation"
                ) from err

            if type_ == int:
                d[attr] = 0
            elif type_ == float:
                d[attr] = 0.0
            elif type_ == bool:
                d[attr] = False
            else:
                msg = (
                    f"Rate variable '{attr}' is type '{type_.__name__}' not float, bool or int. "
                    "Its zero value cannot be determined and it will not be treated by zerofy()."
                )
                self.logger.warning(msg)

        return d

    def zerofy(self):
        """Sets the values of all rate values to zero (int, float) or False (bool)."""

        for attr, value in self._rate_vars_zero.items():
            setattr(self, attr, value)
