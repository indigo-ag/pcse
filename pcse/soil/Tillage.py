from math import sqrt, pi, exp, log, cos
import numpy as np
import math
from ..traitlets import Float, Int, Instance, Enum, Unicode, Bool, List
from ..decorators import prepare_rates, prepare_states
from ..util import limit, Afgen, merge_dict, AfgenTrait
from ..base import ParamTemplate, StatesTemplate, RatesTemplate, SimulationObject
from .. import signals
from .. import exceptions as exc
from .snowmaus import SnowMAUS
from array import array
from ..db.tillage_parameters import (
    cult_factor_ranges,
    cultivation_methods,
    get_tillage_code_params,
)


class TillageSignal(SimulationObject):
    """Handles tillage operations as signals in PCSE."""

    cfitb = Float()  # Cultivation fit parameter B
    XEFCLTE = Float()  # Cultivation effect factor

    def _on_APPLY_TILLAGE(self, **kwargs):
        """Apply tillage"""
        # print everthing inside kwargs

        mixing = kwargs.get("mixing", 0)
        depth = kwargs.get("depth", 0)
        type = kwargs.get("type", 0)
        day = kwargs.get("day", 0)

        self._apply_tillage_effect(type, mixing, depth, day)

    class Parameters(ParamTemplate):
        XEFCLTEF = Float()  # Cultivation effect factor
        MAXCLTEF = Float()  # Maximum cultivation effect
        CFITA = Float()  # Cultivation fit parameter A
        TEFF = List()

    class StateVariables(StatesTemplate):
        CLTFAC = List()  # Cultivation factor for each soil layer
        CULTCNT = Int()
        # CULTRA = List()

    class RateVariables(RatesTemplate):
        bgdefac = Float()  # g C m^-2 day^-1

    def initialize(self, day, kiosk, parvalues):
        """
        Initializes the Tillage object.

        Args:
            day (int): The current day.
            kiosk (Kiosk): The kiosk object.
            parvalues (dict): A dictionary of parameter values.

        Returns:
            None
        """
        self.params = self.Parameters(parvalues)

        self.cfitb = (self.params.MAXCLTEF - 1) / 0.95
        self.XEFCLTE = 2 ** (1 / (self.params.XEFCLTEF * 30.4375)) - 1

        # Initialize soil temperature array with default values
        CLTFAC_init = [1.0] * 4
        # CLTRA_init = [0] * 7
        self.rates = self.RateVariables(kiosk, publish=["bgdefac"])

        self.states = self.StateVariables(
            kiosk,
            CLTFAC=CLTFAC_init,
            # CULTRA = CLTRA_init,
            CULTCNT=0,
            publish=["CLTFAC"],
        )
        self._connect_signal(self._on_APPLY_TILLAGE, signals.apply_tillage)

    def _apply_tillage_effect(self, type, mixing, depth, day):
        """
        Apply tillage effect.

        Args:
            type (str): The type of tillage.
            mixing (float): The mixing factor.
            depth (float): The depth of tillage.
            day (int): The day of the year.

        Returns:
            None

        Raises:
            None
        """
        p = self.params
        s = self.states

        # if type is not in cultivation_methods keys, find the code based on the mixing and depth
        # if there is, then that means a mixed, custom defined tillage is applied
        clteff, cult_factor_key = get_tillage_code_params(
            mixing, depth, type, par="clteff"
        )

        bgdefac = self.rates.bgdefac
        coef = 31.0 * self.XEFCLTE * math.log(2)

        for i in range(4):
            if p.MAXCLTEF > 0:
                if s.CLTFAC[i] == 1:
                    cltefn = abs(clteff[i])
                else:
                    cltef = 1 + (s.CLTFAC[i] - 1) / (coef * (self.XEFCLTE + 1))
                    if p.CFITA == 0:
                        mx1 = (cltef - 1) / self.cfitb
                        mx2 = (abs(clteff[i]) - 1) / self.cfitb
                    else:
                        tmix = self.cfitb / (2 * p.CFITA)
                        mx1 = math.sqrt(tmix**2 + (cltef - 1) / p.CFITA) - tmix
                        mx2 = (
                            math.sqrt(tmix**2 + (abs(clteff[i]) - 1) / p.CFITA) - tmix
                        )
                    if mx1 > 1.0 and mx2 > 1.0:
                        tmix = mx1 * mx2
                    else:
                        tmix = mx1 + (1 - mx1) * mx2
                    cltefn = (p.CFITA * tmix + self.cfitb) * tmix + 1.0
                new_cltfac = 1 + (cltefn - 1) * (coef * (self.XEFCLTE + 1)) / (
                    1 + bgdefac * self.XEFCLTE
                )
            else:
                new_cltfac = abs(clteff[i])
            s.CLTFAC[i] = limit(0.0, 5.0, new_cltfac)
        s.CULTCNT += 1

    @prepare_rates
    def calc_rates(self, day, drv):
        """
        Estimate the rate variable variables.
        bgdefac: decomposition factor based on water and temperature for soil decomposition
        Args:
            day: The current day.
            drv: The driver object containing weather data.

        Returns:
            None

        """
        s = self.states
        p = self.params
        # get the soil temp and factors
        avgstemp = self.kiosk["ST"][2]  # Soil temperature (°C)
        teff = p.TEFF
        # use the rain and et0 to calculate the agwfunc and krainwfunc
        et0 = drv.ET0
        if et0 == 0:
            et0 = 0.0001
        rprpet = drv.RAIN / et0
        agwfunc = 1.0 if rprpet > 9 else 1.0 / (1.0 + 30.0 * math.exp(-8.5 * rprpet))
        # finally calculate the bgdefac

        krainwfunc = self.wfunc_pulse(drv.RAIN, 0)
        tfunc = self.tcalc(avgstemp, teff)
        self.rates.bgdefac = max(0.000001, tfunc[0] * agwfunc * krainwfunc)

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Updates the state variables.

        Args:
            day (float): The current day.
            delt (float, optional): The time step size. Defaults to 1.0.
        """
        s = self.states
        p = self.params

        if self.XEFCLTE > 0:
            CLTFAC = s.CLTFAC.copy()
            for i in range(4):
                if CLTFAC[i] > 1.01:
                    # print(s.CLTFAC[i])
                    CLTFAC[i] = 1 + (CLTFAC[i] - 1) / (
                        1 + self.rates.bgdefac * self.XEFCLTE
                    )
                    # round to 2 decimal places
                    CLTFAC[i] = round(CLTFAC[i], 2)
                elif 1.0 < s.CLTFAC[i] <= 1.01:
                    self.states.CLTFAC[i] = 1.0
        if s.CULTCNT > 0:
            s.CULTCNT = min(s.CULTCNT + 1, 31)
        else:
            s.CULTCNT = 0

        self.states.CLTFAC = CLTFAC

    def tcalc(self, avgstemp, teff):
        """
        Calculate the tillage factor based on the average soil temperature and effective temperature.

        Parameters:
        avgstemp (float): The average soil temperature.
        teff (float): The effective temperature.

        Returns:
        list: A list containing two tillage factors calculated based on the average soil temperature.
        """

        tfunc = [0.0, 0.0]

        tfunc[0] = 0.0418 + (0.0442 * avgstemp) - (0.00162 * avgstemp**2)
        tfunc[1] = 0.06 + (0.027 * avgstemp) - (0.00045 * avgstemp**2)

        return tfunc

    def wfunc_pulse(self, ppt, snow):
        """
        Calculates the water function pulse.

        The water function pulse is calculated based on the precipitation (ppt) and snowfall (snow) values.
        If the sum of ppt and snow is zero, the function returns 1.0.
        Otherwise, it returns 1.0 plus 0.3 times the ratio of ppt to the sum of ppt and snow.

        Parameters:
        - ppt (float): Precipitation value.
        - snow (float): Snowfall value.

        Returns:
        - float: The water function pulse value.

        """
        if ppt + snow == 0:
            return 1.0
        return 1.0 + 0.3 * (ppt / (ppt + snow))
