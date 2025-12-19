import math

from ..base import ParamTemplate, StatesTemplate, SimulationObject


class SoilTemperature(SimulationObject):
    """
    Based on dssat-csm-os Copyright (c) 2021 DSSAT Foundation

    dssat-csm-os is licensed under the terms of the BSD-3-clause license:
    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
    SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
    OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Changes made by Hamze Dokoohaki © Indigo Ag, Inc. 2024-2025

    The SoilTemperature class is a simulation object that calculates the daily soil
    temperature at different depths using the DSSAT STEMP model.

    The model is based on the amplitude and damping of the annual soil temperature wave,
    which is influenced by factors such as air temperature, solar radiation, soil water
    content, and soil thermal properties. The main equations and concepts used in the
    model include:

    1. Calculation of the damping depth (DP) based on soil bulk density and texture.
       The damping depth represents the depth at which the amplitude of the annual soil
       temperature wave is reduced to 1/e (37%) of its surface value.

    2. Estimation of the potential extractable soil water (PESW) based on the soil water
       content and the drained upper limit (DUL) and lower limit (LL) of plant-extractable
       water. PESW influences the thermal conductivity and heat capacity of the soil.

    3. Calculation of the water content function (WC) based on PESW, which affects the
       damping of the annual soil temperature wave.

    4. Calculation of the average annual soil temperature (TAV) and the amplitude of the
       annual soil temperature wave (TAMP) at the surface. These values are estimated
       from the average annual air temperature and the difference between the maximum
       and minimum monthly air temperatures, respectively.

    5. Calculation of the daily soil temperature at different depths using the TAV, TAMP,
       and the damping depth. The model assumes that the annual soil temperature wave
       follows a sinusoidal pattern, with a phase shift and amplitude that decrease
       exponentially with depth.

    The main input variables for the model include:
    - Daily weather data: maximum and minimum air temperature, solar radiation
    - Soil properties: bulk density, drained upper limit, lower limit, soil water content
    - Site characteristics: latitude, average annual air temperature, amplitude of annual
      air temperature variation

    The main output variables of the model are:
    - Daily soil temperature at different depths (ST)
    - Daily soil surface temperature (SRFTEMP)

    The model is initialized with default values for soil temperature and surface
    temperature, and then updates these values daily based on the input variables and
    the model equations. The model also keeps track of the average soil temperature
    over the previous 5 days (TMA) to account for the temporal autocorrelation of soil
    temperature.
    """

    class Parameters(ParamTemplate):

        __slots__ = [
            "NLAYR",
            "TAMP",
            "TAV",
            "MSALB",
            "BD",
            "DS",
            "LL",
            "DLAYR",
        ]

        NLAYR: int  # Number of soil layers
        TAMP: float  # Amplitude of temperature function (°C)
        TAV: float  # Average annual soil temperature (°C)
        MSALB: float  # Soil albedo with mulch and soil water effects (fraction)

        BD: list  # Bulk density (g/cm3)
        DS: list  # Depth of soil layers (cm)
        LL: list  # Lower limit of soil water content (cm3/cm3)
        DLAYR: list  # Thickness of soil layers (cm)

    class StateVariables(StatesTemplate):

        __slots__ = ["ST", "SRFTEMP"]

        ST: list  # Soil temperature for each layer (°C)
        SRFTEMP: float  # Temperature of the soil surface litter (°C)

    def initialize(self, day, kiosk, parvalues):
        self.params = self.Parameters(parvalues)

        # Initialize soil temperature array with default values
        ST_init = [20.0] * self.params.NLAYR

        self.states = self.StateVariables(
            kiosk, ST=ST_init, SRFTEMP=20.0, publish=["ST", "SRFTEMP"]
        )

    def calc_states(self, day, drv):
        """Calculate the soil temperature at the specified depths and time."""
        # Calculate soil temperature at 30 cm depth
        s = self.states
        p = self.params
        ALBEDO = p.MSALB  # (fraction)
        BDs = [
            (p.BD[L] * (p.DS[L] - p.DS[L - 1])) for L in range(1, p.NLAYR)
        ]  # Total bulk density (g/cm3)
        BDs.append(p.BD[0] * p.DS[0])
        TBD = sum(BDs)  # Total bulk density (g/cm3)
        ABD = TBD / p.DS[p.NLAYR - 1]  # Average bulk density (g/cm3)

        FX = ABD / (ABD + 686.0 * math.exp(-5.63 * ABD))  # Exponential decay factor (-)
        B = math.log(500.0 / (1000.0 + 2500.0 * FX))  # Exponential decay factor (-)
        CUMDPT = p.DS[p.NLAYR - 1] * 10.0  # Cumulative depth of soil profile (mm)
        DP = 1000.0 + 2500.0 * FX  # Damping depth parameter (mm)
        HDAY = 20.0 if drv.LAT < 0.0 else 200.0  # Hottest day of the year (DOY)

        # !! FIX THESE TWO THE MULTI LAYER PROBLEM FOR LAYER 0
        SW = self.kiosk["SM"]
        PESW = sum(
            max(0.0, SW - p.LL[L]) * p.DLAYR[L] for L in range(0, p.NLAYR)
        )  # Potential extractable soil water (cm)

        WW = 0.356 - 0.144 * ABD  # Volumetric soil water content parameter (-)
        DSMID = [0.0] * p.NLAYR  # Depth to midpoint of each soil layer (cm)
        DSMID[0] = p.DLAYR[0] / 2.0
        for L in range(1, p.NLAYR):
            DSMID[L] = DSMID[L - 1] + (p.DLAYR[L - 1] + p.DLAYR[L]) / 2.0

        TMA = [
            20.0
        ] * 5  # Initialize array of previous 5 days of average soil temperatures (°C)
        ATOT = sum(TMA)  # Sum of TMA array (°C)

        # Calculate soil temperatures
        # find DOY from drv.DAY if drv.DAY is a date like 2006-01-01
        DOY = drv.DAY.timetuple().tm_yday
        ALX = (DOY - HDAY) * 0.0174  # (radians)
        ATOT = ATOT - TMA[4]
        TMA = TMA[1:] + [TMA[0]]  # Shift TMA array by one day

        tavg = 0.5 * (drv.TMIN + drv.TMAX)
        srd = drv.IRRAD * 1e-6  # |Jm-2day-1| to (MJ/m2-d)
        TMA[0] = (1.0 - ALBEDO) * (
            tavg + (drv.TMAX - tavg) * math.sqrt(srd * 0.03)
        ) + ALBEDO * TMA[
            0
        ]  # (°C)
        TMA[0] = round(
            TMA[0], 4
        )  # Round to 4 decimal places to avoid floating-point errors
        ATOT = ATOT + TMA[0]

        WC = max(0.01, PESW) / (WW * CUMDPT) * 10.0  # Water content function (-)
        FX = math.exp(
            B * ((1.0 - WC) / (1.0 + WC)) ** 2
        )  # Exponential decay factor (-)
        DD = FX * DP  # Damping depth (mm)

        TA = p.TAV + p.TAMP * math.cos(ALX) / 2.0  # Daily normal temperature (°C)
        DT = ATOT / 5.0 - TA  # Temperature difference (°C)

        ST = [0.0] * p.NLAYR
        for L in range(p.NLAYR):
            ZD = -DSMID[L] / DD  # Depth factor (-)
            ST[L] = p.TAV + (p.TAMP / 2.0 * math.cos(ALX + ZD) + DT) * math.exp(
                ZD
            )  # Soil temperature in layer L (°C)
            ST[L] = round(ST[L], 3)  # Round to 3 decimal places

        SRFTEMP = p.TAV + (
            p.TAMP / 2.0 * math.cos(ALX) + DT
        )  # Soil surface temperature (°C)

        s.ST = ST  # Update the ST array with the calculated soil temperatures
        s.SRFTEMP = SRFTEMP

        # s.ST = [20.0] * p.NLAYR
