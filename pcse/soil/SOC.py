import calendar
import math
from math import exp

import numpy as np

from .. import signals
from ..base import ParamTemplate, RatesTemplate, SimulationObject, StatesTemplate
from ..db.tillage_parameters import (
    get_tillage_code_params,
)
from ..decorators import prepare_rates, prepare_states


# Next steps to improve the efficiency :

# Move all the steps that needs to be done only once to the initialize function. An example of this is the calculation of MIMIC parameters.
# some parameters for the emulator. These should also better be moved out of the params and use the soil profile input (these are now inside the soil_data param and not being updated for each site) :
# - bulk_density
# - clay and silt content


class Ensemble_SOC_Indigo(SimulationObject):
    """This wraps the soil water balance for free drainage conditions and NPK balance
    for production conditions limited by both soil water and NPK. Built by Indigo Ag.
    We are also working on adding the soil carbon dynamics.
    """

    __slots__ = [
        "Century_SOC_Indigo",
        "MillennialV2_SOC_Indigo",
        "RothC_SOC_Indigo",
        "MIMICS_SOC_Indigo",
        "IOM",
    ]

    Century_SOC_Indigo: SimulationObject
    MillennialV2_SOC_Indigo: SimulationObject
    RothC_SOC_Indigo: SimulationObject
    MIMICS_SOC_Indigo: SimulationObject
    IOM: float

    def __init__(self, day, kiosk, *args, **kwargs):
        self.IOM = 0.0
        super().__init__(day, kiosk, *args, **kwargs)

    class Parameters(ParamTemplate):

        __slots__ = [
            "Ens_Weights",
            "POM0",
            "LMWC0",
            "AGG0",
            "MIC0",
            "MAOM0",
            "ACTIVE0",
            "SLOW0",
            "PASSIVE0",
            "DPM0",
            "RPM0",
            "BIO0",
            "HUM0",
            "MIC_1_0",
            "MIC_2_0",
            "SOM_1_0",
            "SOM_2_0",
            "SOM_3_0",
            "TIsoc",
        ]

        Ens_Weights: list  # Thickness of soil layers (cm)
        POM0: float  # g C m^-2
        LMWC0: float  # g C m^-2
        AGG0: float  # g C m^-2
        MIC0: float  # g C m^-2
        MAOM0: float  # g C m^-2

        ACTIVE0: float  # g C m^-2
        SLOW0: float  # g C m^-2
        PASSIVE0: float  # g C m^-2

        DPM0: float
        RPM0: float
        BIO0: float
        HUM0: float

        # initialisation values
        MIC_1_0: float
        MIC_2_0: float
        SOM_1_0: float
        SOM_2_0: float
        SOM_3_0: float

        TIsoc: float  # Total initial soc

    class StateVariables(StatesTemplate):

        __slots__ = [
            "Ens_SOC",
            "Century_SOC",
            "MillennialV2_SOC",
            "RothC_SOC",
            "MIMICS_SOC",
        ]

        Ens_SOC: float
        Century_SOC: float
        MillennialV2_SOC: float
        RothC_SOC: float
        MIMICS_SOC: float

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """

        self.MillennialV2_SOC_Indigo = MillennialV2_SOC_Indigo(day, kiosk, parvalues)
        self.Century_SOC_Indigo = Century_SOC_Indigo(day, kiosk, parvalues)
        self.RothC_SOC_Indigo = RothC_SOC_Indigo(day, kiosk, parvalues)
        self.MIMICS_SOC_Indigo = MIMICS_SOC_Indigo(day, kiosk, parvalues)

        self.params = self.Parameters(parvalues)

        # if sum of weights is not 1, normalize the weights
        if sum(self.params.Ens_Weights) != 1:
            self.params.Ens_Weights = [
                w / sum(self.params.Ens_Weights) for w in self.params.Ens_Weights
            ]

        # sum MillennialV2_SOC_Indigo and Century_SOC_Indigo SOC
        M1_Raw = (
            self.params.POM0
            + self.params.LMWC0
            + self.params.AGG0
            + self.params.MIC0
            + self.params.MAOM0
        )
        M1 = M1_Raw * self.params.Ens_Weights[0]

        M2_Raw = self.params.ACTIVE0 + self.params.SLOW0 + self.params.PASSIVE0
        M2 = M2_Raw * self.params.Ens_Weights[1]

        M3_Raw = (
            self.params.DPM0 + self.params.RPM0 + self.params.BIO0 + self.params.HUM0
        )
        M3 = (
            M3_Raw * self.params.Ens_Weights[2]
        )  # in Initialized the initial pool size is sent for gC/m2 for RothC as well as the others.

        M4_Raw = (
            self.params.MIC_1_0
            + self.params.MIC_2_0
            + self.params.SOM_1_0
            + self.params.SOM_2_0
            + self.params.SOM_3_0
        )
        M4 = M4_Raw * self.params.Ens_Weights[3]

        Ens_SOC_weighted = M1 + M2 + M3 + M4

        # print(f"M3 {M3} {self.params.DPM0} {self.params.RPM0} {self.params.BIO0} {self.params.HUM0}")
        # print(f"-- Initialized : Ens_SOC_weighted {Ens_SOC_weighted} IOM {self.kiosk['IOM']} M1 {M1} M2 {M2} M3 {M3}")

        self.states = self.StateVariables(
            kiosk,
            Ens_SOC=Ens_SOC_weighted,
            Century_SOC=M2_Raw,
            MillennialV2_SOC=M1_Raw,
            RothC_SOC=M3_Raw,
            MIMICS_SOC=M4_Raw,
            publish=["Ens_SOC"],
        )

    def calc_rates(self, day, drv):

        self.MillennialV2_SOC_Indigo.calc_rates(day, drv)
        self.Century_SOC_Indigo.calc_rates(day, drv)
        self.RothC_SOC_Indigo.calc_rates(day, drv)
        self.MIMICS_SOC_Indigo.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):

        self.MillennialV2_SOC_Indigo.integrate(day, delt)
        self.Century_SOC_Indigo.integrate(day, delt)
        self.RothC_SOC_Indigo.integrate(day, delt)
        self.MIMICS_SOC_Indigo.integrate(day, delt)

        # We can add the ensemble integration here if needed
        if "POM" in self.kiosk:
            # MillennialV2
            M1_Raw = (
                self.kiosk["POM"]
                + self.kiosk["LMWC"]
                + self.kiosk["AGG"]
                + self.kiosk["MIC"]
                + self.kiosk["MAOM"]
            )
            M1 = M1_Raw * self.params.Ens_Weights[0]

            # CENTURY
            M2_Raw = (
                self.kiosk["SOC_ACTIVE"]
                + self.kiosk["SOC_SLOW"]
                + self.kiosk["SOC_PASSIVE"]
            )
            M2 = M2_Raw * self.params.Ens_Weights[1]

            # RothC
            M3_Raw = (
                self.kiosk["BIO"] + self.kiosk["HUM"] + (self.kiosk["IOM"] * 100)
            )  # self.kiosk['DPM'] + self.kiosk['RPM'] +
            M3 = M3_Raw * self.params.Ens_Weights[2]

            # MIMICS
            M4_Raw = (
                self.kiosk["MIC_1"]
                + self.kiosk["MIC_2"]
                + self.kiosk["SOM_1"]
                + self.kiosk["SOM_2"]
                + self.kiosk["SOM_3"]
            )
            M4 = M4_Raw * self.params.Ens_Weights[3]

            Ens_SOC_weighted = M1 + M2 + M3 + M4

            self.states.Ens_SOC = Ens_SOC_weighted
            self.states.Century_SOC = M2_Raw
            self.states.MillennialV2_SOC = M1_Raw
            self.states.RothC_SOC = M3_Raw
            self.states.MIMICS_SOC = M4_Raw

            # print(f"-- Integrate : Ens_SOC_weighted {Ens_SOC_weighted} IOM {self.kiosk['IOM']} M1 {M1} M2 {M2} M3 {M3} ")


class MIMICS_SOC_Indigo(SimulationObject):
    """
    Based on MIMICS v0.1 Copyright (c) 2015 will wieder

    MIMICS v0.1 is licensed under the terms of the MIT license:
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the “Software”),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software
    is furnished to do so, subject to the following conditions. The above copyright
    notice and this permission notice shall be included in all copies or substantial
    portions of the Software. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF
    ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
    EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    Changes made by Hamze Dokoohaki © Indigo Ag, Inc. 2024-2025:
    adapted for interoperability with the PCSE framework
    """

    __slots__ = [
        "tao",
        "Tao_MOD1",
        "fMET",
        "in_crop_cycle",
        "fMET",
        "fPHYS",
        "fCHEM",
        "fAVAI",
        "desorb",
        "pSCALAR",
        "clayF",
        "MOD1",
        "MOD2",
        "npp_dead_in_season",
        "residue_perc_removed",
        "_increments_SOM1",
        "_increments_SOM2",
        "_increments_SOM3",
        "_increments_MIC1",
        "_increments_MIC2",
    ]

    tao: float
    Tao_MOD1: float  # Modifier based on net primary productivity
    fMET: float  # Fraction of metabolic response
    in_crop_cycle: bool
    fMET: float
    fPHYS: np.ndarray  # fraction to SOMp
    fCHEM: np.ndarray
    fAVAI: np.ndarray
    desorb: float
    pSCALAR: float
    clayF: float
    MOD1: np.ndarray
    MOD2: np.ndarray
    npp_dead_in_season: float
    residue_perc_removed: float

    _increments_SOM1: list
    _increments_SOM2: list
    _increments_SOM3: list
    _increments_MIC1: list
    _increments_MIC2: list

    def __init__(self, day, kiosk, *args, **kwargs):
        self.tao = 0.0
        self.Tao_MOD1 = 0.0
        self.fMET = 0.0
        self.in_crop_cycle = False
        self.fMET = 0.0
        self.fPHYS = np.array([np.nan, np.nan])
        self.fCHEM = np.array([np.nan, np.nan])
        self.fAVAI = np.array([np.nan, np.nan])
        self.desorb = 0.0
        self.pSCALAR = 0.0
        self.clayF = 0.0
        self.MOD1 = np.array([np.nan, np.nan])
        self.MOD2 = np.array([np.nan, np.nan])
        self.npp_dead_in_season = 0.0
        self.residue_perc_removed = 0.0
        self._increments_SOM1 = []
        self._increments_SOM2 = []
        self._increments_SOM3 = []
        self._increments_MIC1 = []
        self._increments_MIC2 = []
        super().__init__(day, kiosk, *args, **kwargs)

    def _on_manage_residue(self, **kwargs):
        """Apply tillage"""
        residue_perc = kwargs.get("residue_percent")
        self.residue_perc_removed = residue_perc / 100

    def _on_CROP_START(self):
        self.in_crop_cycle = True

    def _on_CROP_FINISH(self):
        self.in_crop_cycle = False

    class Parameters(ParamTemplate):

        __slots__ = [
            "depth",
            "Vslope",
            "Vint",
            "Kslope",
            "Kint",
            "CUE",
            "k",
            "a",
            "cMAX",
            "cMIN",
            "cSLOPE",
            "KO",
            "FI",
            "aV",
            "aK",
            "param_clay",
            "param_pH",
            "MIC_1_0",
            "MIC_2_0",
            "SOM_1_0",
            "SOM_2_0",
            "SOM_3_0",
            "ROOT_SHOOT_K",
        ]

        depth: float
        Vslope: list
        Vint: float
        Kslope: list
        Kint: float
        CUE: list  # Carbon Use Efficiency array for different processes
        k: float
        a: float
        cMAX: float
        cMIN: float
        cSLOPE: float
        KO: list
        FI: list
        aV: float  # Modifier for maximum rate of enzymatic activity
        aK: float  # Modifier for the Michaelis constant
        param_clay: float  # Fraction of clay content influencing rates
        param_pH: float  # Soil pH
        # initialisation values
        MIC_1_0: float
        MIC_2_0: float
        SOM_1_0: float
        SOM_2_0: float
        SOM_3_0: float
        ROOT_SHOOT_K: float

    class StateVariables(StatesTemplate):

        __slots__ = [
            "LIT_1",
            "LIT_2",
            "MIC_1",
            "MIC_2",
            "SOM_1",
            "SOM_2",
            "SOM_3",
        ]

        LIT_1: float  # Litter pool 1
        LIT_2: float  # Litter pool 2
        MIC_1: float  # Microbial biomass pool 1
        MIC_2: float  # Microbial biomass pool 2
        SOM_1: float  # Soil organic matter pool 1
        SOM_2: float  # Soil organic matter pool 2
        SOM_3: float  # Soil organic matter pool 3

    class RateVariables(RatesTemplate):

        __slots__ = [
            "dLIT_1",
            "dLIT_2",
            "dMIC_1",
            "dMIC_2",
            "dSOM_1",
            "dSOM_2",
            "dSOM_3",
        ]

        dLIT_1: float  # Litter pool 1
        dLIT_2: float  # Litter pool 2
        dMIC_1: float  # Microbial biomass pool 1
        dMIC_2: float  # Microbial biomass pool 2
        dSOM_1: float  # Soil organic matter pool 1
        dSOM_2: float  # Soil organic matter pool 2
        dSOM_3: float  # Soil organic matter pool 3

    def initialize(self, day, kiosk, parametervalues):
        self.params = self.Parameters(parametervalues)
        self.states = self.StateVariables(
            kiosk,
            LIT_1=0,
            LIT_2=0,
            MIC_1=self.params.MIC_1_0,
            MIC_2=self.params.MIC_2_0,
            SOM_1=self.params.SOM_1_0,
            SOM_2=self.params.SOM_2_0,
            SOM_3=self.params.SOM_3_0,
            publish=["MIC_1", "MIC_2", "SOM_1", "SOM_2", "SOM_3"],
        )
        self.rates = self.RateVariables(
            kiosk,
            publish=[
                "dLIT_1",
                "dLIT_2",
                "dMIC_1",
                "dMIC_2",
                "dSOM_1",
                "dSOM_2",
                "dSOM_3",
            ],
        )
        # Initialize other model-specific states or parameters if necessary
        # search for crop transpiration values
        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_manage_residue, signals.manage_residue)

        # print(f"Initialized MIMICS_SOC_Indigo with {self.params.MIC_1_0} {self.params.MIC_2_0} {self.params.SOM_1_0} {self.params.SOM_2_0} {self.params.SOM_3_0}")
        # Calculate a bunch of parameters ------------------------

        bagCN = np.array(
            [133.3, 92.7, 83.1, 61.8, 50.5, 24.2]
        )  # from Gordon's LitterCharacteristics.txt
        bagLIG = np.array(
            [16.2, 19.2, 26.7, 15.9, 23.5, 10.9]
        )  # % from Gordon's LitterCharacteristics.txt

        calcN = (
            (1 / bagCN) / 2.5 * 100
        )  # Ensure calcN is defined correctly here (it was not provided in the R snippet)

        calcMET = 0.85 - 0.013 * (bagLIG / calcN)  # as calculated in DAYCENT
        self.fMET = np.mean(calcMET)
        # print("fMET", self.fMET)
        self.clayF = self.params.param_clay / 100

        self.fPHYS = (
            np.array([0.3 * np.exp(1.3 * self.clayF), 0.2 * np.exp(0.8 * self.clayF)])
            * 0.6
        )  # fraction to SOMp https://github.com/piersond/MIMICS_HiRes/blob/3c3aa43c50277c635fa816d811f99e201c9fdb9d/MIMICS_ftns/MIMICS_base_ftn.R#L93C16-L93C25
        self.fCHEM = np.array(
            [0.1 * np.exp(-3.0 * self.fMET), 0.3 * np.exp(-3.0 * self.fMET)]
        )  # fraction to SOMc
        self.fAVAI = 1.0 - (self.fPHYS + self.fCHEM)
        self.desorb = (
            1.5e-5 * np.exp(-1.5 * (self.clayF)) * 24 * 2.36935554
        )  # if modified by MIC! https://github.com/piersond/MIMICS_HiRes/blob/3c3aa43c50277c635fa816d811f99e201c9fdb9d/MIMICS_ftns/MIMICS_base_ftn.R#L92
        # - --------
        self.pSCALAR = self.params.a * np.exp(
            -self.params.k * (np.sqrt(self.clayF))
        )  # Scalar for texture effects on SOMp
        self.MOD1 = np.array([10, 2, 10, 3, 3, 2])
        self.MOD2 = np.array([8, 2, 4 * self.pSCALAR, 2, 4, 6 * self.pSCALAR])

        # print(f" pSCALAR {self.pSCALAR} desorb {self.desorb} fPHYS {self.fPHYS} fCHEM {self.fCHEM} fAVAI {self.fAVAI} fMET {self.fMET} clayF {self.clayF}")

    @prepare_rates
    def calc_rates(self, day, drv):
        # Hydrological properties

        forc_st = self.kiosk["ST"][2]  # Soil temperature (°C)
        y = [
            self.states.LIT_1,
            self.states.LIT_2,
            self.states.MIC_1,
            self.states.MIC_2,
            self.states.SOM_1,
            self.states.SOM_2,
            self.states.SOM_3,
        ]

        # C input is either litter during the frowing season or residue at the end of the season
        # dead fall
        if "DMI" not in self.kiosk:
            forc_npp = 0
        else:
            # DMI comes from WOFOST8 crop model
            # DMI Total dry matter increase, calculated as ASRC times a weighted conversion efficiency. Y  |kg ha-1 d-1|
            # 4% dead root and shoot material, 60% of which is C
            forc_npp = (
                self.kiosk["GASST"]
                - self.kiosk["MREST"]
                - (self.npp_dead_in_season * 10)
            )  # Daily plant inputs/daily net primary production needs to be in  (gC/m2/d)
            forc_npp = (
                forc_npp * self.params.ROOT_SHOOT_K * 1000 / 10000
            )  # kg C ha^-1 day^-1 -> g C m^-2 day^-1

        self.npp_dead_in_season += forc_npp  # keep track of the dead in season

        if "DVS" in self.kiosk:
            if "TAGP" in self.kiosk:
                if not self.in_crop_cycle:
                    # (total gross assimilation in C - Total respiration in C) - (50% of above ground * 60% to convert to C)
                    # forc_npp += ((self.kiosk['GASST']- self.kiosk['MREST']) - (0.5 * self.kiosk['TAGP'] * 0.6) - (self.npp_dead_in_season*10))  /10 # kg C ha^-1 day^-1 -> t C ha^-1 day^-1  -> residue -> C
                    total_net_assimilated = (
                        self.kiosk["GASST"] - self.kiosk["MREST"]
                    ) - (self.npp_dead_in_season * 10)
                    yield_C_removed = (
                        0.5 * self.kiosk["TAGP"] * 0.6
                    )  # 50% HI of above ground * 60% to convert to C
                    residue_left = (total_net_assimilated - yield_C_removed) * (
                        1 - self.residue_perc_removed
                    )
                    forc_npp += (
                        residue_left
                    ) / 10  # kg C ha^-1 day^-1 -> t C ha^-1 day^-1  -> residue -> C
                    self.npp_dead_in_season = 0
                    # print(f"day {day} C_input:{forc_npp} residue_left:{residue_left} total_net_assimilated:{total_net_assimilated} yield_C_removed:{yield_C_removed} residue_perc_removed:{self.residue_perc_removed}")

        forc_npp = (forc_npp,)
        # print(f"{day}: forc_npp {forc_npp} {self.npp_dead_in_season} ---------------------------")
        # estimate the rates of change
        diff_eq = self.XEQ_fw(
            y, forc_npp, forc_st
        )  # Use XEQ_fw function to calculate rates

        for i, rate_name in enumerate(
            ["dLIT_1", "dLIT_2", "dMIC_1", "dMIC_2", "dSOM_1", "dSOM_2", "dSOM_3"]
        ):
            setattr(self.rates, rate_name, diff_eq[i])

    @prepare_states
    def integrate(self, day, delt):
        for state_name in [
            "LIT_1",
            "LIT_2",
            "MIC_1",
            "MIC_2",
            "SOM_1",
            "SOM_2",
            "SOM_3",
        ]:
            rate_name = "d" + state_name
            new_value = (
                getattr(self.states, state_name) + getattr(self.rates, rate_name) * delt
            )
            setattr(
                self.states, state_name, max(new_value, 0)
            )  # Ensure non-negative states

    def finalize(self, day):
        # Optionally include any final operations after the simulation
        pass

    def XEQ_fw(self, y, forc_npp, TSOI):  # y=st_state , t=time_step

        params = self.params
        # for param Vslope multiply each value by 24

        LIT_1 = y[0]
        LIT_2 = y[1]
        MIC_1 = y[2]
        MIC_2 = y[3]
        SOM_1 = y[4]
        SOM_2 = y[5]
        SOM_3 = y[6]

        LITmin = np.zeros(4)
        MICtrn = np.zeros(6)
        SOMmin = np.zeros(2)
        DEsorb = np.zeros(1)
        OXIDAT = np.zeros(1)

        depth = params.depth
        ################### estimate parameters
        EST_LIT = np.sum(forc_npp)  # gC/m2/day (Knapp et al. Science 2001)
        # EST_LIT     = EST_LIT_in  * 1e3 / 1e4    #gC/m2/day
        I = np.zeros(2)  # Litter inputs to MET/STR
        I[0] = (EST_LIT / depth) * self.fMET  # partitioned to layers (gC/m3/day)
        I[1] = (EST_LIT / depth) * (1 - self.fMET)

        # print(f" EST_LIT {EST_LIT} I {I} depth {depth*100} fMET {self.fMET}")
        sum_force_npp = np.sum(forc_npp)
        if sum_force_npp < 0:
            sum_force_npp = 0  # to avoid negative values in the sqrt function

        self.Tao_MOD1 = np.sqrt(
            sum_force_npp * 365 / 100.0
        )  # basically standardize against NWT (forc_npp*365=ANPP (gC/m2/y)) # 365 was changed to 1 to make it daily
        # make sure self.Tao_MOD1 is within the range of 0.8 - 1.2 Table B1. If smaller or larger, set to 0.8 or 1.2
        self.Tao_MOD1 = min(1.2, max(0.8, self.Tao_MOD1))
        tao = (
            np.array(
                [5.2e-4 * np.exp(0.3 * self.fMET), 2.4e-4 * np.exp(0.1 * self.fMET)]
            )
            * 24
        )  # unit was per hr so it was *24 to daily

        tao = tao * self.Tao_MOD1  # daily
        vs = (
            np.log(np.exp(params.Vslope) * 24) * 1.69
        )  # https://github.com/piersond/MIMICS_HiRes/blob/3c3aa43c50277c635fa816d811f99e201c9fdb9d/MIMICS_ftns/MIMICS_base_ftn.R#L86
        vint = np.log(np.exp(params.Vint) * 24) * 0.633318

        ks = np.log(np.exp(params.Kslope) * 24) * 1.782366
        kint = np.log(np.exp(params.Kint) * 24) * 0.3609913

        Vmax = np.exp(TSOI * (vs) + (vint)) * params.aV  # daily
        Km = np.exp(np.array(ks) * TSOI + kint) * params.aK
        VMAX = Vmax * self.MOD1
        KM = Km / self.MOD2

        ################### estimating the flows
        som1_till_factor = self.kiosk["CLTFAC"][0]
        som2_till_factor = self.kiosk["CLTFAC"][1]
        som3_till_factor = self.kiosk["CLTFAC"][2]
        som4_till_factor = self.kiosk["CLTFAC"][3]
        # Flows to and from MIC_1
        LITmin[0] = (
            MIC_1 * VMAX[0] * LIT_1 / (KM[0] + LIT_1)
        )  # MIC_1 decomp of MET lit#EQA1
        LITmin[1] = (
            MIC_1 * VMAX[1] * LIT_2 / (KM[1] + LIT_2)
        )  # MIC_1 decomp of STRUC lit#EQA2
        MICtrn[0] = (
            MIC_1 * tao[0] * self.fPHYS[0]
        )  # MIC_1 turnover to PHYSICAL SOM  #EQA4
        MICtrn[1] = (
            MIC_1 * tao[0] * self.fCHEM[0] * som2_till_factor
        )  # MIC_1 turnover to CHEMICAL SOM
        MICtrn[2] = (
            MIC_1 * tao[0] * self.fAVAI[0] * som2_till_factor
        )  # MIC_1 turnover to AVAILABLE SOM
        SOMmin[0] = (
            MIC_1 * VMAX[2] * SOM_3 / (KM[2] + SOM_3)
        )  # decomp of SOMa by MIC_1#EQA3

        # Flows to and from MIC_2
        LITmin[2] = (
            MIC_2 * VMAX[3] * LIT_1 / (KM[3] + LIT_1)
        )  # decomp of MET litter        #EQA5
        LITmin[3] = (
            MIC_2 * VMAX[4] * LIT_2 / (KM[4] + LIT_2)
        )  # decomp of SRUCTURAL litter        #EQA6
        MICtrn[3] = (
            MIC_2 * tao[1] * self.fPHYS[1]
        )  # MIC_2 turnover to PHYSICAL  SOM #EQA8
        MICtrn[4] = MIC_2 * tao[1] * self.fCHEM[1]  # MIC_2 turnover to CHEMICAL  SOM
        MICtrn[5] = MIC_2 * tao[1] * self.fAVAI[1]  # MIC_2 turnover to AVAILABLE SOM

        SOMmin[1] = (
            MIC_2 * VMAX[5] * SOM_3 / (KM[5] + SOM_3)
        )  # decomp of SOMa by MIC_2	    #EQA7

        DEsorb = (
            SOM_1 * self.desorb
        )  # * (MIC_1 + MIC_2)    #EQA9       #desorbtion of PHYS to AVAIL (function of params.param_clay)

        KO = np.array(params.KO) * 1.3
        OXIDAT = (MIC_2 * VMAX[4] * SOM_2 / (KO[1] * KM[4] + SOM_2)) + (
            MIC_1 * VMAX[1] * SOM_2 / (KO[0] * KM[1] + SOM_2)
        )  # oxidation of C to A  #EQA10

        # can make fluxes from CHEM a function of microbial biomass size?
        dLIT_1 = I[0] * (1 - params.FI[0]) - LITmin[0] - LITmin[2]
        dLIT_2 = np.sum(I[1]) * (1.0 - params.FI[1]) - LITmin[1] - LITmin[3]

        dMIC_1 = (
            params.CUE[0] * (LITmin[0] + SOMmin[0])
            + params.CUE[1] * (LITmin[1])
            - np.sum(MICtrn[0:3])
        )
        dMIC_2 = (
            params.CUE[2] * (LITmin[2] + SOMmin[1])
            + params.CUE[3] * (LITmin[3])
            - np.sum(MICtrn[3:6])
        )

        dSOM_1 = (
            I[0] * params.FI[0] + MICtrn[0] + MICtrn[3] - DEsorb
        )  # physcially protected SOC

        dSOM_2 = (
            np.sum(I[1]) * params.FI[1] + MICtrn[1] + MICtrn[4] - OXIDAT
        )  # chemically protected SOC

        dSOM_3 = MICtrn[2] + MICtrn[5] + DEsorb + OXIDAT - SOMmin[0] - SOMmin[1]

        diff_eq = [dLIT_1, dLIT_2, dMIC_1, dMIC_2, dSOM_1, dSOM_2, dSOM_3]

        return diff_eq

    def _set_variable_SOM_1(self, nSOM1):
        """Force the model states based on the input value."""
        increment = nSOM1 - self.states.SOM_1
        self.states.SOM_1 = nSOM1
        self._increments_SOM1.append(increment)
        return {"SOM_1": increment}

    def _set_variable_SOM_2(self, nSOM2):
        """Force the model states based on the input value."""
        increment = nSOM2 - self.states.SOM_2
        self.states.SOM_2 = nSOM2
        self._increments_SOM2.append(increment)
        return {"SOM_2": increment}

    def _set_variable_SOM_3(self, nSOM3):
        """Force the model states based on the input value."""
        increment = nSOM3 - self.states.SOM_3
        self.states.SOM_3 = nSOM3
        self._increments_SOM3.append(increment)
        return {"SOM_3": increment}

    def _set_variable_MIC_1(self, nMIC1):
        """Force the model states based on the input value."""
        increment = nMIC1 - self.states.MIC_1
        self.states.MIC_1 = nMIC1
        self._increments_MIC1.append(increment)
        return {"MIC_1": increment}

    def _set_variable_MIC_2(self, nMIC2):
        """Force the model states based on the input value."""
        increment = nMIC2 - self.states.MIC_2
        self.states.MIC_2 = nMIC2
        self._increments_MIC2.append(increment)
        return {"MIC_2": increment}


# https://zenodo.org/records/10732265


class RothC_SOC_Indigo(SimulationObject):
    """
    Soil Organic Carbon (SOC) Model based on RothC python version v1.0.0
    Copyright (c) 2024 Rothamsted Research.

    RothC_Py v1.0.0 licensed under the Apache License, Version 2.0 (the “License”);
    you may not use RothC_Py v1.0.0 except in compliance with the License. You may
    obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless
    required by applicable law or agreed to in writing, software distributed under the
    License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied. See the License for the specific language governing
    permissions and limitations under the License.

    Changes made by Hamze Dokoohaki © Indigo Ag, Inc. 2024-2025:
    adapted for interoperability with the PCSE framework
    """

    __slots__ = [
        "iom_power",
        "accTSMD",
        "t_param",
        "IOM",
        "C_input",
        "evaps",
        "rain_sum",
        "temp_avg",
        "in_crop_cycle",
        "npp_dead_in_season",
        "residue_perc_removed",
        "_increments_DPM",
        "_increments_RPM",
        "_increments_BIO",
        "_increments_HUM",
        "_increments_IOM",
    ]

    iom_power: float
    accTSMD: float
    t_param: float  # year->month # the K decomposition matrix is multipled by this
    IOM: float
    # ----- collectors
    C_input: float
    evaps: float  # sum evapotration
    rain_sum: float  # sum rain
    temp_avg: float  # Avg temp
    # Flag indicating crop present or not
    in_crop_cycle: bool
    npp_dead_in_season: float
    residue_perc_removed: float

    _increments_DPM: list
    _increments_RPM: list
    _increments_BIO: list
    _increments_HUM: list
    _increments_IOM: list

    def __init__(self, day, kiosk, *args, **kwargs):
        self.iom_power = 1.255
        self.accTSMD = 0.0
        self.t_param = 1.0 / 12.0
        self.IOM = 0.0
        self.C_input = 0.0
        self.evaps = 0.0
        self.rain_sum = 0.0
        self.temp_avg = 0.0
        self.in_crop_cycle = False
        self.npp_dead_in_season = 0.0
        self.residue_perc_removed = 0.0
        self._increments_DPM = []
        self._increments_RPM = []
        self._increments_BIO = []
        self._increments_HUM = []
        self._increments_IOM = []
        super().__init__(day, kiosk, *args, **kwargs)

    def _on_manage_residue(self, **kwargs):
        """Apply tillage"""
        residue_perc = kwargs.get("residue_percent")
        self.residue_perc_removed = residue_perc / 100

    class Parameters(ParamTemplate):

        __slots__ = [
            "DPM_RPM_HUM_frac_inputs",
            "DPM_RPM_HUM_frac_inputs_EOM",
            "c_param",
            "param_temp_in",
            "param_clay",
            "DPM0",
            "RPM0",
            "BIO0",
            "HUM0",
            "kDPM",
            "kRPM",
            "kBIO",
            "kHUM",
            "TIsoc",
            "ROOT_SHOOT_K",
        ]

        DPM_RPM_HUM_frac_inputs: list
        DPM_RPM_HUM_frac_inputs_EOM: list
        c_param: float  # c_vegetated = 0.6
        # t_param: float # t_param=1./12. #year->month
        param_temp_in: float  # param_temp_in=18.27 #C
        param_clay: float  # param_claysilt=0.3
        # initialisation values
        DPM0: float
        RPM0: float
        BIO0: float
        HUM0: float
        # k values
        kDPM: float  # 10.
        kRPM: float  # 0.3
        kBIO: float  # 0.66
        kHUM: float  # 0.02
        TIsoc: float  # Total inital soc
        ROOT_SHOOT_K: float

    def _on_CROP_START(self):
        self.in_crop_cycle = True

    def _on_CROP_FINISH(self):
        self.in_crop_cycle = False

    class StateVariables(StatesTemplate):

        __slots__ = [
            "DPM",
            "RPM",
            "BIO",
            "HUM",
            "IOM",
        ]

        DPM: float
        RPM: float
        BIO: float
        HUM: float
        IOM: float

    class RateVariables(RatesTemplate):

        __slots__ = ["dDPM", "dRPM", "dBIO", "dHUM"]

        dDPM: float
        dRPM: float
        dBIO: float
        dHUM: float

    def is_end_of_month(self, date):
        return date.day == calendar.monthrange(date.year, date.month)[1]

    def initialize(self, day, kiosk, parvalues):
        """Initialize the model."""
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk)

        # Initialize state variables using spinup
        SOC_data_init = self.params.TIsoc  # g C m^-2
        self.IOM = (
            SOC_data_init
            - (
                self.params.DPM0
                + self.params.RPM0
                + self.params.BIO0
                + self.params.HUM0
            )
        ) / 100  # g C m^-2 -> t C ha^-1

        # ss_spinup = self.spinup(SOC_data_init, plantin, EOMin, clay, temp, water, mean_pot_evapot,
        #                        self.params.c_param, self.params.t_param, self.params.param_temp_in,
        #                        self.params.DPM_RPM_HUM_frac_inputs, self.params.DPM_RPM_HUM_frac_inputs_EOM)
        self.states = self.StateVariables(
            kiosk,
            DPM=self.params.DPM0 / 100,
            RPM=self.params.RPM0 / 100,
            BIO=self.params.BIO0 / 100,
            HUM=self.params.HUM0 / 100,
            IOM=self.IOM,  # g C m^-2 -> t C ha^-1
            publish=["DPM", "RPM", "BIO", "HUM", "IOM"],
        )
        self.accTSMD = 0.0
        self.residue_perc_removed = 0
        # Connect to CROP_START/CROP_FINISH signals for water balance to
        # search for crop transpiration values
        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_manage_residue, signals.manage_residue)

    @prepare_rates
    def calc_rates(self, day, drv):

        if "DVS" in self.kiosk:
            # print(f"day {day} {self.kiosk['DVS']} - in_cycle: {self.in_crop_cycle} - TAGP : {self.kiosk['TAGP']}")
            if "TAGP" in self.kiosk:
                if not self.in_crop_cycle:
                    total_net_assimilated = (
                        self.kiosk["GASST"] - self.kiosk["MREST"]
                    ) - (self.npp_dead_in_season * 10)
                    yield_C_removed = (
                        0.5 * self.kiosk["TAGP"] * 0.6
                    )  # 50% HI of above ground * 60% to convert to C
                    residue_left = (total_net_assimilated - yield_C_removed) * (
                        1 - self.residue_perc_removed
                    )

                    self.C_input += (
                        residue_left / 1000
                    )  # ((self.kiosk['GASST']- self.kiosk['MREST']) - (0.5 * self.kiosk['TAGP'] * 0.6) - (self.npp_dead_in_season*10)) / 1000   # kg C ha^-1 day^-1 -> t C ha^-1 day^-1  -> residue -> C
                    # print(f" + + day {day} C_input:{self.C_input} residue_left:{residue_left/1000} total_net_assimilated:{total_net_assimilated} yield_C_removed:{yield_C_removed} residue_perc_removed:{self.residue_perc_removed}")
                    self.npp_dead_in_season = 0
        # -------------------------------------------------- Get input data from the kiosk
        end_month = self.is_end_of_month(day)  # for monthly time step

        if end_month:

            forc_npp = self.C_input  # tC ha^-1 month^-1
            mean_pot_evapot = self.evaps  # mm/month
            water = self.rain_sum  # mm/month
            temp = self.temp_avg / 30  # C

            # print(f"day {day} temp:{temp} water:{water} mean_pot_evapot:{mean_pot_evapot} forc_npp:{forc_npp}")

            """Calculate the rates of change."""
            params = self.params
            states = self.states
            matrix_in = np.zeros(4)
            Kvalues = [params.kDPM, params.kRPM, params.kBIO, params.kHUM]

            plantin = np.array(
                [
                    forc_npp * params.DPM_RPM_HUM_frac_inputs[0],
                    forc_npp * params.DPM_RPM_HUM_frac_inputs[1],
                    forc_npp * params.DPM_RPM_HUM_frac_inputs[2],
                ]
            )

            org_amend = 0  # Organic amendment
            EOMin = np.array(
                [
                    org_amend * params.DPM_RPM_HUM_frac_inputs_EOM[0],
                    org_amend * params.DPM_RPM_HUM_frac_inputs_EOM[1],
                    org_amend * params.DPM_RPM_HUM_frac_inputs_EOM[2],
                ]
            )
            litterin = plantin + EOMin

            for i in range(0, len(litterin)):
                matrix_in[i] = litterin[i]

            clay = params.param_clay  # Clay content (%)

            # ----------------------------------------------------- Calculate A and K matrices
            a_ma = self.A_matrix(
                clay,
                plantin,
                EOMin,
                params.DPM_RPM_HUM_frac_inputs,
                params.DPM_RPM_HUM_frac_inputs_EOM,
            )

            kk_ma = self.K_matrix(
                Kvalues,
                clay,
                temp,
                water,
                mean_pot_evapot,
                params.c_param,
                self.t_param,
                params.param_temp_in,
            )

            # print(f"day {day} a_ma:{a_ma} kk_ma:{kk_ma} ")
            # Calculate rates of change
            matrix_current = np.array([states.DPM, states.RPM, states.BIO, states.HUM])

            matrix_next = (
                matrix_current + matrix_in + np.dot(a_ma, np.dot(kk_ma, matrix_current))
            )

            self.rates.dDPM = matrix_next[0] - states.DPM
            self.rates.dRPM = matrix_next[1] - states.RPM
            self.rates.dBIO = matrix_next[2] - states.BIO
            self.rates.dHUM = matrix_next[3] - states.HUM
            # reset cumulative inputs
            self.C_input = 0.0
            self.evaps = 0.0
            self.rain_sum = 0.0
            self.temp_avg = 0.0
        else:
            if "DMI" in self.kiosk:

                self.C_input += (
                    (
                        self.kiosk["GASST"]
                        - self.kiosk["MREST"]
                        - (self.npp_dead_in_season * 10)
                    )
                    * self.params.ROOT_SHOOT_K
                    / 1000
                )  # shoot and root death at the rate of 0.05

            self.npp_dead_in_season += self.C_input  # keep track of the dead in season

            self.evaps += drv.E0 * 10  # cm -> mm
            self.rain_sum += drv.RAIN * 10  # cm -> mm
            self.temp_avg += (drv.TMAX + drv.TMIN) / 2

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Update the state variables."""
        rates = self.rates
        states = self.states

        states.DPM += rates.dDPM * delt  # t C ha^-1 -> g C m^-2
        states.RPM += rates.dRPM * delt
        states.BIO += rates.dBIO * delt
        states.HUM += rates.dHUM * delt
        states.IOM = self.IOM  # t C ha^-1 -> g C m^-2

        for attr in ["DPM", "RPM", "BIO", "HUM"]:
            if getattr(self.states, attr) < 0:
                setattr(self.states, attr, 0)

    def A_matrix(
        self, clay, plantin, EOMin, DPM_RPM_HUM_frac_inputs, DPM_RPM_HUM_frac_inputs_EOM
    ):
        """Calculate the A matrix."""
        BIO_frac_fromRPM = 0.46
        HUM_frac_fromRPM = 1.0 - BIO_frac_fromRPM

        alpha = BIO_frac_fromRPM * self.BIO_HUM_partition(clay)
        beta = (
            HUM_frac_fromRPM * self.BIO_HUM_partition(clay)
            + plantin[2] * DPM_RPM_HUM_frac_inputs[2]
            + EOMin[2] * DPM_RPM_HUM_frac_inputs_EOM[2]
        )

        A_matrix = np.zeros((4, 4))
        np.fill_diagonal(A_matrix, -1)

        A_matrix[2, 0] = alpha
        A_matrix[2, 1] = alpha
        A_matrix[2, 2] = alpha - 1.0
        A_matrix[2, 3] = alpha

        A_matrix[3, 0] = beta
        A_matrix[3, 1] = beta
        A_matrix[3, 2] = beta
        A_matrix[3, 3] = beta - 1.0

        A_out = A_matrix

        return A_out

    def K_matrix(
        self,
        Kvalues,
        clay,
        temp,
        water,
        mean_pot_evapot,
        c_param,
        t_param,
        param_temp_in,
    ):
        """Calculate the K matrix."""
        # Decomposition rate
        kDPM = Kvalues[0]
        kRPM = Kvalues[1]
        kBIO = Kvalues[2]
        kHUM = Kvalues[3]

        K_matrix = np.zeros((4, 4))
        moist_rate_b, accTSMD_abs = self.control_moist_func(
            clay, water, mean_pot_evapot, c_param
        )
        temp_rate_a = self.control_temp_func(temp, param_temp_in)

        # print(f"temp_rate_a:{temp_rate_a} moist_rate_b:{moist_rate_b} c_param:{c_param} t_param:{t_param}")
        som1_till_factor = self.kiosk["CLTFAC"][0]
        som2_till_factor = self.kiosk["CLTFAC"][1]
        som3_till_factor = self.kiosk["CLTFAC"][2]
        som4_till_factor = self.kiosk["CLTFAC"][3]

        K_matrix[0, 0] = (
            kDPM * c_param * t_param * temp_rate_a * moist_rate_b * som4_till_factor
        )
        K_matrix[1, 1] = kRPM * c_param * t_param * temp_rate_a * moist_rate_b
        K_matrix[2, 2] = (
            kBIO * c_param * t_param * temp_rate_a * moist_rate_b * som2_till_factor
        )
        K_matrix[3, 3] = (
            kHUM * c_param * t_param * temp_rate_a * moist_rate_b * som3_till_factor
        )
        K_out = K_matrix
        return K_out

    def IOM_partition(self, SOC, power=1.139, constant=0.049):
        """Calculate the IOM partition."""
        # tC/ha
        IOM_frac = 0.049 * (SOC ** (power))
        return IOM_frac

    def BIO_HUM_partition(self, clay):
        """Calculate the BIO-HUM partition."""
        # Partitioning of C between CO2 and BIO+HUM
        X_ratio = 1.67 * (1.85 + 1.6 * np.exp(-0.0786 * clay))
        BIO_HUM_frac = 1.0 / (X_ratio + 1.0)
        return BIO_HUM_frac

    def control_temp_func(self, temp_mean, param_temp):
        """Calculate the temperature control function."""
        if temp_mean < -5:
            a = 0.0
        else:
            a = 47.91 / (1.0 + np.exp(106.06 / (temp_mean + param_temp)))

        a = np.minimum(a, 4.0)

        return a

    def control_moist_func(self, clay, rain, mean_pot_evapot, c_param):

        # Maximum Topsoil Moisture Deficit (TSMD)
        # For a soil depth 0-23cm
        # If soil depth is for ex  X(cm) -> maxTSMD=maxTSMD*X/23
        # For vegetated soils
        if c_param < 1.0:
            maxTSMD = -(20.0 + 1.3 * clay - 0.01 * (clay) ** 2)
        # For bare soil divide by 1.8
        else:
            maxTSMD = -(20.0 + 1.3 * clay - 0.01 * (clay) ** 2) / 1.8

        # Accumulated TSMD
        # Attenzione formula originale: invece di mean_pot_evapot c'e (open_pan_evapot*0.75). Siccome uso mean_pot_evapot (from Muller o calculated), open_pan_evapot = mean_pot_evapot/0.75

        # number_of_months = n_month

        pot_ev_month = mean_pot_evapot
        rain_month = rain
        if pot_ev_month < rain_month and self.accTSMD == 0.0:
            self.accTSMD = 0.0
        else:
            self.accTSMD = self.accTSMD + (rain_month - pot_ev_month)
            self.accTSMD = np.maximum(maxTSMD, self.accTSMD)
            self.accTSMD = np.minimum(self.accTSMD, 0.0)

        # Moisture rate modifier (b)
        if np.abs(self.accTSMD) < 0.444 * np.abs(maxTSMD):
            b = 1.0
        else:
            b = 0.2 + (1.0 - 0.2) * (np.abs(maxTSMD) - np.abs(self.accTSMD)) / (
                np.abs(maxTSMD) - 0.444 * np.abs(maxTSMD)
            )
            # b = np.minimum(1.,b)

        return b, np.abs(self.accTSMD)

    def _set_variable_DPM(self, nDPM):
        """Force the model states based on the input value."""
        increment = nDPM - self.states.DPM
        self.states.DPM = nDPM
        self._increments_DPM.append(increment)
        return {"DPM": increment}

    def _set_variable_RPM(self, nRPM):
        """Force the model states based on the input value."""
        increment = nRPM - self.states.RPM
        self.states.RPM = nRPM
        self._increments_RPM.append(increment)
        return {"RPM": increment}

    def _set_variable_BIO(self, nBIO):
        """Force the model states based on the input value."""
        increment = nBIO - self.states.BIO
        self.states.BIO = nBIO
        self._increments_BIO.append(increment)
        return {"BIO": increment}

    def _set_variable_HUM(self, nHUM):
        """Force the model states based on the input value."""
        increment = nHUM - self.states.HUM
        self.states.HUM = nHUM
        self._increments_HUM.append(increment)
        return {"HUM": increment}

    def _set_variable_IOM(self, nIOM):
        """Force the model states based on the input value."""
        increment = nIOM - self.states.IOM
        self.states.IOM = nIOM
        self._increments_IOM.append(increment)
        return {"IOM": increment}


class MillennialV2_SOC_Indigo(SimulationObject):
    """
    Based on Millennial Version 2 Copyright (c) 2021 rabramoff
    Millennial Version 2 is licensed under the terms of the MIT license:
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the “Software”), to deal
    in the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions. The above copyright notice and this permission
    notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    Changes made by Hamze Dokoohaki © Indigo Ag, Inc. 2024-2025:
    adapted for interoperability with the PCSE framework

    The Millennial V2 model simulates the dynamics of soil carbon pools and their interactions.
    It takes into account soil properties, hydrological properties, and decomposition processes.

    State Variables:
    - POM: Particulate Organic Matter (ug C g^-1)
    - LMWC: Low Molecular Weight Carbon (ug C g^-1)
    - AGG: Aggregate Carbon (ug C g^-1)
    - MIC: Microbial Biomass Carbon (ug C g^-1)
    - MAOM: Mineral-Associated Organic Matter (ug C g^-1)

    Parameters:
    - param_p1: Parameter for calculating the binding affinity for LMWC sorption (Equation 10)
    - param_p2: Parameter for calculating the binding affinity for LMWC sorption (Equation 10)
    - kaff_des: Desorption coefficient (day^-1) (Equation 10)
    - param_bulkd: Bulk density (g cm^-3) (Equation 11)
    - param_claysilt: Clay and silt content (Equation 11)
    - param_pc: Parameter for calculating the maximum sorption capacity (Equation 11)
    - porosity: Porosity (mm3/mm3) (Equation 4)
    - lambda_: Dependence of the rate on the matric potential (Equation 15)
    - matpot: Matric potential (MPa) (Equation 15)
    - kamin: Minimum relative rate in saturated soil (Equation 15)
    - alpha_pl: Parameter for calculating vmax_pl (umol ug^-1 day^-1) (Equation 3)
    - eact_pl: Activation energy for calculating vmax_pl (kJ mol^-1) (Equation 3)
    - kaff_pl: Half-saturation constant for POM decomposition (ug C g^-1) (Equation 2)
    - rate_pa: Rate constant for POM to AGG conversion (day^-1) (Equation 5)
    - rate_break: Rate constant for AGG breakup (day^-1) (Equation 6)
    - rate_leach: Rate constant for LMWC leaching (day^-1) (Equation 8)
    - alpha_lb: Parameter for calculating vmax_lb (umol ug^-1 day^-1) (Equation 14)
    - eact_lb: Activation energy for calculating vmax_lb (kJ mol^-1) (Equation 14)
    - kaff_lb: Half-saturation constant for LMWC decomposition (ug C g^-1) (Equation 13)
    - rate_bd: Rate constant for microbial turnover (ug C g^-1 day^-1) (Equation 16)
    - rate_ma: Rate constant for MAOM to AGG conversion (day^-1) (Equation 18)
    - cue_ref: Carbon use efficiency at reference temperature (Equation 21)
    - cue_t: Temperature sensitivity of carbon use efficiency (°C^-1) (Equation 21)
    - tae_ref: Reference temperature for carbon use efficiency (°C) (Equation 21)
    - param_pi: Fraction of plant inputs allocated to POM (Equation 1)
    - param_pa: Fraction of AGG breakup products allocated to POM (Equation 1)
    - param_pb: Fraction of microbial turnover products allocated to MAOM (Equation 19)

    Forcing Variables:
    - forc_st: Soil temperature (°C)
    - forc_sw: Volumetric soil moisture (mm3/mm3)
    - forc_npp: Daily plant inputs (ug C g^-1 day^-1)

    The model simulates the following processes:
    - Decomposition of POM to LMWC and AGG
    - Breakdown of AGG to MAOM and POM
    - Leaching of LMWC out of the system
    - Formation of MAOM from LMWC
    - Decomposition of MAOM to LMWC
    - Microbial growth and turnover
    - Formation of AGG from MAOM
    - Microbial respiration to the atmosphere

    The model equations are based on first-order kinetics and Michaelis-Menten kinetics,
    with various scalars and rate modifiers accounting for environmental factors such as
    soil moisture and temperature.
    """

    __slots__ = [
        "npp_dead_in_season",
        "porosity",
        "param_bulkd",
        "in_crop_cycle",
        "_increments_POM",
        "_increments_LMWC",
        "_increments_AGG",
        "_increments_MIC",
        "_increments_MAOM",
        "residue_perc_removed",
    ]

    npp_dead_in_season: float
    porosity: float
    param_bulkd: float
    in_crop_cycle: bool

    _increments_POM: list
    _increments_LMWC: list
    _increments_AGG: list
    _increments_MIC: list
    _increments_MAOM: list
    residue_perc_removed: float

    def __init__(self, day, kiosk, *args, **kwargs):
        self.npp_dead_in_season = 0.0
        self.porosity = -99.0
        self.param_bulkd = -99.0
        self.in_crop_cycle = False
        self._increments_POM = []
        self._increments_LMWC = []
        self._increments_AGG = []
        self._increments_MIC = []
        self._increments_MAOM = []
        self.residue_perc_removed = 0.0
        super().__init__(day, kiosk, *args, **kwargs)

    def _on_manage_residue(self, **kwargs):
        """Apply tillage"""
        residue_perc = kwargs.get("residue_percent")
        self.residue_perc_removed = residue_perc / 100

    def _on_CROP_START(self):
        self.in_crop_cycle = True

    def _on_CROP_FINISH(self):
        self.in_crop_cycle = False

    class Parameters(ParamTemplate):

        __slots__ = [
            "param_p1",
            "param_p2",
            "kaff_des",
            "param_pH",
            "param_claysilt",
            "param_pc",
            "lambda_",
            "matpot",
            "kamin",
            "alpha_pl",
            "eact_pl",
            "kaff_pl",
            "rate_pa",
            "rate_break",
            "rate_leach",
            "alpha_lb",
            "eact_lb",
            "kaff_lb",
            "rate_bd",
            "rate_ma",
            "cue_ref",
            "cue_t",
            "tae_ref",
            "param_pi",
            "param_pa",
            "param_pb",
            "POM0",
            "LMWC0",
            "AGG0",
            "MIC0",
            "MAOM0",
            "NLAYR",
            "BD",
            "DS",
            "ROOT_SHOOT_K",
        ]

        param_p1: float  # Equation 10
        param_p2: float  # Equation 10
        kaff_des: float  # day^-1, Equation 10
        param_pH: float  # day^-1, Equation 10
        param_claysilt: float  # Equation 11
        param_pc: float  # Equation 11
        # porosity: float  # mm3/mm3, Equation 4

        lambda_: float  # Equation 15
        matpot: float  # MPa, Equation 15
        kamin: float  # Equation 15
        alpha_pl: float  # umol ug^-1 day^-1, Equation 3
        eact_pl: float  # kJ mol^-1, Equation 3
        kaff_pl: float  # ug C g^-1, Equation 2
        rate_pa: float  # day^-1, Equation 5
        rate_break: float  # day^-1, Equation 6
        rate_leach: float  # day^-1, Equation 8
        alpha_lb: float  # umol ug^-1 day^-1, Equation 14
        eact_lb: float  # kJ mol^-1, Equation 14
        kaff_lb: float  # ug C g^-1, Equation 13
        rate_bd: float  # ug C g^-1 day^-1, Equation 16
        rate_ma: float  # day^-1, Equation 18
        cue_ref: float  # Equation 21
        cue_t: float  # °C^-1, Equation 21
        tae_ref: float  # °C, Equation 21
        param_pi: float  # Equation 1
        param_pa: float  # Equation 1
        param_pb: float  # Equation 19
        # initial values
        POM0: float  # ug C g^-1
        LMWC0: float  # ug C g^-1
        AGG0: float  # ug C g^-1
        MIC0: float  # ug C g^-1
        MAOM0: float  # ug C g^-1

        NLAYR: int  # Number of soil layers
        BD: list  # Bulk density (g/cm3)
        DS: list  # Depth of soil layers (cm)
        ROOT_SHOOT_K: float

    class StateVariables(StatesTemplate):

        __slots__ = [
            "POM",
            "LMWC",
            "AGG",
            "MIC",
            "MAOM",
        ]

        POM: float  # ug C g^-1
        LMWC: float  # ug C g^-1
        AGG: float  # ug C g^-1
        MIC: float  # ug C g^-1
        MAOM: float  # ug C g^-1

    class RateVariables(RatesTemplate):

        __slots__ = [
            "dPOM",
            "dLMWC",
            "dAGG",
            "dMIC",
            "dMAOM",
            "f_MB_atm",
        ]

        dPOM: float  # ug C g^-1 day^-1
        dLMWC: float  # ug C g^-1 day^-1
        dAGG: float  # ug C g^-1 day^-1
        dMIC: float  # ug C g^-1 day^-1
        dMAOM: float  # ug C g^-1 day^-1
        f_MB_atm: float  # ug C g^-1 day^-1

    def initialize(self, day, kiosk, parametervalues):
        self.params = self.Parameters(parametervalues)
        self.rates = self.RateVariables(kiosk)
        self.states = self.StateVariables(
            kiosk,
            POM=self.params.POM0,
            LMWC=self.params.LMWC0,
            AGG=self.params.AGG0,
            MIC=self.params.MIC0,
            MAOM=self.params.MAOM0,
            publish=["POM", "LMWC", "AGG", "MIC", "MAOM"],
        )
        self.residue_perc_removed = 0.0
        p = self.params
        BDs = [
            (p.BD[L] * (p.DS[L] - p.DS[L - 1])) for L in range(1, p.NLAYR)
        ]  # Total bulk density (g/cm3)
        BDs.append(p.BD[0] * p.DS[0])
        TBD = sum(BDs)  # Total bulk density (g/cm3)
        AVG_BULK_DENSITY = TBD / p.DS[p.NLAYR - 1]  # Average bulk density (g/cm3)
        self.porosity = (
            1 - AVG_BULK_DENSITY / 2.65
        )  # Average porosity (-) which is 2.65 g/cm3 for soil
        self.param_bulkd = AVG_BULK_DENSITY * 1000.0  # g cm^-3 -> kg soil m^-3
        # print(f"day {day} porosity {self.porosity} param_bulkd {self.param_bulkd} initial POM {self.params.POM0} LMWC {self.params.LMWC0} AGG {self.params.AGG0} MIC {self.params.MIC0} MAOM {self.params.MAOM0}")
        # search for crop transpiration values
        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_manage_residue, signals.manage_residue)

    @prepare_rates
    def calc_rates(self, day, drv):
        p = self.params
        s = self.states

        # Hydrological properties
        forc_sw = self.kiosk["SM"]  # Volumetric soil moisture (mm3/mm3)
        forc_st = self.kiosk["ST"][2]  # Soil temperature (°C)
        forc_sw = min(forc_sw, self.porosity)
        # Tillage effects on decomposition
        som1_till_factor = self.kiosk["CLTFAC"][0]
        som2_till_factor = self.kiosk["CLTFAC"][1]
        som3_till_factor = self.kiosk["CLTFAC"][2]
        som4_till_factor = self.kiosk["CLTFAC"][3]

        # Soil type properties
        # Affinity constant for LMWC sorption to soil minerals (kaff_lm). It's calculated based on soil pH and a desorption constant (p.param_p1, p.param_p2, p.kaff_des).
        kaff_lm = exp(-p.param_p1 * p.param_pH - p.param_p2) * p.kaff_des  # Equation 10

        # Maximum sorption capacity of the soil (param_qmax). It's calculated based on the soil's bulk density, clay/silt content, and carbon content (self.param_bulkd, p.param_claysilt, p.param_pc).
        param_qmax = self.param_bulkd * p.param_claysilt * p.param_pc  # Equation 11
        # print(f"day {day} param_qmax {param_qmax} param_bulkd {self.param_bulkd} param_claysilt {p.param_claysilt} param_pc {p.param_pc}")
        # Moisture scalar for decomposition (scalar_wd). It's calculated based on the soil water content (forc_sw) and the soil porosity (self.porosity). This scalar adjusts the decomposition rates based on the current soil moisture conditions.
        scalar_wd = (forc_sw / self.porosity) ** 0.5  # Equation 4

        # Moisture scalar for microbial biomass (scalar_wb). It's calculated based on the matric potential (p.matpot), a minimum constant (p.kamin), the soil water content (forc_sw), and the soil porosity (self.porosity). This scalar adjusts the microbial biomass based on the current soil moisture conditions.
        scalar_wb = (
            exp(p.lambda_ * -p.matpot)
            * (
                p.kamin
                + (1 - p.kamin) * ((self.porosity - forc_sw) / self.porosity) ** 0.5
            )
            * scalar_wd
        )  # Equation 15

        # Universal gas constant
        gas_const = 8.31446  # J K^-1 mol^-1

        # Maximum decomposition rate of POM (vmax_pl). It's calculated based on a constant (p.alpha_pl) and the activation energy (p.eact_pl), adjusted for the current soil temperature (forc_st).
        vmax_pl = p.alpha_pl * exp(
            -p.eact_pl / (gas_const * (forc_st + 273.15))
        )  # Equation 3

        # Flux from POM to LMWC (f_PO_LM). It's calculated based on the maximum decomposition rate of POM, the moisture scalar, the amount of POM (s.POM), and the amount of microbial biomass (s.MIC). The affinity constant for POM (p.kaff_pl) is used to adjust the flux based on the availability of POM for microbial decomposition.
        f_PO_LM = (
            vmax_pl * som1_till_factor * scalar_wd * s.POM * s.MIC / (p.kaff_pl + s.MIC)
            if s.POM > 0 and s.MIC > 0
            else 1e-4
        )  # Equation 2

        # print(f"day {day} vmax_pl {vmax_pl}  scalar_wd {scalar_wd}  s.LMWC {s.LMWC} s.POM {s.POM} s.MIC {s.MIC} vmax_pl {vmax_pl} f_PO_LM {f_PO_LM} (p.kaff_pl + s.MIC) {(p.kaff_pl + s.MIC)}")
        # Flux from POM to AGG (f_PO_AG). It's calculated based on a rate constant (p.rate_pa), the moisture scalar, and the amount of POM (s.POM).
        f_PO_AG = p.rate_pa * scalar_wd * s.POM if s.POM > 0 else 0  # Equation 5

        # Flux from AGG to MAOM and POM (f_AG_break). It's calculated based on a rate constant (p.rate_break), the moisture scalar, and the amount of AGG (s.AGG).
        f_AG_break = (
            som3_till_factor * p.rate_break * scalar_wd * s.AGG if s.AGG > 0 else 0
        )  # Equation 6

        # Flux from LMWC out of the system via leaching (f_LM_leach). It's calculated based on a rate constant (p.rate_leach), the moisture scalar, and the amount of LMWC (s.LMWC).
        f_LM_leach = (
            p.rate_leach * som4_till_factor * scalar_wd * s.LMWC if s.LMWC > 0 else 0
        )  # Equation 8

        # Flux from LMWC to MAOM (f_LM_MA). It's calculated based on the moisture scalar, the affinity constant for LMWC sorption to soil minerals, the amount of LMWC (s.LMWC), and the amount of MAOM (s.MAOM). The maximum sorption capacity of the soil (param_qmax) is used to adjust the flux based on the availability of sorption sites for LMWC.
        f_LM_MA = (
            scalar_wd * kaff_lm * s.LMWC * (1 - s.MAOM / param_qmax)
            if s.LMWC > 0 and s.MAOM > 0
            else 0
        )  # Equation 9

        # print(f"day {day} scalar_wd {scalar_wd} scalar_wb {scalar_wb} kaflm {kaff_lm}  s.LMWC {s.LMWC} s.MAOM {s.MAOM} param_qmax {param_qmax} vmax_pl {vmax_pl} f_PO_LM {f_PO_LM} f_PO_AG {f_PO_AG} f_AG_break {f_AG_break} f_LM_leach {f_LM_leach} f_LM_MA {f_LM_MA}")
        # Flux from MAOM to LMWC (f_MA_LM). It's calculated based on a desorption constant (p.kaff_des), the amount of MAOM (s.MAOM), and the maximum sorption capacity of the soil (param_qmax).
        f_MA_LM = (
            p.kaff_des * som2_till_factor * s.MAOM / param_qmax if s.MAOM > 0 else 0
        )  # Equation 12

        # Maximum decomposition rate of LMWC (vmax_lb). It's calculated based on a constant (p.alpha_lb) and the activation energy (p.eact_lb), adjusted for the current soil temperature (forc_st).
        vmax_lb = (
            som4_till_factor
            * p.alpha_lb
            * exp(-p.eact_lb / (gas_const * (forc_st + 273.15)))
        )  # Equation 14

        # Flux from LMWC to MIC (f_LM_MB). It's calculated based on the maximum decomposition rate of LMWC, the moisture scalar for microbial biomass, the amount of microbial biomass (s.MIC), and the amount of LMWC (s.LMWC). The affinity constant for LMWC (p.kaff_lb) is used to adjust the flux based on the availability of LMWC for microbial decomposition.
        f_LM_MB = (
            vmax_lb * scalar_wb * s.MIC * s.LMWC / (p.kaff_lb + s.LMWC)
            if s.LMWC > 0 and s.MIC > 0
            else 0
        )  # Equation 13

        # Flux from MIC to MAOM and LMWC (f_MB_turn). It's calculated based on a rate constant (p.rate_bd) and the amount of microbial biomass (s.MIC).
        f_MB_turn = p.rate_bd * s.MIC**2.0 if s.MIC > 0 else 0  # Equation 16

        # Flux from MAOM to AGG (f_MA_AG). It's calculated based on a rate constant (p.rate_ma), the moisture scalar, and the amount of MAOM (s.MAOM).
        f_MA_AG = p.rate_ma * scalar_wd * s.MAOM if s.MAOM > 0 else 0  # Equation 18

        # Flux from MIC to the atmosphere (f_MB_atm). It's calculated based on the flux from LMWC to MIC and the carbon use efficiency (p.cue_ref, p.cue_t, forc_st, p.tae_ref).
        f_MB_atm = (
            som2_till_factor
            * f_LM_MB
            * (1 - (p.cue_ref - p.cue_t * (forc_st - p.tae_ref)))
            if s.MIC > 0 and s.LMWC > 0
            else 0
        )  # Equation 21

        # if s.MIC != 0:
        #    print(f"{day}: f_MB_atm: {f_MB_atm} f_LM_MB:{f_LM_MB} vmax:{vmax_lb} lmwc{s.LMWC} s.MIC{s.MIC} vmax_lb {vmax_lb}")
        # Update rate variables
        # dead fall
        if "DMI" not in self.kiosk:
            forc_npp = 0
        else:
            # DMI comes from WOFOST8 crop model
            # DMI Total dry matter increase, calculated as ASRC times a weighted conversion efficieny. Y  |kg ha-1 d-1|
            # 4% dead root and shoot material, 60% of which is C
            forc_npp = (
                self.kiosk["GASST"]
                - self.kiosk["MREST"]
                - (self.npp_dead_in_season * 10)
            )  # Daily plant inputs/daily net primary production needs to be in  (gC/m2/d)
            forc_npp = (
                forc_npp * self.params.ROOT_SHOOT_K * 1000 / 10000
            )  # kg C ha^-1 day^-1 -> g C m^-2 day^-1

        self.npp_dead_in_season += forc_npp  # keep track of the dead in season

        if "DVS" in self.kiosk:
            # print(f"day {day} {self.kiosk['DVS']} - in_cycle: {self.in_crop_cycle} - TAGP : {self.kiosk['TAGP']}")
            if "TAGP" in self.kiosk:
                if not self.in_crop_cycle:
                    # (total gross assimulation in C - Total respiration in C) - (50% of above ground * 60% to convert to C)
                    # forc_npp += ((self.kiosk['GASST']- self.kiosk['MREST']) - (0.5 * self.kiosk['TAGP'] * 0.6) - (self.npp_dead_in_season*10))  /10 # kg C ha^-1 day^-1 -> t C ha^-1 day^-1  -> residue -> C
                    total_net_assimilated = (
                        self.kiosk["GASST"] - self.kiosk["MREST"]
                    ) - (self.npp_dead_in_season * 10)
                    yield_C_removed = (
                        0.5 * self.kiosk["TAGP"] * 0.6
                    )  # 50% HI of above ground * 60% to convert to C
                    residue_left = (total_net_assimilated - yield_C_removed) * (
                        1 - self.residue_perc_removed
                    )
                    forc_npp += (
                        residue_left
                    ) / 10  # kg C ha^-1 day^-1 -> t C ha^-1 day^-1  -> residue -> C
                    self.npp_dead_in_season = 0

        self.rates.dPOM = (
            forc_npp * p.param_pi + f_AG_break * p.param_pa - f_PO_AG - f_PO_LM
        )  # Equation 1
        # print(f"forc_npp {forc_npp} f_LM_leach {f_LM_leach} f_PO_LM {f_PO_LM} f_LM_MA {f_LM_MA} f_LM_MB {f_LM_MB} f_MB_turn {f_MB_turn} f_MA_LM {f_MA_LM} p.param_pi")

        # Change in LMWC (dLMWC): It's calculated based on the net primary productivity (forc_npp) not going to POM (p.param_pi), the flux from LMWC out of the system via leaching (f_LM_leach), the flux from POM to LMWC (f_PO_LM), the flux from LMWC to MAOM (f_LM_MA), the flux from LMWC to MIC (f_LM_MB), the flux from MIC to MAOM and LMWC not going to MAOM (f_MB_turn * (1. - p.param_pb)), and the flux from MAOM to LMWC (f_MA_LM).
        self.rates.dLMWC = (
            forc_npp * (1.0 - p.param_pi)
            - f_LM_leach
            + f_PO_LM
            - f_LM_MA
            - f_LM_MB
            + f_MB_turn * (1.0 - p.param_pb)
            + f_MA_LM
        )  # Equation 7

        # Change in AGG (dAGG): It's calculated based on the flux from MAOM to AGG (f_MA_AG), the flux from POM to AGG (f_PO_AG), and the flux from AGG to MAOM and POM (f_AG_break).
        self.rates.dAGG = f_MA_AG + f_PO_AG - f_AG_break  # Equation 17

        # Change in MIC (dMIC): It's calculated based on the flux from LMWC to MIC (f_LM_MB), the flux from MIC to MAOM and LMWC (f_MB_turn), and the flux from MIC to the atmosphere (f_MB_atm).
        self.rates.dMIC = f_LM_MB - f_MB_turn - f_MB_atm  # Equation 20

        # Change in MAOM (dMAOM): It's calculated based on the flux from LMWC to MAOM (f_LM_MA), the flux from MAOM to LMWC (f_MA_LM), the flux from MIC to MAOM (f_MB_turn * p.param_pb), the flux from MAOM to AGG (f_MA_AG), and the flux from AGG to MAOM (f_AG_break * (1. - p.param_pa)).
        self.rates.dMAOM = (
            f_LM_MA
            - f_MA_LM
            + f_MB_turn * p.param_pb
            - f_MA_AG
            + f_AG_break * (1.0 - p.param_pa)
        )  # Equation 19
        # print(f"day {day} f_LM_MA {f_LM_MA} f_MA_LM {f_MA_LM} f_MB_turn {f_MB_turn} f_MA_AG {f_MA_AG} f_AG_break {f_AG_break} p.param_pb {p.param_pb} p.param_pa {p.param_pa}")
        # print("-----------")
        # Flux from MIC to the atmosphere (f_MB_atm).
        self.rates.f_MB_atm = f_MB_atm  # Equation 22

    @prepare_states
    def integrate(self, day, delt):
        self.states.POM += self.rates.dPOM * delt
        self.states.LMWC += self.rates.dLMWC * delt
        self.states.AGG += self.rates.dAGG * delt
        self.states.MIC += self.rates.dMIC * delt
        self.states.MAOM += self.rates.dMAOM * delt

        for attr in ["POM", "LMWC", "AGG", "MIC", "MAOM"]:
            if getattr(self.states, attr) < 0:
                setattr(self.states, attr, 1e-3)

    def finalize(self, day):
        pass

    def _set_variable_POM(self, nPOM):
        """Force the model states based on the input value."""
        increment = nPOM - self.states.POM
        self.states.POM = nPOM
        self._increments_POM.append(increment)
        return {"POM": increment}

    def _set_variable_LMWC(self, nLMWC):
        """Force the model states based on the input value."""
        increment = nLMWC - self.states.LMWC
        self.states.LMWC = nLMWC
        self._increments_LMWC.append(increment)
        return {"LMWC": increment}

    def _set_variable_AGG(self, nAGG):
        """Force the model states based on the input value."""
        increment = nAGG - self.states.AGG
        self.states.AGG = nAGG
        self._increments_AGG.append(increment)
        return {"AGG": increment}

    def _set_variable_MIC(self, nMIC):
        """Force the model states based on the input value."""
        increment = nMIC - self.states.MIC
        self.states.MIC = nMIC
        self._increments_MIC.append(increment)
        return {"MIC": increment}

    def _set_variable_MAOM(self, nMAOM):
        """Force the model states based on the input value."""
        increment = nMAOM - self.states.MAOM
        self.states.MAOM = nMAOM
        self._increments_MAOM.append(increment)
        return {"MAOM": increment}


class Century_SOC_Indigo(SimulationObject):
    """
    Based on derivs_V2_Century.R Copyright (c) 2021 rabramoff

    derivs_V2_Century.R is licensed under the terms of the MIT license:
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the “Software”), to deal in the
    Software without restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions. The above copyright notice and this permission notice shall
    be included in all copies or substantial portions of the Software. THE SOFTWARE IS
    PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
    AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.

    Changes made by Hamze Dokoohaki © Indigo Ag, Inc. 2024-2025:
    adapted for interoperability with the PCSE framework

    The Century model simulates the dynamics of soil carbon pools and their interactions,
    considering the influence of soil temperature, moisture, and plant inputs.

    State Variables:
    - StrLitter: Structural Litter (g C m^-2)
    - MetLitter: Metabolic Litter (g C m^-2)
    - ACTIVE: Active Carbon Pool (g C m^-2)
    - SLOW: Slow Carbon Pool (g C m^-2)
    - PASSIVE: Passive Carbon Pool (g C m^-2)

    Parameters:
    - t1: x-axis location of inflection point (°C) (Equation B1)
    - t2: y-axis location of inflection point (-) (Equation B1)
    - t3: Distance from the maximum point to the minimum point (step size) (-) (Equation B1)
    - t4: Slope of line at inflection point (-) (Equation B1)
    - w1: Water scalar parameter (-) (Equation B2)
    - w2: Water scalar parameter (-) (Equation B2)
    - c1: Intercept of clay fraction relationship (-) (Equation B3)
    - c2: Slope of clay fraction relationship (-) (Equation B3)
    - k_strlitter: Turnover rate of structural litter pool (day^-1)
    - k_metlitter: Turnover rate of metabolic litter pool (day^-1)
    - k_active: Turnover rate of active pool (day^-1)
    - k_slow: Turnover rate of slow pool (day^-1)
    - k_passive: Turnover rate of passive pool (day^-1)
    - LigFrac: Fraction of litter that is lignin (-)
    - input_to_strlitter: Proportion of plant residue to structural litter pool (-)
    - strlitter_to_active: Fraction of structural litter to active pool (-)
    - metlitter_to_active: Fraction of metabolic litter to active pool (-)
    - slow_to_active: Fraction of slow pool to active pool (-)
    - passive_to_active: Fraction of passive pool to active pool (-)
    - strlitter_to_slow: Fraction of structural litter to slow pool (-)
    - active_to_passive: Fraction of active pool to passive pool (-)
    - slow_to_passive: Fraction of slow pool to passive pool (-)

    Forcing Variables:
    - forc_st: Soil temperature (°C)
    - forc_sw: Volumetric soil moisture (mm3/mm3)
    - forc_npp: Daily plant inputs (g C m^-2 day^-1)

    The model simulates the following processes:
    - Decomposition of Structural Litter and Metabolic Litter
    - Transfer of carbon from litter pools to the ACTIVE, SLOW, and PASSIVE pools
    - Decomposition of the ACTIVE, SLOW, and PASSIVE pools
    - Transfer of carbon between the ACTIVE, SLOW, and PASSIVE pools

    The model equations use temperature and moisture scalars to modify the decomposition rates
    based on environmental conditions.
    """

    __slots__ = [
        "AVGDUL",
        "in_crop_cycle",
        "npp_dead_in_season",
        "residue_perc_removed",
        "_increments_active",
        "_increments_slow",
        "_increments_passive",
    ]

    AVGDUL: float
    in_crop_cycle: bool
    npp_dead_in_season: float
    residue_perc_removed: float
    _increments_active: list
    _increments_slow: list
    _increments_passive: list

    def __init__(self, day, kiosk, *args, **kwargs):
        self.AVGDUL = -99.0
        self.in_crop_cycle = False
        self.npp_dead_in_season = 0.0
        self.residue_perc_removed = 0.0
        self._increments_active = []
        self._increments_slow = []
        self._increments_passive = []
        super().__init__(day, kiosk, *args, **kwargs)

    def _on_manage_residue(self, **kwargs):
        """Apply tillage"""
        residue_perc = kwargs.get("residue_percent")
        self.residue_perc_removed = residue_perc / 100

    def _on_APPLY_TILLAGE(self, **kwargs):
        """Apply tillage"""
        mixing = kwargs.get("mixing", 0.0)
        depth = kwargs.get("depth", 0.0)
        type = kwargs.get("type", 0.0)
        day = kwargs.get("day", 0.0)
        cultra, cult_factor_key = get_tillage_code_params(
            mixing, depth, type, par="cultra"
        )

    def _on_CROP_START(self):
        self.in_crop_cycle = True

    def _on_CROP_FINISH(self):
        self.in_crop_cycle = False

    def take_avg(self, values, depths):
        """Calculate average value of soil properties"""
        value_list = [
            values[L] * (depths[L] - depths[L - 1]) for L in range(1, self.params.NLAYR)
        ]
        value_list.append(values[0] * depths[0])
        return sum(value_list) / depths[self.params.NLAYR - 1]

    class Parameters(ParamTemplate):

        __slots__ = [
            "t1",
            "t2",
            "t3",
            "t4",
            "w1",
            "w2",
            "c1",
            "c2",
            "k_strlitter",
            "k_metlitter",
            "k_active",
            "k_slow",
            "k_passive",
            "LigFrac",
            "input_to_strlitter",
            "strlitter_to_active",
            "metlitter_to_active",
            "slow_to_active",
            "passive_to_active",
            "strlitter_to_slow",
            "active_to_passive",
            "slow_to_passive",
            "ROOT_SHOOT_K",
            "param_clay",
            "ACTIVE0",
            "SLOW0",
            "PASSIVE0",
            "DS",
            "NLAYR",
            "DUL",
        ]

        t1: float  # °C, Equation B1
        t2: float  # -, Equation B1
        t3: float  # -, Equation B1
        t4: float  # -, Equation B1
        w1: float  # -, Equation B2
        w2: float  # -, Equation B2
        c1: float  # -, Equation B3
        c2: float  # -, Equation B3
        k_strlitter: float  # day^-1
        k_metlitter: float  # day^-1
        k_active: float  # day^-1
        k_slow: float  # day^-1
        k_passive: float  # day^-1
        LigFrac: float  # -
        input_to_strlitter: float  # -
        strlitter_to_active: float  # -
        metlitter_to_active: float  # -
        slow_to_active: float  # -
        passive_to_active: float  # -
        strlitter_to_slow: float  # -
        active_to_passive: float  # -
        slow_to_passive: float  # -
        ROOT_SHOOT_K: float
        param_clay: float  # Equation 11 percent clay and silt

        ACTIVE0: float  # g C m^-2
        SLOW0: float  # g C m^-2
        PASSIVE0: float  # g C m^-2

        DS: list  # Depth of soil layers (cm)
        NLAYR: int  # Number of soil layers
        DUL: list  # Lower limit of soil water content (cm3/cm3)

    class StateVariables(StatesTemplate):

        __slots__ = [
            "StrLitter",
            "MetLitter",
            "SOC_ACTIVE",
            "SOC_SLOW",
            "SOC_PASSIVE",
        ]

        StrLitter: float  # g C m^-2
        MetLitter: float  # g C m^-2
        SOC_ACTIVE: float  # g C m^-2
        SOC_SLOW: float  # g C m^-2
        SOC_PASSIVE: float  # g C m^-2

    class RateVariables(RatesTemplate):

        __slots__ = [
            "dStrLitter",
            "dMetLitter",
            "dACTIVE",
            "dSLOW",
            "dPASSIVE",
        ]

        dStrLitter: float  # g C m^-2 day^-1
        dMetLitter: float  # g C m^-2 day^-1
        dACTIVE: float  # g C m^-2 day^-1
        dSLOW: float  # g C m^-2 day^-1
        dPASSIVE: float  # g C m^-2 day^-1

    def initialize(self, day, kiosk, parametervalues):
        # unless this is modified by the signal we remove nothing
        self.residue_perc_removed = 0
        self.params = self.Parameters(parametervalues)
        self.rates = self.RateVariables(
            kiosk, publish=["dACTIVE", "dSLOW", "dPASSIVE", "dStrLitter", "dMetLitter"]
        )
        self.states = self.StateVariables(
            kiosk,
            StrLitter=0.0,
            MetLitter=0.0,
            SOC_ACTIVE=self.params.ACTIVE0,
            SOC_SLOW=self.params.SLOW0,
            SOC_PASSIVE=self.params.PASSIVE0,
            publish=["StrLitter", "MetLitter", "SOC_ACTIVE", "SOC_SLOW", "SOC_PASSIVE"],
        )
        self.AVGDUL = self.take_avg(self.params.DUL, self.params.DS)
        # Connect to CROP_START/CROP_FINISH signals for water balance to
        # search for crop transpiration values
        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_APPLY_TILLAGE, signals.apply_tillage)
        self._connect_signal(self._on_manage_residue, signals.manage_residue)

    @prepare_rates
    def calc_rates(self, day, drv):
        p = self.params
        s = self.states

        # testing tillage

        forc_sw = self.kiosk["SM"]  # Volumetric soil moisture (mm3/mm3)
        forc_st = self.kiosk["ST"][2]  # Soil temperature (°C)
        # Tillage effects on decomposition
        som1_till_factor = self.kiosk["CLTFAC"][0]
        som2_till_factor = self.kiosk["CLTFAC"][1]
        som3_till_factor = self.kiosk["CLTFAC"][2]
        som4_till_factor = self.kiosk["CLTFAC"][3]

        # print(f"day {day} - forc_sw {forc_sw} - forc_st {forc_st} - som1_till_factor {som1_till_factor} - som2_till_factor {som2_till_factor} - som3_till_factor {som3_till_factor} - som4_till_factor {som4_till_factor}")
        # Temperature scalar (Equation B1): This equation calculates a scaling factor for temperature,
        # which adjusts the decomposition rates based on the current soil temperature (forc_st).
        # The parameters p.t1, p.t2, p.t3, and p.t4 are used to shape the response curve.
        t_scalar = (
            p.t2 + (p.t3 / math.pi) * math.atan(math.pi * p.t4 * (forc_st - p.t1))
        ) / (p.t2 + (p.t3 / math.pi) * math.atan(math.pi * p.t4 * (30.0 - p.t1)))

        # Moisture scalar (Equation B2): This equation calculates a scaling factor for soil moisture,
        # which adjusts the decomposition rates based on the current soil water content (forc_sw).
        # The parameters p.w1 and p.w2 are used to shape the response curve, and self.
        # AVGDUL represents the average soil moisture at field capacity.
        w_scalar = 1.0 / (1.0 + p.w1 * math.exp(-p.w2 * forc_sw / self.AVGDUL))

        # Soil texture effect (Equation B3): This equation calculates a factor that represents the effect of soil texture on decomposition.
        # It is based on the proportion of clay and silt in the soil (p.param_claysilt), with parameters p.c1 and p.c2 used to shape the response.
        f_TEX = p.c1 - p.c2 * p.param_clay * 0.01  # Equation B3

        # print(f"t_scalar: {t_scalar} - w_scalar: {w_scalar} - f_TEX: {f_TEX}")
        # Decomposition fluxes (Equations B4-B8)
        # Structural Litter Flux: Amount of structural litter (organic material resistant to decomposition)
        # times the decomposition rate of structural litter, scaled by temperature and moisture,
        # and adjusted for the proportion of lignin (which decomposes slowly)
        f_StrLitter = (
            s.StrLitter
            * p.k_strlitter
            * t_scalar
            * w_scalar
            * math.exp(-3 * p.LigFrac)
            * som4_till_factor
        )  # Equation B4

        # Metabolic Litter Flux: Amount of metabolic litter (organic material that decomposes quickly)
        # times the decomposition rate of metabolic litter, scaled by temperature and moisture
        f_MetLitter = s.MetLitter * p.k_metlitter * t_scalar * w_scalar  # Equation B5

        # Active SOC Flux: Amount of active SOC (SOC readily available for microbial decomposition)
        # times the decomposition rate of active SOC, scaled by temperature, moisture, and soil texture
        f_ACTIVE = (
            s.SOC_ACTIVE * p.k_active * t_scalar * w_scalar * f_TEX * som1_till_factor
        )  # Equation B6

        # Slow SOC Flux: Amount of slow SOC (SOC that decomposes slowly over time)
        # times the decomposition rate of slow SOC, scaled by temperature and moisture
        f_SLOW = (
            s.SOC_SLOW * p.k_slow * t_scalar * w_scalar * som2_till_factor
        )  # Equation B7

        # Passive SOC Flux: Amount of passive SOC (SOC that decomposes very slowly over long periods)
        # times the decomposition rate of passive SOC, scaled by temperature and moisture
        f_PASSIVE = (
            s.SOC_PASSIVE * p.k_passive * t_scalar * w_scalar * som3_till_factor
        )  # Equation B8

        # dead fall
        if "DMI" not in self.kiosk:
            forc_npp = 0
        else:
            # DMI comes from WOFOST8 crop model
            # DMI Total dry matter increase, calculated as ASRC times a weighted conversion efficieny. Y  |kg ha-1 d-1|
            # 4% dead root and shoot material, 60% of which is C
            forc_npp = (
                self.kiosk["GASST"]
                - self.kiosk["MREST"]
                - (self.npp_dead_in_season * 10)
            )  # Daily plant inputs/daily net primary production needs to be in  (gC/m2/d)
            forc_npp = (
                forc_npp * self.params.ROOT_SHOOT_K * 1000 / 10000
            )  # kg C ha^-1 day^-1 -> g C m^-2 day^-1

        self.npp_dead_in_season += forc_npp  # keep track of the dead in season

        if "DVS" in self.kiosk:
            # This is basically event KILL in DAYCENT without needing to have a KILL everytime.
            if "TAGP" in self.kiosk:
                if not self.in_crop_cycle:
                    # (total gross assimulation in C - Total respiration in C) - (50% of above ground * 60% to convert to C) - (dead in season already accounted for)
                    total_net_assimilated = (
                        self.kiosk["GASST"] - self.kiosk["MREST"]
                    ) - (self.npp_dead_in_season * 10)
                    yield_C_removed = (
                        0.5 * self.kiosk["TAGP"] * 0.6
                    )  # 50% HI of above ground * 60% to convert to C
                    residue_left = (total_net_assimilated - yield_C_removed) * (
                        1 - self.residue_perc_removed
                    )
                    forc_npp += (
                        residue_left
                    ) / 10  # kg C ha^-1 day^-1 -> t C ha^-1 day^-1  -> residue -> C
                    self.npp_dead_in_season = 0

        # Update rate variables (Equations B9-B13)
        # Change in Structural Litter: Input to structural litter (proportion of net primary productivity)
        # minus the flux of structural litter
        self.rates.dStrLitter = (
            p.input_to_strlitter * forc_npp - f_StrLitter
        )  # Equation B9

        # Change in Metabolic Litter: Input to metabolic litter (remaining proportion of net primary productivity)
        # minus the flux of metabolic litter
        self.rates.dMetLitter = (
            1 - p.input_to_strlitter
        ) * forc_npp - f_MetLitter  # Equation B10

        # Change in Active SOC: Fluxes from structural litter, metabolic litter, slow SOC, and passive SOC
        # to active SOC, minus the flux from active SOC
        self.rates.dACTIVE = (
            ((1 - p.LigFrac) * p.strlitter_to_active * f_StrLitter)
            + (p.metlitter_to_active * f_MetLitter)
            + (f_SLOW * p.slow_to_active)
            + (f_PASSIVE * p.passive_to_active)
            - f_ACTIVE
        )  # Equation B11

        # Change in Slow SOC: Flux from structural litter to slow SOC and from active SOC to slow SOC
        # (adjusted for soil texture and the proportion going to passive SOC), minus the flux from slow SOC
        self.rates.dSLOW = (
            p.LigFrac * p.strlitter_to_slow * f_StrLitter
            + f_ACTIVE * (1 - f_TEX - p.active_to_passive)
            - f_SLOW
        )  # Equation B12

        # Change in Passive SOC: Flux from active SOC and slow SOC to passive SOC, minus the flux from passive SOC
        self.rates.dPASSIVE = (
            f_ACTIVE * p.active_to_passive + f_SLOW * p.slow_to_passive - f_PASSIVE
        )  # Equation B13

    @prepare_states
    def integrate(self, day, delt):
        self.states.StrLitter += self.rates.dStrLitter * delt
        self.states.MetLitter += self.rates.dMetLitter * delt

        self.states.SOC_ACTIVE += self.rates.dACTIVE * delt
        self.states.SOC_SLOW += self.rates.dSLOW * delt
        self.states.SOC_PASSIVE += self.rates.dPASSIVE * delt

        for attr in ["StrLitter", "MetLitter", "SOC_ACTIVE", "SOC_SLOW", "SOC_PASSIVE"]:
            if getattr(self.states, attr) < 0:
                setattr(self.states, attr, 0)

    def finalize(self, day):
        pass

    def _set_variable_SOC_ACTIVE(self, nactive):
        """Force the model states based on the given soil SOC_ACTIVE value.

        Further, the increment made to SOC_ACTIVE is added to self._increments_SOC_ACTIVE
        """
        increment = nactive - self.states.SOC_ACTIVE
        self.states.SOC_ACTIVE = nactive
        self._increments_active.append(increment)
        return {"SOC_ACTIVE": increment}

    def _set_variable_SOC_SLOW(self, nslow):
        """Force the model states based on the given soil SOC_SLOW value.

        Further, the increment made to SOC_SLOW is added to self._increments_SOC_SLOW
        """
        increment = nslow - self.states.SOC_SLOW
        self.states.SOC_SLOW = nslow
        self._increments_slow.append(increment)
        return {"SOC_SLOW": increment}

    def _set_variable_SOC_PASSIVE(self, npassive):
        """Force the model states based on the given soil SOC_PASSIVE value.

        Further, the increment made to SOC_PASSIVE is added to self._increments_SOC_PASSIVE
        """
        increment = npassive - self.states.SOC_PASSIVE
        self.states.SOC_PASSIVE = npassive
        self._increments_passive.append(increment)
        return {"SOC_PASSIVE": increment}
