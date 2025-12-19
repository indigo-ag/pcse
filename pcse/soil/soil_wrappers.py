# -*- coding: utf-8 -*-
# Copyright (c) 2004-2020 Alterra, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), September 2020
"""This module wraps the soil components for water and nutrients so that they run jointly
within the same model.
"""
from pcse.base import SimulationObject
from .classic_waterbalance import WaterbalanceFD, WaterbalancePP, IndigoWaterbalanceFD
from .npk_soil_dynamics import (
    NPK_Soil_Dynamics,
    NPK_PotentialProduction,
    Indigo_NPK_Soil_Dynamics,
)
from .n_soil_dynamics import N_Soil_Dynamics

from .Soil_Temp import SoilTemperature
from .Tillage import TillageSignal
from .SOC import Ensemble_SOC_Indigo
from .SOCMineralization import SOCMineralization_Indigo


class SoilModuleWrapper_PP(SimulationObject):
    """This wraps the soil water balance and soil NPK balance for potential production."""

    __slots__ = ["WaterbalancePP", "NPK_PotentialProduction"]

    WaterbalancePP: SimulationObject
    NPK_PotentialProduction: SimulationObject

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        self.WaterbalancePP = WaterbalancePP(day, kiosk, parvalues)
        self.NPK_PotentialProduction = NPK_PotentialProduction(day, kiosk, parvalues)

    def calc_rates(self, day, drv):
        self.WaterbalancePP.calc_rates(day, drv)
        self.NPK_PotentialProduction.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):
        self.WaterbalancePP.integrate(day, delt)
        self.NPK_PotentialProduction.integrate(day, delt)


class SoilModuleWrapper_WLP_FD(SimulationObject):
    """This wraps the soil water balance for free drainage conditions and NPK balance
    for production conditions limited by soil water only.
    """

    __slots__ = ["WaterbalanceFD", "NPK_Soil_Dynamics"]

    WaterbalanceFD: SimulationObject
    NPK_Soil_Dynamics: SimulationObject

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        self.WaterbalanceFD = WaterbalanceFD(day, kiosk, parvalues)
        self.NPK_Soil_Dynamics = NPK_PotentialProduction(day, kiosk, parvalues)

    def calc_rates(self, day, drv):
        self.WaterbalanceFD.calc_rates(day, drv)
        self.NPK_Soil_Dynamics.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):
        self.WaterbalanceFD.integrate(day, delt)
        self.NPK_Soil_Dynamics.integrate(day, delt)


class SoilModuleWrapper_NPK_WLP_FD(SimulationObject):
    """This wraps the soil water balance for free drainage conditions and NPK balance
    for production conditions limited by both soil water and NPK.
    """

    __slots__ = ["WaterbalanceFD", "NPK_Soil_Dynamics"]

    WaterbalanceFD: SimulationObject
    NPK_Soil_Dynamics: SimulationObject

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        self.WaterbalanceFD = WaterbalanceFD(day, kiosk, parvalues)
        self.NPK_Soil_Dynamics = NPK_Soil_Dynamics(day, kiosk, parvalues)

    def calc_rates(self, day, drv):
        self.WaterbalanceFD.calc_rates(day, drv)
        self.NPK_Soil_Dynamics.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):
        self.WaterbalanceFD.integrate(day, delt)
        self.NPK_Soil_Dynamics.integrate(day, delt)


class SoilModuleWrapper_N_WLP_FD(SimulationObject):
    """This wraps the soil water balance for free drainage conditions and N balance
    for production conditions limited by both soil water and N.
    """

    __slots__ = ["WaterbalanceFD", "N_Soil_Dynamics"]

    WaterbalanceFD: SimulationObject
    N_Soil_Dynamics: SimulationObject

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        self.WaterbalanceFD = WaterbalanceFD(day, kiosk, parvalues)
        self.N_Soil_Dynamics = N_Soil_Dynamics(day, kiosk, parvalues)

    def calc_rates(self, day, drv):
        self.WaterbalanceFD.calc_rates(day, drv)
        self.N_Soil_Dynamics.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):
        self.WaterbalanceFD.integrate(day, delt)
        self.N_Soil_Dynamics.integrate(day, delt)


class SoilModuleWrapper_Indigo(SimulationObject):
    """This wraps the soil water balance for free drainage conditions and NPK balance
    for production conditions limited by both soil water and NPK. Built by Indigo Ag.
    We are also working on adding the soil carbon dynamics.
    """

    __slots__ = [
        "IndigoWaterbalanceFD",
        "NPK_Soil_Dynamics",
        "SoilTemperature",
        "TillageSignal",
        "Ensemble_SOC_Indigo",
        "SOCMineralization_Indigo",
    ]

    IndigoWaterbalanceFD: SimulationObject
    NPK_Soil_Dynamics: SimulationObject
    SoilTemperature: SimulationObject
    TillageSignal: SimulationObject
    Ensemble_SOC_Indigo: SimulationObject
    SOCMineralization_Indigo: SimulationObject

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        self.IndigoWaterbalanceFD = IndigoWaterbalanceFD(day, kiosk, parvalues)
        self.NPK_Soil_Dynamics = Indigo_NPK_Soil_Dynamics(day, kiosk, parvalues)
        self.SoilTemperature = SoilTemperature(day, kiosk, parvalues)
        self.TillageSignal = TillageSignal(day, kiosk, parvalues)
        self.Ensemble_SOC_Indigo = Ensemble_SOC_Indigo(day, kiosk, parvalues)
        self.SOCMineralization_Indigo = SOCMineralization_Indigo(day, kiosk, parvalues)

    def calc_rates(self, day, drv):
        self.IndigoWaterbalanceFD.calc_rates(day, drv)
        self.SoilTemperature.calc_states(day, drv)
        self.TillageSignal.calc_rates(day, drv)  # this doesn't have the cal_rate
        self.Ensemble_SOC_Indigo.calc_rates(day, drv)
        self.SOCMineralization_Indigo.calc_rates(day, drv)
        self.NPK_Soil_Dynamics.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):
        self.IndigoWaterbalanceFD.integrate(day, delt)
        self.NPK_Soil_Dynamics.integrate(day, delt)
        # SoilTemperature directly estimates the states without the need for integration, we put
        # in the rates to let it use the drivers
        self.TillageSignal.integrate(day, delt)
        self.Ensemble_SOC_Indigo.integrate(day, delt)
