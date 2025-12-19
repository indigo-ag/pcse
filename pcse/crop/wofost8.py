#!/usr/bin/env python

import datetime


from ..decorators import prepare_rates, prepare_states
from ..base import ParamTemplate, StatesTemplate, RatesTemplate, SimulationObject
from .. import signals
from .. import exceptions as exc
from .phenology import DVS_Phenology as Phenology
from .respiration import WOFOST_Maintenance_Respiration as MaintenanceRespiration
from .stem_dynamics import WOFOST_Stem_Dynamics as Stem_Dynamics
from .root_dynamics import WOFOST_Root_Dynamics as Root_Dynamics
from .leaf_dynamics import WOFOST_Leaf_Dynamics_NPK as Leaf_Dynamics
from .storage_organ_dynamics import (
    WOFOST_Storage_Organ_Dynamics as Storage_Organ_Dynamics,
)
from .assimilation import WOFOST_Assimilation2 as Assimilation
from .partitioning import DVS_Partitioning_NPK as Partitioning
from .evapotranspiration import EvapotranspirationCO2 as Evapotranspiration

from .npk_dynamics import NPK_Crop_Dynamics as NPK_crop
from .nutrients.npk_stress import NPK_Stress as NPK_Stress


class Wofost80(SimulationObject):
    """Top level object organizing the different components of the WOFOST crop
    simulation including the implementation of N/P/K dynamics.

    The CropSimulation object organizes the different processes of the crop
    simulation. Moreover, it contains the parameters, rate and state variables
    which are relevant at the level of the entire crop. The processes that are
    implemented as embedded simulation objects consist of:

        1. Phenology (self.pheno)
        2. Partitioning (self.part)
        3. Assimilation (self.assim)
        4. Maintenance respiration (self.mres)
        5. Evapotranspiration (self.evtra)
        6. Leaf dynamics (self.lv_dynamics)
        7. Stem dynamics (self.st_dynamics)
        8. Root dynamics (self.ro_dynamics)
        9. Storage organ dynamics (self.so_dynamics)
        10. N/P/K crop dynamics (self.npk_crop_dynamics)
        12. N/P/K stress (self.npk_stress)

    **Simulation parameters:**

    ======== =============================================== =======  ==========
     Name     Description                                     Type     Unit
    ======== =============================================== =======  ==========
    CVL      Conversion factor for assimilates to leaves       SCr     -
    CVO      Conversion factor for assimilates to storage      SCr     -
             organs.
    CVR      Conversion factor for assimilates to roots        SCr     -
    CVS      Conversion factor for assimilates to stems        SCr     -
    ======== =============================================== =======  ==========


    **State variables:**

    ============  ================================================= ==== ===============
     Name          Description                                      Pbl      Unit
    ============  ================================================= ==== ===============
    TAGP          Total above-ground Production                      N    |kg ha-1|
    GASST         Total gross assimilation                           N    |kg CH2O ha-1|
    MREST         Total gross maintenance respiration                N    |kg CH2O ha-1|
    CTRAT         Total crop transpiration accumulated over the
                  crop cycle                                         N    cm
    CEVST         Total soil evaporation accumulated over the
                  crop cycle                                         N    cm
    HI            Harvest Index (only calculated during              N    -
                  `finalize()`)
    DOF           Date representing the day of finish of the crop    N    -
                  simulation.
    FINISH_TYPE   String representing the reason for finishing the   N    -
                  simulation: maturity, harvest, leave death, etc.
    ============  ================================================= ==== ===============


     **Rate variables:**

    =======  ================================================ ==== =============
     Name     Description                                      Pbl      Unit
    =======  ================================================ ==== =============
    GASS     Assimilation rate corrected for water stress       N  |kg CH2O ha-1 d-1|
    PGASS    Potential assimilation rate                        N  |kg CH2O ha-1 d-1|
    MRES     Actual maintenance respiration rate, taking into
             account that MRES <= GASS.                         N  |kg CH2O ha-1 d-1|
    PMRES    Potential maintenance respiration rate             N  |kg CH2O ha-1 d-1|
    ASRC     Net available assimilates (GASS - MRES)            N  |kg CH2O ha-1 d-1|
    DMI      Total dry matter increase, calculated as ASRC
             times a weighted conversion efficiency.            Y  |kg ha-1 d-1|
    ADMI     Aboveground dry matter increase                    Y  |kg ha-1 d-1|
    =======  ================================================ ==== =============

    """

    __slots__ = [
        "pheno",
        "part",
        "assim",
        "mres",
        "evtra",
        "lv_dynamics",
        "st_dynamics",
        "ro_dynamics",
        "so_dynamics",
        "npk_crop_dynamics",
        "npk_stress",
    ]

    # sub-model components for crop simulation
    pheno: SimulationObject
    part: SimulationObject
    assim: SimulationObject
    mres: SimulationObject
    evtra: SimulationObject
    lv_dynamics: SimulationObject
    st_dynamics: SimulationObject
    ro_dynamics: SimulationObject
    so_dynamics: SimulationObject
    npk_crop_dynamics: SimulationObject
    npk_stress: SimulationObject

    # Parameters, rates and states which are relevant at the main crop
    # simulation level
    class Parameters(ParamTemplate):

        __slots__ = ["CVL", "CVO", "CVR", "CVS"]

        CVL: float
        CVO: float
        CVR: float
        CVS: float

    class StateVariables(StatesTemplate):

        __slots__ = [
            "TAGP",
            "GASST",
            "MREST",
            "CTRAT",
            "CEVST",
            "HI",
            "DOF",
            "FINISH_TYPE",
        ]

        TAGP: float
        GASST: float
        MREST: float
        CTRAT: float  # Crop total transpiration
        CEVST: float
        HI: float
        DOF: datetime.date
        FINISH_TYPE: str | None

    class RateVariables(RatesTemplate):

        __slots__ = [
            "GASS",
            "PGASS",
            "MRES",
            "ASRC",
            "DMI",
            "ADMI",
        ]

        GASS: float
        PGASS: float
        MRES: float
        ASRC: float
        DMI: float
        ADMI: float

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE model instance
        :param parvalues: dictionary with parameter key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["DMI", "ADMI"])
        self.kiosk = kiosk

        # Initialize components of the crop
        self.pheno = Phenology(day, kiosk, parvalues)
        self.part = Partitioning(day, kiosk, parvalues)
        self.assim = Assimilation(day, kiosk, parvalues)
        self.mres = MaintenanceRespiration(day, kiosk, parvalues)
        self.evtra = Evapotranspiration(day, kiosk, parvalues)
        self.ro_dynamics = Root_Dynamics(day, kiosk, parvalues)
        self.st_dynamics = Stem_Dynamics(day, kiosk, parvalues)
        self.so_dynamics = Storage_Organ_Dynamics(day, kiosk, parvalues)
        self.lv_dynamics = Leaf_Dynamics(day, kiosk, parvalues)
        # Added for book keeping of N/P/K in crop and soil
        self.npk_crop_dynamics = NPK_crop(day, kiosk, parvalues)
        self.npk_stress = NPK_Stress(day, kiosk, parvalues)

        # Initial total (living+dead) above-ground biomass of the crop
        TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        self.states = self.StateVariables(
            kiosk,
            publish=["TAGP", "GASST", "MREST", "HI"],
            TAGP=TAGP,
            GASST=0.0,
            MREST=0.0,
            CTRAT=0.0,
            HI=0.0,
            CEVST=0.0,
            DOF=None,
            FINISH_TYPE=None,
        )

        # Check partitioning of TDWI over plant organs
        checksum = parvalues["TDWI"] - self.states.TAGP - self.kiosk.TWRT
        if abs(checksum) > 0.0001:
            msg = "Error in partitioning of initial biomass (TDWI)!"
            raise exc.PartitioningError(msg)

        # assign handler for CROP_FINISH signal
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

    @staticmethod
    def _check_carbon_balance(day, DMI, GASS, MRES, CVF, pf):
        (FR, FL, FS, FO) = pf
        checksum = (
            (GASS - MRES - (FR + (FL + FS + FO) * (1.0 - FR)) * DMI / CVF)
            * 1.0
            / (max(0.0001, GASS))
        )
        if abs(checksum) >= 0.0001:
            msg = "Carbon flows not balanced on day %s\n" % day
            msg += "Checksum: %f, GASS: %f, MRES: %f\n" % (checksum, GASS, MRES)
            msg += "FR,L,S,O: %5.3f,%5.3f,%5.3f,%5.3f, DMI: %f, CVF: %f\n" % (
                FR,
                FL,
                FS,
                FO,
                DMI,
                CVF,
            )
            raise exc.CarbonBalanceError(msg)

    @prepare_rates
    def calc_rates(self, day, drv):
        params = self.params
        rates = self.rates
        k = self.kiosk

        # Phenology
        self.pheno.calc_rates(day, drv)
        crop_stage = self.pheno.get_variable("STAGE")

        # if before emergence there is no need to continue
        # because only the phenology is running.
        if crop_stage == "emerging":
            return

        # Potential assimilation
        rates.PGASS = self.assim(day, drv)

        # (evapo)transpiration rates
        self.evtra(day, drv)

        # nutrient status and reduction factor
        NNI, NPKI, RFNPK = self.npk_stress(day, drv)

        # Select minimum of nutrient and water/oxygen stress
        reduction = min(RFNPK, k.RFTRA)

        rates.GASS = rates.PGASS * reduction

        # Respiration
        PMRES = self.mres(day, drv)
        rates.MRES = min(rates.GASS, PMRES)

        # Net available assimilates
        rates.ASRC = rates.GASS - rates.MRES

        # DM partitioning factors (pf), conversion factor (CVF),
        # dry matter increase (DMI) and check on carbon balance
        pf = self.part.calc_rates(day, drv)
        CVF = 1.0 / (
            (pf.FL / params.CVL + pf.FS / params.CVS + pf.FO / params.CVO)
            * (1.0 - pf.FR)
            + pf.FR / params.CVR
        )
        rates.DMI = CVF * rates.ASRC
        self._check_carbon_balance(day, rates.DMI, rates.GASS, rates.MRES, CVF, pf)

        # distribution over plant organ

        # Below-ground dry matter increase and root dynamics
        self.ro_dynamics.calc_rates(day, drv)
        # Aboveground dry matter increase and distribution over stems,
        # leaves, organs
        rates.ADMI = (1.0 - pf.FR) * rates.DMI
        self.st_dynamics.calc_rates(day, drv)
        self.so_dynamics.calc_rates(day, drv)
        self.lv_dynamics.calc_rates(day, drv)

        # Update nutrient rates in crop and soil
        self.npk_crop_dynamics.calc_rates(day, drv)

    @prepare_states
    def integrate(self, day, delt=1.0):
        rates = self.rates
        states = self.states

        # crop stage before integration
        crop_stage = self.pheno.get_variable("STAGE")

        # Phenology
        self.pheno.integrate(day, delt)

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # Just run a touch() to to ensure that all state variables are available
        # in the kiosk
        if crop_stage == "emerging":
            self.touch()
            return

        # Partitioning
        self.part.integrate(day, delt)

        # Integrate states on leaves, storage organs, stems and roots
        self.ro_dynamics.integrate(day, delt)
        self.so_dynamics.integrate(day, delt)
        self.st_dynamics.integrate(day, delt)
        self.lv_dynamics.integrate(day, delt)

        # Update nutrient states in crop and soil
        self.npk_crop_dynamics.integrate(day, delt)

        # Integrate total (living+dead) above-ground biomass of the crop
        states.TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        # total gross assimilation and maintenance respiration
        states.GASST += rates.GASS
        states.MREST += rates.MRES

        # total crop transpiration and soil evaporation
        states.CTRAT += self.kiosk.TRA
        states.CEVST += self.kiosk.EVS

    @prepare_states
    def finalize(self, day):

        # Calculate Harvest Index
        if self.states.TAGP > 0:
            self.states.HI = self.kiosk.TWSO / self.states.TAGP
        else:
            msg = "Cannot calculate Harvest Index because TAGP=0"
            self.logger.warning(msg)
            self.states.HI = -1.0

        SimulationObject.finalize(self, day)

    def _on_CROP_FINISH(self, day, finish_type=None):
        """Handler for setting day of finish (DOF) and reason for
        crop finishing (FINISH).
        """
        self._for_finalize["DOF"] = day
        self._for_finalize["FINISH_TYPE"] = finish_type
