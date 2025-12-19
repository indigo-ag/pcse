# -*- coding: utf-8 -*-
# Copyright (c) 2004-2015 Alterra, Wageningen-UR
# Allard de Wit and Iwan Supit (allard.dewit@wur.nl), July 2015
# Approach based on LINTUL N/P/K made by Joost Wolf

from ...decorators import prepare_rates, prepare_states
from ...base import ParamTemplate, StatesTemplate, RatesTemplate, SimulationObject


class NPK_Translocation(SimulationObject):
    """Does the bookkeeping for translocation of N/P/K from the roots, leaves
    and stems towards the storage organs of the crop.

    First the routine calculates the state of the translocatable amount of N/P/K.
    This translocatable amount is defined as the amount of N/P/K above the
    residual N/P/K amount calculated as the residual concentration times the
    living biomass. The residual amount is locked into the plant structural biomass
    and cannot be mobilized anymore. The translocatable amount is calculated for
    stems, roots and leaves and published as the state variables
    Ntranslocatable, Ptranslocatable and Ktranslocatable.

    The overall translocation rate is calculated as the minimum of supply (the
    translocatable amount) and demand from the storage organs as calculated in
    the component on Demand_Uptake.
    The actual rate of N/P/K translocation from the different plant organs is
    calculated assuming that the uptake rate is distributed over roots, stems and
    leaves in proportion to the translocatable amount for each organ.

    **Simulation parameters**

    ===============  =============================================  ======================
     Name             Description                                    Unit
    ===============  =============================================  ======================
    NRESIDLV          Residual N fraction in leaves                 kg N kg-1 dry biomass
    PRESIDLV          Residual P fraction in leaves                 kg P kg-1 dry biomass
    KRESIDLV          Residual K fraction in leaves                 kg K kg-1 dry biomass

    NRESIDST          Residual N fraction in stems                  kg N kg-1 dry biomass
    PRESIDST          Residual P fraction in stems                  kg P kg-1 dry biomass
    KRESIDST          Residual K fraction in stems                  kg K kg-1 dry biomass

    NPK_TRANSLRT_FR   NPK translocation from roots as a fraction     -
                      of resp. total NPK amounts translocated
                      from leaves and stems
    ===============  =============================================  ======================


    **State variables**

    ===================  ================================================= ===== ============
     Name                  Description                                      Pbl      Unit
    ===================  ================================================= ===== ============
    NtranslocatableLV     Translocatable N amount in living leaves           N    |kg N ha-1|
    PtranslocatableLV     Translocatable P amount in living leaves           N    |kg P ha-1|
    KtranslocatableLV     Translocatable K amount in living leaves           N    |kg K ha-1|
    NtranslocatableST     Translocatable N amount in living stems            N    |kg N ha-1|
    PtranslocatableST     Translocatable P amount in living stems            N    |kg P ha-1|
    KtranslocatableST     Translocatable K amount in living stems            N    |kg K ha-1|
    NtranslocatableRT     Translocatable N amount in living roots            N    |kg N ha-1|
    PtranslocatableRT     Translocatable P amount in living roots            N    |kg P ha-1|
    KtranslocatableRT     Translocatable K amount in living roots            N    |kg K ha-1|
    Ntranslocatable       Total N amount that can be translocated to the     Y    [kg N ha-1]
                          storage organs
    Ptranslocatable       Total P amount that can be translocated to the     Y    [kg P ha-1]
                          storage organs
    Ktranslocatable       Total K amount that can be translocated to the     Y    [kg K ha-1]
                          storage organs
    ===================  ================================================= ===== ============


    **Rate variables**


    ===================  ================================================= ==== ==============
     Name                 Description                                      Pbl      Unit
    ===================  ================================================= ==== ==============
    RNtranslocationLV     Weight increase (N) in leaves                     Y    |kg ha-1 d-1|
    RPtranslocationLV     Weight increase (P) in leaves                     Y    |kg ha-1 d-1|
    RKtranslocationLV     Weight increase (K) in leaves                     Y    |kg ha-1 d-1|
    RNtranslocationST     Weight increase (N) in stems                      Y    |kg ha-1 d-1|
    RPtranslocationST     Weight increase (P) in stems                      Y    |kg ha-1 d-1|
    RKtranslocationST     Weight increase (K) in stems                      Y    |kg ha-1 d-1|
    RNtranslocationRT     Weight increase (N) in roots                      Y    |kg ha-1 d-1|
    RPtranslocationRT     Weight increase (P) in roots                      Y    |kg ha-1 d-1|
    RKtranslocationRT     Weight increase (K) in roots                      Y    |kg ha-1 d-1|
    ===================  ================================================= ==== ==============

    **Signals send or handled**

    None

    **External dependencies:**

    ===========  ================================ ======================  ===========
     Name         Description                      Provided by             Unit
    ===========  ================================ ======================  ===========
    DVS           Crop development stage           DVS_Phenology           -
    WST           Dry weight of living stems       WOFOST_Stem_Dynamics   |kg ha-1|
    WLV           Dry weight of living leaves      WOFOST_Leaf_Dynamics   |kg ha-1|
    WRT           Dry weight of living roots       WOFOST_Root_Dynamics   |kg ha-1|
    NamountLV     Amount of N in leaves            NPK_Crop_Dynamics      |kg ha-1|
    NamountST     Amount of N in stems             NPK_Crop_Dynamics      |kg ha-1|
    NamountRT     Amount of N in roots             NPK_Crop_Dynamics      |kg ha-1|
    PamountLV     Amount of P in leaves            NPK_Crop_Dynamics      |kg ha-1|
    PamountST     Amount of P in stems             NPK_Crop_Dynamics      |kg ha-1|
    PamountRT     Amount of P in roots             NPK_Crop_Dynamics      |kg ha-1|
    KamountLV     Amount of K in leaves            NPK_Crop_Dynamics      |kg ha-1|
    KamountST     Amount of K in stems             NPK_Crop_Dynamics      |kg ha-1|
    KamountRT     Amount of K in roots             NPK_Crop_Dynamics      |kg ha-1|
    ===========  ================================ ======================  ===========
    """

    class Parameters(ParamTemplate):

        __slots__ = [
            "NRESIDLV",
            "NRESIDST",
            "NRESIDRT",
            "PRESIDLV",
            "PRESIDST",
            "PRESIDRT",
            "KRESIDLV",
            "KRESIDST",
            "KRESIDRT",
            "NPK_TRANSLRT_FR",
        ]

        NRESIDLV: float  # residual N fraction in leaves [kg N kg-1 dry biomass]
        NRESIDST: float  # residual N fraction in stems [kg N kg-1 dry biomass]
        NRESIDRT: float  # residual N fraction in roots [kg N kg-1 dry biomass]

        PRESIDLV: float  # residual P fraction in leaves [kg P kg-1 dry biomass]
        PRESIDST: float  # residual P fraction in stems [kg P kg-1 dry biomass]
        PRESIDRT: float  # residual P fraction in roots [kg P kg-1 dry biomass]

        KRESIDLV: float  # residual K fraction in leaves [kg P kg-1 dry biomass]
        KRESIDST: float  # residual K fraction in stems [kg P kg-1 dry biomass]
        KRESIDRT: float  # residual K fraction in roots [kg P kg-1 dry biomass]

        NPK_TRANSLRT_FR: float  # NPK translocation from roots as a fraction of
        # resp. total NPK amounts translocated from leaves
        # and stems

    class RateVariables(RatesTemplate):

        __slots__ = [
            "RNtranslocationLV",
            "RNtranslocationST",
            "RNtranslocationRT",
            "RPtranslocationLV",
            "RPtranslocationST",
            "RPtranslocationRT",
            "RKtranslocationLV",
            "RKtranslocationST",
            "RKtranslocationRT",
        ]

        RNtranslocationLV: float  # N translocation rate from leaves [kg ha-1 d-1]
        RNtranslocationST: float  # N translocation rate from stems [kg ha-1 d-1]
        RNtranslocationRT: float  # N translocation rate from roots [kg ha-1 d-1]

        RPtranslocationLV: float  # P translocation rate from leaves [kg ha-1 d-1]
        RPtranslocationST: float  # P translocation rate from stems [kg ha-1 d-1]
        RPtranslocationRT: float  # P translocation rate from roots [kg ha-1 d-1]

        RKtranslocationLV: float  # K translocation rate from leaves [kg ha-1 d-1]
        RKtranslocationST: float  # K translocation rate from stems [kg ha-1 d-1]
        RKtranslocationRT: float  # K translocation rate from roots [kg ha-1 d-1]

    class StateVariables(StatesTemplate):

        __slots__ = [
            "NtranslocatableLV",
            "NtranslocatableST",
            "NtranslocatableRT",
            "PtranslocatableLV",
            "PtranslocatableST",
            "PtranslocatableRT",
            "KtranslocatableLV",
            "KtranslocatableST",
            "KtranslocatableRT",
            "Ntranslocatable",
            "Ptranslocatable",
            "Ktranslocatable",
        ]

        NtranslocatableLV: float  # translocatable N amount in leaves [kg N ha-1]
        NtranslocatableST: float  # translocatable N amount in stems [kg N ha-1]
        NtranslocatableRT: float  # translocatable N amount in roots [kg N ha-1]

        PtranslocatableLV: float  # translocatable P amount in leaves [kg N ha-1]
        PtranslocatableST: float  # translocatable P amount in stems [kg N ha-1]
        PtranslocatableRT: float  # translocatable P amount in roots [kg N ha-1]

        KtranslocatableLV: float  # translocatable K amount in leaves [kg N ha-1
        KtranslocatableST: float  # translocatable K amount in stems [kg N ha-1]
        KtranslocatableRT: float  # translocatable K amount in roots [kg N ha-1]

        Ntranslocatable: float  # Total N amount that can be translocated to the storage organs [kg N ha-1]
        Ptranslocatable: float  # Total P amount that can be translocated to the storage organs [kg P ha-1]
        Ktranslocatable: float  # Total K amount that can be translocated to the storage organs [kg K ha-1]

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with WOFOST cropdata key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(
            kiosk,
            publish=[
                "RNtranslocationLV",
                "RNtranslocationST",
                "RNtranslocationRT",
                "RPtranslocationLV",
                "RPtranslocationST",
                "RPtranslocationRT",
                "RKtranslocationLV",
                "RKtranslocationST",
                "RKtranslocationRT",
            ],
        )

        self.states = self.StateVariables(
            kiosk,
            NtranslocatableLV=0.0,
            NtranslocatableST=0.0,
            NtranslocatableRT=0.0,
            PtranslocatableLV=0.0,
            PtranslocatableST=0.0,
            PtranslocatableRT=0.0,
            KtranslocatableLV=0.0,
            KtranslocatableST=0.0,
            KtranslocatableRT=0.0,
            Ntranslocatable=0.0,
            Ptranslocatable=0.0,
            Ktranslocatable=0.0,
            publish=["Ntranslocatable", "Ptranslocatable", "Ktranslocatable"],
        )
        self.kiosk = kiosk

    @prepare_rates
    def calc_rates(self, day, drv):
        r = self.rates
        s = self.states
        k = self.kiosk

        # partitioning of the uptake for storage organs from the leaves, stems, roots
        # assuming equal distribution of N/P/K from each organ.
        # If amount of translocatable N/P/K = 0 then translocation rate is 0
        if s.Ntranslocatable > 0.0:
            r.RNtranslocationLV = k.RNuptakeSO * s.NtranslocatableLV / s.Ntranslocatable
            r.RNtranslocationST = k.RNuptakeSO * s.NtranslocatableST / s.Ntranslocatable
            r.RNtranslocationRT = k.RNuptakeSO * s.NtranslocatableRT / s.Ntranslocatable
        else:
            r.RNtranslocationLV = r.RNtranslocationST = r.RNtranslocationRT = 0.0

        if s.Ptranslocatable > 0:
            r.RPtranslocationLV = k.RPuptakeSO * s.PtranslocatableLV / s.Ptranslocatable
            r.RPtranslocationST = k.RPuptakeSO * s.PtranslocatableST / s.Ptranslocatable
            r.RPtranslocationRT = k.RPuptakeSO * s.PtranslocatableRT / s.Ptranslocatable
        else:
            r.RPtranslocationLV = r.RPtranslocationST = r.RPtranslocationRT = 0.0

        if s.Ktranslocatable > 0:
            r.RKtranslocationLV = k.RKuptakeSO * s.KtranslocatableLV / s.Ktranslocatable
            r.RKtranslocationST = k.RKuptakeSO * s.KtranslocatableST / s.Ktranslocatable
            r.RKtranslocationRT = k.RKuptakeSO * s.KtranslocatableRT / s.Ktranslocatable
        else:
            r.RKtranslocationLV = r.RKtranslocationST = r.RKtranslocationRT = 0.0

    @prepare_states
    def integrate(self, day, delt=1.0):
        p = self.params
        s = self.states
        k = self.kiosk

        # translocatable N amount in the organs [kg N ha-1]
        s.NtranslocatableLV = max(0.0, k.NamountLV - k.WLV * p.NRESIDLV)
        s.NtranslocatableST = max(0.0, k.NamountST - k.WST * p.NRESIDST)
        s.NtranslocatableRT = max(0.0, k.NamountRT - k.WRT * p.NRESIDRT)

        # translocatable P amount in the organs [kg P ha-1]
        s.PtranslocatableLV = max(0.0, k.PamountLV - k.WLV * p.PRESIDLV)
        s.PtranslocatableST = max(0.0, k.PamountST - k.WST * p.PRESIDST)
        s.PtranslocatableRT = max(0.0, k.PamountRT - k.WRT * p.PRESIDRT)

        # translocatable K amount in the organs [kg K ha-1]
        s.KtranslocatableLV = max(0.0, k.KamountLV - k.WLV * p.KRESIDLV)
        s.KtranslocatableST = max(0.0, k.KamountST - k.WST * p.KRESIDST)
        s.KtranslocatableRT = max(0.0, k.KamountRT - k.WRT * p.KRESIDRT)

        # total translocatable NPK amount in the organs [kg N ha-1]
        s.Ntranslocatable = (
            s.NtranslocatableLV + s.NtranslocatableST + s.NtranslocatableRT
        )
        s.Ptranslocatable = (
            s.PtranslocatableLV + s.PtranslocatableST + s.PtranslocatableRT
        )
        s.Ktranslocatable = (
            s.KtranslocatableLV + s.KtranslocatableST + s.KtranslocatableRT
        )
