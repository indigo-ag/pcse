from ..decorators import prepare_rates, prepare_states
from ..base import ParamTemplate, RatesTemplate, SimulationObject


class SOCMineralization_Indigo(SimulationObject):
    """
    The SOCMineralization module estimates nitrogen mineralization from the decomposition
    of soil organic carbon (SOC) pools. It combines two approaches:

    1. Reading the decomposed carbon from the Ensemble_SOC_Indigo module, which provides
       the SOC pools and their decomposition rates.

    2. Using pool-specific carbon-to-nitrogen (C:N) ratios and mineralization efficiency
       factors to calculate the amount of nitrogen mineralized from each SOC pool.

    The module considers four main SOC pools:
    - Particulate Organic Matter (POM)
    - Low Molecular Weight Carbon (LMWC)
    - Microbial Biomass
    - Humic Substances

    For each SOC pool, the module calculates the nitrogen mineralized by dividing the
    decomposed carbon by the corresponding C:N ratio and multiplying by the mineralization
    efficiency factor. The mineralization efficiency factor represents the fraction of
    organic nitrogen that is converted to mineral nitrogen during decomposition.

    The total nitrogen mineralized from all SOC pools is then summed up and communicated
    to the mineral N module of PCSE, which updates its ammonium or nitrate pools accordingly.

    The module requires the following parameters:
    - C:N ratios for each SOC pool (CN_ratio_POM, CN_ratio_LMWC, CN_ratio_microbial, CN_ratio_humic)
    - Mineralization efficiency factors for each SOC pool (efficiency_POM, efficiency_LMWC,
      efficiency_microbial, efficiency_humic)

    Default parameter values are provided based on typical ranges reported in the literature,
    but it is recommended to use site-specific values when available or to calibrate the
    parameters using experimental data.

    The SOCMineralization module provides a simplified representation of nitrogen mineralization
    from SOC decomposition and can be integrated with other modules in the PCSE framework to
    simulate soil nitrogen dynamics.
    """

    class Parameters(ParamTemplate):

        __slots__ = [
            "CN_ratio_active",
            "CN_ratio_slow",
            "CN_ratio_passive",
            "efficiency_active",
            "efficiency_slow",
            "efficiency_passive",
        ]

        CN_ratio_active: float  # C:N ratio of particulate organic matter
        CN_ratio_slow: float  # C:N ratio of low molecular weight carbon
        CN_ratio_passive: float  # C:N ratio of microbial biomass

        efficiency_active: float  # Mineralization efficiency factor for POM
        efficiency_slow: float  # Mineralization efficiency factor for LMWC
        efficiency_passive: float  # Mineralization efficiency factor for microbial biomass

    class RateVariables(RatesTemplate):

        __slots__ = ["total_mineralized_N"]

        total_mineralized_N: float  # Total nitrogen mineralized from all SOC pools (kg N ha^-1 day^-1)

    def initialize(self, day, kiosk, parvalues):

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["total_mineralized_N"])

    @prepare_rates
    def calc_rates(self, day, drv):

        # if dPOM is in the kiosk, then we can calculate the mineralization
        if "dACTIVE" in self.kiosk:
            # Read decomposed carbon from each SOC pool
            decomp_carbon_active = self.kiosk["dACTIVE"]
            decomp_carbon_slow = self.kiosk["dSLOW"]
            decomp_carbon_passive = self.kiosk["dPASSIVE"]
        else:
            # If decomposed carbon is not available, set it to zero
            decomp_carbon_active = 0
            decomp_carbon_slow = 0
            decomp_carbon_passive = 0

        # Calculate nitrogen mineralization for each SOC pool
        mineralized_N_active = (
            decomp_carbon_active
            / self.params.CN_ratio_active
            * self.params.efficiency_active
        )
        mineralized_N_slow = (
            decomp_carbon_slow / self.params.CN_ratio_slow * self.params.efficiency_slow
        )
        mineralized_N_passive = (
            decomp_carbon_passive
            / self.params.efficiency_passive
            * self.params.efficiency_passive
        )

        # Sum up mineralized nitrogen from all SOC pools
        self.rates.total_mineralized_N = 0  # (
        # mineralized_N_active + mineralized_N_slow + mineralized_N_passive
        # ) * 10 # Convert from g N m^-2 day^-1 to kg N ha^-1 day^-1
        # print(f"Total mineralized N: {self.rates.total_mineralized_N} decomp_carbon_active {decomp_carbon_active} decomp_carbon_slow {decomp_carbon_slow} decomp_carbon_passive {decomp_carbon_passive}")

    @prepare_states
    def integrate(self, day, delt=1.0):
        # No state variables to update in this module
        pass
