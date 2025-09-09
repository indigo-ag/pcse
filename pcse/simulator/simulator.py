# pylint: skip-file
# type: ignore
import concurrent.futures
import copy
import json
import logging
import multiprocessing as mp
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dask
import numpy as np
import pandas as pd
import yaml
from dask.distributed import Client, progress
from pcse.base import ParameterProvider
from pcse.fileinput import (
    CABOFileReader,
    YAMLAgroManagementReader,
    YAMLCropDataProvider,
)
from pcse.models import Indigo_Wofost
from scipy.interpolate import interp1d
from tqdm import tqdm as tqdm_func
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def calculate_soc_pool_fractions(clay_silt_percent, maom_modifier=1.0):
    """
    Calculate the fraction of total SOC for each pool based on clay & silt percentage,
    with a modifier for the MAOM fraction. The AGG fraction is adjusted to ensure the total sum is 1.
    All fractions are ensured to be non-negative.

    :param clay_silt_percent: Percentage of clay & silt in the soil
    :param maom_modifier: Modifier for the MAOM fraction (default: 1.0, no modification)
    :return: Dictionary of pool fractions
    """

    # Linear relationships for pool fractions (estimated from the graph)
    fractions = {
        "MAOM": max(0, min(1, (0.0015 * clay_silt_percent + 0.25) * maom_modifier)),
        "POM": max(0, min(1, -0.0005 * clay_silt_percent + 0.15)),
        "MIC": max(0, min(1, 0.0002 * clay_silt_percent + 0.05)),
        "LMWC": max(0, min(1, 0.0001 * clay_silt_percent + 0.02)),
    }

    # Calculate the total sum of fractions excluding AGG
    total_excluding_agg = sum(fractions.values())

    # Adjust AGG fraction to ensure the total sum remains 1
    fractions["AGG"] = max(0, 1 - total_excluding_agg)

    # If the sum still exceeds 1, normalize all fractions
    total = sum(fractions.values())
    if total > 1:
        for key in fractions:
            fractions[key] /= total

    return fractions


def breakdown_isoc(total_soc, clay_silt_percent, MFHP_Mill):
    """
    Break down total SOC into different pools based on clay & silt percentage.

    Parameters:
    total_soc: Total SOC value
    clay_silt_percent: Percentage of clay & silt

    Returns:
    Dictionary of SOC values for each pool
    """
    fractions = calculate_soc_pool_fractions(clay_silt_percent, MFHP_Mill)
    return {pool: total_soc * fraction for pool, fraction in fractions.items()}


def calculate_soc_pools(total_soc, clay_percent):
    # Data extracted from the graph (approximate values)
    clay_percentages = [10, 20, 30, 40, 50, 60, 70, 80]

    lmwc_data = [0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17]
    mic_data = [0.10, 0.15, 0.18, 0.20, 0.22, 0.23, 0.24, 0.25]
    pom_data = [0.15, 0.20, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]
    agg_data = [0.20, 0.30, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85]
    maom_data = [0.80, 1.20, 1.45, 1.70, 1.95, 2.15, 2.30, 2.45]

    # Create interpolation functions
    lmwc_func = interp1d(
        clay_percentages, lmwc_data, kind="linear", fill_value="extrapolate"
    )
    mic_func = interp1d(
        clay_percentages, mic_data, kind="linear", fill_value="extrapolate"
    )
    pom_func = interp1d(
        clay_percentages, pom_data, kind="linear", fill_value="extrapolate"
    )
    agg_func = interp1d(
        clay_percentages, agg_data, kind="linear", fill_value="extrapolate"
    )
    maom_func = interp1d(
        clay_percentages, maom_data, kind="linear", fill_value="extrapolate"
    )

    # Calculate proportions
    total_graph = (
        lmwc_func(clay_percent)
        + mic_func(clay_percent)
        + pom_func(clay_percent)
        + agg_func(clay_percent)
        + maom_func(clay_percent)
    )

    lmwc = (lmwc_func(clay_percent) / total_graph) * total_soc
    mic = (mic_func(clay_percent) / total_graph) * total_soc
    pom = (pom_func(clay_percent) / total_graph) * total_soc
    agg = (agg_func(clay_percent) / total_graph) * total_soc
    maom = (maom_func(clay_percent) / total_graph) * total_soc

    # Ensure the sum matches the total initial SOC
    total = lmwc + mic + pom + agg + maom
    adjustment_factor = total_soc / total

    return {
        "LMWC": lmwc * adjustment_factor,
        "MIC": mic * adjustment_factor,
        "POM": pom * adjustment_factor,
        "AGG": agg * adjustment_factor,
        "MAOM": maom * adjustment_factor,
    }


def run_simulations_dask(simulations):
    futures = [dask.delayed(sim.run_simulation)() for sim in simulations]
    return dask.compute(*futures)


def run_simulation_chunk(chunk, observed_assim_data):
    updated_sims = []
    obs_assim = None
    for sim_id, sim in chunk:
        # Check if the simulation is already set up
        if observed_assim_data is not None:
            obs_assim = observed_assim_data[sim_id]
        else:
            obs_assim = sim.prepare_inital_soc_assimilation()

        sim.run_simulation(obs_assim)
        updated_sims.append((sim_id, sim))
    return updated_sims


class SimulationStatus(Enum):
    """
    Enum class to represent the status of a simulation.
    """

    NOT_STARTED = auto()
    SETUP_FAILED = auto()
    SETUP_SUCCESSFUL = auto()
    RUN_FAILED = auto()
    RUN_SUCCESSFUL = auto()
    PARAMETERS_CHANGED = auto()
    PARAMETERS_NOT_CHANGED = auto()
    NO_INPUT_DIR = auto()

    @property
    def is_failed(self):
        """
        Property that returns True if the status represents a failure state.
        """
        return self in {
            SimulationStatus.SETUP_FAILED,
            SimulationStatus.RUN_FAILED,
            SimulationStatus.NO_INPUT_DIR,
        }


class SimulationRunType(Enum):
    """
    Enum class to represent the type of the simulation run.
    """

    CALIBRATION_VALIDATION = auto()
    CAR1459 = auto()


class SiteSimulation:
    def __init__(
        self, obs_id: str, site: str, treatment_schedule: str, experiment_id: int
    ):
        self.obs_id = obs_id
        self.site = site
        self.treatment_schedule = treatment_schedule
        self.experiment_id = experiment_id
        self.siteID = None

        self.parameters = None
        self.agromanagement = None
        self.weather_data = None
        self.results = None
        self.status = SimulationStatus.NOT_STARTED
        self.run_info = None
        self.duration = None
        self.modified_parameters = {}
        self.residue_removal = None
        self.soc_initialization_df = None
        self.additionality_date = None

    def get_info(self):
        return {
            "obs_id": self.obs_id,
            "site": self.site,
            "treatment_schedule": self.treatment_schedule,
            "experiment_id": self.experiment_id,
            "end_year": self.run_info["end_year"],
            "parameters": print(self.parameters),
            "Status": self.status,
            "Row": self.run_info[["Variable", "Starting_Value"]],
        }

    def modify_residue_percent(self, agromanagement):
        new_residue_percent = self.residue_removal["residue_removal"].values[0]

        try:
            for i, item in enumerate(agromanagement):
                for date, campaign_data in item.items():
                    timed_events = campaign_data.get("TimedEvents", [])
                    for event in timed_events:
                        events_table = event.get("events_table", [])
                        for event_item in events_table:
                            for event_date, event_data in event_item.items():
                                if "residue_percent" in event_data:
                                    event_data["residue_percent"] = new_residue_percent
        except Exception:
            agromanagement = agromanagement
            # print(agromanagement)
            # print(f"Error modifying residue percent for {self.obs_id} - {self.treatment_schedule} - {new_residue_percent}: {e}")
        return agromanagement

    def filter_agromanagement(self, agromanagement):
        """
        This function does the following:
        1. Remove all items with null CropCalendar
        2. Check the dates of the CropCalendar and TimedEvents and find the latest date
        3. If the latest date is not the last date of the year, add a new item to the list with the latest date moved to the last day of the year
        This helps to ensure that the simulation runs for the entire year in case the last campaign does not cover the entire year
        and also removes any null CropCalendar items that are not the last item in the list

        :param agromanagement: List of agromanagement items
        :return: Filtered list of agromanagement items
        """

        # Keep all items except those with null CropCalendar that are not the last item
        filtered_list = []
        for index, item in enumerate(agromanagement):
            crop_calendar = list(item.values())[0].get("CropCalendar")
            if crop_calendar is not None:
                filtered_list.append(item)

        # print("here")
        # finding the last date from here.
        if filtered_list:
            last_campaign = filtered_list[-1]
            last_campaign_data = next(iter(last_campaign.values()), {})
            latest_date = list(last_campaign.keys())[0]
            # print(latest_date)
            # Check CropCalendar dates
            crop_calendar = last_campaign_data.get("CropCalendar", {})
            # print(crop_calendar)
            if crop_calendar:
                for date_str in [
                    crop_calendar.get("crop_end_date"),
                    crop_calendar.get("crop_start_date"),
                ]:
                    if date_str:
                        date_crop = date_str
                        if latest_date is None or date_crop > latest_date:
                            latest_date = date_crop

            # print(latest_date)
            # Check TimedEvents dates
            timed_events = last_campaign_data.get("TimedEvents", [])
            # print(timed_events)
            # if time events is not None
            if timed_events is not None:
                for event in timed_events:
                    events_table = event.get("events_table", [])

                    for event_date in events_table:
                        date_str = next(iter(event_date.keys()), None)
                        # print(date_str)
                        if date_str:
                            ev_date = (
                                date_str  # datetime.strptime(date_str, '%Y-%m-%d')
                            )
                            if latest_date is None or ev_date > latest_date:
                                latest_date = ev_date

            # check if the latest date is greater than the date of the last campaign

            if latest_date:
                # 365 - day of year of latest date
                # data collection year:
                end_year_date = pd.to_datetime(
                    f"{self.run_info['end_year']}-01-01"
                ) + pd.Timedelta(days=int(self.run_info["end_doy"]) - 1)
                days_to_add = 365 - latest_date.timetuple().tm_yday
                new_date = latest_date + pd.Timedelta(days=days_to_add)
                # print(f" {end_year_date.date()}- {new_date}")
                if new_date < end_year_date.date():
                    new_date = end_year_date.date()

                # print(f"Latest date in the last campaign: {latest_date} moved to {new_date} {end_year_date.date()}")
                filtered_list.append({new_date: None})

        # print(filtered_list[-1])
        return filtered_list

    def setup_simulation(
        self, pcse_input_dir: str, data_dir: str, verbose: bool = False
    ):
        if self.RunType == SimulationRunType.CALIBRATION_VALIDATION:
            if not os.path.exists(f"{pcse_input_dir}/experiment_{self.experiment_id}"):
                self.status = SimulationStatus.NO_INPUT_DIR
                # print(f"Input directory not found: {pcse_input_dir}/experiment_{self.experiment_id}")
                return
            try:
                # ----------------------- CROP DATA PROVIDER -----------------------
                cropd = YAMLCropDataProvider(
                    fpath=Path(__file__).parent / "resources" / "WOFOST_crop_parameters"
                )
                # ----------------------- SITE DATA PROVIDER -----------------------
                sitedata = CABOFileReader(
                    Path(__file__).parent / "resources" / "wofost_npk.site"
                )

                # ----------------------- SOIL DATA PROVIDER -----------------------
                soil_file = os.path.join(
                    pcse_input_dir,
                    f"experiment_{self.experiment_id}",
                    f"experiment_{self.experiment_id}_soils.yaml",
                )
                with open(soil_file) as stream:
                    soil_data = yaml.safe_load(stream)

                soil_data["Ens_Weights"] = [0.25, 0.25, 0.25, 0.25]
                # soil_data['param_pc'] = 0.05
                soil_data["kaff_lb"] = 1200  # 0038

                soil_data["KO"] = [12, 12]
                soil_data["CUE"] = [0.55, 0.25, 0.55, 0.25]

                # ----------------------- PARAMETER PROVIDER -----------------------
                self.parameters = ParameterProvider(
                    cropdata=cropd, soildata=soil_data, sitedata=sitedata
                )
                # ----------------------- AGROMANAGEMENT PROVIDER -----------------------
                agromanagement_file = os.path.join(
                    pcse_input_dir,
                    f"experiment_{self.experiment_id}",
                    f"experiment_{self.experiment_id}.yaml",
                )
                agromanagement = YAMLAgroManagementReader(agromanagement_file)
                # Filter the agromanagement data
                self.agromanagement = self.filter_agromanagement(agromanagement)

                if self.residue_removal is not None:
                    self.agromanagement = self.modify_residue_percent(
                        self.agromanagement
                    )

                self.status = SimulationStatus.SETUP_SUCCESSFUL
            except Exception as e:
                self.status = SimulationStatus.SETUP_FAILED
                print(
                    f"Setup failed for experiment {self.experiment_id} - {self.treatment_schedule}: {e}"
                )

        elif self.RunType == SimulationRunType.CAR1459:
            try:
                # ----------------------- CROP DATA PROVIDER -----------------------
                cropd = YAMLCropDataProvider(
                    fpath=Path(__file__).parent / "resources" / "wofost_crop_parameters"
                )

                # ----------------------- SITE DATA PROVIDER -----------------------
                sitedata = CABOFileReader(
                    Path(__file__).parent / "resources" / "wofost_npk.site"
                )
                # ----------------------- SOIL DATA PROVIDER -----------------------
                # Use the soil file path from run_info
                soil_data = self.run_info["soil_data"]

                # Apply standard soil parameters for CAR1459
                soil_data["Ens_Weights"] = [0.25, 0.25, 0.25, 0.25]
                soil_data["kaff_lb"] = 1200
                soil_data["KO"] = [12, 12]
                soil_data["CUE"] = [0.55, 0.25, 0.55, 0.25]

                # ----------------------- PARAMETER PROVIDER -----------------------
                self.parameters = ParameterProvider(
                    cropdata=cropd, soildata=soil_data, sitedata=sitedata
                )
                # ----------------------- AGROMANAGEMENT PROVIDER -----------------------
                # Use the experiment event data from run_info if provided, otherwise use sample data
                agromanagement = self.run_info["experiment_data"]["AgroManagement"]

                # Filter the agromanagement data
                self.agromanagement = (
                    agromanagement  # self.filter_agromanagement(agromanagement)
                )

                if verbose:
                    print(
                        f"Setup successful for management zone {self.run_info['Site']}, "
                        f"ensemble {self.treatment_schedule}, run {self.obs_id}"
                    )

            except Exception as e:
                self.status = SimulationStatus.SETUP_FAILED
                print(
                    f"Setup failed for management zone {self.run_info['Site']}, "
                    f"ensemble {self.treatment_schedule}: {str(e)}"
                )

    def reset_parameters(self):
        # deep copy the parameters
        tmp_params = copy.deepcopy(self.parameters)
        print(self.modified_parameters.keys())
        # clear the override parameters
        for key in list(self.modified_parameters.keys()):
            # print(f"Clearing override for {key}")
            tmp_params.clear_override(key)

        return tmp_params

    def run_simulation(self, observed_assim_data=None, verbose: bool = False):
        if self.status not in [
            SimulationStatus.SETUP_FAILED,
            SimulationStatus.NO_INPUT_DIR,
        ]:
            # deep copy the parameters
            # the expected format of observed_assim_data is a dict with keys of dates with
            # for each key we have a dict of (variable, value)
            # as long as there are dates that we need to assimilate we do
            # 1- run_till(obs_date)
            # 2- assimilate the data using set_variable(variable, value)
            # 3- run_till_terminate()
            try:
                if verbose:
                    print(self.parameters)

                wofsim = Indigo_Wofost(
                    copy.deepcopy(self.parameters),
                    self.weather_data,
                    self.agromanagement,
                )
                # ------------------------------------ Mode RUN -----------------------------------
                # -------------------------------- Data Assimilation ------------------------------
                if observed_assim_data is None:
                    wofsim.run_till_terminate()
                else:  # data assimilation
                    # print(self.parameters)
                    # Run till the first observed date
                    # print("Running under data assimilation mode")
                    for obs_date, obs_data in observed_assim_data.items():
                        # print(f"Running till {obs_date} and assimilating data")
                        # Convert obs_date to datetime if it's a string
                        if isinstance(obs_date, str):
                            obs_date = pd.to_datetime(obs_date)
                        # Run till the observation date
                        wofsim.run_till(obs_date)
                        # Assimilate the data
                        for variable, value in obs_data.items():
                            # print(f"Assimilating {variable} with value {value} on {obs_date}")
                            wofsim.set_variable(variable, value)

                    # Run till termination
                    wofsim.run_till_terminate()

                self.results = pd.DataFrame(wofsim.get_output())
                # Ensure the 'day' column is parsed as datetime
                self.results["day"] = pd.to_datetime(self.results["day"])
                # Set 'day' as the index
                self.results.set_index("day", inplace=True)
                self.status = SimulationStatus.RUN_SUCCESSFUL
                # Reset the parameter change status after a successful run
                if self.status == SimulationStatus.PARAMETERS_CHANGED:
                    self.status = SimulationStatus.PARAMETERS_NOT_CHANGED

                # clear the override parameters
                for key in list(self.modified_parameters.keys()):
                    # print(f"Clearing override for {key}")
                    self.parameters.clear_override(key)
                    # remove the key from the modified parameters
                    self.modified_parameters.pop(key)

            except Exception as e:
                self.status = SimulationStatus.RUN_FAILED
                raise ValueError(
                    f"Simulation failed for experiment {self.experiment_id} - {self.treatment_schedule}: {e}"
                )
        else:
            print(
                f"{self.status} - Simulation not run for experiment {self.experiment_id} - {self.treatment_schedule}"
            )

    def modify_parameter(
        self, parameter_name: str, value: Any, initial_value: bool = False
    ):
        if self.parameters is not None:
            # raise ValueError("Parameters not set up. Run setup_simulation first.")
            # if value is a dict then, it is modifying an array of parameters and it has both value, index and the original value
            if isinstance(value, dict):
                print(value)
                original_array = value["original_value"]
                modified_array = original_array.copy()
                modified_array[value["index"]] = value["value"]
                self.parameters.set_override(
                    parameter_name, modified_array, check=False
                )
                self.status = SimulationStatus.PARAMETERS_CHANGED
                if not initial_value:
                    self.modified_parameters[parameter_name] = modified_array
            else:
                self.parameters.set_override(parameter_name, value, check=False)
                self.status = SimulationStatus.PARAMETERS_CHANGED
                if not initial_value:
                    self.modified_parameters[parameter_name] = value

    def prepare_inital_soc_assimilation(self):
        if self.soc_initialization_df is not None:
            # use the self.additionality_date as the date for the assimilation
            if self.additionality_date is not None:
                self.additionality_date = pd.to_datetime(self.additionality_date)
                # create a dict of the values to be estimated as a dict of date, variable, value
                # each column in soc_initialization_df is a variable that needs to be assimilated
                soc_assimilation = {}
                soc_assimilation[self.additionality_date] = {}
                for index, row in self.soc_initialization_df.iterrows():
                    variable = row["Parameter"]
                    value = row["Value"]
                    # create the dict
                    soc_assimilation[self.additionality_date][variable] = value

                return soc_assimilation
            else:
                return None

    def initialize_simulation(
        self, params: Dict[str, Any] = None, verbose: bool = False
    ):
        if self.RunType == SimulationRunType.CALIBRATION_VALIDATION:
            # if row info is not none then we can initialize the simulation using information from the run table
            if self.status not in [
                SimulationStatus.SETUP_FAILED,
                SimulationStatus.NO_INPUT_DIR,
            ]:
                if (self.run_info is not None) and (self.parameters is not None):
                    if verbose:
                        print(
                            f"Initializing simulation for experiment {self.experiment_id} - {self.treatment_schedule} {params}"
                        )

                    # Initialize the nitrogen pool to the initial values from experiment table
                    self.modify_parameter(
                        "NAVAILI", params["NAVAILI"], initial_value=True
                    )
                    self.modify_parameter(
                        "NSOILBASE", params["NAVAILI"], initial_value=True
                    )
                    clay_content = self.parameters.get(
                        "param_clay"
                    )  # convert to fraction
                    claysilt_content = self.parameters.get(
                        "param_claysilt"
                    )  # convert to fraction

                    if "somsc" in self.run_info["Variable"]:
                        FHP = params["FHP"]
                        MFHP_Mill = params["MFHP_Mill"]
                        intial_soc = self.run_info["Starting_Value"]
                        print(
                            f"Initialized with : {params} {clay_content} {intial_soc}"
                        )
                        # ------- Millenna model
                        # init_MillV1 = calculate_soc_pools(intial_soc, clay_content)
                        init_Mill = breakdown_isoc(
                            intial_soc, claysilt_content, MFHP_Mill
                        )
                        # print(init_MillV1)
                        # print(init_Mill)
                        # print("-----------------------------")
                        mom0 = init_Mill["MAOM"]
                        agg0 = init_Mill["AGG"]
                        pom0 = init_Mill["POM"]
                        mb0 = init_Mill["MIC"]
                        lmwc0 = init_Mill["LMWC"]
                        # if any of the pools is negative print
                        if any(value < 0 for value in init_Mill.values()):
                            print(f"Negative value in initial pools: {init_Mill}")
                        # if verbose:
                        # print(f"claysilt:{claysilt_content} - SOCi: {intial_soc} - MOAM:{mom0} - AGG:{agg0} - POM:{pom0} - MB:{mb0} - LMWC:{lmwc0}, FHP{FHP}")
                        self.modify_parameter("AGG0", agg0, initial_value=True)
                        self.modify_parameter(
                            "MIC0", mb0, initial_value=True
                        )  # Microbial biomass is not a large pool, typically < 5% of SOC
                        self.modify_parameter(
                            "MAOM0", mom0, initial_value=True
                        )  # MAOM accounts for a large proportion (50–85%) of the total SOC stock in bulk soil and in most soils SOC in MAOM has a longer mean turnover time than other measurable soil fractions such as aggregates and POM
                        self.modify_parameter(
                            "LMWC0", lmwc0, initial_value=True
                        )  # i.e., free fragments of plant detritus # i.e., root exudates and the by-products of exoenzyme activity;
                        self.modify_parameter(
                            "POM0", pom0, initial_value=True
                        )  # i.e., free fragments of plant detritus
                        # ------ ROTH-C
                        bio0 = 0.04 * intial_soc
                        hum0 = (FHP) * (intial_soc - (0.04 * intial_soc))
                        intial_soc - (bio0 + hum0)
                        self.modify_parameter("DPM0", 1.0, initial_value=True)
                        self.modify_parameter("RPM0", 1.5, initial_value=True)
                        self.modify_parameter("HUM0", hum0, initial_value=True)
                        self.modify_parameter("BIO0", bio0, initial_value=True)
                        self.modify_parameter("TIsoc", intial_soc, initial_value=True)
                        # ------- Century
                        act0 = 0.04 * intial_soc
                        pas0 = (FHP) * (intial_soc - (act0))
                        slow0 = intial_soc - (act0 + (pas0))

                        self.modify_parameter("ACTIVE0", act0, initial_value=True)
                        self.modify_parameter("SLOW0", slow0, initial_value=True)
                        self.modify_parameter("PASSIVE0", pas0, initial_value=True)
                        # --------  MIMICs
                        self.modify_parameter("MIC_1_0", mb0, initial_value=True)
                        self.modify_parameter("MIC_2_0", lmwc0, initial_value=True)
                        self.modify_parameter("SOM_1_0", agg0, initial_value=True)
                        self.modify_parameter("SOM_2_0", mom0, initial_value=True)
                        self.modify_parameter("SOM_3_0", pom0, initial_value=True)
                        # collecting all the values used in initializatiob and keep it internally
                        all_params = {
                            # Mill
                            "AGG": agg0,
                            "MIC": mb0,
                            "MAOM": mom0,
                            "LMWC": lmwc0,
                            "POM": pom0,
                            # ROTH-C
                            "DPM": 1.0,
                            "RPM": 1.5,
                            "HUM": hum0,
                            "BIO": bio0,
                            "IOM": intial_soc,
                            # Century
                            "SOC_ACTIVE": act0,
                            "SOC_SLOW": slow0,
                            "SOC_PASSIVE": pas0,
                            # MIMICs
                            "MIC_1": mb0,
                            "MIC_2": lmwc0,
                            "SOM_1": agg0,
                            "SOM_2": mom0,
                            "SOM_3": pom0,
                        }

                        # Convert to DataFrame
                        self.soc_initialization_df = pd.DataFrame(
                            list(all_params.items()), columns=["Parameter", "Value"]
                        )

                        if verbose:
                            print("<Initializtion> ---------------------------------")
                            print(
                                f"Initialized with FHP: {params} and initial SOC: {intial_soc}"
                            )
                            print(
                                f"Initial SOC pools: {self.parameters.get('ACTIVE0')}, {self.parameters.get('SLOW0')}, {self.parameters.get('PASSIVE0')}"
                            )
                            print(
                                f"Initial SOM pools: {self.parameters.get('SOM_1_0')}, {self.parameters.get('SOM_2_0')}, {self.parameters.get('SOM_3_0')} ,{self.parameters.get('MIC_1_0')}, {self.parameters.get('MIC_2_0')}"
                            )
                            print(
                                f"Initial ROTH-C pools: {self.parameters.get('DPM0')}, {self.parameters.get('RPM0')}, {self.parameters.get('HUM0')}, {self.parameters.get('BIO0')}, {self.parameters.get('TIsoc')}"
                            )
                            print(
                                f"Initial Millenna pools: {self.parameters.get('AGG0')}, {self.parameters.get('MIC0')}, {self.parameters.get('MAOM0')}, {self.parameters.get('LMWC0')}, {self.parameters.get('POM0')}"
                            )
                            print("</Initializtion> ---------------------------------")
                else:
                    self.status = SimulationStatus.SETUP_FAILED

        elif self.RunType == SimulationRunType.CAR1459:
            if self.status not in [
                SimulationStatus.SETUP_FAILED,
                SimulationStatus.NO_INPUT_DIR,
            ]:
                if (self.run_info is not None) and (self.parameters is not None):
                    if verbose:
                        print(
                            f"Initializing simulation for management zone {self.run_info['Site']}, "
                            f"ensemble {self.treatment_schedule}"
                        )

                    # Initialize the nitrogen pools
                    self.modify_parameter(
                        "NAVAILI", params["NAVAILI"], initial_value=True
                    )
                    self.modify_parameter(
                        "NSOILBASE", params["NAVAILI"], initial_value=True
                    )

                    soil_propr = self.run_info["soil_sample"][0]

                    # Collect initial SOC (gC/m2)
                    intial_soc = soil_propr["SOIL_CARBON_CONCENTRATION"]
                    self.run_info["intial_soc"] = intial_soc

                    FHP = params["FHP"]
                    MFHP_Mill = params["MFHP_Mill"]
                    # Get soil properties
                    clay_content = self.parameters.get("param_clay")
                    claysilt_content = self.parameters.get("param_claysilt")
                    # ------- Millenna model initialization
                    init_Mill = breakdown_isoc(intial_soc, claysilt_content, MFHP_Mill)

                    # Check for negative values in initial pools
                    if any(value < 0 for value in init_Mill.values()):
                        print(f"Warning: Negative value in initial pools: {init_Mill}")

                    # Initialize Millenna pools
                    self.modify_parameter("AGG0", init_Mill["AGG"], initial_value=True)
                    self.modify_parameter("MIC0", init_Mill["MIC"], initial_value=True)
                    self.modify_parameter(
                        "MAOM0", init_Mill["MAOM"], initial_value=True
                    )
                    self.modify_parameter(
                        "LMWC0", init_Mill["LMWC"], initial_value=True
                    )
                    self.modify_parameter("POM0", init_Mill["POM"], initial_value=True)

                    # ------ ROTH-C initialization
                    bio0 = 0.04 * intial_soc
                    hum0 = FHP * (intial_soc - bio0)
                    intial_soc - (bio0 + hum0)
                    # print(f"ROTHC intial_soc: {intial_soc} - hum0: {hum0} - bio0: {bio0}")
                    self.modify_parameter("DPM0", 1.0, initial_value=True)
                    self.modify_parameter("RPM0", 1.5, initial_value=True)
                    self.modify_parameter("HUM0", hum0, initial_value=True)
                    self.modify_parameter("BIO0", bio0, initial_value=True)
                    self.modify_parameter("TIsoc", intial_soc, initial_value=True)

                    # ------- Century initialization
                    act0 = 0.04 * intial_soc
                    pas0 = FHP * (intial_soc - act0)
                    slow0 = intial_soc - (act0 + pas0)

                    self.modify_parameter("ACTIVE0", act0, initial_value=True)
                    self.modify_parameter("SLOW0", slow0, initial_value=True)
                    self.modify_parameter("PASSIVE0", pas0, initial_value=True)

                    # -------- MIMICs initialization
                    self.modify_parameter(
                        "MIC_1_0", init_Mill["MIC"], initial_value=True
                    )
                    self.modify_parameter(
                        "MIC_2_0", init_Mill["LMWC"], initial_value=True
                    )
                    self.modify_parameter(
                        "SOM_1_0", init_Mill["AGG"], initial_value=True
                    )
                    self.modify_parameter(
                        "SOM_2_0", init_Mill["MAOM"], initial_value=True
                    )
                    self.modify_parameter(
                        "SOM_3_0", init_Mill["POM"], initial_value=True
                    )

                    all_params = {
                        # Mill
                        "AGG": init_Mill["AGG"],
                        "MIC": init_Mill["MIC"],
                        "MAOM": init_Mill["MAOM"],
                        "LMWC": init_Mill["LMWC"],
                        "POM": init_Mill["POM"],
                        # ROTH-C
                        "DPM": 1.0,
                        "RPM": 1.5,
                        "HUM": hum0,
                        "BIO": bio0,
                        "IOM": intial_soc,
                        # Century
                        "SOC_ACTIVE": act0,
                        "SOC_SLOW": slow0,
                        "SOC_PASSIVE": pas0,
                        # MIMICs
                        "MIC_1": init_Mill["MIC"],
                        "MIC_2": init_Mill["LMWC"],
                        "SOM_1": init_Mill["AGG"],
                        "SOM_2": init_Mill["MAOM"],
                        "SOM_3": init_Mill["POM"],
                    }

                    # Convert to DataFrame
                    self.soc_initialization_df = pd.DataFrame(
                        list(all_params.items()), columns=["Parameter", "Value"]
                    )

                    if verbose:
                        print("<Initializtion> ---------------------------------")
                        print(
                            f"Initialized with FHP: {FHP}, MFHP_Mill: {MFHP_Mill} and initial SOC: {intial_soc}"
                        )
                        print(
                            f"Initial SOC pools: {self.parameters.get('ACTIVE0')}, {self.parameters.get('SLOW0')}, {self.parameters.get('PASSIVE0')}"
                        )
                        print(
                            f"Initial SOM pools: {self.parameters.get('SOM_1_0')}, {self.parameters.get('SOM_2_0')}, {self.parameters.get('SOM_3_0')}, {self.parameters.get('MIC_1_0')}, {self.parameters.get('MIC_2_0')}"
                        )
                        print(
                            f"Initial ROTH-C pools: {self.parameters.get('DPM0')}, {self.parameters.get('RPM0')}, {self.parameters.get('HUM0')}, {self.parameters.get('BIO0')}, {self.parameters.get('TIsoc')}"
                        )
                        print(
                            f"Initial Millenna pools: {self.parameters.get('AGG0')}, {self.parameters.get('MIC0')}, {self.parameters.get('MAOM0')}, {self.parameters.get('LMWC0')}, {self.parameters.get('POM0')}"
                        )
                        print("</Initializtion> ---------------------------------")

                else:
                    self.status = SimulationStatus.SETUP_FAILED

    def get_results(self) -> pd.DataFrame:
        return self.results

    def get_results_runtable(self) -> pd.DataFrame:
        """
        Get filtered simulation results based on start and end dates from run_info.

        :return: Filtered DataFrame with simulation results
        """
        if self.results is None or self.run_info is None:
            return pd.DataFrame()

        start_date = pd.to_datetime(
            f"{self.run_info['start_year']}-01-01"
        ) + pd.Timedelta(days=int(self.run_info["start_doy"]) - 1)
        if self.additionality_date is not None:
            start_date = pd.to_datetime(self.additionality_date) - pd.Timedelta(days=7)
        end_date = pd.to_datetime(f"{self.run_info['end_year']}-01-01") + pd.Timedelta(
            days=int(self.run_info["end_doy"]) - 1
        )

        filtered_results = self.results.loc[start_date:end_date]

        return filtered_results

    def get_results_emulator(self, n_samples=50) -> pd.DataFrame:
        """
        Get filtered simulation results based on start and end dates from run_info.

        :return: Filtered DataFrame with simulation results
        """

        if self.results is None or self.run_info is None:
            return pd.DataFrame()

        start_date = pd.to_datetime(
            f"{self.run_info['start_year']}-01-01"
        ) + pd.Timedelta(days=int(self.run_info["start_doy"]) - 1)
        end_date = pd.to_datetime(f"{self.run_info['end_year']}-01-01") + pd.Timedelta(
            days=int(self.run_info["end_doy"]) - 1
        )

        # print("here")
        # Ensure the dates are within the range of the results index
        start_date = max(start_date, self.results.index.min())
        end_date = min(end_date, self.results.index.max())
        # print("here2")
        # Create a DatetimeIndex with equally spaced samples, excluding start and end
        if n_samples > 2:
            inner_range = pd.date_range(
                start=start_date, end=end_date, periods=n_samples - 2
            )
            date_range = [start_date] + list(inner_range[1:-1]) + [end_date]
        else:
            date_range = [start_date, end_date]
        # print("here3")
        # Find the nearest available dates in the results DataFrame
        nearest_dates = [
            self.results.index.get_indexer([date], method="nearest")[0]
            for date in date_range
        ]
        sampled_results = self.results.iloc[nearest_dates]
        # print("here4")
        # Remove potential duplicates (in case start or end date was the nearest for multiple points)
        sampled_results = sampled_results[
            ~sampled_results.index.duplicated(keep="first")
        ]

        # Calculate the distance in days from the start date
        sampled_results["days_from_start"] = (sampled_results.index - start_date).days
        # print("here5")
        return sampled_results


class MultiSiteSimulator:
    """
    A class representing a multi-site simulator.

    Attributes:
        db_path (str): The path to the SQLite database.
        simulations (Dict[int, SiteSimulation]): A dictionary of SiteSimulation objects representing the simulations.
        observed_values (Dict[int, float]): A dictionary of observed values.
        run_table (pd.DataFrame): A DataFrame representing the run table.
        max_workers (int): The maximum number of workers for multi-threading.
        chunk_size (int): The chunk size for dividing simulations.

    Methods:
        load_data_from_db(filter_conditions: List[Dict[str, Any]] = None) -> None:
            Load data from the database with optional filtering.

        run_single_simulation(sim_id, simulation) -> Tuple[int, Any]:
            Run a single simulation.

        setup_all_simulations(pcse_input_dir: str, data_dir: str) -> None:
            Set up all simulations.

        run_all_simulations_mp() -> None:
            Run all simulations using multiprocessing.

        run_all_simulations_mt() -> None:
            Run all simulations using multi-threading.

        modify_parameters(parameters: Dict[str, Any], experiment_specific: bool = False) -> None:
            Modify parameters for all simulations or for specific experiments.

        get_results(output_variables: List[str], aggregation_func: Callable = None,
                    time_period: str = None) -> pd.DataFrame:
            Fetch results for specified output variables across all simulations.

        get_results_runtable(output_variables: List[str],
                            aggregation_func: Union[str, Callable] = 'last',
                            time_period: str = None) -> pd.DataFrame:
            Fetch results for specified output variables across all simulations from the run table.
    """

    def __init__(self, db_path: str, pcse_input_dir: str):
        self.db_path = db_path
        self.pcse_input_dir = pcse_input_dir
        self.simulations: Dict[int, SiteSimulation] = {}
        self.observed_values: Dict[int, float] = {}
        self.run_table: pd.DataFrame = None
        self.max_workers = 64
        self.chunk_size = 2  # Default chunk size, can be adjusted
        self.failed_sims = []
        self.setting_path = None
        self.logs = None
        self.residue_df = None

        # if Obs_IDS_with_residue_removal.csv exists in db_path read that file

        residue_file = os.path.join(
            os.path.dirname(self.db_path), "Obs_IDS_with_residue_removal.csv"
        )

        if os.path.exists(residue_file):
            self.residue_df = pd.read_csv(residue_file)
        self.manage_settings(
            {
                "# ZEROFY": False,
                "# LOG_LEVEL_FILE": '"ERROR"',
            }
        )

    def manage_settings(self, new_settings):
        # Possible locations for the settings file
        locations = [
            os.path.expanduser("~/.pcse/user_settings.py"),
            "/tmp/.pcse/user_settings.py",
        ]

        # Find the settings file
        settings_file = None
        for loc in locations:
            if os.path.exists(loc):
                settings_file = loc
                # find the dir of the settings file
                self.setting_path = os.path.dirname(settings_file)
                break

        if settings_file is None:
            raise FileNotFoundError(
                "settings.py file not found in the specified locations."
            )

        # Read the current settings
        with open(settings_file) as file:
            content = file.read()

        # Create a pattern to match the setting
        for setting_name, new_value in new_settings.items():
            # Create a pattern to match the setting
            pattern = re.compile(rf"^{setting_name}\s*=\s*(.*?)$", re.MULTILINE)

            # Check if the setting exists
            if not pattern.search(content):
                print(
                    f"Warning: Setting '{setting_name}' not found in the file. It will be added."
                )
                content += f"\n{setting_name} = {new_value}\n"
            else:
                # Replace the old value with the new one
                content = pattern.sub(f"{setting_name} = {new_value}", content)

            print(f"Setting '{setting_name}' updated to '{new_value}'")

        # Write the updated content back to the file
        with open(settings_file, "w") as file:
            file.write(content)

        # look for the log files in the output directory + /log
        log_dir = os.path.join(self.setting_path, "logs")
        # remove all the log files in the log directory
        for file in os.listdir(log_dir):
            print(f"Removing log file: {file}")
            os.remove(os.path.join(log_dir, file))

        print(f"All settings updated in {settings_file}")

    def extract_ids(self, paths):
        # grab the file name from the path received
        soil_file_name = paths.get("soil", "").split("/")[-1]
        soil_match = re.search(
            r"^-?[\d.]+_-?[\d.]+_([a-zA-Z0-9-_]+)_\d+_soils\.yaml$", soil_file_name
        )
        soil_id = soil_match.group(1) if soil_match else None

        # grab the file name from the path received
        exp_file_name = paths.get("experiment", "").split("/")[-1]
        # print(f"exp_file_name: {exp_file_name}, paths: {paths['experiment']}")
        # remove _events.yaml
        experiment_id = exp_file_name.replace("_events.yaml", "")

        return {"soil_id": soil_id, "experiment_id": experiment_id}

    def load_data_from_json(
        self, json_path: str, filter_conditions: Dict[str, Any] = None
    ):
        """
        Load simulation configurations from a JSON file with optional filtering.

        Args:
            json_path (str): Path to the JSON configuration file
            filter_conditions (Dict[str, Any], optional): Dictionary containing filter conditions:
                - management_zone: Filter by specific zone
                - ensemble_type: Filter by ensemble type (baseline/project)
                - soil_pattern: Filter by soil file pattern
        """

        # Load JSON configuration
        with open(json_path) as f:
            config_data = json.load(f)

        # Initialize empty lists to build run_table DataFrame
        duplicates_sim = []
        run_records = []
        initial_count = 0

        # Process each management zone
        for mgmt_zone, zone_data in config_data.items():
            site_info = zone_data["Site"]
            # # collect all the run IDs from runs
            # baseline_ids = [baseline_run["ID"] for baseline_run in zone_data['runs']['baseline'].items()]
            # project_ids = [project_run["ID"] for project_run in zone_data['runs']['project'].items()]
            # # combine the two lists
            # all_ids = baseline_ids + project_ids

            # Check management zone filter
            if filter_conditions and "management_zone" in filter_conditions:
                if mgmt_zone != filter_conditions["management_zone"]:
                    continue

            # Process runs for each ensemble type
            for ensemble_type, ensemble_runs in zone_data["runs"].items():
                # Check ensemble type filter
                if filter_conditions and "ensemble_type" in filter_conditions:
                    if ensemble_type != filter_conditions["ensemble_type"]:
                        continue

                # Process individual runs
                for run_id, run_config in ensemble_runs.items():
                    initial_count += 1

                    # Check soil pattern filter
                    if filter_conditions and "soil_pattern" in filter_conditions:
                        if filter_conditions["soil_pattern"] not in run_config["soil"]:
                            continue

                    if filter_conditions and "Thread_id" in filter_conditions:
                        if run_config["ID"] not in filter_conditions["Thread_id"]:
                            continue

                    soil_file = run_config["soil"]
                    baseline_file = run_config["experiment"]
                    id_parts = self.extract_ids(
                        {"soil": soil_file, "experiment": baseline_file}
                    )
                    # Create unique observation ID
                    obs_id = f"{ensemble_type}|{id_parts['experiment_id']}|{id_parts['soil_id']}"

                    # if the same id is already in the simulations dict then skip it
                    if obs_id in self.simulations.keys():
                        duplicates_sim.append(obs_id)
                        continue

                    # if the soil sample in the run_config is empty then skip it
                    if len(run_config["soil_sample "]) == 0:
                        continue

                    # Build record for run_table
                    record = {
                        "Obs_ID": obs_id,
                        "Site": mgmt_zone,
                        "SiteID": site_info["MnGZoneID"],
                        "treatment_schedule": ensemble_type,
                        "soil_file": run_config["soil"],
                        "experiment_file": run_config["experiment"],
                        "location": site_info["location"],
                        "soil_sample": run_config["soil_sample "],
                        "start_year": 2020,
                        "start_doy": 1,
                        "end_year": 2023,
                        "end_doy": 365,
                    }
                    run_records.append(record)

                    # Create SiteSimulation object
                    sim = SiteSimulation(
                        obs_id=obs_id,
                        site=mgmt_zone,
                        treatment_schedule=ensemble_type,
                        experiment_id=run_id,
                    )

                    # Add additional information to simulation object
                    sim.run_info = record
                    sim.siteID = site_info["MnGZoneID"]
                    sim.RunType = SimulationRunType.CAR1459
                    print(
                        f"Obs_ID: {obs_id} - Site: {mgmt_zone} - Treatment: {ensemble_type} - Experiment: {run_id}"
                    )
                    print(run_config["soil_sample "])
                    print("-------------------------------------------------")
                    sim.additionality_date = pd.to_datetime(
                        run_config["soil_sample "][0]["additionality_start"]
                    )
                    # Store simulation object
                    self.simulations[obs_id] = sim

                    # Handle residue data if available
                    if (
                        self.residue_df is not None
                        and obs_id in self.residue_df["Obs_ID"].values
                    ):
                        sim.residue_removal = self.residue_df[
                            self.residue_df["Obs_ID"] == obs_id
                        ]

        # Create run_table DataFrame
        self.run_table = pd.DataFrame(run_records)
        self.RunType = SimulationRunType.CAR1459
        # Report results
        after_filter_row_count = len(self.run_table)
        # Report duplicates
        if duplicates_sim:
            print(
                f"++ {len(duplicates_sim)} duplicate simulations found: \n {duplicates_sim}"
            )
        # Report the number of simulations created
        print(
            f"Initial # of runs: {initial_count} - Rows after applying filters: {after_filter_row_count} - "
            f"Number of simulations created: {len(self.simulations)}"
        )

    def load_data_from_db(self, filter_conditions: List[Dict[str, Any]] = None):
        """
        Load data from the database with optional filtering.

        :param filter_conditions: A list of dictionaries, each containing:
                                  - 'column': The column name to filter on
                                  - 'operator': The comparison operator ('==', '!=', '>', '<', '>=', '<=', 'in')
                                  - 'value': The value to compare against
        """
        # Load site information and run table from SQLite database
        conn = sqlite3.connect(self.db_path)

        # Load experiments table
        experiments_df = pd.read_sql_query("SELECT * FROM Experiments", conn)

        # Load sites table
        sites_df = pd.read_sql_query("SELECT * FROM Sites", conn)

        # Load run_table
        run_table_df = pd.read_sql_query("SELECT * FROM run_table_obs", conn)
        initial_row_count = len(run_table_df)

        # Join experiments with sites
        experiments_sites_df = experiments_df.merge(sites_df, on="SiteID", how="inner")

        # Join the result with run_table
        self.run_table = run_table_df.merge(
            experiments_sites_df,
            left_on=["Site", "treatment_schedule"],
            right_on=["SiteName", "ExperimentName"],
            how="inner",
        )

        # Apply filtering if conditions are provided
        if filter_conditions:
            for condition in filter_conditions:
                column = condition["column"]
                operator = condition["operator"]
                value = condition["value"]

                if column in self.run_table.columns:
                    if operator == "==":
                        self.run_table = self.run_table[self.run_table[column] == value]
                    elif operator == "!=":
                        self.run_table = self.run_table[self.run_table[column] != value]
                    elif operator == ">":
                        self.run_table = self.run_table[self.run_table[column] > value]
                    elif operator == "<":
                        self.run_table = self.run_table[self.run_table[column] < value]
                    elif operator == ">=":
                        self.run_table = self.run_table[self.run_table[column] >= value]
                    elif operator == "<=":
                        self.run_table = self.run_table[self.run_table[column] <= value]
                    elif operator == "in":
                        self.run_table = self.run_table[
                            self.run_table[column].isin(value)
                        ]
                    else:
                        print(
                            f"Warning: Unsupported operator '{operator}'. Ignoring this filter condition."
                        )
                else:
                    print(
                        f"Warning: Column '{column}' not found in run table. Ignoring this filter condition."
                    )

        after_filter_row_count = len(self.run_table)

        # Create SiteSimulation objects
        for _, row in self.run_table.iterrows():
            sim = SiteSimulation(
                obs_id=row["Obs_ID"],
                site=row["Site"],
                treatment_schedule=row["treatment_schedule"],
                experiment_id=row["ExperimentID"],
            )
            sim.run_info = row
            sim.siteID = row["SiteID"]
            sim.RunType = SimulationRunType.CALIBRATION_VALIDATION
            # try to add the duration , which is an indication of how long the simulation will run to the simulation object
            start_date = pd.to_datetime(
                f"{sim.run_info['start_year']}-01-01"
            ) + pd.Timedelta(days=int(sim.run_info["start_doy"]) - 1)
            end_date = pd.to_datetime(
                f"{sim.run_info['end_year']}-01-01"
            ) + pd.Timedelta(days=int(sim.run_info["end_doy"]) - 1)
            # find the difference between the start and end dates in days
            sim.duration = (end_date - start_date).days
            self.simulations[row["Obs_ID"]] = sim
            # if the residue_df is not none and row['Obs_ID'] is in the residue_df return that row and put it in the simulation object

            if (
                self.residue_df is not None
                and row["Obs_ID"] in self.residue_df["Obs_ID"].values
            ):
                sim.residue_removal = self.residue_df[
                    self.residue_df["Obs_ID"] == row["Obs_ID"]
                ]

            if "Value" in row:
                self.observed_values[row["Obs_ID"]] = row["Value"]

        conn.close()
        self.RunType = SimulationRunType.CALIBRATION_VALIDATION
        # Report the results
        print(
            f"Initial # of rows: {initial_row_count} - Rows after applying filters: {after_filter_row_count} - Number of simulations created: {len(self.simulations)}"
        )

    @staticmethod
    def run_single_simulation(sim_id, simulation):
        try:
            return sim_id, simulation.run_simulation()
        except Exception as e:
            print(
                f"Simulation for experiment {sim_id} generated an exception: {str(e)}"
            )
            return sim_id, None

    def setup_chunk(self, data_dir: str, verbose: bool, chunk):
        results = []
        for sim_id, simulation in chunk:
            simulation.setup_simulation(self.pcse_input_dir, data_dir, verbose)
            results.append((sim_id, simulation))
        return results

    def setup_seq_all_simulations(self, data_dir: str, verbose: bool = False):
        # this just reads the input files and sets up the simulations parameters
        for simulation in tqdm(
            self.simulations.values(), desc="Setting up simulations"
        ):
            simulation.setup_simulation(self.pcse_input_dir, data_dir, verbose)

    def setup_all_simulations(self, data_dir: str, verbose: bool = False):
        cpu_count = mp.cpu_count()
        chunk_size = 1  # max(1, len(self.simulations) // (cpu_count * 10))  # Adjust this factor as needed
        chunks = [
            list(self.simulations.items())[i : i + chunk_size]
            for i in range(0, len(self.simulations), chunk_size)
        ]

        setup_func = partial(self.setup_chunk, data_dir, verbose)

        with mp.Pool(processes=cpu_count) as pool:
            results = list(
                tqdm(
                    pool.imap(setup_func, chunks),
                    total=len(chunks),
                    desc="Setting up simulation chunks",
                )
            )

        # Flatten results and update simulations
        self.simulations = dict([item for sublist in results for item in sublist])

    def initialize_simulation(
        self, params=None, experiment_specific=False, verbose=False
    ):
        print(f"Initializing simulations with params: {params}")
        if self.db_path != "":
            conn = sqlite3.connect(self.db_path)
        else:
            conn = None

        if experiment_specific:
            for exp_id, param_dict in params.items():
                if exp_id in self.simulations:
                    simulation = self.simulations[exp_id]
                    self._initialize_single_simulation(
                        simulation, param_dict, conn, verbose
                    )
        else:
            for simulation in tqdm(
                self.simulations.values(), desc="Initializing simulations"
            ):
                self._initialize_single_simulation(simulation, params, conn, verbose)

        if conn is not None:
            conn.close()

    def _initialize_single_simulation(self, simulation, params, conn, verbose):
        if conn is not None:
            site_prop = pd.read_sql_query(
                "SELECT * FROM SiteProperty WHERE SiteID = ? AND Parameter LIKE 'MINERL%'",
                conn,
                params=(simulation.siteID,),
            )

            if site_prop.empty:
                totN = 100  # default value
            else:
                site_prop["Value"] = site_prop["Value"].astype(float)
                totN = site_prop["Value"].sum() * 10  # convert gN/m2 to kg/ha
        else:
            totN = 100  # default value

        if params is None:
            params = {}
        params["NAVAILI"] = totN
        simulation.initialize_simulation(params, verbose)

    def run_all_simulations_mp(
        self, observed_assim_data=None, adaptive_chunk_size: bool = True
    ):
        self.failed_sims = []

        cpu_count = self.max_workers
        print(f"Running simulations using {cpu_count} cores")

        if adaptive_chunk_size:
            # Sort simulations by duration
            sorted_sims = sorted(
                self.simulations.items(), key=lambda x: x[1].duration, reverse=True
            )

            # Calculate total duration and target duration per chunk
            total_duration = sum(sim.duration for _, sim in sorted_sims)
            target_chunk_duration = total_duration / (
                cpu_count * self.chunk_size
            )  # Aim for 4 chunks per core

            chunks = []
            current_chunk = []
            current_chunk_duration = 0

            for sim_id, sim in sorted_sims:
                if (
                    current_chunk_duration + sim.duration > target_chunk_duration
                    and current_chunk
                ):
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_chunk_duration = 0
                current_chunk.append((sim_id, sim))
                current_chunk_duration += sim.duration

            if current_chunk:
                chunks.append(current_chunk)

            print(f"Created {len(chunks)} chunks based on simulation duration")
        else:
            sim_items = list(self.simulations.items())
            chunks = [
                sim_items[i : i + self.chunk_size]
                for i in range(0, len(sim_items), self.chunk_size)
            ]
            print(f"chunk size: {self.chunk_size}")
            print(f"Running simulations in {len(chunks)} chunks")

        results = []
        pbar = tqdm(total=len(chunks), desc="Running simulation chunks")

        def update_pbar(result):
            pbar.update(1)
            results.extend(result)

        # with mp.Pool(processes=cpu_count) as pool:
        # Create a list to store all the async results
        async_results = []

        # Submit all chunks to the pool
        for chunk in chunks:
            async_result = run_simulation_chunk(chunk, observed_assim_data)
            update_pbar(async_result)
            async_results.append(async_result)

        # The pool will automatically close and join when exiting the context manager
        # But we need to wait for all tasks to complete
        # for async_result in async_results:
        #     async_result.wait()

        pbar.close()

        # Update the simulations in the main process
        for sim_id, sim in results:
            self.simulations[sim_id] = sim

        self.failed_sims = self.get_failed_simulations()

        self.logs = self.read_log_files()

        # print("All simulations completed")
        # print(f"Number of chunks: {len(chunks)}")
        # print(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.2f} simulations")
        # print(f"Average chunk duration: {sum(sum(sim.duration for _, sim in chunk) for chunk in chunks) / len(chunks):.2f} days")

    def run_all_simulations_mt(self):
        print(
            f"Running simulations using multi-threading with max {self.max_workers} workers"
        )
        mp.cpu_count()
        sim_items = list(self.simulations.items())
        # chunk_size = max(1, len(sim_items) // (cpu_count * 4))
        # chunks = [sim_items[i:i + chunk_size] for i in range(0, len(sim_items), chunk_size)]
        chunks = [
            sim_items[i : i + self.chunk_size]
            for i in range(0, len(sim_items), self.chunk_size)
        ]
        print(f"chunk size: {self.chunk_size}")
        print(f"Running simulations in {len(chunks)} chunks")

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(chunks))
        ) as executor:
            futures = [executor.submit(run_simulation_chunk, chunk) for chunk in chunks]

            for future in tqdm_func(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing chunks",
            ):
                chunk_result = future.result()
                for sim_id, sim in chunk_result:
                    self.simulations[sim_id] = sim

        print("All simulations completed")

    def modify_parameters(
        self,
        parameters: Dict[str, Any],
        experiment_specific: bool = False,
        initial_value: bool = False,
    ):
        """
        Modify parameters for all simulations or for specific experiments.

        :param parameters: Dict of parameter names and values
        :param experiment_specific: If True, parameters should be a dict of experiment IDs to parameter dicts

        Example:
        # Modify parameters for specific experiments
        #simulator.modify_parameters({
        #    1: {'param1': 0.1, 'param2': 0.2},
        #    2: {'param1': 0.3, 'param2': 0.4}
        #}, experiment_specific=True)
        """
        if experiment_specific:
            for exp_id, param_dict in parameters.items():
                if exp_id in self.simulations:
                    for param_name, value in param_dict.items():
                        self.simulations[exp_id].modify_parameter(param_name, value)
        else:
            for simulation in self.simulations.values():
                for param_name, value in parameters.items():
                    simulation.modify_parameter(
                        param_name, value, initial_value=initial_value
                    )

    def get_results(
        self,
        output_variables: List[str],
        aggregation_func: Optional[Callable] = None,
        time_period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch results for specified output variables across all simulations.
        This function handles potential broken simulations and provides detailed logging.

        :param output_variables: List of variable names to extract
        :param aggregation_func: Optional function to aggregate results (e.g., np.mean, np.sum)
        :param time_period: Optional time period for aggregation ('D' for daily, 'M' for monthly, 'Y' for yearly)
        :return: DataFrame with results for all simulations
        """
        all_results = []
        failed_simulations = []

        for exp_id, simulation in self.simulations.items():
            # print(f"Processing results for simulation {exp_id} {simulation.status}")
            if simulation.results is None:
                # print(f"Simulation {exp_id} has no results")
                failed_simulations.append(exp_id)
                continue
            # print(f"Simulation {exp_id} has {simulation.results.shape[0]} rows")
            try:
                sim_results = simulation.get_results()
                # print(f"Results for simulation {exp_id}: {sim_results.shape[0]} rows")
                if not isinstance(sim_results.index, pd.DatetimeIndex):
                    sim_results.index = pd.to_datetime(sim_results.index)

                if aggregation_func and time_period:
                    sim_results = sim_results.resample(time_period).agg(
                        aggregation_func
                    )

                result = sim_results[output_variables].copy()
                result["experiment_id"] = exp_id
                result["site"] = simulation.site
                result["treatment_schedule"] = simulation.treatment_schedule
                result["Obs_ID"] = simulation.obs_id

                all_results.append(result.reset_index())
            except Exception as e:
                print(f"Error processing results for simulation {exp_id}: {str(e)}")
                failed_simulations.append(exp_id)

        if not all_results:
            print("Warning: No valid results found.")
            return pd.DataFrame()

        combined_results = pd.concat(all_results, ignore_index=True)

        if failed_simulations:
            print(
                f"Warning: {len(failed_simulations)} simulations failed or had no results."
            )
            print(f"Failed simulation experiment IDs: {failed_simulations}")

        return combined_results

    def get_results_runtable_sampled(
        self,
        output_variables: List[str],
        aggregation_func: Union[str, Callable] = "last",
        time_period: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch results for specified output variables across all simulations and add them to the run table.

        :param output_variables: List of variable names to extract from simulations
        :param aggregation_func: Function to aggregate results. Can be:
                                 - 'last' (default): Take the last value
                                 - 'mean': Take the mean value
                                 - 'sum': Take the sum
                                 - A custom function that takes a Series and returns a single value
        :param time_period: Optional time period for aggregation ('D' for daily, 'M' for monthly, 'Y' for yearly)
                            If None, aggregation is done over the entire simulation period
        :return: DataFrame with original run table data and added simulation results
        """
        self.run_table.copy()
        failed_simulations = []
        out_simulations = []
        output_variables.append("days_from_start")

        for obs_id, simulation in self.simulations.items():
            if simulation.results is None:
                # print(f"Simulation {obs_id} has no results")
                failed_simulations.append(obs_id)
                continue

            try:
                sim_results = (
                    simulation.get_results_emulator()
                )  # This now returns filtered results
                # add days_from_start to the output variables

                # select output_variables from sim_results
                sim_results = sim_results[output_variables]

                sim_results["Obs_ID"] = obs_id
                out_simulations.append(sim_results)
            except Exception as e:
                print(f"Error processing results for simulation {obs_id}: {str(e)}")
                failed_simulations.append(obs_id)

        if failed_simulations and verbose:
            print(
                f"Warning: {len(failed_simulations)} simulations failed or had no results."
            )
            # print(f"Failed simulation Obs_IDs: {failed_simulations}")

        # print(pd.concat(out_simulations))
        # print("here")
        return pd.concat(out_simulations)

    def get_results_runtable(
        self,
        output_variables: List[str],
        aggregation_func: Union[str, Callable] = "last",
        time_period: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch results for specified output variables across all simulations and add them to the run table.

        :param output_variables: List of variable names to extract from simulations
        :param aggregation_func: Function to aggregate results. Can be:
                                 - 'last' (default): Take the last value
                                 - 'mean': Take the mean value
                                 - 'sum': Take the sum
                                 - A custom function that takes a Series and returns a single value
        :param time_period: Optional time period for aggregation ('D' for daily, 'M' for monthly, 'Y' for yearly)
                            If None, aggregation is done over the entire simulation period
        :return: DataFrame with original run table data and added simulation results
        """
        results = self.run_table.copy()
        failed_simulations = []

        for obs_id, simulation in self.simulations.items():
            if simulation.results is None:
                # print(f"Simulation {obs_id} has no results")
                failed_simulations.append(obs_id)
                continue

            try:
                sim_results = (
                    simulation.get_results_runtable()
                )  # This now returns filtered results

                if time_period:
                    sim_results = sim_results.resample(time_period).agg(
                        aggregation_func
                    )

                for var in output_variables:
                    if var in sim_results.columns:
                        start_value = sim_results[var].iloc[0]

                        if aggregation_func == "last":
                            value = sim_results[var].iloc[-1]
                        elif aggregation_func == "mean":
                            value = sim_results[var].mean()
                        elif aggregation_func == "sum":
                            value = sim_results[var].sum()
                        elif callable(aggregation_func):
                            value = aggregation_func(sim_results[var])
                        else:
                            raise ValueError(
                                f"Unsupported aggregation function: {aggregation_func}"
                            )

                        results.loc[
                            results["Obs_ID"] == obs_id, f"Simulated_{var}"
                        ] = value
                        results.loc[
                            results["Obs_ID"] == obs_id, f"Start_Simulated_{var}"
                        ] = start_value
                    else:
                        print(
                            f"Warning: Variable '{var}' not found in simulation results for Obs_ID {obs_id}"
                        )
            except Exception as e:
                print(f"Error processing results for simulation {obs_id}: {str(e)}")
                failed_simulations.append(obs_id)

        # Calculate overall statistics
        for var in output_variables:
            if f"Simulated_{var}" in results.columns and "Value" in results.columns:
                valid_results = results.dropna(subset=[f"Simulated_{var}", "Value"])
                np.sqrt(
                    np.mean(
                        (valid_results[f"Simulated_{var}"] - valid_results["Value"])
                        ** 2
                    )
                )
                np.mean(
                    np.abs(valid_results[f"Simulated_{var}"] - valid_results["Value"])
                )
                # results[f'Diff_{var}'] = results[f'Simulated_{var}'] - results['Value']

                # print(f"Statistics for {var}:")
                # print(f"RMSE: {rmse}")
                # print(f"MAE: {mae}")

        if failed_simulations and verbose:
            print(
                f"Warning: {len(failed_simulations)} simulations failed or had no results."
            )
            # print(f"Failed simulation Obs_IDs: {failed_simulations}")

        return {"results": results}

    def run_all_simulations_dask(
        self, n_workers: Optional[int] = None, threads_per_worker: int = 1
    ) -> None:
        """
        Run all simulations using Dask for distributed computing.

        :param n_workers: Number of Dask workers to use. If None, Dask will use all available cores.
        :param threads_per_worker: Number of threads per worker. Default is 1 to avoid GIL contention.
        """
        start_time = time.time()

        print(
            f"Starting Dask cluster with {n_workers or 'all available'} workers "
            f"and {threads_per_worker} threads per worker"
        )

        client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
        print(f"Dask dashboard available at: {client.dashboard_link}")

        try:
            # Convert dict to list of simulations
            simulations = list(self.simulations.values())

            print(f"Submitting {len(simulations)} simulations to Dask")
            futures = client.map(lambda sim: sim.run_simulation(), simulations)

            # Display progress bar
            progress(futures)

            results = client.gather(futures)

            # Update the simulations with results
            for sim, result in zip(simulations, results):
                sim.results = result

            print(f"All simulations completed successfully")
        except Exception as e:
            print(f"An error occurred during Dask execution: {str(e)}")
        finally:
            client.close()
            end_time = time.time()
            print(f"Total execution time: {end_time - start_time:.2f} seconds")

    def get_simulation_status_dask(self) -> dict:
        """
        Get the status of all simulations after running with Dask.

        :return: A dictionary with simulation IDs as keys and their status as values.
        """
        return {
            sim_id: "Completed" if sim.results is not None else "Not Run"
            for sim_id, sim in self.simulations.items()
        }

    def objective_function_soc(self, parameters: Dict[str, float]) -> float:
        """
        Objective function to be minimized by an external optimizer.

        Args:
            parameters (Dict[str, float]): A dictionary where keys are parameter names
                                        and values are the parameter values to evaluate.
            {'Vint': 3.19, 'Kint': 3.55, 'param_pc': 0.075}

        Returns:
            float: The metric value to be minimized (lower is better).
        """
        # Modify the model parameters
        self.modify_parameters(parameters)

        try:
            # Run the simulation with the current parameters
            self.run_all_simulations_mp()
            soc_out_df, all_rmses, obs_mean = self.prepare_SOC_df()

            # Calculate the metric to be minimized
            metric = np.mean(list(all_rmses.values()))
        except Exception as e:
            print(f"Error running simulations for parameters {parameters}: {str(e)}")
            metric = float("inf")  # Return a large value in case of error

        return metric

    def prepare_SOC_df(self, outlier_sites=[], time_period=None, extra_outputs=[]):
        """
        Prepare the SOC (Soil Organic Carbon) dataframe.

        This method retrieves the SOC results from the runtable and performs some data processing steps.
        It calculates the difference between the simulated and observed SOC for each model and calculates
        the root mean square error (RMSE) for each model. It also removes outliers from the dataframe.

        Returns:
        - soc_results_df_out_removed (pandas.DataFrame): The processed SOC dataframe with outliers removed.
        - all_rmse (dict): A dictionary containing the RMSE values for each model.
        - mean_value (float): The mean value of the 'Value' column in the processed dataframe.
        """
        sim_outs = [
            "MIMICS_SOC",
            "MillennialV2_SOC",
            "RothC_SOC",
            "Century_SOC",
            "Ens_SOC",
        ]
        out_vars = sim_outs + extra_outputs

        # Get the results
        soc_results = self.get_results_runtable(
            out_vars, aggregation_func="last", time_period=time_period
        )

        # find all the columns that start with Start_Simulated_
        start_cols = [
            col
            for col in soc_results["results"].columns
            if col.startswith("Start_Simulated_")
        ]

        if str(self.RunType) == str(SimulationRunType.CAR1459):
            final_cols = ["Simulated_" + i for i in out_vars] + [
                "Obs_ID",
                "treatment_schedule",
                "end_year",
                "start_year",
            ]
            # add the start columns to the final columns
            final_cols += start_cols

        else:
            final_cols = ["Simulated_" + i for i in out_vars] + [
                "Obs_ID",
                "SiteName",
                "treatment_schedule",
                "end_year",
                "start_year",
                "Value",
            ]

        # keep only the columns we need
        soc_results_df = soc_results["results"][final_cols]

        if self.RunType == SimulationRunType.CAR1459:
            soc_results_df_out_removed = soc_results_df[
                ~soc_results_df["Obs_ID"].isin(outlier_sites)
            ].dropna()
            return soc_results_df_out_removed, {}, np.nan
        else:
            soc_results_df_out_removed = soc_results_df[
                ~soc_results_df["SiteName"].isin(outlier_sites)
            ].dropna()
            # for each model calculate the difference between the simulated and observed SOC
            all_rmse = {}
            for sim in [
                "Simulated_MIMICS_SOC",
                "Simulated_RothC_SOC",
                "Simulated_Century_SOC",
                "Simulated_MillennialV2_SOC",
                "Simulated_Ens_SOC",
            ]:
                all_rmse[sim] = np.sqrt(
                    np.mean(
                        (
                            soc_results_df_out_removed["Value"]
                            - soc_results_df_out_removed[sim]
                        )
                        ** 2
                    )
                )

            return (
                soc_results_df_out_removed,
                all_rmse,
                np.mean(soc_results_df_out_removed["Value"]),
            )

    def get_all_sims_params(self, params=[]):
        all_data_dict = {}

        for sim in self.simulations.values():
            all_data_dict[sim.obs_id] = {
                "Obs_ID": sim.obs_id,
                "ExperimentID": sim.experiment_id,
            }
            for param in params:
                if sim.parameters is not None:
                    tmp_val = sim.parameters.get(param)
                    # if tmp_val is arry then tak the mean
                    if isinstance(tmp_val, list):
                        tmp_val = np.mean(tmp_val)
                    all_data_dict[sim.obs_id][param] = tmp_val

        all_data_dict_df = pd.DataFrame(all_data_dict).T
        # remove the index
        all_data_dict_df.reset_index(drop=True, inplace=True)
        # round the values to 2 decimal places
        all_data_dict_df = all_data_dict_df.round(3)

        return all_data_dict_df

    def get_simulation_status(self) -> Dict[int, SimulationStatus]:
        """
        Get the status of all simulations.

        :return: A dictionary with simulation IDs as keys and their status as values.
        """
        return {sim_id: sim.status for sim_id, sim in self.simulations.items()}

    def get_failed_simulations(self) -> Dict[int, SimulationStatus]:
        """
        Get all simulations that failed either in setup or run.

        :return: A dictionary with simulation IDs as keys and their status as values for failed simulations.

        """

        return {
            sim_id: sim.status
            for sim_id, sim in self.simulations.items()
            if sim.status
            in [
                SimulationStatus.SETUP_FAILED,
                SimulationStatus.RUN_FAILED,
                SimulationStatus.NO_INPUT_DIR,
            ]
        }

    def read_log_files(self, write_to_file: bool = False) -> List[str]:
        """
        Read log files from the PCSE output directory.
        """
        if self.setting_path is not None:
            # look for the log files in the output directory + /log
            log_dir = os.path.join(self.setting_path, "logs")
            # list files and read and stack them
            log_files = os.listdir(log_dir)
            log_data = []
            for file in log_files:
                with open(os.path.join(log_dir, file)) as f:
                    log_data.append(f.read())

            if write_to_file:
                with open("simulator_logs.txt", "w") as f:
                    for log in log_data:
                        f.write(log)

            return log_data
