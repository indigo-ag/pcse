from datetime import datetime

from pcse.simulator.simulator import MultiSiteSimulator


def run_simulation_for_zone(zone: str, data_path: str) -> MultiSiteSimulator:
    print(f"\n{datetime.utcnow()} - starting in {data_path}")
    simulator = MultiSiteSimulator("", f"{data_path}")
    simulator.load_data_from_json(
        json_path=f"{data_path}/Run_map.json",
        filter_conditions={"management_zone": zone},
    )
    print(f"\n{datetime.utcnow()} - setup")
    simulator.setup_seq_all_simulations(f"{data_path}")

    init_params = {}
    # Example:
    # init_params["project|5PKxIePDYQQ|6fIUSiYCRMxRR"] = {"FHP": 0.25, "MFHP_Mill": 0.45}
    print(f"\n{datetime.utcnow()} - init")
    simulator.initialize_simulation(
        params=init_params, experiment_specific=True
    )  # initialize the simulation

    print(
        f"\n{datetime.utcnow()} - Number of simulations: {len(simulator.simulations)} - and {len(simulator.get_failed_simulations())} are failed"
    )
    for key in simulator.get_failed_simulations():
        print(
            f"{key} : {simulator.simulations[key].experiment_id} -- {simulator.simulations[key].site} -- {simulator.simulations[key].treatment_schedule} -- {simulator.simulations[key].status}"
        )

    simulator.run_all_simulations_mp(adaptive_chunk_size=False)

    print(
        f"\n{datetime.utcnow()} - Number of simulations: {len(simulator.simulations)} - and {len(simulator.get_failed_simulations())} are failed"
    )

    return simulator


if __name__ == "__main__":
    # Folder location containing the run_map.json
    run_map_path = "/tmp/pcse"
    management_zone = "AD_8Jczb5T2qtxz"
    sim = run_simulation_for_zone(management_zone, run_map_path)
