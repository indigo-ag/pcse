import itertools
import os
from collections import defaultdict


def _extract_soil_id(filename: str) -> str:
    # Remove the '_soils.yaml' suffix
    base = filename.removesuffix("_soils.yaml")
    # Take the last 13 characters of the remaining string
    return base[-13:]


def extract_ids(paths):
    # grab the file name from the path received
    soil_file_name = paths.get("soil", "").split("/")[-1]
    soil_id = _extract_soil_id(soil_file_name)

    # grab the file name from the path received
    exp_file_name = paths.get("experiment", "").split("/")[-1]
    # print(f"exp_file_name: {exp_file_name}, paths: {paths['experiment']}")
    # remove _events.yaml
    experiment_id = exp_file_name.replace("_events.yaml", "")

    return {"soil_id": soil_id, "experiment_id": experiment_id}


def analyze_cse_directory(root_dir, soil_sample_df):
    # Initialize nested defaultdict
    cse_files = defaultdict(
        lambda: {
            "Site": {"MnGZoneID": "", "location": ""},
            "soils": set(),
            "weather": set(),
            "baseline": set(),
            "events": set(),
            "runs": {"baseline": {}, "project": {}},
        }
    )

    # Walk through the directory
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name.startswith("__"):
                continue

            dir_path = os.path.join(root, dir_name)
            # Temporary lists to store files for creating combinations
            soils_files = []
            weather_files = []
            baseline_files = []
            events_files = []

            # List all files in the directory
            for file_path in os.listdir(dir_path):
                # Set Site information
                if "events.yaml" in file_path and "baseline" not in file_path:
                    site_id = file_path.replace("_events.yaml", "")
                    location = dir_name.replace("_", ",")

                    cse_files[dir_name]["Site"]["MnGZoneID"] = site_id
                    cse_files[dir_name]["Site"]["location"] = location

                # add the directory name to the file path
                file_path = os.path.join(dir_path.split("/")[-1], file_path)
                # Categorize files
                if "_soils.yaml" in file_path:
                    cse_files[dir_name]["soils"].add(file_path)
                    soils_files.append(file_path)
                elif "_weather.csv" in file_path:
                    cse_files[dir_name]["weather"].add(file_path)
                    weather_files.append(file_path)
                elif "baseline" in file_path.lower():
                    cse_files[dir_name]["baseline"].add(file_path)
                    baseline_files.append(file_path)
                elif "events.yaml" in file_path:
                    cse_files[dir_name]["events"].add(file_path)
                    if "baseline" not in file_path:
                        events_files.append(file_path)

            # Create runs combinations
            # Baseline runs
            for i, combo in enumerate(
                itertools.product(soils_files, baseline_files, weather_files)
            ):
                soil_file, baseline_file, weather_file = combo
                id_parts = extract_ids({"soil": soil_file, "experiment": baseline_file})
                soil_id_part = id_parts["soil_id"]
                experiment_id = id_parts["experiment_id"]
                # filter soil_sample_df on POINT_ID and look for the soil_id_part
                soil_sample_df_filtered = soil_sample_df[
                    soil_sample_df["POINT_ID"] == soil_id_part
                ]
                soil_sample_df_filtered = soil_sample_df_filtered.to_dict(
                    orient="records"
                )
                cse_files[dir_name]["runs"]["baseline"][i] = {
                    "soil": soil_file,
                    "experiment": baseline_file,
                    "weather": weather_file,
                    "soil_sample ": soil_sample_df_filtered,
                    "ID": f"baseline|{experiment_id}|{soil_id_part}",
                }

            # Project runs
            for i, combo in enumerate(
                itertools.product(soils_files, events_files, weather_files)
            ):
                soil_file, events_file, weather_file = combo
                id_parts = extract_ids({"soil": soil_file, "experiment": events_file})
                soil_id_part = id_parts["soil_id"]
                experiment_id = id_parts["experiment_id"]
                # filter soil_sample_df on POINT_ID and look for the soil_id_part
                soil_sample_df_filtered = soil_sample_df[
                    soil_sample_df["POINT_ID"] == soil_id_part
                ]
                soil_sample_df_filtered = soil_sample_df_filtered.to_dict(
                    orient="records"
                )

                cse_files[dir_name]["runs"]["project"][i] = {
                    "soil": soil_file,
                    "experiment": events_file,
                    "weather": weather_file,
                    "soil_sample ": soil_sample_df_filtered,
                    "ID": f"project|{experiment_id}|{soil_id_part}",
                }

    # Convert defaultdict to regular dict and sets to sorted lists
    result = {}
    for dir_name, categories in cse_files.items():
        result[dir_name] = {
            "Site": categories["Site"],
            "runs": categories["runs"],
            **{
                category: sorted(list(files))
                for category, files in categories.items()
                if files and category not in ["Site", "runs"]
            },
        }

    return result
