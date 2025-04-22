# if type is not in cultivation_methods keys, find the code based on the mixing and depth
# if there is, then that means a mixed, custom defined tillage is applied
def get_tillage_code_params(mixing, depth, type, par="cultra"):
    """
    Get the tillage code parameters based on the mixing, depth, type, and optional parameter.

    Parameters:
    mixing (float): The mixing factor.
    depth (float): The depth of tillage.
    type (str): The type of tillage.
    par (str, optional): The optional parameter. Defaults to "cultra".

    Returns:
    list: A list containing the tillage code parameter and the tillage factor key.
    """
    if type not in cultivation_methods.keys():
        cult_factor = mixing * depth / 200
        # find in what range does cult_factor fall in cult_factor_ranges dictionary
        for key, value in cult_factor_ranges.items():
            if value[0] <= cult_factor <= value[1]:
                cult_factor_key = key
                break
    else:
        cult_factor_key = type

    cultivation_methods_params = cultivation_methods.get(cult_factor_key)
    cult_param = cultivation_methods_params.get(par)

    return [cult_param, cult_factor_key]


cult_factor_ranges = {
    "A": (0.001, 0.02),
    "B": (0.02, 0.06),
    "C": (0.06, 0.12),
    "D": (0.12, 0.17),
    "E": (0.17, 0.22),
    "F": (0.22, 0.26),
    "G": (0.26, 0.31),
    "H": (0.31, 0.38),
    "I": (0.38, 0.41),
    "J": (0.41, 0.60),
    "K": (0.60, 1.00),
}

cultivation_methods = {
    "A": {
        "name": "RodWeed",
        "cultra": [0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2],
        "clteff": [1.1872, 1.1872, 1.00, 1.1872],
        "notes": "century 1.066",
    },
    "B": {
        "name": "Planters and Cultivators",
        "cultra": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0],
        "clteff": [2.5028, 2.5028, 1.00, 2.5028],
        "notes": "century 1.521",
    },
    "C": {
        "name": "Field Cultivators and Planters",
        "cultra": [0.0, 0.0, 0.0, 0.1, 0.5, 0.17, 0.0],
        "clteff": [3.4414, 3.4414, 1.00, 3.4414],
        "notes": "century 1.843",
    },
    "D": {
        "name": "Field and Row Cultivators",
        "cultra": [0.0, 0.4, 0.6, 0.1, 0.6, 0.25, 0.7],
        "clteff": [4.4814, 4.4814, 1.00, 4.4814],
        "notes": "century 2.203",
    },
    "E": {
        "name": "SWEEPS and TANDEM DISKS",
        "cultra": [0.0, 0.0, 0.0, 0.1, 0.6, 0.3, 0.0],
        "clteff": [5.1171, 5.1171, 1.00, 5.1171],
        "notes": "century 2.426",
    },
    "F": {
        "name": "Field Cultivator and Tandem Disk",
        "cultra": [0.0, 0.05, 0.2, 0.05, 0.2, 0.37, 0.3],
        "clteff": [6.0843, 6.0843, 1.00, 6.0843],
        "notes": "century 2.762",
    },
    "G": {
        "name": "Multiple Tandem",
        "cultra": [0, 0.05, 0.65, 0.05, 0.65, 0.45, 0.7],
        "clteff": [7.1529, 7.1529, 1.000, 7.1529],
        "notes": "century 3.132",
    },
    "H": {
        "name": "DisksPoint Chisel Tandem Disk",
        "cultra": [0.0, 0.05, 0.65, 0.05, 0.65, 0.53, 0.8],
        "clteff": [8.3086, 8.3086, 1.00, 8.3086],
        "notes": "century 3.529",
    },
    "I": {
        "name": "Offset and Tandem Disks",
        "cultra": [0.0, 0.05, 0.2, 0.4, 0.6, 0.6, 0.44],
        "clteff": [9.19, 9.19, 1.00, 9.19],
        "notes": "century 3.837",
    },
    "J": {
        "name": "Pint Chisel Offset Disk",
        "cultra": [0.0, 0.08, 0.92, 0.08, 0.92, 0.92, 1.0],
        "clteff": [13.6243, 13.6243, 1.00, 13.6243],
        "notes": "century 5.372",
    },
    "K": {
        "name": "Moldboard Plow",
        "cultra": [0.0, 0.05, 0.95, 0.05, 0.95, 0.95, 1.0],
        "clteff": [14, 14, 1.00, 14],
        "notes": "century 5.5",
    },
    "CRMP": {
        "name": "CRIMPER",
        "cultra": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "clteff": [1.0942, 1.0942, 1.00, 1.0942],
        "notes": "century 1.066",
    },
    "CSG": {
        "name": "75%_aboveground_biomass_harvested",
        "cultra": [1.00, 0.00, 0.0, 0.75, 0.0, 0.0, 0.0],
        "clteff": [1, 1, 1, 1],
    },
    "CTIL": {
        "name": "Conventional Tillage",
        "cultra": [0.0, 0.05, 0.95, 0.05, 0.95, 0.95, 1.0],
        "clteff": [9.74, 9.74, 1.00, 9.74],
    },
    "CULT": {
        "name": "CULTIVATOR",
        "cultra": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        "clteff": [2.69, 2.69, 1.00, 2.69],
    },
    "DRILL": {
        "name": "DRILL",
        "cultra": [0.05, 0.05, 0.1, 0.05, 0.15, 0.2, 0.2],
        "clteff": [1.13, 1.13, 1.1, 1.13],
    },
    "HERB": {
        "name": "HERBICIDE",
        "cultra": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "IK": {
        "name": "Offset and Tandem Disks modified for 100% kill",
        "cultra": [0.05, 0.45, 0.5, 0.4, 0.6, 0.6, 1.0],
        "clteff": [6.67, 6.67, 1.00, 6.67],
        "notes": "century 3.837",
    },
    "KILL": {
        "name": "End crop growth",
        "cultra": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "MOW": {
        "name": "Mow event for Ram",
        "cultra": [0.0, 0.75, 0.0, 0.9, 0.0, 0.0, 0.0],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "NDRIL": {
        "name": "NO-TILL-DRILL",
        "cultra": [0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 0.1],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "NOTILL": {
        "name": "New Notill Drill",
        "cultra": [0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        "clteff": [1.16, 1.16, 1.00, 1.16],
        "notes": "50% mixing * 20% coverage * 32% (2.5 in) of 20cm depth",
    },
    "P": {
        "name": "PLOWING",
        "cultra": [0.0, 0.1, 0.9, 0.1, 0.9, 0.9, 1.0],
        "clteff": [10.0, 10.0, 1.00, 10.0],
    },
    "PINE": {
        "name": "pineapple (slip removal)",
        "cultra": [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "R": {
        "name": "RODWEEDER",
        "cultra": [0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 1.0],
        "clteff": [1.13, 1.13, 1.00, 1.13],
    },
    "ROOTH": {
        "name": "Root Harvest (beets, potatoes)",
        "cultra": [0.0, 0.95, 0.10, 0.90, 0.10, 0.20, 1.00],
        "clteff": [2.00, 2.00, 1.00, 2.00],
    },
    "ROW": {
        "name": "ROW-CULTIVATOR",
        "cultra": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        "clteff": [3.41, 3.41, 1.00, 3.41],
    },
    "S": {
        "name": "SWEEP",
        "cultra": [0.7, 0.25, 0.05, 0.1, 0.1, 0.1, 1.0],
        "clteff": [3.85, 3.85, 1.0, 3.85],
    },
    "SHRD": {
        "name": "Shredder",
        "cultra": [0.1, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "SUMF": {
        "name": "Summer Fallow (HERBICIDE)",
        "cultra": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
        "clteff": [1.0, 1.0, 1.0, 1.0],
    },
    "W": {
        "name": "WORM_MIXING",
        "cultra": [0.0, 0.00, 0.00, 0.00, 0.00, 0.50, 0.0],
        "clteff": [1.000, 1.000, 1.000, 1.000],
    },
}
