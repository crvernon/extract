import os
import calendar
import glob
import itertools
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


def get_xanthos_coordinates(xanthos_reference_file: str) -> np.ndarray:
    """Generate an array of xanthos latitude, longitude values from the input xanthos reference file.

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :returns:                                       Array of latitude, longitude values corresponding with ordered
                                                    xanthos grid cells (67,420)

    """

    # read in the xanthos reference file to a data frame
    df = pd.read_csv(xanthos_reference_file)

    # generate an array of lat, lon for xanthos land grids
    return df[["latitude", "longitude"]].values


def generate_coordinate_reference(xanthos_lat_lon: np.ndarray,
                                  climate_lat_arr: np.ndarray,
                                  climate_lon_arr: np.ndarray):
    """Create a data frame of extracted data from the source climate product to the Xanthos
    input structure.

    :param xanthos_lat_lon:                         Array of latitude, longitude values corresponding with ordered
                                                    xanthos grid cells (67,420).
    :type xanthos_lat_lon:                          np.ndarray

    :param climate_lat_arr:                         Climate latitude array associated with each latitude from Xanthos.
    :type climate_lat_arr:                          np.ndarray

    :param climate_lon_arr:                         Climate longitude array associated with each longitude from Xanthos.
    :type climate_lon_arr:                          np.ndarray

    :returns:                                       [0] list of climate latitude index values associated with each
                                                        xanthos grid cell
                                                    [1] list of climate longitude index values associated with each
                                                        xanthos grid cell

    """

    climate_lat_idx = []
    climate_lon_idx = []

    # get the climate grid index associated with each xanthos grid centroid via lat, lon
    for index, coords in enumerate(xanthos_lat_lon):
        # break out lat, lon from xanthos coordinate pairs for each grid
        lat, lon = coords

        # get the index pair in the climate data associated with xanthos coordinates
        lat_idx = np.where(climate_lat_arr == lat)[0][0]
        lon_idx = np.where(climate_lon_arr == lon)[0][0]

        # append the climate grid index associated with each lat, lon from Xanthos
        climate_lat_idx.append(lat_idx)
        climate_lon_idx.append(lon_idx)

    return climate_lat_idx, climate_lon_idx


def get_days_in_month(file_basename: str) -> list:
    """Generate a list of the number of days in each month of the record of interest including leap years.

    :param file_basename:                           Basename of file with underscore separated start year and through
                                                    year in the first and second positions.
    :type file_basename:                            str

    :returns:                                       A list of days in the month for the period of interest.

    """

    days_in_month_standard = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_in_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    split_name = file_basename.split("_")

    try:
        start_year = int(split_name[0])
    except ValueError:
        raise (
            f"Expected year in YYYY format as first position in underscore separated file name.  Received:  '{split_name[0]}'")

    try:
        through_year = int(split_name[1])
    except ValueError:
        raise (
            f"Expected year in YYYY format as second position in underscore separated file name.  Received:  '{split_name[1]}'")

    year_list = range(start_year, through_year + 1, 1)

    days_in_month = []
    for i in year_list:

        if calendar.isleap(i):
            days_in_month.extend(days_in_month_leap)
        else:
            days_in_month.extend(days_in_month_standard)

    return days_in_month


def extract_climate_data(ds: xr.Dataset,
                         target_variable_dict: dict,
                         climate_lat_idx: list,
                         climate_lon_idx: list) -> dict:
    """Extract target variables for each xanthos grid cell.

    :param ds:                                      Input xarray dataset from the climate NetCDF file.
    :type ds:                                       xr.Dataset

    :param target_variable_dict:                    Dictionary of variables to extract data for and their target units.
    :type target_variable_dict:                     dict

    :param climate_lat_idx:                         List of index values from the climate data corresponding with
                                                    xanthos grid cell latitudes.
    :type climate_lat_idx:                          list

    :param climate_lon_idx:                         List of index values from the climate data corresponding with
                                                    xanthos grid cell longitudes.
    :type climate_lon_idx:                          list

    :return:                                        A dictionary of variable to extracted data.

    """

    return {i: ds[i].values[:, climate_lat_idx, climate_lon_idx].T for i in target_variable_dict.keys()}


def run_extraction(climate_file: str,
                   xanthos_reference_file: str,
                   target_variables: dict,
                   output_directory: str,
                   scenario: str,
                   model: str) -> str:
    """Workhorse function to extract target variables at each xanthos grid cell and write to a compressed
    numpy array.

    :param climate_file:                            Full path with file name and extension to the input climate file.
    :type climate_file:                             str

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :param target_variables:                        Dict of variables to extract data for and their target units.
    :type target_variables:                         dict

    :param output_directory:                        Full path to the directory where the output file will be stored.
    :type output_directory:                         str

    :param scenario:                                Scenario name to process.
    :type scenario:                                 str

    :param model:                                   Model name to process.
    :type model:                                    str

    :returns:                                       Full path with file name and extension to the output file.

    """
    # read in climate NetCDF to an xarray dataset
    ds = xr.open_dataset(climate_file)

    # generate an array of lat, lon for xanthos land grid cells
    xanthos_lat_lon = get_xanthos_coordinates(xanthos_reference_file)

    # generate lists of lat, lon indices from the climate data associated with xanthos grid cells
    climate_lat_idx, climate_lon_idx = generate_coordinate_reference(xanthos_lat_lon=xanthos_lat_lon,
                                                                     climate_lat_arr=ds.LAT.values,
                                                                     climate_lon_arr=ds.LON.values)

    # generate a dictionary of variable to extracted array of xanthos grid cell locations
    data = extract_climate_data(ds=ds,
                                target_variable_dict=target_variables,
                                climate_lat_idx=climate_lat_idx,
                                climate_lon_idx=climate_lon_idx)

    # create output file name from input file
    basename = os.path.splitext(os.path.basename(climate_file))[0]
    output_filename = os.path.join(output_directory, f"{scenario}__{model}__{basename}.npz")

    # convert units for temperature variables from K to C
    data["Tair"] += -273.15
    data["Tair_trend"] += -273.15
    data["Tmin"] += -273.15
    data["Tmin_trend"] += -273.15
    data["Tmax"] += -273.15
    data["Tmax_trend"] += -273.15

    # convert units for precipitation from mm/day to mm/month; assumes start month of January
    days_in_month_list = get_days_in_month(basename)
    data["PRECTmmd"] *= days_in_month_list
    data["PRECTmmd_trend"] *= days_in_month_list

    # write selected data to a compressed numpy structure
    np.savez_compressed(file=output_filename,
                        flds=data["FLDS"],
                        flds_trend=data["FLDS_trend"],
                        fsds=data["FSDS"],
                        fsds_trend=data["FSDS_trend"],
                        hurs=data["Hurs"],
                        hurs_trend=data["Hurs_trend"],
                        huss=data["Huss"],
                        huss_trend=data["Huss_trend"],
                        prectmmd=data["PRECTmmd"],
                        prectmmd_trend=data["PRECTmmd_trend"],
                        tair=data["Tair"],
                        tair_trend=data["Tair_trend"],
                        tmin=data["Tmin"],
                        tmin_trend=data["Tmin_trend"],
                        tmax=data["Tmax"],
                        tmax_trend=data["Tmax_trend"],
                        wind=data["WIND"],
                        wind_trend=data["WIND_trend"])

    return output_filename


def run_extraction_parallel(data_directory: str,
                            xanthos_reference_file: str,
                            target_variables: dict,
                            output_directory: str,
                            scenario: str,
                            model: str,
                            njobs=-1):
    """Extract target variables at each xanthos grid cell and write to a compressed
    numpy array for each file in parallel.

    :param data_directory:                          Directory containing the input climate data directory structure.
    :type data_directory:                           str

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :param target_variables:                        Dictionary of variables to extract data for and their target units.
    :type target_variables:                         dict

    :param output_directory:                        Full path to the directory where the output file will be stored.
    :type output_directory:                         str

    :param scenario:                                Scenario name to process.
    :type scenario:                                 str

    :param model:                                   Model name to process.
    :type model:                                    str
    """

    # get a list of target files to process in parallel
    target_files = glob.glob(os.path.join(data_directory, scenario, model, "*_0.5_e00*_monthly.nc"))

    print(f"Processing files:  {target_files}")

    # process all files for a model and scenario in parallel
    results = Parallel(n_jobs=njobs, backend="loky")(delayed(run_extraction)(climate_file=i,
                                                                             xanthos_reference_file=xanthos_reference_file,
                                                                             target_variables=target_variables,
                                                                             output_directory=output_directory,
                                                                             scenario=scenario,
                                                                             model=model) for i in target_files)

    return results


if __name__ == "__main__":

    # task index from SLURM array to run specific scenario, model combinations
    task_id = int(sys.argv[1])

    # number of jobs per node to use for parallel processing; -1 is all
    njobs = int(sys.argv[2])

    # data directory where climate data directory structure is housed
    data_dir = sys.argv[3]

    # directory to store the outputs in
    output_directory = sys.argv[4]

    # xanthos reference file path with filename and extension
    xanthos_reference_file = sys.argv[5]

    # dict of target variables to extract with TARGET units, not native units; some require conversion in the code
    target_variables = {"FLDS": "w-per-m2",  # surface incident longwave radiation
                        "FLDS_trend": "w-per-m2",
                        "FSDS": "w-per-m2",  # surface incident shortwave radiation
                        "FSDS_trend": "w-per-m2",
                        "Hurs": "percent",  # near surface relative humidity
                        "Hurs_trend": "percent",
                        "Huss": "percent",  # near surface specific humidity,
                        "Huss_trend": "percent",
                        "PRECTmmd": "mm-per-month",  # precipitation rate (native units mm/day)
                        "PRECTmmd_trend": "mm-per-month",
                        "Tair": "degrees-C",  # near surface air temperature (native units K)
                        "Tair_trend": "degrees-C",
                        "Tmax": "degrees-C",  # monthly mean of daily maximum near surface air temperature (native units K)
                        "Tmax_trend": "degrees-C",
                        "Tmin": "degrees-C",  # monthly mean of daily minimum near surface air temperature (native units K)
                        "Tmin_trend": "degrees-C",
                        "WIND": "m-per-sec",  # near surface wind speed
                        "WIND_trend": "m-per-sec"}

    # scenario name to process; should mirror the associated directory name
    scenario_list = ['PARIS_2C', 'PFCOV.1', 'BASECOV', 'PARIS_1p5C.1', 'PARIS_1p5C', 'PARIS_2C.1', 'PFCOV']

    # list of model names to process
    model_list = ['ACCESS-ESM1-5', 'CMCC-ESM2', 'AWI-ESM-1-1-LR', 'GISS-E2-2-G', 'INM-CM5-0', 'UKESM1-0-LL',
                  'FGOALS-g3', 'IPSL-CM6A-LR', 'CanESM5', 'FIO-ESM-2-0', 'HadGEM3-GC31-MM', 'SAM0-UNICON',
                  'CNRM-ESM2-1', 'MRI-ESM2-0', 'BCC-CSM2-MR', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'EC-Earth3-Veg']

    # create cross product list of scenario, model
    scenario_model_list = [i for i in itertools.product(scenario_list, model_list)]

    # get the scenario, model to process based off of the task id
    scenario, model = scenario_model_list[task_id]

    # run extraction of climate data target variables per realization to xanthos grid cells in parallel
    results = run_extraction_parallel(data_directory=data_dir,
                                      xanthos_reference_file=xanthos_reference_file,
                                      target_variables=target_variables,
                                      output_directory=output_directory,
                                      scenario=scenario,
                                      model=model,
                                      njobs=njobs)

    print(f"Created files:  {results}")
