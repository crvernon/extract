import os
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


def extract_climate_data(ds: xr.Dataset,
                         target_variable_list: list,
                         climate_lat_idx: list,
                         climate_lon_idx: list) -> dict:
    """Extract target variables for each xanthos grid cell.

    :param ds:                                      Input xarray dataset from the climate NetCDF file.
    :type ds:                                       xr.Dataset

    :param target_variable_list:                    List of variables to extract data for.
    :type target_variable_list:                     list

    :param climate_lat_idx:                         List of index values from the climate data corresponding with
                                                    xanthos grid cell latitudes.
    :type climate_lat_idx:                          list

    :param climate_lon_idx:                         List of index values from the climate data corresponding with
                                                    xanthos grid cell longitudes.
    :type climate_lon_idx:                          list

    :return:                                        A dictionary of variable to extracted data.

    """

    return {i: ds[i].values[:, climate_lat_idx, climate_lon_idx].T for i in target_variable_list}


def run_extraction(climate_file: str,
                   xanthos_reference_file: str,
                   target_variables: list,
                   output_directory: str,
                   scenario: str,
                   model: str) -> str:
    """Workhorse function to extract target variables at each xanthos grid cell and write to a compressed
    numpy array.

    :param climate_file:                            Full path with file name and extension to the input climate file.
    :type climate_file:                             str

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :param target_variables:                        List of variables to extract data for.
    :type target_variables:                         list

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
                                target_variable_list=target_variables,
                                climate_lat_idx=climate_lat_idx,
                                climate_lon_idx=climate_lon_idx)

    # create output file name from input file
    basename = os.path.splitext(os.path.basename(climate_file))[0]
    output_filename = os.path.join(output_directory, f"{scenario}__{model}__{basename}.npz")

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
                            target_variables: list,
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

    :param target_variables:                        List of variables to extract data for.
    :type target_variables:                         list

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

    # list of target variables to extract
    target_variables = ["FLDS", "FLDS_trend", "FSDS", "FSDS_trend", "Hurs", "Hurs_trend", "Huss", "Huss_trend",
                        "PRECTmmd", "PRECTmmd_trend", "Tair", "Tair_trend", "Tmax", "Tmax_trend",
                        "Tmin", "Tmin_trend", "WIND", "WIND_trend"]

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
