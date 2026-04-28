"""
BMI Forcings Engine standalone mode wrapper script.

Provides ability to run the BMI Forcings Engine pipeline in standalone mode using a single command.

example usage: python bmi_wrapper.py short_range Gage_01011000.gpkg
"""

import argparse
import os
import tempfile
import subprocess
from datetime import datetime, timedelta

import yaml
from types import SimpleNamespace
from git_util import print_git_info_all

import forcing_extraction
import esmf_creation


def execute(forcing_config_input: str, config_input: str = None):
    """
    Execute the preprocessing steps for NextGen Forcings Engine pipeline in standalone mode.

    Modules executed: ESMF Mesh Conversion and Forcing Extraction.

    This method accepts expects all the output pathways to already be hard coded into
    the forcing configuration input file, which is te requriment for this driver script.
    It handles mesh conversion and forcing extraction using the specified parameters.

    :param forcing_config_input: Path to forcing engine configuration file for forecast run
    :param config_input: Optional path to the wrapper config file.
    :return: None
    """
    print_git_info_all()

    # Read in the configuration file to access paths and settings
    if config_input:
        config_read = config_input
    else:
        config_read = './wrapper_config.yml'

    with open(config_read, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Read in forcing engine configuration file
    with open(forcing_config_input, 'r') as forcing_config_file:
        forcing_config = yaml.safe_load(forcing_config_file)

    # Wrap config dict into simplenamespace to match esmf creation ConfigOptions format
    esmf_cfg = SimpleNamespace(geopackage=forcing_config['Geopackage'],
                               geogrid=forcing_config['GeogridIn'],
                               parquet=config['global']['parquet_path'])

    # Wrap config dict into simplenamespace to match forcing extraction ConfigOptions format
    extract_cfg = SimpleNamespace(b_date_proc=datetime.strptime(forcing_config['RefcstBDateProc'], "%Y%m%d%H%M"),
                                  input_forcings=forcing_config['InputForcings'],
                                  supp_precip_forcings=forcing_config['SuppPcp'],
                                  input_force_dirs=forcing_config['InputForcingDirectories'],
                                  supp_precip_dirs=forcing_config['SuppPcpDirectories'],
                                  fcst_input_horizons=forcing_config['ForecastInputHorizons'],
                                  cfsv2EnsMember=forcing_config['cfsEnsNumber'],
                                  ana_flag=forcing_config['AnAFlag'],
                                  look_back=forcing_config['LookBack'])

    # Create mesh file
    esmf_creation.create_mesh(esmf_cfg)

    # Extract forcing
    forcing_extraction.retrieve_forcing(extract_cfg)

def main():
    """
    Main function to handle command-line execution.

    This function parses command-line arguments and calls the execute() method.
    It allows the script to be run both programmatically or from the command line.

    :return: None
    """
    # Parse command-line arguments
    args = get_options()

    # Call execute with parsed arguments
    execute(forcing_config_input=args.forcing_config_input,config_input=args.config_input)


def get_options():
    """
    Function to accept and parse arguments.

    This function handles the command-line argument parsing and returns the parsed arguments.

    :return: An argparse.Namespace object containing the parsed arguments
    """
    # TODO keyword arguments should start with --
    parser = argparse.ArgumentParser()
    parser.add_argument('forcing_config_input',
                        type=str,
                        help='Path to forcing engine configuration file for forecast run')
    parser.add_argument('-config_input',
                        type=str,
                        help='Path to wrapper config file. If omitted, defaults to ./wrapper_config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    main()
