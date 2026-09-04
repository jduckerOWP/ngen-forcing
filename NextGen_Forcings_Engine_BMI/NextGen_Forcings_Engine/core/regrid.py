"""Regridding module file for regridding input forcing files."""

import functools
import hashlib
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from time import monotonic, time
from pathlib import Path
from .. import os_utils

import shapely
from mpi4py.futures import MPICommExecutor

try:
    import esmpy as ESMF
except ImportError:
    import ESMF

import logging

from ..esmf_utils import (
    esmf_field_retry,
    esmf_grid_retry,
    esmf_mesh_retry,
    esmf_regrid_retry,
    esmf_regridfromfile_retry,
    esmf_regridobj_call_retry,
)

import dask
import dask.delayed
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from . import (
    err_handler,
    ioMod,
    timeInterpMod,
)
from .config import (
    ConfigOptions,
)
from .geoMod import (
    GeoMetaWrfHydro,
)
from .parallel import MpiConfig


# Get log level string from environment variable (defaults to 'INFO' if unset)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

# Convert string ('DEBUG', 'INFO', 'WARNING', etc.) to logging level constant
log_level = logging.getLevelName(log_level_str)

# Fallback check in case an invalid string was passed in the environment variable
if not isinstance(log_level, int):
    log_level = logging.INFO

# Configure logging directly to stdout
logging.basicConfig(
    level=log_level,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOG = logging.getLogger()

if "WGRIB2" not in os.environ:
    WGRIB2_env = False
else:
    WGRIB2_env = True

NETCDF = "NETCDF"
GRIB2 = "GRIB2"

next_file_number = 0


@contextmanager
def timing_block(step_str: str):
    """Context manager for timing code execution."""
    start = time()
    yield
    end = time()
    LOG.debug(f"  Execution time for {step_str}: {round(end - start, 2)} seconds")


def mkfilename():
    """Create a unique filename suffix."""
    global next_file_number
    next_file_number += 1
    return f"{next_file_number}"


def static_vars(**kwargs):
    """Add static variables to a function."""

    def decorate(func):
        """Add static variables to a function."""
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def create_link(name, input_file, tmpFile, config_options, mpi_config):
    """Create a symbolic link to the input file for processing."""
    if mpi_config.rank == 0:
        try:
            config_options.statusMsg = name + " file being used: " + input_file
            err_handler.log_msg(config_options, mpi_config, True)

            os.symlink(input_file, tmpFile)
        except:
            config_options.errMsg = (
                "Unable to create link: " + input_file + " to: " + tmpFile
            )
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def get_mrms_subhourly_avg(config_options, supplemental_precip, mpi_config):

    supplemental_precip.file_in1_grib2 = []
    for mrms_gzfile in supplemental_precip.file_in1:
        mrms_grib2_name = mrms_gzfile.stem
        mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
        ioMod.unzip_file(mrms_gzfile, mrms_tmp_grib2, config_options, mpi_config)
        supplemental_precip.file_in1_grib2.append(mrms_tmp_grib2)

    if len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0:
        supplemental_precip.file_in2_grib2 = []
        for mrms_gzfile in supplemental_precip.file_in2:
            mrms_grib2_name = mrms_gzfile.stem
            mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
            ioMod.unzip_file(mrms_gzfile, mrms_tmp_grib2, config_options, mpi_config)
            supplemental_precip.file_in2_grib2.append(mrms_tmp_grib2)

    mrms_grib2_name = supplemental_precip.file_in1_bc.stem
    mrms_hourly_tmp1_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
    ioMod.unzip_file(supplemental_precip.file_in1_bc, mrms_hourly_tmp1_grib2, config_options, mpi_config)

    if len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0:
        mrms_grib2_name = supplemental_precip.file_in2_bc.stem
        mrms_hourly_tmp2_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
        ioMod.unzip_file(supplemental_precip.file_in2_bc, mrms_hourly_tmp2_grib2, config_options, mpi_config)

    ds_subhourly = xr.open_mfdataset(supplemental_precip.file_in1_grib2, engine="cfgrib", concat_dim="time", combine="nested")
    ds_subhourly['precip_rate'] = ds_subhourly['unknown'].where(ds_subhourly['unknown'] >= 0, np.nan)
    subhourly_15m = ds_subhourly.resample(time="15min", closed="left").mean(dim="time")
    subhourly_15m_depth = subhourly_15m['precip_rate'] * 0.25
    subhourly_accumulation = subhourly_15m_depth.sum(dim="time", skipna=True)

    ds_hourly = xr.open_mfdataset(mrms_hourly_tmp1_grib2, engine="cfgrib")
    mrms_hourly = ds_hourly["unknown"].where(ds_hourly["unknown"] >= 0, np.nan)

    bias_ratio = mrms_hourly / (subhourly_accumulation + 1e-05)

    meaningful_rain_mask = (mrms_hourly > 0.1) & (subhourly_accumulation > 0.1)

    bounded_bias_ratio = bias_ratio.where(meaningful_rain_mask, 1.0)
    bounded_bias_ratio = bounded_bias_ratio.clip(min=0.1, max=10.0)

    bias_per_15min_depth = gaussian_filter(bounded_bias_ratio.compute().values, sigma=5)

    bias_depth_xr = xr.DataArray(bias_per_15min_depth, coords=[ds_subhourly.latitude.compute().values, ds_subhourly.longitude.compute().values], dims=["latitude", "longitude"])

    corrected_15m_depth = subhourly_15m_depth * bias_depth_xr

    subhourly_15m_tmp1 = corrected_15m_depth.where(corrected_15m_depth >= 0, 0.0)

    if len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0:
        ds_subhourly = xr.open_mfdataset(supplemental_precip.file_in2_grib2, engine="cfgrib", concat_dim="time", combine="nested")
        ds_subhourly['precip_rate'] = ds_subhourly['unknown'].where(ds_subhourly['unknown'] >= 0, np.nan)
        subhourly_15m = ds_subhourly.resample(time="15min", closed="left").mean(dim="time")
        subhourly_15m_depth = subhourly_15m['precip_rate'] * 0.25
        subhourly_accumulation = subhourly_15m_depth.sum(dim="time", skipna=True)

        ds_hourly = xr.open_mfdataset(mrms_hourly_tmp2_grib2, engine="cfgrib")
        mrms_hourly = ds_hourly["unknown"].where(ds_hourly["unknown"] >= 0, np.nan)

        bias_ratio = mrms_hourly / (subhourly_accumulation + 1e-05)

        meaningful_rain_mask = (mrms_hourly > 0.1) & (subhourly_accumulation > 0.1)

        bounded_bias_ratio = bias_ratio.where(meaningful_rain_mask, 1.0)
        bounded_bias_ratio = bounded_bias_ratio.clip(min=0.1, max=10.0)

        bias_per_15min_depth = gaussian_filter(bounded_bias_ratio.compute().values, sigma=5)

        bias_depth_xr = xr.DataArray(bias_per_15min_depth, coords=[ds_subhourly.latitude.compute().values, ds_subhourly.longitude.compute().values], dims=["latitude", "longitude"])

        corrected_15m_depth = subhourly_15m_depth * bias_depth_xr

        subhourly_15m_tmp2 = corrected_15m_depth.where(corrected_15m_depth >= 0, 0.0)

        subhourly_15m_tmp1 = subhourly_15m_tmp1.isel(time=supplemental_precip.tmpFile1_mrms_subhourly_timesteps)
        subhourly_15m_tmp2 = subhourly_15m_tmp2.isel(time=supplemental_precip.tmpFile2_mrms_subhourly_timesteps)

        combined_subhourly = xr.concat([subhourly_15m_tmp1, subhourly_15m_tmp2], dim="time")

        mrms_subhourly_final = combined_subhourly.mean(dim="time", skipna=True).compute().values

        del subhourly_15m_tmp2
        del combined_subhourly

    else:
        mrms_subhourly_final = subhourly_15m_tmp1.mean(dim="time", skipna=True).compute().values

    del ds_subhourly
    del subhourly_15m
    del subhourly_15m_depth
    del subhourly_accumulation
    del mrms_hourly
    del ds_hourly
    del bias_ratio
    del bounded_bias_ratio
    del meaningful_rain_mask
    del bias_per_15min_depth
    del bias_depth_xr
    del corrected_15m_depth
    del subhourly_15m_tmp1

    for f in supplemental_precip.file_in1_grib2:
        if os.path.isfile(f):
            try:
                os_utils.os_remove_retry(f)
            except OSError:
                config_options.errMsg = f"Unable to remove scratch file: {f}"
                LOG.warning(config_options.errMsg)

    if os.path.isfile(mrms_hourly_tmp1_grib2):
        try:
            os_utils.os_remove_retry(mrms_hourly_tmp1_grib2)
        except OSError:
            config_options.errMsg = f"Unable to remove scratch file: {mrms_hourly_tmp1_grib2}"
            LOG.warning(config_options.errMsg)

    if len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0:
        for f in supplemental_precip.file_in2_grib2:
            if os.path.isfile(f):
                try:
                    os_utils.os_remove_retry(f)
                except OSError:
                    config_options.errMsg = f"Unable to remove scratch file: {f}"
                    LOG.warning(config_options.errMsg)

        if os.path.isfile(mrms_hourly_tmp2_grib2):
            try:
                os_utils.os_remove_retry(mrms_hourly_tmp2_grib2)
            except OSError:
                config_options.errMsg = f"Unable to remove scratch file: {mrms_hourly_tmp2_grib2}"
                LOG.warning(config_options.errMsg)

    return mrms_subhourly_final


def get_subhourly_avg(input_forcings, var, id_tmp, id_tmp2):

    if len(input_forcings.file_in2_indices) != 0:

        ds1 = xr.open_dataset(id_tmp)
        ds2 = xr.open_dataset(id_tmp2)

        da1_var = ds1[var]
        da2_var = ds2[var]

        da1_subset = da1_var.isel(time=input_forcings.file_in1_indices)
        da2_subset = da2_var.isel(time=input_forcings.file_in2_indices)

        combined_series = xr.concat([da1_subset, da2_subset], dim='time')

        if var != "APCP":
            spatial_avg = combined_series.mean(dim='time', keep_attrs=True)
        else:
            spatial_avg = combined_series.sum(dim='time', keep_attrs=True)

        spatial_avg_computed = spatial_avg.compute().values

        ds1.close()
        ds2.close()
        da1_subset.close()
        da2_subset.close()
        combined_series.close()
        spatial_avg.close()

    else:
        ds1 = xr.open_dataset(id_tmp)

        da1_var = ds1[var]

        da1_subset = da1_var.isel(time=input_forcings.file_in1_indices)

        if var != "APCP":
            spatial_avg = da1_subset.mean(dim='time', keep_attrs=True)
        else:
            spatial_avg = da1_subset.sum(dim='time', keep_attrs=True)

        spatial_avg_computed = spatial_avg.compute().values

        ds1.close()
        da1_subset.close()
        spatial_avg.close()

    return spatial_avg_computed


@dask.delayed
def compute(id_tmp, nc_var):
    """Compute masked array for a given NetCDF variable."""
    return id_tmp[nc_var].to_masked_array()


def regrid_ak_ext_ana(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Read in and regrid Alaska ExtAna data."""
    esmf_grid_retry_partial = functools.partial(
        esmf_grid_retry, mpi_config, config_options, err_handler
    )
    esmf_mesh_retry_partial = functools.partial(
        esmf_mesh_retry, mpi_config, config_options, err_handler
    )

    ds = None

    try:
        if not os.path.isfile(input_forcings.file_in2):
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "No AK AnA in_2 file found for this timestep."
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            return

        if input_forcings.regridComplete:
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "No AK AnA regridding required for this timestep."
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            return

        if mpi_config.rank == 0:
            from netCDF4 import Dataset

            try:
                ds = Dataset(input_forcings.file_in2, "r")
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to open input NetCDF file: {input_forcings.file_in2} ({e})"
                )
                err_handler.log_critical(config_options, mpi_config)

            ds.set_auto_scale(True)
            ds.set_auto_mask(False)

        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            if mpi_config.rank == 0:
                input_forcings.ny_global = ds.dimensions["y"].size
                input_forcings.nx_global = ds.dimensions["x"].size

            input_forcings.ny_global = mpi_config.broadcast_parameter(
                input_forcings.ny_global, config_options, param_type=int
            )
            err_handler.check_program_status(config_options, mpi_config)
            input_forcings.nx_global = mpi_config.broadcast_parameter(
                input_forcings.nx_global, config_options, param_type=int
            )
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                try:
                    input_forcings.esmf_grid_in = esmf_grid_retry_partial(
                        np.array([input_forcings.ny_global, input_forcings.nx_global]),
                        staggerloc=ESMF.StaggerLoc.CENTER,
                        coord_sys=ESMF.CoordSys.SPH_DEG,
                    )
                except ESMF.ESMPyException as esmf_error:
                    config_options.errMsg = f"Unable to create source ESMF grid from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.x_lower_bound = (
                        input_forcings.esmf_grid_in.lower_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][1]
                    )
                    input_forcings.x_upper_bound = (
                        input_forcings.esmf_grid_in.upper_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][1]
                    )
                    input_forcings.y_lower_bound = (
                        input_forcings.esmf_grid_in.lower_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][0]
                    )
                    input_forcings.y_upper_bound = (
                        input_forcings.esmf_grid_in.upper_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][0]
                    )
                    input_forcings.nx_local = (
                        input_forcings.x_upper_bound - input_forcings.x_lower_bound
                    )
                    input_forcings.ny_local = (
                        input_forcings.y_upper_bound - input_forcings.y_lower_bound
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract local X/Y boundaries from global grid from netCDF file: {input_forcings.file_in} ({str(err)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.esmf_grid_in = esmf_mesh_retry_partial(
                        filename=config_options.geogrid,
                        filetype=ESMF.FileFormat.ESMFMESH,
                    )
                    input_forcings.esmf_grid_in_elem = esmf_mesh_retry_partial(
                        filename=config_options.geogrid,
                        filetype=ESMF.FileFormat.ESMFMESH,
                    )
                except ESMF.ESMPyException as esmf_error:
                    config_options.errMsg = f"Unable to create source ESMF Mesh from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.nx_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.ny_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.nx_local_elem = len(
                        input_forcings.esmf_grid_in_elem.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.ny_local_elem = len(
                        input_forcings.esmf_grid_in_elem.esmf_grid_in.coords[0][1]
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract local X/Y boundaries from global mesh file: {input_forcings.file_in} ({str(err)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.esmf_grid_in = esmf_mesh_retry_partial(
                        filename=config_options.geogrid,
                        filetype=ESMF.FileFormat.ESMFMESH,
                    )
                except ESMF.ESMPyException as esmf_error:
                    config_options.errMsg = f"Unable to create source ESMF Mesh from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.nx_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.ny_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract local X/Y boundaries from global mesh file: {input_forcings.file_in} ({str(err)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            force_count = 9 if config_options.include_lqfrac else 8

            if config_options.grid_type == "gridded":
                input_forcings.regridded_forcings1 = np.empty(
                    [force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                    np.float32,
                )
                input_forcings.regridded_forcings2 = np.empty(
                    [force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                    np.float32,
                )
            elif config_options.grid_type == "unstructured":
                input_forcings.regridded_forcings1 = np.empty(
                    [force_count, wrf_hydro_geo_meta.ny_local], np.float32
                )
                input_forcings.regridded_forcings2 = np.empty(
                    [force_count, wrf_hydro_geo_meta.ny_local], np.float32
                )
                input_forcings.regridded_forcings1_elem = np.empty(
                    [force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32
                )
                input_forcings.regridded_forcings2_elem = np.empty(
                    [force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32
                )

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            var_tmp = None
            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = f"Processing input AK AnA variable: {nc_var} from {input_forcings.file_in2}"
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                LOG.debug(f"{config_options.statusMsg}")
                if config_options.grid_type == "gridded":
                    try:
                        var_tmp = ds.variables[nc_var][0, :, :]
                        var_tmp = np.float32(var_tmp)

                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    try:
                        var_tmp = ds.variables[nc_var][0, :]
                        var_tmp = np.float32(var_tmp)

                        var_tmp_elem = ds.variables[nc_var][0, :]
                        var_tmp_elem = np.float32(var_tmp_elem)
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    try:
                        var_tmp = ds.variables[nc_var][0, :]
                        var_tmp = np.float32(var_tmp)

                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract ExtAnA forcing data from the AK AnA field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_tmp[wrf_hydro_geo_meta.mesh_inds]
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_tmp[wrf_hydro_geo_meta.mesh_inds_elem]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract ExtAnA forcing data from the AK AnA mesh field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_tmp[wrf_hydro_geo_meta.mesh_inds]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract ExtAnA forcing data from the AK AnA mesh field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
    finally:
        if mpi_config.rank == 0 and ds is not None:
            try:
                ds.close()
            except OSError:
                config_options.errMsg = (
                    f"Unable to close input NetCDF file: {input_forcings.file_in2}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

def _regrid_ak_ext_ana_pcp_stage4(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid Alaska ExtAna supplemental Stage IV precip data."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.exists(supplemental_precip.file_in1):
        return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "No StageIV regridding required for this timestep."
            )
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"STAGEIV_AK_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    stage4_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    lat_var = "latitude"
    lon_var = "longitude"

    id_tmp = None
    try:
        if supplemental_precip.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(stage4_tmp_nc):
                    config_options.statusMsg = (
                        "Found old temporary file: "
                        + stage4_tmp_nc
                        + " - Removing....."
                    )
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(stage4_tmp_nc)
                    except OSError:
                        config_options.errMsg = (
                            f"Unable to remove temporary file: {stage4_tmp_nc}"
                        )
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "APCP:surface:0-6 hour acc fcst" {supplemental_precip.file_in2} -netcdf {stage4_tmp_nc}'
            else:
                cmd = "APCP:surface:0-6 hour acc fcst"

            if mpi_config.rank == 0:
                config_options.statusMsg = f"WGRIB2 command: {cmd}"
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            id_tmp = ioMod.open_grib2(
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "STAGEIV-PCP",
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                stage4_tmp_nc, config_options, mpi_config, False, lat_var, lon_var
            )

        calc_regrid_flag = check_supp_pcp_regrid_status(
            id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating STAGE IV regridding weights."
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            calculate_supp_pcp_weights(
                supplemental_precip,
                id_tmp,
                stage4_tmp_nc,
                config_options,
                mpi_config,
                lat_var,
                lon_var,
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp_elem = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                supplemental_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = (
                    supplemental_precip.regridded_precip2[:, :]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = (
                    esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj_elem,
                        supplemental_precip.esmf_field_in_elem,
                        supplemental_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[
                    np.where(supplemental_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = (
                supplemental_precip.esmf_field_out_elem.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2_elem
                    != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2_elem[ind_valid] = (
                    supplemental_precip.regridded_precip2_elem[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = (
                    supplemental_precip.regridded_precip2_elem[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {stage4_tmp_nc}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(stage4_tmp_nc)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {stage4_tmp_nc}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {stage4_tmp_nc}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_ak_ext_ana_pcp(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid Alaska ExtAna supplemental precip data."""
    if supplemental_precip.ext_ana == "STAGE4":
        supplemental_precip.netcdf_var_names.append("APCP_surface")
        _regrid_ak_ext_ana_pcp_stage4(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()
    else:
        supplemental_precip.netcdf_var_names.append(
            "MultiSensorQPE01H_0mabovemeansealevel"
        )
        regrid_mrms_hourly(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()


def _regrid_conus_ext_ana_pcp_stage4(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid Alaska ExtAna supplemental Stage IV precip data."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.exists(supplemental_precip.file_in1):
        return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "No StageIV regridding required for this timestep."
            )
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"STAGEIV_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    stage4_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    lat_var = "latitude"
    lon_var = "longitude"

    id_tmp = None
    try:
        if supplemental_precip.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(stage4_tmp_nc):
                    config_options.statusMsg = (
                        "Found old temporary file: "
                        + stage4_tmp_nc
                        + " - Removing....."
                    )
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(stage4_tmp_nc)
                    except OSError:
                        config_options.errMsg = (
                            f"Unable to remove temporary file: {stage4_tmp_nc}"
                        )
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "APCP:surface:0-1 hour acc fcst" {supplemental_precip.file_in2} -netcdf {stage4_tmp_nc}'
            else:
                cmd = "APCP:surface:0-1 hour acc fcst"

            if mpi_config.rank == 0:
                config_options.statusMsg = f"WGRIB2 command: {cmd}"
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            id_tmp = ioMod.open_grib2(
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "STAGEIV-PCP",
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                stage4_tmp_nc, config_options, mpi_config, False, lat_var, lon_var
            )

        calc_regrid_flag = check_supp_pcp_regrid_status(
            id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating STAGE IV regridding weights."
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            calculate_supp_pcp_weights(
                supplemental_precip,
                id_tmp,
                stage4_tmp_nc,
                config_options,
                mpi_config,
                lat_var,
                lon_var,
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :].data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp_elem = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :].data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                supplemental_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract precipitation from STAGE IV file: "
                        + supplemental_precip.file_in1
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 < 1e10
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                invalid = np.where(supplemental_precip.regridded_precip2 > 1e10)
                supplemental_precip.regridded_precip2[invalid] = (
                    config_options.globalNdv
                )
                del invalid
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = (
                    supplemental_precip.regridded_precip2[:, :]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 < 1e10
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                invalid = np.where(supplemental_precip.regridded_precip2 > 1e10)
                supplemental_precip.regridded_precip2[invalid] = (
                    config_options.globalNdv
                )
                del invalid
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = (
                    esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj_elem,
                        supplemental_precip.esmf_field_in_elem,
                        supplemental_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[
                    np.where(supplemental_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = (
                supplemental_precip.esmf_field_out_elem.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2_elem
                    != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2_elem[ind_valid] = (
                    supplemental_precip.regridded_precip2_elem[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = (
                    supplemental_precip.regridded_precip2_elem[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place STAGE IV precipitation into local ESMF field: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to run mask search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = (
                    "Unable to run NDV search on STAGE IV supplemental precipitation: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {stage4_tmp_nc}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(stage4_tmp_nc)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {stage4_tmp_nc}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {stage4_tmp_nc}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_conus_ext_ana_pcp(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid CONUS ExtAna supplemental precip data."""
    if supplemental_precip.ext_ana == "STAGE4":
        supplemental_precip.netcdf_var_names.append("APCP_surface")
        _regrid_conus_ext_ana_pcp_stage4(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()
    else:
        supplemental_precip.netcdf_var_names.append(
            "MultiSensorQPE01H_0mabovemeansealevel"
        )
        regrid_mrms_hourly(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()


def regrid_conus_hrrr(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid CONUS HRRR data."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in1) and input_forcings.product_name == "HRRR_15min":
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR file_in1 file found for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    if not os.path.isfile(input_forcings.file_in2):
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR file_in2 file found for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"HRRR_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    if input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
        file_name = f"HRRR_CONUS_TMP2-{mkfilename()}.nc"
        file_uuid = str(mpi_config.uid64)
        input_forcings.tmpFile2 = str(
            Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
        )

    id_tmp = None
    id_tmp2 = None
    try:
        config_options.statusMsg = "Regrid CONUS HRRR"
        err_handler.log_msg(config_options, mpi_config)

        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0 and os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = (
                    f"Found old temporary file: {input_forcings.tmpFile} - Removing..."
                )
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    config_options.errMsg = (
                        f"Unable to remove temporary file: {input_forcings.tmpFile}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0 and input_forcings.tmpFile2 is not None and os.path.isfile(input_forcings.tmpFile2) and input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                config_options.statusMsg = (
                    f"Found old temporary file: {input_forcings.tmpFile2} - Removing..."
                )
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile2)
                except OSError:
                    config_options.errMsg = (
                        f"Unable to remove temporary file: {input_forcings.tmpFile2}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Converting HRRR Variable: {grib_var}"
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                if 0 < input_forcings.cycle_freq < 60:
                    time_str = (
                        f"{input_forcings.fcst_min1}-{input_forcings.fcst_min2} min acc fcst"
                        if grib_var == "APCP"
                        else f"{input_forcings.fcst_min2} min fcst"
                    )
                    sub_rem = int(input_forcings.fcst_min1) % 60
                    sub_id = int(sub_rem / input_forcings.cycle_freq)
                else:
                    time_str = (
                        f"{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst"
                        if grib_var == "APCP"
                        else f"{input_forcings.fcst_hour2} hour fcst"
                    )
                if input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                    fields.append(
                        ":("
                        + grib_var
                        + "):("
                        + input_forcings.grib_levels[force_count]
                        + "):"
                    )
                else:
                    fields.append(
                        ":"
                        + grib_var
                        + ":"
                        + input_forcings.grib_levels[force_count]
                        + ":"
                        + time_str
                        + ":"
                    )
            fields.append(":(HGT):(surface):")

            if WGRIB2_env:
                pattern = "|".join(fields)
                cmd = f'$WGRIB2 -match "({pattern})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                if WGRIB2_env:
                    pattern = "|".join(fields)
                    cmd = f'$WGRIB2 -match "({pattern})" {input_forcings.file_in1} -netcdf {input_forcings.tmpFile2}'
                else:
                    cmd = "(" + "|".join(fields) + ")"

                id_tmp2 = ioMod.open_grib2(
                    input_forcings.file_in1,
                    input_forcings.tmpFile2,
                    cmd,
                    config_options,
                    mpi_config,
                    inputVar=None,
                    special_case=False,
                )

                err_handler.check_program_status(config_options, mpi_config)

        else:
            create_link(
                "HRRR",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing HRRR Variable: " + grib_var
                err_handler.log_msg(
                    config_options, mpi_config, True
                )

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating HRRR regridding weights."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp = get_subhourly_avg(input_forcings,"HGT_surface",input_forcings.tmpFile2,input_forcings.tmpFile)
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HRRR elevation from "
                                + input_forcings.tmpFile
                                + ": "
                                + str(err)
                            )

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place input NetCDF HRRR data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid HRRR surface elevation using ESMF: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform HRRR mask search on elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract regridded HRRR elevation data from ESMF: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp = get_subhourly_avg(input_forcings,"HGT_surface",input_forcings.tmpFile2,input_forcings.tmpFile)
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HRRR elevation from "
                                + input_forcings.tmpFile
                                + ": "
                                + str(err)
                            )

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place input NetCDF HRRR data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid HRRR surface elevation using ESMF: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform HRRR mask search on elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract regridded HRRR elevation data from ESMF: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp_elem = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp_elem = get_subhourly_avg(input_forcings,"HGT_surface",input_forcings.tmpFile2,input_forcings.tmpFile)
                            else:
                                var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HRRR elevation from "
                                + input_forcings.tmpFile
                                + ": "
                                + str(err)
                            )

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place input NetCDF HRRR data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out_elem = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid HRRR surface elevation using ESMF: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform HRRR mask search on elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract regridded HRRR elevation data from ESMF: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp = get_subhourly_avg(input_forcings,"HGT_surface",input_forcings.tmpFile2,input_forcings.tmpFile)
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HRRR elevation from "
                                + input_forcings.tmpFile
                                + ": "
                                + str(err)
                            )

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place input NetCDF HRRR data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid HRRR surface elevation using ESMF: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform HRRR mask search on elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract regridded HRRR elevation data from ESMF: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Processing input HRRR variable: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp = get_subhourly_avg(input_forcings,input_forcings.netcdf_var_names[force_count],input_forcings.tmpFile2,input_forcings.tmpFile)
                        else:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (
                                100 - var_tmp[var_tmp >= 0]
                            ) / 100
                            var_tmp[var_tmp < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place input HRRR data into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input HRRR Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input HRRR forcing data: " + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to perform mask test on regridded HRRR forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract regridded HRRR forcing data from the ESMF field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]

                mask[:] = prev_mask

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Processing input HRRR variable: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp = get_subhourly_avg(input_forcings,input_forcings.netcdf_var_names[force_count],input_forcings.tmpFile2,input_forcings.tmpFile)
                        else:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (
                                100 - var_tmp[var_tmp >= 0]
                            ) / 100
                            var_tmp[var_tmp < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place input HRRR data into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input HRRR Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input HRRR forcing data: " + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to perform mask test on regridded HRRR forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract regridded HRRR forcing data from the ESMF field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Processing input HRRR variable: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp_elem = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp_elem = get_subhourly_avg(input_forcings,input_forcings.netcdf_var_names[force_count],input_forcings.tmpFile2,input_forcings.tmpFile)
                        else:
                            var_tmp_elem = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp_elem /= 3600
                        if grib_var == "CPOFP":
                            var_tmp_elem[var_tmp_elem >= 0] = (
                                100 - var_tmp_elem[var_tmp_elem >= 0]
                            ) / 100
                            var_tmp_elem[var_tmp_elem < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask_elem = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask_elem = np.copy(mask_elem)
                if grib_var == 'CPOFP':
                    mask_elem[np.isnan(var_sub_tmp_elem)] = 0

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place input HRRR data into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input HRRR Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out_elem = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input HRRR forcing data: " + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out_elem.data[np.isnan(input_forcings.esmf_field_out_elem.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to perform mask test on regridded HRRR forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract regridded HRRR forcing data from the ESMF field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]

                mask[:] = prev_mask
                mask_elem[:] = prev_mask_elem

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Processing input HRRR variable: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp = get_subhourly_avg(input_forcings,input_forcings.netcdf_var_names[force_count],input_forcings.tmpFile2,input_forcings.tmpFile)
                        else:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (
                                100 - var_tmp[var_tmp >= 0]
                            ) / 100
                            var_tmp[var_tmp < 0] = np.nan    
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place input HRRR data into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input HRRR Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input HRRR forcing data: " + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to perform mask test on regridded HRRR forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract regridded HRRR forcing data from the ESMF field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]

                mask[:] = prev_mask

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0 and id_tmp2 is not None:
            try:
                id_tmp2.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {input_forcings.tmpFile2} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile2)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {input_forcings.tmpFile2}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {input_forcings.tmpFile2} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

def regrid_conus_rap(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid CONUS RAP 13km data."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No RAP regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"RAP_CONUS_BGRB-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    file_name = f"RAP_CONUS_PGRB-{mkfilename()}.nc"
    input_forcings.tmpFile2 = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        config_options.statusMsg = "Regrid CONUS RAP"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(input_forcings.tmpFile):
                    config_options.statusMsg = (
                        "Found old temporary file: "
                        + input_forcings.tmpFile
                        + " - Removing....."
                    )
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(input_forcings.tmpFile)
                    except OSError:
                        config_options.errMsg = (
                            "Unable to remove file: " + input_forcings.tmpFile
                        )
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Converting CONUS RAP Variable: " + grib_var
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                time_str = (
                    "{}-{} hour acc fcst".format(
                        input_forcings.fcst_hour1, input_forcings.fcst_hour2
                    )
                    if grib_var in ("APCP")
                    else str(input_forcings.fcst_hour2) + " hour fcst"
                )
                fields.append(
                    ":"
                    + grib_var
                    + ":"
                    + input_forcings.grib_levels[force_count]
                    + ":"
                    + time_str
                    + ":"
                )
            fields.append(":(HGT):(surface):")

            fields.append(":(CFRZR):(surface):")
            fields.append(":(CICEP):(surface):")
            fields.append(":(CSNOW):(surface):")
            fields.append(":(CRAIN):(surface):")

            if input_forcings.t2dDownscaleOpt == 3:
                fields.append(":(HGT):(12 hybrid level):")
                fields.append(":(TMP):(12 hybrid level):")

            if WGRIB2_env:
                cmd = (
                    '$WGRIB2 -match "('
                    + "|".join(fields)
                    + ')" '
                    + input_forcings.file_in2
                    + " -netcdf "
                    + input_forcings.tmpFile
                )

                cmd2 = (
                    '$WGRIB2 -match "('
                    + "|".join(fields)
                    + ')" '
                    + input_forcings.file_in2.replace("bgrb", "pgrb")
                    + " -netcdf "
                    + input_forcings.tmpFile2
                )
            else:
                cmd = "(" + "|".join(fields) + ")"
                cmd2 = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)

            id_tmp2 = ioMod.open_grib2(
                input_forcings.file_in2.replace("bgrb", "pgrb"),
                input_forcings.tmpFile2,
                cmd2,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)

        else:
            create_link(
                "RAP",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

            create_link(
                "RAP",
                input_forcings.file_in2.replace("bgrb", "pgrb"),
                input_forcings.tmpFile2,
                config_options,
                mpi_config,
            )
            id_tmp2 = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile2, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing Conus RAP Variable: " + grib_var
                err_handler.log_msg(
                    config_options, mpi_config, True
                )

            if grib_var != "LQFRAC":
                calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
                err_handler.check_program_status(config_options, mpi_config)
            else:
                calc_regrid_flag = False

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating RAP regridding weights."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from : "
                                + id_tmp
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place temporary RAP elevation variable into ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on RAP elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place RAP ESMF elevation field into local array: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from : "
                                + id_tmp
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place temporary RAP elevation variable into ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on RAP elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place RAP ESMF elevation field into local array: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from : "
                                + id_tmp
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place temporary RAP elevation variable into ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out_elem = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on RAP elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place RAP ESMF elevation field into local array: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from : "
                                + id_tmp
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place temporary RAP elevation variable into ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on RAP elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place RAP ESMF elevation field into local array: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'LQFRAC':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local RAP array into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input RAP Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask calculation on RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                        input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local gridded array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'LQFRAC':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local RAP array into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input RAP Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask calculation on RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = \
                        input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local unstructured node array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp_elem = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp_elem = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp_elem)
                        else:
                            var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp_elem /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask_elem = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask_elem = np.copy(mask_elem)
                if grib_var == 'LQFRAC':
                    mask_elem[np.isnan(var_sub_tmp_elem)] = 0

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local RAP array into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input RAP Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out_elem = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out_elem.data[np.isnan(input_forcings.esmf_field_out_elem.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask calculation on RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = \
                        input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF element data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]

                mask[:] = prev_mask
                mask_elem[:] = prev_mask_elem

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'LQFRAC':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local RAP array into ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input RAP Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv

                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50

                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask calculation on RAP variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = \
                        input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local hydrofabric elem array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]

                mask[:] = prev_mask

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0 and id_tmp2 is not None:
            try:
                id_tmp2.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {input_forcings.tmpFile2} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile2)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {input_forcings.tmpFile2}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {input_forcings.tmpFile2} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_cfsv2(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid global CFSv2 forecast data."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No need to read in new CFSv2 data at this time."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"CFSv2_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        config_options.statusMsg = "Regrid CFSv2"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(input_forcings.tmpFile):
                    config_options.statusMsg = (
                        "Found old temporary file: "
                        + input_forcings.tmpFile
                        + " - Removing....."
                    )
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(input_forcings.tmpFile)
                    except OSError as err:
                        config_options.errMsg = (
                            "Unable to remove previous temporary file: "
                            + input_forcings.tmpFile
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting CFSv2 Variable: " + grib_var
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                fields.append(
                    ":"
                    + grib_var
                    + ":"
                    + input_forcings.grib_levels[force_count]
                    + ":"
                    + str(input_forcings.fcst_hour2)
                    + " hour fcst:"
                )
            fields.append(":(HGT):(surface):")

            if WGRIB2_env:
                cmd = (
                    '$WGRIB2 -match "('
                    + "|".join(fields)
                    + ')" '
                    + input_forcings.file_in2
                    + " -netcdf "
                    + input_forcings.tmpFile
                )
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "CFSv2",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing CFSv2 Variable: " + grib_var
                err_handler.log_msg(
                    config_options, mpi_config, True
                )

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculate CFSv2 regridding weights."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )

                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from file: "
                                + input_forcings.file_in2
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 elevation data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = (
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )

                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract CFSv2 regridded elevation data from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from file: "
                                + input_forcings.file_in2
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 elevation data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = (
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )

                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract CFSv2 regridded elevation data from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from file: "
                                + input_forcings.file_in2
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 elevation data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = (
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )

                    try:
                        input_forcings.esmf_field_out_elem = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract CFSv2 regridded elevation data from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract HGT_surface from file: "
                                + input_forcings.file_in2
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 elevation data into the ESMF field object: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = (
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )

                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract CFSv2 regridded elevation data from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = (
                            "Regridding CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from file: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1 is None
                    ):
                        input_forcings.coarse_input_forcings1 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    if (
                        input_forcings.coarse_input_forcings2 is None
                    ):
                        input_forcings.coarse_input_forcings2 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    try:
                        input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place local CFSv2 input variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " into local numpy array. ("
                            + str(err)
                            + ")"
                        )

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 forcing data into temporary ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(ve)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(npe)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract ESMF field data for CFSv2: " + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = config_options.globalNdv

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = (
                            "Regridding CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from file: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1 is None
                    ):
                        input_forcings.coarse_input_forcings1 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    if (
                        input_forcings.coarse_input_forcings2 is None
                    ):
                        input_forcings.coarse_input_forcings2 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    try:
                        input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place local CFSv2 input variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " into local numpy array. ("
                            + str(err)
                            + ")"
                        )

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 forcing data into temporary ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(ve)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(npe)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract ESMF field data for CFSv2: " + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = (
                            "Regridding CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from file: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1_elem is None
                    ):
                        input_forcings.coarse_input_forcings1_elem = np.empty(
                            [9, var_sub_tmp_elem.shape[0], var_sub_tmp_elem.shape[1]],
                            np.float64,
                        )

                    if (
                        input_forcings.coarse_input_forcings2_elem is None
                    ):
                        input_forcings.coarse_input_forcings2_elem = np.empty(
                            [9, var_sub_tmp_elem.shape[0], var_sub_tmp_elem.shape[1]],
                            np.float64,
                        )

                    try:
                        input_forcings.coarse_input_forcings2_elem[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place local CFSv2 input variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " into local numpy array. ("
                            + str(err)
                            + ")"
                        )

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1_elem[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2_elem[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2_elem = None
                    input_forcings.coarse_input_forcings1_elem = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 forcing data into temporary ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(ve)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(npe)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract ESMF field data for CFSv2: " + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1_elem[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = (
                            "Regridding CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from file: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1 is None
                    ):
                        input_forcings.coarse_input_forcings1 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    if (
                        input_forcings.coarse_input_forcings2 is None
                    ):
                        input_forcings.coarse_input_forcings2 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    try:
                        input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place local CFSv2 input variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " into local numpy array. ("
                            + str(err)
                            + ")"
                        )

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place CFSv2 forcing data into temporary ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(ve)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to run mask calculation on CFSv2 variable: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " ("
                            + str(npe)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract ESMF field data for CFSv2: " + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_nwm(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid custom input NetCDF hourly forcing files."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if config_options.aws:
        regrid_nwm_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
        return

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "No NWM NetCDF regridding required for this timestep."
            )
            err_handler.log_msg(config_options, mpi_config, True)
        return

    id_tmp = ioMod.open_netcdf_forcing(
        input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True
    )

    config_options.statusMsg = "Regrid NWM Custom NetCDF Forcing Variables"
    err_handler.log_msg(config_options, mpi_config)

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "Processing Custom NetCDF Forcing Variable: " + nc_var
            )
            err_handler.log_msg(config_options, mpi_config, True)
        calc_regrid_flag = check_regrid_status(
            id_tmp,
            force_count,
            input_forcings,
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
        )

        if calc_regrid_flag:
            calculate_weights(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                mpi_config,
                wrf_hydro_geo_meta,
            )

        input_forcings.height = None
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                f"Unable to locate HGT_surface in: {input_forcings.file_in2}. "
                f"Downscaling will not be available."
            )
            err_handler.log_msg(config_options, mpi_config, True)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom netCDF input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom netCDF input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom netCDF input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp_elem = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                input_forcings, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj_elem,
                    input_forcings.esmf_field_in_elem,
                    input_forcings.esmf_field_out_elem,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom netCDF input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = (
                "Unable to close NetCDF file: " + input_forcings.tmpFile
            )
            err_handler.err_out(config_options)


def regrid_nwm_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid AWS NWM Forcing Data Downloaded from Server."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "No NWM NetCDF regridding required for this timestep."
            )
            err_handler.log_msg(config_options, mpi_config, True)
        return
    mpi_config.comm.barrier()
    with MPICommExecutor(comm=mpi_config.comm, root=0) as executor:
        with dask.config.set(scheduler=executor):
            if mpi_config.rank == 0:
                id_tmp = config_options.aws_obj
            else:
                id_tmp = None
    mpi_config.comm.barrier()

    if mpi_config.rank == 0 and id_tmp is not None:
        if "x" in id_tmp.coords and "y" in id_tmp.coords:
            nwm_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            transformer = Transformer.from_crs(nwm_crs, "EPSG:4326", always_xy=True)

            x_coords, y_coords = np.meshgrid(id_tmp.x.values, id_tmp.y.values)
            lon_coords, lat_coords = transformer.transform(x_coords, y_coords)

            id_tmp = id_tmp.assign_coords(
                longitude=(["y", "x"], lon_coords), latitude=(["y", "x"], lat_coords)
            )

    config_options.statusMsg = "Regrid NWM Custom zarr Forcing Variables"
    err_handler.log_msg(config_options, mpi_config)

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "Processing Custom zarr Forcing Variable: " + nc_var
            )
            err_handler.log_msg(config_options, mpi_config, True)
        calc_regrid_flag = check_regrid_status(
            id_tmp,
            force_count,
            input_forcings,
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
        )
        if calc_regrid_flag:
            calculate_weights(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                mpi_config,
                wrf_hydro_geo_meta,
            )

        input_forcings.height = None
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                f"Unable to locate HGT_surface in: {input_forcings.file_in2}. "
                f"Downscaling will not be available."
            )
            err_handler.log_msg(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom zarr input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom zarr forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom zarr input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom zarr forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom zarr input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp_elem = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                input_forcings, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj_elem,
                    input_forcings.esmf_field_in_elem,
                    input_forcings.esmf_field_out_elem,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom zarr forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding Custom zarr input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if input_forcings.product_name == "NWM":
                var_tmp = np.asarray(var_tmp, dtype=np.float64)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input Custom zarr forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = (
                "Unable to close NetCDF file: " + input_forcings.tmpFile
            )
            err_handler.err_out(config_options)

def regrid_custom_hourly_netcdf(
    input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Regrid Custom Hourly NetCDF Forcing Data."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    with timing_block("Regrid AORC AWS"):
        if config_options.aws:
            regrid_aorc_aws(
                input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
            )
            return

    with timing_block("Regrid Custom Hourly NetCDF Forcing Data"):
        if not os.path.isfile(input_forcings.file_in2):
            return

        if input_forcings.regridComplete:
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "No Custom Hourly NetCDF regridding required for this timestep."
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            return

        id_tmp = ioMod.open_netcdf_forcing(
            input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True
        )

        fill_values = {
            "TMP": 288.0,
            "SPFH": 0.005,
            "PRES": 101300.0,
            "APCP": 0,
            "UGRD": 1.0,
            "VGRD": 1.0,
            "DSWRF": 80.0,
            "DLWRF": 310.0,
        }

        config_options.statusMsg = "Regrid Custom Hourly NetCDF Forcing Variables"
        err_handler.log_msg(config_options, mpi_config)

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Processing Custom NetCDF Forcing Variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )

            if calc_regrid_flag:
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )

                if 23 in config_options.input_forcings:
                    input_forcings.regridded_mask_AORC = input_forcings.regridded_mask
                    if config_options.grid_type == "unstructured":
                        input_forcings.regridded_mask_elem_AORC = (
                            input_forcings.regridded_mask_elem
                        )

                if "HGT_surface" in id_tmp.variables.keys():
                    if config_options.grid_type == "gridded":
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            input_forcings, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to place NetCDF elevation data into the ESMF field object: "
                                + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = (
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                            err_handler.log_msg(
                                config_options, mpi_config, True
                            )
                        try:
                            input_forcings.esmf_field_out = (
                                esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj,
                                    input_forcings.esmf_field_in,
                                    input_forcings.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            config_options.errMsg = (
                                "Unable to regrid elevation data to the WRF-Hydro domain "
                                "using ESMF: " + str(ve)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_out.data[
                                np.where(input_forcings.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = (
                                "Unable to compute mask on elevation data: " + str(npe)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:, :] = (
                                input_forcings.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract ESMF regridded elevation data to a local "
                                "array: " + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                    elif config_options.grid_type == "unstructured":
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            input_forcings, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to place NetCDF elevation data into the ESMF field object: "
                                + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = (
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                            err_handler.log_msg(
                                config_options, mpi_config, True
                            )
                        try:
                            input_forcings.esmf_field_out = (
                                esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj,
                                    input_forcings.esmf_field_in,
                                    input_forcings.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            config_options.errMsg = (
                                "Unable to regrid elevation data to the WRF-Hydro domain "
                                "using ESMF: " + str(ve)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_out.data[
                                np.where(input_forcings.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = (
                                "Unable to compute mask on elevation data: " + str(npe)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:] = (
                                input_forcings.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract ESMF regridded elevation data to a local "
                                "array: " + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp_elem = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp_elem = mpi_config.scatter_array(
                            input_forcings, var_tmp_elem, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in_elem.data[:, :] = (
                                var_sub_tmp_elem
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to place NetCDF elevation data into the ESMF field object: "
                                + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = (
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                            err_handler.log_msg(
                                config_options, mpi_config, True
                            )
                        try:
                            input_forcings.esmf_field_out_elem = (
                                esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj_elem,
                                    input_forcings.esmf_field_in_elem,
                                    input_forcings.esmf_field_out_elem,
                                )
                            )
                        except ValueError as ve:
                            config_options.errMsg = (
                                "Unable to regrid elevation data to the WRF-Hydro domain "
                                "using ESMF: " + str(ve)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_out_elem.data[
                                np.where(input_forcings.regridded_mask_elem == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = (
                                "Unable to compute mask on elevation data: " + str(npe)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height_elem[:] = (
                                input_forcings.esmf_field_out_elem.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract ESMF regridded elevation data to a local "
                                "array: " + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                    elif config_options.grid_type == "hydrofabric":
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            input_forcings, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to place NetCDF elevation data into the ESMF field object: "
                                + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = (
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                            err_handler.log_msg(
                                config_options, mpi_config, True
                            )
                        try:
                            input_forcings.esmf_field_out = (
                                esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj,
                                    input_forcings.esmf_field_in,
                                    input_forcings.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            config_options.errMsg = (
                                "Unable to regrid elevation data to the WRF-Hydro domain "
                                "using ESMF: " + str(ve)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_out.data[
                                np.where(input_forcings.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = (
                                "Unable to compute mask on elevation data: " + str(npe)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:] = (
                                input_forcings.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract ESMF regridded elevation data to a local "
                                "array: " + str(err)
                            )
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                else:
                    input_forcings.height = None
                    if mpi_config.rank == 0:
                        config_options.statusMsg = (
                            f"Unable to locate HGT_surface in: {input_forcings.file_in2}. "
                            f"Downscaling will not be available."
                        )
                        err_handler.log_msg(config_options, mpi_config)

                if mpi_config.rank != 0:
                    id_tmp.close()

            if config_options.grid_type == "gridded":
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Custom netCDF input variable: " + nc_var
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        config_options.statusMsg = (
                            f"Using {fill} to replace missing values in input"
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = (
                            "Unable to extract "
                            + nc_var
                            + " from: "
                            + input_forcings.file_in2
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local array into local ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to calculate mask from input Custom netCDF regridded forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        config_options.errMsg = (
                            "Unable to run NDV search on Custom netCDF precipitation: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local ESMF regridded data into local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Custom netCDF input variable: " + nc_var
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        config_options.statusMsg = (
                            f"Using {fill} to replace missing values in input"
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = (
                            "Unable to extract "
                            + nc_var
                            + " from: "
                            + input_forcings.file_in2
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local array into local ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to calculate mask from input Custom netCDF regridded forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        config_options.errMsg = (
                            "Unable to run NDV search on Custom netCDF precipitation: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local ESMF regridded data into local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Custom netCDF input variable: " + nc_var
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        config_options.statusMsg = (
                            f"Using {fill} to replace missing values in input"
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                        var_tmp_elem = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = (
                            "Unable to extract "
                            + nc_var
                            + " from: "
                            + input_forcings.file_in2
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local array into local ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to calculate mask from input Custom netCDF regridded forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "APCP_surface":
                    try:
                        ind_valid_elem = np.where(
                            input_forcings.esmf_field_out_elem.data != fill
                        )
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = (
                            input_forcings.esmf_field_out_elem.data[ind_valid_elem]
                            / 3600.0
                        )
                        del ind_valid_elem
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        config_options.errMsg = (
                            "Unable to run NDV search on Custom netCDF precipitation: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local ESMF regridded data into local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Custom netCDF input variable: " + nc_var
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    try:
                        config_options.statusMsg = (
                            f"Using {fill} to replace missing values in input"
                        )
                        err_handler.log_msg(
                            config_options, mpi_config, True
                        )
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = (
                            "Unable to extract "
                            + nc_var
                            + " from: "
                            + input_forcings.file_in2
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local array into local ESMF field: " + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid input Custom netCDF forcing variables using ESMF: "
                        + str(ve)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to calculate mask from input Custom netCDF regridded forcings: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "DSWRF_surface":
                    ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                    input_forcings.esmf_field_out.data[ind_valid] = 0.0
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                        input_forcings.esmf_field_out.data[ind_valid] = 0.0
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        config_options.errMsg = (
                            "Unable to run NDV search on Custom netCDF precipitation: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local ESMF regridded data into local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)
        if mpi_config.rank == 0:
            try:
                id_tmp.close()
            except OSError:
                config_options.errMsg = (
                    "Unable to close NetCDF file: " + input_forcings.tmpFile
                )
                err_handler.err_out(config_options)


def regrid_era5(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid ERA5-Interim Forcing Variables."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in1):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "No ERA5-Interim regridding required for this timestep."
            )
            err_handler.log_msg(config_options, mpi_config, True)
        return

    id_tmp = ioMod.open_netcdf_forcing(
        input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True
    )

    time = nc.num2date(
        id_tmp.variables["time"][:].data,
        units=id_tmp.variables["time"].units,
        only_use_cftime_datetimes=False,
    )
    seconds_index = np.abs(
        (
            pd.to_datetime(time) - pd.to_datetime(config_options.current_time)
        ).total_seconds()
    )
    ind = np.where(seconds_index == np.min(seconds_index))[0][0]

    config_options.statusMsg = "Regrid Custom Hourly NetCDF Forcing Variables"
    err_handler.log_msg(config_options, mpi_config)

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "Processing ERA-5 Interim Forcing Variable: " + nc_var
            )
            err_handler.log_msg(config_options, mpi_config, True)
        calc_regrid_flag = check_regrid_status(
            id_tmp,
            force_count,
            input_forcings,
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
        )
        if calc_regrid_flag:
            calculate_weights(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                mpi_config,
                wrf_hydro_geo_meta,
            )

            if "Geopotential" in id_tmp.variables.keys():
                LOG.info("Found geopotential height in ERA5-Interim data")
            else:
                input_forcings.height = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        f"Unable to locate Geopoential height in: {input_forcings.file_in2}. "
                        f"Downscaling will not be available."
                    )
                    err_handler.log_msg(config_options, mpi_config)

            if mpi_config.rank != 0:
                id_tmp.close()

        if config_options.grid_type == "gridded":
            var_tmp = None

            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding ERA5-Interim input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    config_options.statusMsg = (
                        "Using -9999. to replace missing values in input"
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input ERA5-Interim forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to calculate mask from input ERA5-Interim regridded forcings: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = (
                        "Unable to run NDV search on ERA5-Interim precipitation: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding ERA5-Interim input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    config_options.statusMsg = (
                        "Using -9999. to replace missing values in input"
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input ERA5-Interim forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to calculate mask from input ERA5-Interim regridded forcings: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = (
                        "Unable to run NDV search on ERA5-Interim precipitation: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding ERA5-Interim input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    config_options.statusMsg = (
                        "Using -9999. to replace missing values in input"
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    var_tmp_elem = id_tmp.variables[nc_var][:].filled(-9999.0)[
                        ind, :, :
                    ]
                    if nc_var == "d2m":
                        var_tmp_elem = var_tmp_elem - 273.15
                        e = 6.112 * np.exp(
                            (17.67 * var_tmp_elem) / (var_tmp_elem + 243.5)
                        )
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                input_forcings, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj_elem,
                    input_forcings.esmf_field_in_elem,
                    input_forcings.esmf_field_out_elem,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input ERA5-Interim forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem.data[
                    np.where(input_forcings.regridded_mask_elem == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to calculate mask from input Custom netCDF regridded forcings: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if nc_var == "mtpr":
                try:
                    ind_valid_elem = np.where(
                        input_forcings.esmf_field_out_elem.data != -9999.0
                    )
                    input_forcings.esmf_field_out_elem.data[ind_valid_elem] = (
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] / 3600.0
                    )
                    del ind_valid_elem
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = (
                        "Unable to run NDV search on ERA5-Interim precipitation: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out_elem.data

            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = (
                    "Regridding ERA5-Interim input variable: " + nc_var
                )
                err_handler.log_msg(
                    config_options, mpi_config, True
                )
                try:
                    config_options.statusMsg = (
                        "Using -9999. to replace missing values in input"
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = (
                        "Unable to extract "
                        + nc_var
                        + " from: "
                        + input_forcings.file_in2
                        + " ("
                        + str(err)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local array into local ESMF field: " + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = (
                    "Unable to regrid input ERA5-Interim forcing variables using ESMF: "
                    + str(ve)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = (
                    "Unable to calculate mask from input ERA5-Interim regridded forcings: "
                    + str(npe)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = (
                        "Unable to run NDV search on ERA5-Interim precipitation: "
                        + str(npe)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = (
                    "Unable to place local ESMF regridded data into local array: "
                    + str(err)
                )
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = (
                "Unable to close NetCDF file: " + input_forcings.tmpFile
            )
            err_handler.err_out(config_options)


@static_vars(last_file=None)
def regrid_gfs(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid GFS Forcing Variables."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = (
                "No 13km GFS regridding required for this timestep."
            )
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_uuid = str(mpi_config.uid64)
    file_name = f"GFS_TMP-{mkfilename()}.nc"
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    id_tmp = None
    try:
        config_options.statusMsg = "Regridding 13km GFS Variables."
        err_handler.log_msg(config_options, mpi_config)

        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0 and os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = (
                    "Found old temporary file: "
                    + input_forcings.tmpFile
                    + " - Removing....."
                )
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    config_options.errMsg = (
                        "Unable to remove file: " + input_forcings.tmpFile
                    )
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Converting 13km GFS Variable: " + grib_var
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                if grib_var == "PRATE":
                    if input_forcings.fcst_hour2 <= 384:
                        tmp_hr_current = input_forcings.fcst_hour2

                        diff_tmp = tmp_hr_current % 6 if tmp_hr_current % 6 > 0 else 6
                        tmp_hr_previous = tmp_hr_current - diff_tmp

                    else:
                        tmp_hr_previous = input_forcings.fcst_hour1

                    fields.append(
                        ":"
                        + grib_var
                        + ":"
                        + input_forcings.grib_levels[force_count]
                        + ":"
                        + str(tmp_hr_previous)
                        + "-"
                        + str(input_forcings.fcst_hour2)
                        + " hour ave fcst:"
                    )
                else:
                    fields.append(
                        ":"
                        + grib_var
                        + ":"
                        + input_forcings.grib_levels[force_count]
                        + ":"
                        + str(input_forcings.fcst_hour2)
                        + " hour fcst:"
                    )

            fields.append(":(HGT):(surface):")
            if WGRIB2_env:
                cmd = (
                    '$WGRIB2 -match "('
                    + "|".join(fields)
                    + ')" '
                    + input_forcings.file_in2
                    + " -netcdf "
                    + input_forcings.tmpFile
                )
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "GFS",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing 13km GFS Variable: " + grib_var
                err_handler.log_msg(
                    config_options, mpi_config, True
                )

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Calculating 13km GFS regridding weights."
                    )
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract GFS elevation from: "
                                + input_forcings.tmpFile
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract GFS elevation from: "
                                + input_forcings.tmpFile
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract GFS elevation from: "
                                + input_forcings.tmpFile
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = (
                                "Unable to extract GFS elevation from: "
                                + input_forcings.tmpFile
                                + " ("
                                + str(err)
                                + ")"
                            )
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place local GFS array into an ESMF field: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding 13km GFS surface elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(
                        config_options, mpi_config, True
                    )
                if config_options.grid_type == "gridded":
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid GFS elevation data: " + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on GFS elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid GFS elevation data with ESMF mesh nodes: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to place local GFS array into an ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid GFS elevation data with ESMF mesh elements: "
                            + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on GFS elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on GFS elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    try:
                        input_forcings.esmf_field_out = (
                            esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        config_options.errMsg = (
                            "Unable to regrid GFS elevation data: " + str(ve)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = (
                            "Unable to perform mask search on GFS elevation data: "
                            + str(npe)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract GFS elevation array from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract GFS elevation array from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract GFS elevation array from ESMF field with mesh elements: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract GFS elevation array from ESMF field: "
                            + str(err)
                        )
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = (
                            "Unable to extract: "
                            + input_forcings.netcdf_var_names[force_count]
                            + " from: "
                            + input_forcings.tmpFile
                            + " ("
                            + str(err)
                            + ")"
                        )
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if input_forcings.product_name == "GFS_Production_GRIB2":
                if grib_var == "PRATE":
                    if mpi_config.rank == 0:
                        if config_options.grid_type == "gridded":
                            input_forcings.globalPcpRate2 = var_tmp
                            var_tmp = timeInterpMod.gfs_pcp_time_interp(
                                input_forcings, config_options, mpi_config
                            )
                        elif config_options.grid_type == "unstructured":
                            input_forcings.globalPcpRate2 = var_tmp
                            input_forcings.globalPcpRate2_elem = var_tmp_elem
                            var_tmp, var_tmp_elem = timeInterpMod.gfs_pcp_time_interp(
                                input_forcings, config_options, mpi_config
                            )
                        elif config_options.grid_type == "hydrofabric":
                            input_forcings.globalPcpRate2 = var_tmp
                            var_tmp = timeInterpMod.gfs_pcp_time_interp(
                                input_forcings, config_options, mpi_config
                            )

            if grib_var == "CPOFP":
                if mpi_config.rank == 0:
                    var_tmp[var_tmp >= 0] = (
                        100 - var_tmp[var_tmp >= 0]
                    ) / 100
                    var_tmp[var_tmp < 0] = np.nan
                    if config_options.grid_type == "unstructured":
                        var_tmp_elem[var_tmp_elem >= 0] = (
                            100 - var_tmp_elem[var_tmp_elem >= 0]
                        ) / 100
                        var_tmp_elem[var_tmp_elem < 0] = np.nan

            if config_options.grid_type == "gridded":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place GFS local array into ESMF element field object: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place GFS local array into ESMF element field object: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place GFS local array into ESMF element field object: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to place GFS local array into ESMF element field object: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input 13km GFS Field: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    begin = time.monotonic()
                    input_forcings.esmf_field_out = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                    end = time.monotonic()
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Regridding took {end - begin} seconds"
                        err_handler.log_msg(config_options, mpi_config)
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid GFS variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(ve)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask search on GFS variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract GFS ESMF field data to local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input 13km GFS Field for Mesh nodes: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    begin = time.monotonic()
                    input_forcings.esmf_field_out = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                    end = time.monotonic()
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Node Regridding took {end - begin} seconds"
                        err_handler.log_msg(config_options, mpi_config)
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid GFS variable for Mesh Nodes: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(ve)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask search on GFS variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract GFS ESMF field data to local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input 13km GFS Field for Mesh elements: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    begin = time.monotonic()
                    input_forcings.esmf_field_out_elem = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                    end = time.monotonic()
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Element Regridding took {end - begin} seconds"
                        err_handler.log_msg(config_options, mpi_config)
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid GFS variable for Mesh Elements: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(ve)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask search on GFS variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract GFS ESMF field data to local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                if mpi_config.rank == 0:
                    config_options.statusMsg = (
                        "Regridding Input 13km GFS Field for Mesh Elements: "
                        + input_forcings.netcdf_var_names[force_count]
                    )
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    begin = time.monotonic()
                    input_forcings.esmf_field_out = (
                        esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                    end = time.monotonic()
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Element Regridding took {end - begin} seconds"
                        err_handler.log_msg(config_options, mpi_config)
                except ValueError as ve:
                    config_options.errMsg = (
                        "Unable to regrid GFS variable for Mesh Element: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(ve)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = (
                        "Unable to run mask search on GFS variable: "
                        + input_forcings.netcdf_var_names[force_count]
                        + " ("
                        + str(npe)
                        + ")"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = (
                        "Unable to extract GFS ESMF field data to local array: "
                        + str(err)
                    )
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = (
                    f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                )
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = (
                    f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n"
                    f"{traceback.format_exc()}"
                )
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_nam_nest(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid input NAM nest data from GRIB2 files."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        config_options.statusMsg = "No regridding of NAM nest data necessary for this timestep - already completed."
        err_handler.log_msg(config_options, mpi_config)
        return

    input_forcings.tmpFile = config_options.scratch_dir + "/" + "NAM_NEST_TMP-{}.nc".format(mkfilename())
    err_handler.check_program_status(config_options, mpi_config)
    if input_forcings.fileType != NETCDF:

        if mpi_config.rank == 0:
            if os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = "Found old temporary file: " + \
                                           input_forcings.tmpFile + " - Removing....."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    err_handler.err_out(config_options)
        err_handler.check_program_status(config_options, mpi_config)

        fields = []
        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Converting NAM-Nest Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config)
            fields.append(':' + grib_var + ':' +
                          input_forcings.grib_levels[force_count] + ':'
                          + str(input_forcings.fcst_hour2) + " hour fcst:")
        fields.append(":(HGT):(surface):")

        if(WGRIB2_env):
            cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
                  " -netcdf " + input_forcings.tmpFile
        else:
            cmd = '(' + '|'.join(fields) + ')'

        id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                  config_options, mpi_config, inputVar=None, special_case=False)
        err_handler.check_program_status(config_options, mpi_config)
    else:
        create_link("NAM-Nest", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
        id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing NAM Nest Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating NAM nest regridding weights...."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
            err_handler.check_program_status(config_options, mpi_config)

            if(config_options.grid_type == "gridded"):
                if mpi_config.rank == 0:
                    var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            elif(config_options.grid_type == "unstructured"):
                if mpi_config.rank == 0:
                    var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    var_tmp_elem = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp_elem = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj_elem,
                        input_forcings.esmf_field_in_elem,
                        input_forcings.esmf_field_out_elem,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            elif(config_options.grid_type == "hydrofabric"):
                if mpi_config.rank == 0:
                    var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

        err_handler.check_program_status(config_options, mpi_config)

        if(config_options.grid_type == "gridded"):
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding NAM nest input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif(config_options.grid_type == "unstructured"):
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding NAM nest input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding NAM nest input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj_elem,
                    input_forcings.esmf_field_in_elem,
                    input_forcings.esmf_field_out_elem,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

        elif(config_options.grid_type == "hydrofabric"):
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding NAM nest input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
        try:
            os.remove(input_forcings.tmpFile)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_mrms_hourly(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid hourly MRMS precipitation."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not config_options.use_data_at_current_time:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Exceeded max hours for MRMS precipitation"
            err_handler.log_msg(config_options, mpi_config)
        return

    if not os.path.isfile(supplemental_precip.file_in2):
        if os.path.isfile(supplemental_precip.file_in1):
            supplemental_precip.file_in2 = supplemental_precip.file_in1
        else:
            return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return

    mrms_tmp_grib2 = config_options.scratch_dir + "/MRMS_PCP_TMP-{}.grib2".format(mkfilename())
    mrms_tmp_nc = config_options.scratch_dir + "/MRMS_PCP_TMP-{}.nc".format(mkfilename())
    mrms_tmp_rqi_grib2 = config_options.scratch_dir + "/MRMS_RQI_TMP-{}.grib2".format(mkfilename())
    mrms_tmp_rqi_nc = config_options.scratch_dir + "/MRMS_RQI_TMP-{}.nc".format(mkfilename())

    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will " \
                                       "not be used."
            err_handler.log_msg(config_options, mpi_config)
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        if(config_options.grid_type == "unstructured"):
            supplemental_precip.regridded_precip2_elem = None
            supplemental_precip.regridded_precip1_elem = None
        return

    if mpi_config.rank == 0:
        if os.path.isfile(mrms_tmp_grib2):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_grib2 + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(mrms_tmp_grib2)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_grib2
                err_handler.log_critical(config_options, mpi_config)
        if os.path.isfile(mrms_tmp_nc):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_nc + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(mrms_tmp_nc)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_nc
                err_handler.log_critical(config_options, mpi_config)
        if os.path.isfile(mrms_tmp_rqi_grib2):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_rqi_grib2 + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(mrms_tmp_rqi_grib2)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_rqi_grib2
                err_handler.log_critical(config_options, mpi_config)
        if os.path.isfile(mrms_tmp_rqi_nc):
            config_options.statusMsg = "Found old temporary file: " + \
                                       mrms_tmp_rqi_nc + " - Removing....."
            err_handler.log_warning(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(mrms_tmp_rqi_nc)
            except OSError:
                config_options.errMsg = "Unable to remove file: " + mrms_tmp_rqi_nc
                err_handler.log_critical(config_options, mpi_config)

    err_handler.check_program_status(config_options, mpi_config)

    if supplemental_precip.fileType != NETCDF:
        ioMod.unzip_file(supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod == 1:
            ioMod.unzip_file(supplemental_precip.rqi_file_in2, mrms_tmp_rqi_grib2, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if(WGRIB2_env):
            cmd1 = "$WGRIB2 " + mrms_tmp_grib2 + " -netcdf " + mrms_tmp_nc
        else:
            cmd1 = mrms_tmp_grib2

        id_mrms = ioMod.open_grib2(mrms_tmp_grib2, mrms_tmp_nc, cmd1, config_options,
                                   mpi_config, supplemental_precip.netcdf_var_names[0], special_case=True)
        err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod == 1:
            if(WGRIB2_env):
                cmd2 = "$WGRIB2 " + mrms_tmp_rqi_grib2 + " -netcdf " + mrms_tmp_rqi_nc
            else:
                cmd2 = mrms_tmp_rqi_grib2

            id_mrms_rqi = ioMod.open_grib2(mrms_tmp_rqi_grib2, mrms_tmp_rqi_nc, cmd2, config_options,
                                           mpi_config, supplemental_precip.rqi_netcdf_var_names[0], special_case=True)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            id_mrms_rqi = None

        if mpi_config.rank == 0:
            try:
                os_utils.os_remove_retry(mrms_tmp_grib2)
            except OSError:
                config_options.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_grib2
                err_handler.log_critical(config_options, mpi_config)

            if supplemental_precip.rqiMethod == 1:
                try:
                    os_utils.os_remove_retry(mrms_tmp_rqi_grib2)
                except OSError:
                    config_options.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_rqi_grib2
                    err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)
    else:
        create_link("MRMS", supplemental_precip.file_in2, mrms_tmp_nc, config_options, mpi_config)
        id_mrms = ioMod.open_netcdf_forcing(mrms_tmp_nc, config_options, mpi_config)
        if supplemental_precip.rqiMethod == 1:
            create_link("RQI", supplemental_precip.rqi_file_in2, mrms_tmp_rqi_nc, config_options, mpi_config)
            id_mrms_rqi = ioMod.open_netcdf_forcing(mrms_tmp_rqi_nc, config_options, mpi_config)
        else:
            id_mrms_rqi = None

    calc_regrid_flag = check_supp_pcp_regrid_status(id_mrms, supplemental_precip, config_options,
                                                    wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating MRMS regridding weights."
            err_handler.log_msg(config_options, mpi_config)
        calculate_supp_pcp_weights(supplemental_precip, id_mrms, mrms_tmp_nc, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    if supplemental_precip.rqiMethod == 1:
        if(config_options.grid_type != "unstructured"):
            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + \
                                            " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding MRMS RQI Field."
                err_handler.log_msg(config_options, mpi_config)
            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                n_masked = len((supplemental_precip.regridded_mask == 0))
                if n_masked > 0:
                    if mpi_config == 0:
                        config_options.statusMsg = f"{n_masked} masked cells in RQI field, will remove"
                        err_handler.log_msg(config_options, mpi_config)

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        elif(config_options.grid_type == "unstructured"):
            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + \
                                            " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding MRMS RQI Field."
                err_handler.log_msg(config_options, mpi_config)
            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj,
                    supplemental_precip.esmf_field_in,
                    supplemental_precip.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                n_masked = len((supplemental_precip.regridded_mask == 0))
                if n_masked > 0:
                    if mpi_config == 0:
                        config_options.statusMsg = f"{n_masked} masked cells in RQI field, will remove"
                        err_handler.log_msg(config_options, mpi_config)

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                try:
                    var_tmp_elem = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + \
                                            " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding MRMS RQI Field."
                err_handler.log_msg(config_options, mpi_config)
            try:
                supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj_elem,
                    supplemental_precip.esmf_field_in_elem,
                    supplemental_precip.esmf_field_out_elem,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                n_masked = len((supplemental_precip.regridded_mask_elem == 0))
                if n_masked > 0:
                    if mpi_config == 0:
                        config_options.statusMsg = f"{n_masked} masked cells in RQI field, will remove"
                        err_handler.log_msg(config_options, mpi_config)

                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

    if not supplemental_precip.rqiMethod:
        if(config_options.grid_type == "gridded"):
            supplemental_precip.regridded_rqi2[:, :] = 1.0
        elif(config_options.grid_type == "unstructured"):
            supplemental_precip.regridded_rqi2[:] = 1.0
            supplemental_precip.regridded_rqi2_elem[:] = 1.0
        elif(config_options.grid_type == "hydrofabric"):
            supplemental_precip.regridded_rqi2[:] = 1.0

        if mpi_config.rank == 0:
            config_options.statusMsg = "MRMS Will not be filtered using RQI values."
            err_handler.log_msg(config_options, mpi_config)

    elif supplemental_precip.rqiMethod == 2:
        ioMod.read_rqi_monthly_climo(config_options, mpi_config, supplemental_precip, wrf_hydro_geo_meta)
    elif supplemental_precip.rqiMethod == 1:
        if(config_options.grid_type == "gridded"):
            supplemental_precip.regridded_rqi2[:, :] = supplemental_precip.esmf_field_out.data
        elif(config_options.grid_type == "unstructured"):
            supplemental_precip.regridded_rqi2[:] = supplemental_precip.esmf_field_out.data
            supplemental_precip.regridded_rqi2_elem[:] = supplemental_precip.esmf_field_out_elem.data
        elif(config_options.grid_type == "hydrofabric"):
            supplemental_precip.regridded_rqi2[:] = supplemental_precip.esmf_field_out.data

    err_handler.check_program_status(config_options, mpi_config)

    if supplemental_precip.rqiMethod == 1:
        if mpi_config.rank == 0:
            try:
                id_mrms_rqi.close()
            except OSError:
                config_options.errMsg = "Unable to close NetCDF file: " + mrms_tmp_rqi_nc
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(mrms_tmp_rqi_nc)
            except OSError:
                config_options.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_rqi_nc
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    if(config_options.grid_type == "gridded"):
        var_tmp = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + \
                                        " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                supplemental_precip.regridObj,
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
            )
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

            supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2[:, :] = \
            supplemental_precip.esmf_field_out.data
        err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod > 0:
            try:
                ind_filter = np.where(supplemental_precip.regridded_rqi2 < supplemental_precip.rqiThresh)
                if len(ind_filter) > 0:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                        err_handler.log_msg(config_options, mpi_config)
                supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
                del ind_filter
            except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.keyValue != 14:
            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1[:, :] = \
                supplemental_precip.regridded_precip2[:, :]
            supplemental_precip.regridded_rqi1[:, :] = \
                supplemental_precip.regridded_rqi2[:, :]

    elif(config_options.grid_type == "unstructured"):
        var_tmp = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + \
                                        " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                supplemental_precip.regridObj,
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
            )
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

            supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2[:] = \
            supplemental_precip.esmf_field_out.data
        err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod > 0:
            try:
                ind_filter = np.where(supplemental_precip.regridded_rqi2 < supplemental_precip.rqiThresh)
                if len(ind_filter) > 0:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                        err_handler.log_msg(config_options, mpi_config)
                supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
                del ind_filter
            except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.keyValue != 14:
            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1[:] = \
                supplemental_precip.regridded_precip2[:]
            supplemental_precip.regridded_rqi1[:] = \
                supplemental_precip.regridded_rqi2[:]

        var_tmp_elem = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp_elem = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + \
                                        " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                supplemental_precip.regridObj_elem,
                supplemental_precip.esmf_field_in_elem,
                supplemental_precip.esmf_field_out_elem,
            )
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv
            
            if len(np.argwhere(supplemental_precip.esmf_field_out_elem.data < 0)) > 0:
                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.esmf_field_out_elem.data < 0)] = config_options.globalNdv

        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2_elem[:] = \
            supplemental_precip.esmf_field_out_elem.data
        err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod > 0:
            try:
                ind_filter_elem = np.where(supplemental_precip.regridded_rqi2_elem < supplemental_precip.rqiThresh)
                if len(ind_filter_elem) > 0:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Removing {len(ind_filter_elem)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                        err_handler.log_msg(config_options, mpi_config)
                supplemental_precip.regridded_precip2_elem[ind_filter_elem] = config_options.globalNdv
                del ind_filter_elem
            except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        try:
            ind_valid_elem = np.where(supplemental_precip.regridded_precip2_elem != config_options.globalNdv)
            supplemental_precip.regridded_precip2_elem[ind_valid_elem] = supplemental_precip.regridded_precip2_elem[ind_valid_elem] / 3600.0
            del ind_valid_elem
        except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
            config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1_elem[:] = \
                supplemental_precip.regridded_precip2_elem[:]
            supplemental_precip.regridded_rqi1_elem[:] = \
                supplemental_precip.regridded_rqi2_elem[:]

    elif(config_options.grid_type == "hydrofabric"):
        var_tmp = None
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
            err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + \
                                        " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
                supplemental_precip.regridObj,
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
            )
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

            supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2[:] = \
            supplemental_precip.esmf_field_out.data
        err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod > 0:
            try:
                ind_filter = np.where(supplemental_precip.regridded_rqi2 < supplemental_precip.rqiThresh)
                if len(ind_filter) > 0:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                        err_handler.log_msg(config_options, mpi_config)
                supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
                del ind_filter
            except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.keyValue != 14:
            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1[:] = \
                supplemental_precip.regridded_precip2[:]
            supplemental_precip.regridded_rqi1[:] = \
                supplemental_precip.regridded_rqi2[:]

    if mpi_config.rank == 0:
        try:
            id_mrms.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + mrms_tmp_nc
            err_handler.log_critical(config_options, mpi_config)

        try:
            os_utils.os_remove_retry(mrms_tmp_nc)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_nc
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_mrms_precip_flag(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid SBCv2 Liquid Water Precip forcing files."""
    if not os.path.exists(supplemental_precip.file_in2):
        return

    if supplemental_precip.regridComplete:
        return

    fileno = mkfilename()
    mrms_tmp_grib2 = config_options.scratch_dir + f"/MRMS_PCP_FLAG_TMP_{fileno}.grib2"
    mrms_tmp_nc    = config_options.scratch_dir + f"/MRMS_PCP_FLAG_TMP_{fileno}.nc"
    ioMod.unzip_file(supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    cmd1 = "$WGRIB2 " + mrms_tmp_grib2 + " -netcdf " + mrms_tmp_nc
    id_tmp = ioMod.open_grib2(mrms_tmp_grib2, mrms_tmp_nc, cmd1, config_options,
                                mpi_config, supplemental_precip.netcdf_var_names[0])
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            os_utils.os_remove_retry(mrms_tmp_grib2)
        except OSError:
            config_options.errMsg = "Unable to remove GRIB2 file: " + mrms_tmp_grib2
            err_handler.log_critical(config_options, mpi_config)

    calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options,
                                                    wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating MRMS PrecipFlag regridding weights."
            err_handler.log_msg(config_options, mpi_config)
        calculate_supp_pcp_weights(supplemental_precip, id_tmp, supplemental_precip.file_in2,
                                   config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    var_tmp = None
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding MRMS PrecipFlag Fraction."
            err_handler.log_msg(config_options, mpi_config)
        try:
            var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[0]][0,:,:]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract PrecipFlag from file: " + \
                                    supplemental_precip.file_in2 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        var_sub_tmp[var_sub_tmp <= 0] = 1.0
        var_sub_tmp[var_sub_tmp == 3] = 0.0
        var_sub_tmp[var_sub_tmp == 7] = 0.0
        var_sub_tmp[var_sub_tmp >  0] = 1.0

        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to place MRMS PrecipFlag into local ESMF field: " + str(err)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(
            supplemental_precip.regridObj,
            supplemental_precip.esmf_field_in,
            supplemental_precip.esmf_field_out,
        )
    except ValueError as ve:
        config_options.errMsg = "Unable to regrid MRMS PrecipFlag: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = 1.0
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = 1.0
    except (ValueError, ArithmeticError) as npe:
        config_options.errMsg = "Unable to run mask search on MRMS PrecipFlag: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.current_output_step == 1:
        supplemental_precip.regridded_precip1[:] = \
            supplemental_precip.regridded_precip2[:]
    err_handler.check_program_status(config_options, mpi_config)

    if(config_options.grid_type == "unstructured"):

        var_tmp_elem = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding MRMS PrecipFlag Fraction."
                err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp_elem = id_tmp.variables[supplemental_precip.netcdf_var_names[0]][0,:,:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract PrecipFlag from file: " + \
                                        supplemental_precip.file_in2 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            var_sub_tmp_elem[var_sub_tmp_elem <= 0] = 1.0
            var_sub_tmp_elem[var_sub_tmp_elem == 3] = 0.0
            var_sub_tmp_elem[var_sub_tmp_elem == 7] = 0.0
            var_sub_tmp_elem[var_sub_tmp_elem >  0] = 1.0

            supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place MRMS PrecipFlag into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                supplemental_precip.regridObj_elem,
                supplemental_precip.esmf_field_in_elem,
                supplemental_precip.esmf_field_out_elem,
            )
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS PrecipFlag: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = 1.0
            supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.esmf_field_out_elem.data < 0)] = 1.0
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on MRMS PrecipFlag: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1_elem[:] = \
                supplemental_precip.regridded_precip2_elem[:]
        err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + supplemental_precip.file_in2
            err_handler.log_critical(config_options, mpi_config)
        try:
            os_utils.os_remove_retry(mrms_tmp_nc)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + mrms_tmp_nc
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_hourly_wrf_arw(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid input WRF-ARW data from GRIB2 files."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        config_options.statusMsg = "No regridding of WRF-ARW nest data necessary for this timestep - already completed."
        err_handler.log_msg(config_options, mpi_config)
        return

    input_forcings.tmpFile = config_options.scratch_dir + "/" + "ARW_TMP-{}.nc".format(mkfilename())
    err_handler.check_program_status(config_options, mpi_config)

    if input_forcings.fileType != NETCDF:
        if mpi_config.rank == 0:
            if os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = "Found old temporary file: " + \
                                           input_forcings.tmpFile + " - Removing....."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    err_handler.err_out(config_options)
        err_handler.check_program_status(config_options, mpi_config)

        fields = []
        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Converting WRF-ARW Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config)
            time_str = "{}-{} hour acc fcst".format(input_forcings.fcst_hour1, input_forcings.fcst_hour2) \
                if grib_var == 'APCP' else str(input_forcings.fcst_hour2) + " hour fcst"
            fields.append(':' + grib_var + ':' +
                          input_forcings.grib_levels[force_count] + ':'
                          + time_str + ":")
        fields.append(":(HGT):(surface):")

        if(WGRIB2_env):
            cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
                  " -netcdf " + input_forcings.tmpFile
        else:
            cmd = '(' + '|'.join(fields) + ')'

        id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                  config_options, mpi_config, inputVar=None, special_case=False)
        err_handler.check_program_status(config_options, mpi_config)
    else:
        create_link("WRF-ARW", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
        id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

    for force_count, grib_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing WRF-ARW Variable: " + grib_var
            err_handler.log_msg(config_options, mpi_config)

        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating WRF-ARW regridding weights...."
                err_handler.log_msg(config_options, mpi_config)
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
            err_handler.check_program_status(config_options, mpi_config)

            if(config_options.grid_type == "gridded"):
                if mpi_config.rank == 0:
                    var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " \
                                        + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            elif(config_options.grid_type == "unstructured"):
                if mpi_config.rank == 0:
                    var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    var_tmp_elem = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp_elem = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj_elem,
                        input_forcings.esmf_field_in_elem,
                        input_forcings.esmf_field_out_elem,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            elif(config_options.grid_type == "hydrofabric"):
                if mpi_config.rank == 0:
                    var_tmp = id_tmp.variables['HGT_surface'][0, :, :]
                else:
                    var_tmp = None
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj,
                        input_forcings.esmf_field_in,
                        input_forcings.esmf_field_out,
                    )
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

        err_handler.check_program_status(config_options, mpi_config)

        if(config_options.grid_type == "gridded"):
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding WRF-ARW input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif(config_options.grid_type == "unstructured"):
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding WRF-ARW input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding WRF-ARW input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj_elem,
                    input_forcings.esmf_field_in_elem,
                    input_forcings.esmf_field_out_elem,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

        elif(config_options.grid_type == "hydrofabric"):
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding WRF-ARW input variable: " + \
                                           input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config)
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + \
                                            " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                    config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
        try:
            os.remove(input_forcings.tmpFile)
        except OSError:
            config_options.errMsg = "Unable to remove NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_sbcv2_lwf(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid SBCv2 Liquid Water Precip forcing files."""
    esmf_regridobj_call_retry_partial = functools.partial(
        esmf_regridobj_call_retry, mpi_config, config_options, err_handler
    )

    if not os.path.exists(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        return

    id_tmp = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config, False,
                                        "latitude", "longitude")
    err_handler.check_program_status(config_options, mpi_config)

    calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, input_forcings, config_options,
                                                    wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating SBCV2_LWF regridding weights."
            err_handler.log_msg(config_options, mpi_config)
        calculate_supp_pcp_weights(input_forcings, id_tmp, input_forcings.file_in2,
                                   config_options, mpi_config, "latitude", "longitude")
        err_handler.check_program_status(config_options, mpi_config)

    var_tmp = None
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding SBCV2_LWF Fraction."
            err_handler.log_msg(config_options, mpi_config)
        try:
            var_tmp = id_tmp.variables['liquid_water_fraction'][0,:,:]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract liquid_water_fraction from file: " + \
                                    input_forcings.file_in2 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to place SBCV2_LWF into local ESMF field: " + str(err)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(
            input_forcings.regridObj,
            input_forcings.esmf_field_in,
            input_forcings.esmf_field_out,
        )
    except ValueError as ve:
        config_options.errMsg = "Unable to regrid SBCV2_LWF: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = 1.0
        input_forcings.esmf_field_out.data[np.where(input_forcings.esmf_field_out.data < 0)] = 1.0
    except (ValueError, ArithmeticError) as npe:
        config_options.errMsg = "Unable to run mask search on SBCV2_LWF: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    input_forcings.regridded_precip2[:] = input_forcings.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.current_output_step == 1:
        input_forcings.regridded_precip1[:] = \
            input_forcings.regridded_precip2[:]
    err_handler.check_program_status(config_options, mpi_config)

    if(config_options.grid_type == "unstructured"):

        var_tmp_elem = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding SBCV2_LWF Fraction."
                err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp_elem = id_tmp.variables['liquid_water_fraction'][0,:,:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract liquid_water_fraction from file: " + \
                                        input_forcings.file_in2 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place SBCV2_LWF into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(
                input_forcings.regridObj_elem,
                input_forcings.esmf_field_in_elem,
                input_forcings.esmf_field_out_elem,
            )
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid SBCV2_LWF: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = 1.0
            input_forcings.esmf_field_out_elem.data[np.where(input_forcings.esmf_field_out_elem.data < 0)] = 1.0
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on SBCV2_LWF: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        input_forcings.regridded_precip2_elem[:] = input_forcings.esmf_field_out_elem.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            input_forcings.regridded_precip1_elem[:] = \
                input_forcings.regridded_precip2_elem[:]
        err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.file_in2
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def check_regrid_status(id_tmp, force_count, input_forcings, config_options,
                         wrf_hydro_geo_meta, mpi_config):
    """Check regridding status for input forcing fields."""
    calc_regrid_flag = False

    if input_forcings.esmf_grid_in is None:
        calc_regrid_flag = True
    else:
        if mpi_config.rank == 0:
            try:
                lat_var = "latitude" if "latitude" in id_tmp.variables else "lat"
                lon_var = "longitude" if "longitude" in id_tmp.variables else "lon"
                lat_tmp = id_tmp.variables[lat_var][:]
                lon_tmp = id_tmp.variables[lon_var][:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract lat/lon coordinates from input file: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            if lat_tmp.ndim == 1:
                ny_in = len(lat_tmp)
                nx_in = len(lon_tmp)
            else:
                ny_in = lat_tmp.shape[0]
                nx_in = lat_tmp.shape[1]
        else:
            ny_in = None
            nx_in = None

        ny_in = mpi_config.broadcast_parameter(ny_in, config_options, param_type=int)
        nx_in = mpi_config.broadcast_parameter(nx_in, config_options, param_type=int)

        if ny_in != input_forcings.ny_global or nx_in != input_forcings.nx_global:
            calc_regrid_flag = True

    return calc_regrid_flag


def calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta):
    """Calculate ESMF regridding weights for primary input forcing fields."""
    esmf_grid_retry_partial = functools.partial(
        esmf_grid_retry, mpi_config, config_options, err_handler
    )
    esmf_field_retry_partial = functools.partial(
        esmf_field_retry, mpi_config, config_options, err_handler
    )
    esmf_regrid_retry_partial = functools.partial(
        esmf_regrid_retry, mpi_config, config_options, err_handler
    )

    if mpi_config.rank == 0:
        try:
            lat_var = "latitude" if "latitude" in id_tmp.variables else "lat"
            lon_var = "longitude" if "longitude" in id_tmp.variables else "lon"
            lat_tmp = id_tmp.variables[lat_var][:]
            lon_tmp = id_tmp.variables[lon_var][:]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract lat/lon coordinates for weight calculation: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        if lat_tmp.ndim == 1:
            ny_in = len(lat_tmp)
            nx_in = len(lon_tmp)
        else:
            ny_in = lat_tmp.shape[0]
            nx_in = lat_tmp.shape[1]
    else:
        ny_in = None
        nx_in = None

    input_forcings.ny_global = mpi_config.broadcast_parameter(ny_in, config_options, param_type=int)
    input_forcings.nx_global = mpi_config.broadcast_parameter(nx_in, config_options, param_type=int)

    try:
        input_forcings.esmf_grid_in = esmf_grid_retry_partial(
            np.array([input_forcings.ny_global, input_forcings.nx_global]),
            staggerloc=ESMF.StaggerLoc.CENTER,
            coord_sys=ESMF.CoordSys.SPH_DEG
        )
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = f"Unable to create source ESMF grid: ({str(esmf_error)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
        input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_lower_bound
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = f"Unable to extract local boundaries from ESMF grid: ({str(err)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        if lat_tmp.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lon_tmp, lat_tmp)
        else:
            lon_2d = lon_tmp
            lat_2d = lat_tmp
    else:
        lon_2d = None
        lat_2d = None

    lat_sub = mpi_config.scatter_array(input_forcings, lat_2d, config_options)
    lon_sub = mpi_config.scatter_array(input_forcings, lon_2d, config_options)

    grid_lat = input_forcings.esmf_grid_in.get_coords(0)
    grid_lon = input_forcings.esmf_grid_in.get_coords(1)

    grid_lat[...] = lat_sub
    grid_lon[...] = lon_sub

    try:
        input_forcings.esmf_field_in = esmf_field_retry_partial(
            input_forcings.esmf_grid_in,
            name="input_field",
            staggerloc=ESMF.StaggerLoc.CENTER
        )
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = f"Unable to create input ESMF field: ({str(esmf_error)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        try:
            input_forcings.esmf_field_out = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid,
                name="output_field",
                staggerloc=ESMF.StaggerLoc.CENTER
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create output ESMF field: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridObj = esmf_regrid_retry_partial(
                input_forcings.esmf_field_in,
                input_forcings.esmf_field_out,
                regrid_method=input_forcings.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create ESMF regrid object: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "unstructured":
        try:
            input_forcings.esmf_field_out = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid,
                name="output_field_nodes",
                meshloc=ESMF.MeshLoc.NODE
            )
            input_forcings.esmf_field_in_elem = esmf_field_retry_partial(
                input_forcings.esmf_grid_in,
                name="input_field_elem",
                staggerloc=ESMF.StaggerLoc.CENTER
            )
            input_forcings.esmf_field_out_elem = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid_elem,
                name="output_field_elems",
                meshloc=ESMF.MeshLoc.ELEMENT
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create mesh ESMF fields: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridObj = esmf_regrid_retry_partial(
                input_forcings.esmf_field_in,
                input_forcings.esmf_field_out,
                regrid_method=input_forcings.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
            input_forcings.regridObj_elem = esmf_regrid_retry_partial(
                input_forcings.esmf_field_in_elem,
                input_forcings.esmf_field_out_elem,
                regrid_method=input_forcings.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create ESMF mesh regrid objects: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "hydrofabric":
        try:
            input_forcings.esmf_field_out = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid,
                name="output_field_hydrofabric",
                meshloc=ESMF.MeshLoc.ELEMENT
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create hydrofabric ESMF field: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridObj = esmf_regrid_retry_partial(
                input_forcings.esmf_field_in,
                input_forcings.esmf_field_out,
                regrid_method=input_forcings.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create hydrofabric regrid object: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Check regridding status for supplemental precipitation fields."""
    calc_regrid_flag = False

    if supplemental_precip.esmf_grid_in is None:
        calc_regrid_flag = True
    else:
        if mpi_config.rank == 0:
            try:
                lat_var = "latitude" if "latitude" in id_tmp.variables else "lat"
                lon_var = "longitude" if "longitude" in id_tmp.variables else "lon"
                lat_tmp = id_tmp.variables[lat_var][:]
                lon_tmp = id_tmp.variables[lon_var][:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract lat/lon from supp pcp file: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            if lat_tmp.ndim == 1:
                ny_in = len(lat_tmp)
                nx_in = len(lon_tmp)
            else:
                ny_in = lat_tmp.shape[0]
                nx_in = lat_tmp.shape[1]
        else:
            ny_in = None
            nx_in = None

        ny_in = mpi_config.broadcast_parameter(ny_in, config_options, param_type=int)
        nx_in = mpi_config.broadcast_parameter(nx_in, config_options, param_type=int)

        if ny_in != supplemental_precip.ny_global or nx_in != supplemental_precip.nx_global:
            calc_regrid_flag = True

    return calc_regrid_flag


def calculate_supp_pcp_weights(supplemental_precip, id_tmp, tmp_file, config_options, mpi_config, lat_var="latitude", lon_var="longitude"):
    """Calculate ESMF regridding weights for supplemental precipitation fields."""
    esmf_grid_retry_partial = functools.partial(
        esmf_grid_retry, mpi_config, config_options, err_handler
    )
    esmf_field_retry_partial = functools.partial(
        esmf_field_retry, mpi_config, config_options, err_handler
    )
    esmf_regrid_retry_partial = functools.partial(
        esmf_regrid_retry, mpi_config, config_options, err_handler
    )

    if mpi_config.rank == 0:
        try:
            lat_tmp = id_tmp.variables[lat_var][:]
            lon_tmp = id_tmp.variables[lon_var][:]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract supp pcp lat/lon coordinates: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        if lat_tmp.ndim == 1:
            ny_in = len(lat_tmp)
            nx_in = len(lon_tmp)
        else:
            ny_in = lat_tmp.shape[0]
            nx_in = lat_tmp.shape[1]
    else:
        ny_in = None
        nx_in = None

    supplemental_precip.ny_global = mpi_config.broadcast_parameter(ny_in, config_options, param_type=int)
    supplemental_precip.nx_global = mpi_config.broadcast_parameter(nx_in, config_options, param_type=int)

    try:
        supplemental_precip.esmf_grid_in = esmf_grid_retry_partial(
            np.array([supplemental_precip.ny_global, supplemental_precip.nx_global]),
            staggerloc=ESMF.StaggerLoc.CENTER,
            coord_sys=ESMF.CoordSys.SPH_DEG
        )
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = f"Unable to create supp pcp source ESMF grid: ({str(esmf_error)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.x_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.x_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.y_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.y_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.nx_local = supplemental_precip.x_upper_bound - supplemental_precip.x_lower_bound
        supplemental_precip.ny_local = supplemental_precip.y_upper_bound - supplemental_precip.y_lower_bound
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = f"Unable to extract local boundaries from supp pcp ESMF grid: ({str(err)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        if lat_tmp.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lon_tmp, lat_tmp)
        else:
            lon_2d = lon_tmp
            lat_2d = lat_tmp
    else:
        lon_2d = None
        lat_2d = None

    lat_sub = mpi_config.scatter_array(supplemental_precip, lat_2d, config_options)
    lon_sub = mpi_config.scatter_array(supplemental_precip, lon_2d, config_options)

    grid_lat = supplemental_precip.esmf_grid_in.get_coords(0)
    grid_lon = supplemental_precip.esmf_grid_in.get_coords(1)

    grid_lat[...] = lat_sub
    grid_lon[...] = lon_sub

    try:
        supplemental_precip.esmf_field_in = esmf_field_retry_partial(
            supplemental_precip.esmf_grid_in,
            name="supp_pcp_input_field",
            staggerloc=ESMF.StaggerLoc.CENTER
        )
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = f"Unable to create supp pcp input ESMF field: ({str(esmf_error)})"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        try:
            supplemental_precip.esmf_field_out = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid,
                name="supp_pcp_output_field",
                staggerloc=ESMF.StaggerLoc.CENTER
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create supp pcp output ESMF field: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                regrid_method=supplemental_precip.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create supp pcp regrid object: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "unstructured":
        try:
            supplemental_precip.esmf_field_out = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid,
                name="supp_pcp_output_field_nodes",
                meshloc=ESMF.MeshLoc.NODE
            )
            supplemental_precip.esmf_field_in_elem = esmf_field_retry_partial(
                supplemental_precip.esmf_grid_in,
                name="supp_pcp_input_field_elem",
                staggerloc=ESMF.StaggerLoc.CENTER
            )
            supplemental_precip.esmf_field_out_elem = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid_elem,
                name="supp_pcp_output_field_elems",
                meshloc=ESMF.MeshLoc.ELEMENT
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create supp pcp mesh ESMF fields: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                regrid_method=supplemental_precip.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
            supplemental_precip.regridObj_elem = esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in_elem,
                supplemental_precip.esmf_field_out_elem,
                regrid_method=supplemental_precip.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create supp pcp mesh regrid objects: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "hydrofabric":
        try:
            supplemental_precip.esmf_field_out = esmf_field_retry_partial(
                wrf_hydro_geo_meta.esmf_grid,
                name="supp_pcp_output_field_hydrofabric",
                meshloc=ESMF.MeshLoc.ELEMENT
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create supp pcp hydrofabric ESMF field: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                regrid_method=supplemental_precip.regrid_opt,
                unmapped_action=ESMF.UnmappedAction.IGNORE
            )
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = f"Unable to create supp pcp hydrofabric regrid object: ({str(esmf_error)})"
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

