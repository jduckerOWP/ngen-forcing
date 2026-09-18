# NextGen Forcings Engine BMI Overview

The **NextGen Forcings Engine BMI** workflow is a meteorological forcing provider for the NextGen Water Resources Modeling Framework. It can be integrated directly as a supporting Basic Model Interface (BMI) forcing provider within a given NextGen formulation or executed as a standalone workflow to produce NextGen-compatible forcing files (NetCDF4 or catchment CSV files).

This application streamlines forcing file production for all National Water Model (NWM) v3.1 operational configurations across gridded domains, unstructured meshes, and the NextGen hydrofabric geopackage.

---

## Workflow Execution Pipeline

```text
[ config.yml ] 
      │
      ▼
[ ESMF Mesh Production (Target NextGen Hydrofabric / Domain) ]
      │
      ▼
[ Forcing File Download (Based on NWMv3.0 Operational Config) ]
      │
      ▼
[ ESMF Bilinear Regridding (Grid Cells, Nodes, Elements) ]
      │
      ├──► NextGen-Compatible Lumped Forcing Files (NetCDF or Catchment CSVs)
      └──► BMI Advertisement of Regridded Output Fields (via Catchment IDs)
```


## Component Directories

| Directory | Description |
| :--- | :--- |
| `Forcing_Extraction_Scripts` | Contains domain-specific scripts (CONUS, Alaska, Puerto Rico, Hawaii) to download required meteorological forcing data products from AWS S3 buckets or NOMADS servers for each regional NWM v3.1 setup. |
| `ESMF_Mesh_Domain_Configuration_Production` | Utility scripts focused on converting raw model domain file formats (NextGen Hydrofabric Geopackages, D-FlowFM, SCHISM) into ESMF-mesh-compliant NetCDF files. |
| `BMI_NextGen_Configs/` | Contains `config.yml` templates pre-configured for NWM v3.1 operational runs across gridded, coastal unstructured, and hydrofabric domains. |
| `NextGen_Domains/` | Contains sample domain geogrid files for gridded domains, coastal unstructured meshes, and the NextGen Hydrofabric (VPU 06). |
| `NWM_Params/` | Directory designated for supporting climatology files used in bias calibration and downscaling routines. |
| `NextGen_Forcings_Engine/` | Directory harboring the NextGen Forcings Engine BMI code (`bmi_model.py`), model implementation (`model.py`), and supporting Python tools for the Forcing Engine. |
| `NextGen_Forcings_Engine/core/` | Core Python supporting modules required by `model.py` to drive regridding, interpolation, and I/O. |

---

## Setup and Execution Guide

1. **Configure Execution Options**:
   Navigate to `./BMI_NextGen_Configs/config_templates` and select the appropriate template for your domain and NWM v3.0 configuration. Copy the selected `config.yml` into the working directory containing `bmi_wrapper.py`.

2. **Execute the BMI Workflow**:
   Run the driver script via `mpirun` specifying the path to your configuration file, output directory, and total MPI processes:

   ```bash
   mpirun -n 1 python bmi_wrapper.py ./short_range_config.yml -output_path ./Scratch/Short_Range -np 2

* **`./short_range_config.yml`**: Pathway to the target configuration file.
   * **`-output_path ./Scratch/Short_Range`**: Directory path for downloaded forcing inputs and regridded outputs.
   * **`-np 2`**: Number of internal MPI worker processes allocated to the Forcings Engine.

---

## Core System Architecture

### 1. Initialization Phase (`model.py`)
1. **Option Parsing (`config.py`)**: Reads the input configuration file, initializes `ConfigOptions`, and sets operational parameters, directory pathways, and operational flags.
2. **Execution Metadata**: Captures operational arguments (`nwm_version`, `nwm_config`) and sets logging pathways.
3. **Parallel Context (`parallel.py`)**: Sets up MPI communicator ranks and sizes (`MPI_COMM_WORLD`).
4. **Domain Processing (`geoMod.py`)**: Opens the modeling domain NetCDF/Geopackage, reads spatial coordinates and grid spacing, and broadcasts local processor grid boundaries and ESMF grid objects to worker ranks.
5. **Coordinate Reference Metadata**: Rank 0 extracts global CRS and geospatial metadata needed for NetCDF output generation.
6. **I/O Grid Allocation (`ioMod.py`)**: Allocates local slabs (`ny_local`, `nx_local`) on each rank to hold output array slices and initializes output NetCDF dataset structures.
7. **Input Class Allocation (`forcingInputMod.py`)**: Maps source data product characteristics (grid size, variables, projection) and creates ESMF input grid/mesh objects for downscaling and regridding.
8. **Supplemental Precip Setup (`suppPrecipMod.py`)**: Initializes supplemental precipitation dataset variables, disaggregation flags, and input ESMF regrid objects if enabled.
9. **BMI Execution Invocation**: Passes configured classes to `model.py` to drive time-stepping execution.

### 2. Operational Forecast Regridding Loop
1. **Supplemental Disaggregation Check (`disaggregateMod.py`)**: If supplemental 6-hourly accumulation data is supplied, disaggregates fields down to 1-hourly intervals prior to regridding.
2. **Forecast Cycle Configuration (`time_handling.py`)**: Computes current cycle numbers, creates output forecast directories, and configures lookback windows for Analysis and Assimilation (AnA) runs.
3. **Timestep Processing Loop**:
   * **Reset Grids**: Re-initializes destination arrays to missing values (`globalNdv`).
   * **Input Regridding (`forcingInputMod.py` / `regrid.py`)**: Identifies surrounding input cycle files, converts incoming GRIB2 files to NetCDF via `$WGRIB2`, and regrids forcing variables using predefined ESMF bilinear objects.
   * **Unit Conversion & Masking**: Converts average precipitation rates to instantaneous rates and sets unmapped out-of-domain cells to missing values.
   * **Post-Processing (`timeInterpMod.py` / `bias_correction.py` / `downscale.py`)**: Applies temporal interpolation to the target timestep, followed by bias correction and terrain downscaling routines where applicable.
   * **Supplemental Layering (`layeringMod.py` / `suppPrecipMod.py`)**: Regrids supplemental precipitation fields, applies temporal disaggregation/interpolation, and overlays (relayers) the result onto the output object.
4. **Finalization (`ioMod.py`)**: Closes dataset handles, writes output NetCDF/CSV files or advertises fields via BMI pointers, purges scratch directories, and terminates the MPI communicator.

---

## Key Workflow Files

* **`./NextGen_Forcings_Engine/bmi_model.py`**: The main BMI class definition exposing standard BMI control endpoints.
* **`./NextGen_Forcings_Engine/model.py`**: The core execution driver that controls data movement and transformation logic.
* **`bmi_wrapper.py`**: The primary entry-point script that automates ESMF mesh generation, forcing extraction, and execution.
* **`run_bmi_model.py`**: Lightweight script demonstrating standalone initialization, update, and finalization calls.
* **`run_bmi_unit_test.py`**: Comprehensive unit-test suite verifying BMI interface compliance.
* **`environment.yml`**: Conda environment definition containing all library dependencies (ESMF, NetCDF4, PyProj, Dask, mpi4py).

---

## Environment Setup

To create and activate the required Python environment:

```bash
conda env create -f environment.yml
conda activate bmi_test
