#Scripts originally from reading_routines_noresm.py and general_util_funcs.py by author: @adagj
#Modified by @andrear to be able to read NorESM2 data from other users

import xarray as xr
import numpy as np
def make_filelist_cmor(
    self,
    var,
    activity_id="CMIP",
    realm="Amon",
    grid="gn",
    path_to_data="/projects/NS9034K/CMIP6/",
):
    """
    Function for providing a list of all files in order to read a cmorized variable for a given experiment.
    There is usually only one grid for the atmosphere; gn (grid native)
    For the ocean and sea-ice files there are often both native (gn) and regridded files (gr, grz, gm). If not gn is used it is necessary state which grid label is wanted in self.grid_label_ocean

    Parameters
    ----------
    self: model object
    var :          str, name of variable
    realm:         str, e.g. Amon, AERmon, Omon, SImon
    grid:          str, e.g. gn, gr, grz
    activity_id :  str, which MIP the experiment belongs to. Default is 'CMIP'
    path_to_data : str, path to the data folders. Default is '/projects/NS9034K/CMIP6/'.

    Returns
    -------
    Sets a list if filename(s) as an attribute of the model object; self.filenames

    """
    import glob

    self.variable = var
    self.activity_id = activity_id
    self.realm = realm
    self.grid = grid
    # This is kind of a selection function since the ocean grid names are kind of variable, can also have multiple grids for one given variable:
    grid_label_paths = sorted(
        glob.glob(
            path_to_data
            + "/"
            + self.activity_id
            + "/"
            + self.institute
            + "/"
            + self.name
            + "/"
            + self.expid
            + "/"
            + self.realiz
            + "/"
            + self.realm
            + "/"
            + self.variable
            + "/*"
        )
    )
    grid_labels = [
        grid_label_paths[i].split("/")[-1] for i in range(0, len(grid_label_paths))
    ]
    if not grid_labels:
        print(
            "No grid labels found for the given path: %s. Please chekc variable name and path"
            % (
                path_to_data
                + "/"
                + self.activity_id
                + "/"
                + self.institute
                + "/"
                + self.name
                + "/"
                + self.expid
                + "/"
                + self.realiz
                + "/"
                + self.realm
                + "/"
                + self.variable
                + "/*"
            )
        )
        self.filenames = None
    else:
        if self.grid in grid_labels:
            self.gridlabel = self.grid
        else:
            self.gridlabel = grid_labels[0]
            print(
                "\nPLEASE NOTE! Grid label is set to %s for variable %s. If not correct, please reset the grid info. Available grid(s): "
                % (self.gridlabel, self.variable)
                + " ".join(map(str, grid_labels))
            )
        self.path = (
            path_to_data
            + "/"
            + self.activity_id
            + "/"
            + self.institute
            + "/"
            + self.name
            + "/"
            + self.expid
            + "/"
            + self.realiz
            + "/"
            + self.realm
            + "/"
            + self.variable
            + "/"
            + self.gridlabel
            + "/"
        )
        # We need to fix the multiple version challenge. Not all files are necessarily located in the 'latest' folder
        versions = sorted(glob.glob(self.path + "*"))
        if versions:
            fnames = sorted(
                glob.glob(
                    versions[0]
                    + "/"
                    + self.variable
                    + "_"
                    + self.realm
                    + "_"
                    + self.name
                    + "_"
                    + self.expid
                    + "_"
                    + self.realiz
                    + "_"
                    + self.gridlabel
                    + "*.nc" #original: "_*.nc"
                )
            )
        else:
            fnames = None
        if len(versions) > 1:
            for version in versions[1:]:
                files = sorted(
                    glob.glob(
                        version
                        + "/"
                        + self.variable
                        + "_"
                        + self.realm
                        + "_"
                        + self.name
                        + "_"
                        + self.expid
                        + "_"
                        + self.realiz
                        + "_"
                        + self.gridlabel
                        + "*.nc" #original: "_*.nc"
                    )
                )
                for file in files:
                    if versions[0] + "/" + file.split("/")[-1] not in fnames:
                        fnames.append(file)
        
        if fnames:
            if len(fnames) > 1 and self.activity_id[-3:] == "mon":
                # test that the files contained in the filelist covers all years in consecutive order
                # only works for monthly files
                fnames = sorted(fnames, key=lambda x: extract_number(x))
                checkConsecutive(fnames)
            self.filenames = fnames

        if not fnames:
            self.filenames = None
        #if not fnames:
        #    raise Exception(
        #        "Variable %s not prestent in output folder for model %s\n"
        #        % (self.variable, self.name)
        #    )

        
def read_noresm_cmor(
    model,
    varlist,
    realm="Amon",
    grid="gn",
    dim="time",
    transform_func=None,
    path_to_data="/projects/NS9034K/CMIP6/",
):
    """
    This function reads cmorized data from NorESM.
    Please NOTE: This is not a general CMIP6 reading routine as it contains NorESM specific bug fixes
    which may mess up model results from other CMIP6 models.

    Parameters
    ----------
    model :          python object, with experiment details as attributes (generated by class Modelinfo )
    varlist:         list, list of variable names which will be read and loaded into one xarray.Dataset
    realm:           str, Realm: e.g. Amon, AERmon, CFmon, Omon, SImon
    grid :           str, which grid resolution should be used.  e.g. 'gn', 'gr', 'gm', 'grz'
                     gn: native grid, 'gr': regridded somehow - not obvious
                     The grid is not really needed unless you want to specify one particular grid (out of several options)
    dim:             str, concatenate files by this dimension (if there are several files for one single variable) . Default is 'time'
    transform_func:  python function(s), which will be applied to all variables in the variable list

    Returns
    -------
    ds_out : xarray.Dataset

    """
    ds_out = None
    for var in varlist:
        make_filelist_cmor(
            model,
            var,
            activity_id=model.activity_id,
            realm=realm,
            grid=grid,
            path_to_data=path_to_data,
        )
        if isinstance(model.filenames,list):
            # NOTE! if the reading crashes due to memory issues you may add chunks, i.e. parallel=True, chunks={"time":12}
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            with xr.open_mfdataset(
                model.filenames,
                concat_dim=dim,
                combine="nested",
                data_vars="minimal",
                coords="minimal",
                join="override",
                compat="override", #added this because of the merge problem with sea ice (ANDREA)
                decode_times=time_coder, #added this because of serialisation warning with future time variables (ANDREA)
            ) as ds:
                if "height" in ds.variables:
                    # Datasets for variables @ a given height e.g. 2 m temperature, 10 m surface temperature
                    # contain a 'height' variable which causes problems when combining datasets.
                    # i.e. xarray.core.merge.MergeError: conflicting values for variable 'height' on objects to be combined.
                    # The height is not really used for anything, so it is safe to drop it
                    ds = ds.drop("height")

                if "lon_bnds" in ds.variables and model.name in [
                    "NorESM2-LM",
                    "NorESM2-MM",
                    "NorCPM1",
                ]:
                    if ds.lon_bnds.isel(bnds=0).values[0] == 0:
                        # many of the cmorized variables contain wrong longitude boundary values
                        # which causes problems when combining datasets.
                        # Wrong boundary values can potentially really mess up regridding as well
                        # Over write with correct boundary values
                        lon_b = np.concatenate(
                            (
                                np.array(
                                    [
                                        ds.lon[0].values
                                        - 0.5 * ds.lon.diff(dim="lon").values[0]
                                    ]
                                ),
                                0.5 * ds.lon.diff(dim="lon").values + ds.lon.values[:-1],
                                np.array(
                                    [
                                        ds.lon[-1].values
                                        + 0.5 * ds.lon.diff(dim="lon").values[-1]
                                    ]
                                ),
                            )
                        )
                        lon_b = np.reshape(
                            np.concatenate([lon_b[:-1], lon_b[1:]]), [2, len(ds.lon.values)]
                        ).T
                        ds["lon_bnds"] = xr.DataArray(
                            lon_b, dims=("lon", "bnds"), coords={"lon": ds.lon}
                        )
                if "sector" in ds.variables:
                    # the problem here is that we operate with a very variable number of sting lenghts (from 70 to 1100)
                    # make them identical by getting rid of white spaces
                    ds["sector"] = ds.sector.str.strip()
                # transform_func should do some sort of selection on aggregation
                if transform_func is not None:
                    ds = transform_func(ds)
                if isinstance(ds_out, xr.Dataset):
                    ds_out = xr.merge([ds, ds_out],compat="override")
                else:
                    ds_out = ds
        else:
            print("No files found for variable %s \n"%var)
            continue
    return ds_out




class Modelinfo:
    """
    Sets the details of the model experiment
    """

    def __init__(
        self,
        name="NorESM2-LM",
        institute="NCC",
        activity_id="CMIP",
        expid="piControl",
        realiz=["r1i1p1f1"],
        branchtime_year=0,
    ):
        """
        The attributes need to be general and not file specific thus detail like Amon, grid label should not be included

        Attributes
        ----------
        name :        str, name of the CMIP model - typically the Source ID
        institute :   str, Institution ID
        activity_id:  str, activity ID for experiment
        expid :       str, Experiment ID, e.g. piControl, abrupt-4xCO2
        realiz :      str, variant labels saying something about which realization (ensemble member), initial conditions, forcings etc.
                      e.g. 'r1i1p1f1'
        branchtime_year : int, when simulation was branched off from parent.
                          Useful when anomalies are considered e.g. abrupt-4xCO2, historical
                          then you only consider data from piControl for the corresponding period
                          e.g. xarray.DataArray(year = slice(model.branchtime_year,None))

        """
        self.name = name
        self.institute = institute
        self.activity_id = activity_id
        self.expid = expid
        self.realiz = realiz
        self.branchtime_year = branchtime_year


        

def global_avg(ds):
    """Calculates globally averaged values

    Parameters
    ----------
    ds : xarray.DataArray i.e. ds[var]

    Returns
    -------
    ds_out :  xarray.DataArray with globally averaged values
    """
    # to include functionality for subsets or regional averages:
    #if "time" in ds.dims:
    #    #weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat)) * ds.notnull().mean(dim=("lon", "time"))
    #    weights = np.cos(np.deg2rad(ds.lat))#* ds.notnull().mean(dim=("lon", "time"))
    #elif "month" in ds.dims:
    #    #weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat)) * ds.notnull().mean(dim=("lon", "month"))
    #    weights = np.cos(np.deg2rad(ds.lat))#* ds.notnull().mean(dim=("lon", "month"))
    #else:
    #    #weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat)) * ds.notnull().mean(dim=("lon"))
    #    weights = np.cos(np.deg2rad(ds.lat))#* ds.notnull().mean(dim=("lon"))
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    ds_out = ds.weighted(weights).mean(("lon", "lat"))
    #ds_out = (ds.mean(dim="lon") * weights).sum(dim="lat") / weights.sum()
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = "Globally averaged " + ds.long_name
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = ds.standard_name
    return ds_out



def consistent_naming(ds):
    """
    The naming convention for coordinates and dimensions are not the same 
    for noresm raw output and cmorized variables. This function rewrites the 
    coords and dims names to be consistent and the functions thus work on all
    Choose the cmor naming convention.

    Parameters
    ----------
    ds : xarray.Dataset 

    Returns
    -------
    ds : xarray.Dataset

    """
    #print(ds)
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "region" in ds.dims:
        ds = ds.rename(
            {"region": "basin"}
        )  # are we sure that it is the dimension and not the variable which is renamed? Probably both
        # note in BLOM raw, region is both a dimension and a variable. Not sure how xarray handles that
        # in cmorized variables sector is the char variable with the basin names, and basin is the dimension
        # don't do this -> ds = ds.rename({'region':'sector'})
    if "x" in ds.dims:
        ds = ds.rename({"x": "i"})
    if "y" in ds.dims:
        ds = ds.rename({"y": "j"})
    if "ni" in ds.dims:
        ds = ds.rename({"ni": "i"})
    if "nj" in ds.dims:
        ds = ds.rename({"nj": "j"})
    if "nlat" in ds.dims:
        ds = ds.rename({"nlat": "j"})
    if "nlon" in ds.dims:
        ds = ds.rename({"nlon": "i"})
    if "depth" in ds.dims:
        ds = ds.rename({"depth": "lev"})
    if "depth_bnds" in ds.dims:
        ds = ds.rename({"depth_bnds": "lev_bnds"})
    if "nbnd" in ds.dims:
        ds = ds.rename({"nbnd": "bnds"})
    if "hist_interval" in ds.dims:
        ds = ds.rename({"hist_interval": "bnds"})
    if "nbounds" in ds.dims:
        ds = ds.rename({"nbounds": "bnds"})
    if "bounds" in ds.dims:
        ds = ds.rename({"bounds": "bnds"})
    if "type" in ds.coords:
        ds = ds.drop("type")
    if 'latitude_bnds' in ds.variables:
        ds = ds.rename({'latitude_bnds':'lat_bnds'})
    if 'longitude_bnds' in ds.variables:
        ds = ds.rename({'longitude_bnds':'lon_bnds'})
    if 'nav_lat' in ds.coords:
        ds = ds.rename({'nav_lon':'lon','nav_lat':'lat'})
    if 'bounds_nav_lat' in ds.variables:
        ds = ds.rename({'bounds_nav_lat':'vertices_latitude','bounds_nav_lon':'vertices_longitude'})

    return ds