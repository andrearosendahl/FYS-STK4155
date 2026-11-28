
import numpy as np
from reading_noresm import read_noresm_cmor, Modelinfo, global_avg
import functions_heatbudget as fhb
import xarray as xr

#Reading the NorESM data, for a specific model and time period
def read_data(period="historical"):
    """    INPUT    """
    modelname = "NorESM2-MM"  # name of model
    realiz = "r1i1p1f1"  # ensemble member
    """ END OF INPUT """

    if period == "historical":
        #Information of experiments in each period
        exps = "historical"
        activity_id = "CMIP"

    elif period == "ssp585":
        #Information of experiments in each period
        exps = "ssp585"
        activity_id = "ScenarioMIP"
    elif period == "sample":
        #Information of experiments in each period
        exps = "historical"
        activity_id = "CMIP"
    else:
        raise ValueError("Period must be either 'historical', 'ssp585', or 'sample'.")

    model = Modelinfo(name=modelname, activity_id=activity_id, expid=exps, realiz=realiz)

    #Reading the different experiments and getting the 30 year periods
    if period == "historical":
        var = fhb.read_atmos_data(model=model,first_year = 1985,last_year = 2014)
        base = fhb.read_atmos_data(model=model,first_year = 1850,last_year = 1879)
    elif period == "ssp585":
        var = fhb.read_atmos_data(model=model,first_year = 2071,last_year = 2100)
        model_base = Modelinfo(name=modelname, activity_id="CMIP", expid="historical", realiz=realiz)
        base = fhb.read_atmos_data(model=model_base,first_year = 1985,last_year = 2014)
    elif period == "sample":
        var = xr.open_dataset("ds_sample.nc")
        base = xr.open_dataset("base_sample.nc")

    return var, base



# Load SEB data, split into train and test and scale the data based on the training set
def load_SEB_data():
    """Returns numpy arrays (x_tr, y_tr, x_te, y_te) with shapes (N, 6) and (N, 1).
    Works directly with the own NN code (numpy/JAX); use torch.tensor(...) for the PyTorch model."""
    
    ds, base = read_data(period="historical")
    
    ds_mean = global_avg(ds.drop_vars(["time_bnds","lat_bnds","lon_bnds"]))  # Global average of the dataset
    base_mean = global_avg(base.drop_vars(["time_bnds","lat_bnds","lon_bnds"])).mean(dim="time")  # Global average of the dataset        
    ptc_temp = fhb.partial_temperature_contributions(ds_mean,base_mean)


    SAF = ptc_temp.SAF #surface albedo feedback
    CRE = ptc_temp.CRE #cloud radiative effect
    HSTORE = ptc_temp.HSTORE #heat storage
    HFLUX = ptc_temp.HFLUX  #heat fluxes
    SWCS = ptc_temp.SWCS #shortwave clear-sky
    LWCS = ptc_temp.LWCS #longwave clear-sky
    dts = ptc_temp.dts #change in surface temperature (dts = ts - ts_base)
    
    # Stack inputs and normalize
    input_tr = np.stack([SAF.sel(time=slice('1985-01-01', '2004-12-31')).values,
                       CRE.sel(time=slice('1985-01-01', '2004-12-31')).values,
                       HSTORE.sel(time=slice('1985-01-01', '2004-12-31')).values,
                       HFLUX.sel(time=slice('1985-01-01', '2004-12-31')).values,
                       SWCS.sel(time=slice('1985-01-01', '2004-12-31')).values,
                       LWCS.sel(time=slice('1985-01-01', '2004-12-31')).values], axis=1)

    input_te = np.stack([SAF.sel(time=slice('2005-01-01', '2014-12-31')).values,
                       CRE.sel(time=slice('2005-01-01', '2014-12-31')).values,
                       HSTORE.sel(time=slice('2005-01-01', '2014-12-31')).values,
                       HFLUX.sel(time=slice('2005-01-01', '2014-12-31')).values,
                       SWCS.sel(time=slice('2005-01-01', '2014-12-31')).values,
                       LWCS.sel(time=slice('2005-01-01', '2014-12-31')).values], axis=1)

    # Targets as numpy arrays of shape (N, 1)
    target_tr = dts.sel(time=slice('1985-01-01', '2004-12-31')).values.reshape(-1, 1)
    target_te = dts.sel(time=slice('2005-01-01', '2014-12-31')).values.reshape(-1, 1)

    # Standardise both sets with the statistics of the TRAINING set
    x_mean, x_std = input_tr.mean(axis=0), input_tr.std(axis=0)
    y_mean, y_std = target_tr.mean(), target_tr.std()

    input_tr = (input_tr - x_mean) / x_std
    input_te = (input_te - x_mean) / x_std
    target_tr = (target_tr - y_mean) / y_std
    target_te = (target_te - y_mean) / y_std

    return input_tr, target_tr, input_te, target_te