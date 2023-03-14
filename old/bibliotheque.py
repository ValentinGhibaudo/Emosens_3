import numpy as np
import mne
import xarray as xr
import ghibtools as gh

def mne_to_xarray(raw):
    data = raw.get_data()
    srate = raw.info['sfreq']
    da = xr.DataArray(data=data, dims = ['chan','time'], coords = {'chan':raw.info['ch_names'], 'time':gh.time_vector(data[0,:], srate)}, attrs={'srate':srate})
    return da