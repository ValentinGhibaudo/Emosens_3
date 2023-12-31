from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from compute_psd import psd_baselined_job

def compute_bandpower(run_key, **p):
    """
    Compute bandpower for each frequency band (EEG)
    """
    participant, session = run_key.split('_')
    psd = psd_baselined_job.get(run_key)['psd_baselined'] # load baselined spectrum
    
    rows = []
    
    for chan in psd['chan'].values: # loop on chans
        total_power = psd.loc[chan, p['total_band'][0]:p['total_band'][0]].mean('freq') # compute total mean power
        for band, bornes in p['fbands'].items(): # loop on frequency bands
            Psd_chan_band = psd.loc[chan, bornes[0]:bornes[1]] # slice the frequency band
            power_mean = Psd_chan_band.mean('freq') # mean value
            power_median = Psd_chan_band.median('freq') # median value
            power_integral = Psd_chan_band.integrate('freq') # integrate (power)
            rel_power = power_integral / total_power # relative power by dividing with total power

            row = [participant, session, chan, band, power_mean, power_median, power_integral, rel_power] # store row in a list
            rows.append(row) # store row in list
    
    bandpowers = pd.DataFrame(rows, columns = ['participant','session','chan','band','power_mean','power_median','power_integral','relative_power']) # make dataframe from list of lists
    ds_bandpower = xr.Dataset(bandpowers) # dataframe to dataset
    return ds_bandpower

def test_compute_bandpower():
    run_key = 'P02_music'
    ds_bandpower = compute_bandpower(run_key, **bandpower_params)
    print(ds_bandpower)
    

bandpower_job = jobtools.Job(precomputedir, 'bandpower', bandpower_params, compute_bandpower)
jobtools.register_job(bandpower_job)

def compute_all():
    jobtools.compute_job_list(bandpower_job, stim_keys, force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    # test_compute_bandpower()
    compute_all()


