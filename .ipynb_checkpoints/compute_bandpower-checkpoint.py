from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from compute_psd import psd_eeg_job

def compute_bandpower(run_key, **p):
    participant, session = run_key.split('_')
    psd_eeg = psd_eeg_job.get(run_key)['interp']
    srate = psd_eeg.attrs['srate']
    
    rows = []
    
    for chan in psd_eeg.coords['chan'].values:

        Pxx = psd_eeg.sel(chan=chan)

        for band, bornes in p['fbands'].items():

            power_mean = float(Pxx.loc[bornes[0]:bornes[1]].mean('freq'))
            power_median = float(Pxx.loc[bornes[0]:bornes[1]].median('freq'))
            power_integral = float(Pxx.loc[bornes[0]:bornes[1]].integrate('freq'))
            total_power = float(Pxx.loc[ p['total_band'][0] : p['total_band'][1] ].integrate('freq'))
            rel_power = power_integral / total_power

            row = [participant, session, chan, band, power_mean, power_median, power_integral, rel_power]
            rows.append(row)
    
    bandpowers = pd.DataFrame(rows, columns = ['participant','session','chan','band','power_mean','power_median','power_integral','relative_power'])
    ds_bandpower = xr.Dataset(bandpowers)
    return ds_bandpower

def test_compute_bandpower():
    run_key = 'P02_baseline'
    ds_bandpower = compute_bandpower(run_key, **bandpower_params)
    

bandpower_job = jobtools.Job(precomputedir, 'bandpower', bandpower_params, compute_bandpower)
jobtools.register_job(bandpower_job)

def compute_all():
    jobtools.compute_job_list(bandpower_job, run_keys, force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    test_compute_bandpower()
    # compute_all()


