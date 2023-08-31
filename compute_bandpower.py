from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from compute_psd import psd_baselined_job

def compute_bandpower(run_key, **p):
    participant, session = run_key.split('_')
    psd = psd_baselined_job.get(run_key)['psd_baselined']
    
    rows = []
    
    for chan in psd['chan'].values:
        total_power = psd.loc[chan, p['total_band'][0]:p['total_band'][0]].mean('freq')
        for band, bornes in p['fbands'].items():
            Psd_chan_band = psd.loc[chan, bornes[0]:bornes[1]]
            power_mean = Psd_chan_band.mean('freq')
            power_median = Psd_chan_band.median('freq')
            power_integral = Psd_chan_band.integrate('freq')
            rel_power = power_integral / total_power

            row = [participant, session, chan, band, power_mean, power_median, power_integral, rel_power]
            rows.append(row)
    
    bandpowers = pd.DataFrame(rows, columns = ['participant','session','chan','band','power_mean','power_median','power_integral','relative_power'])
    ds_bandpower = xr.Dataset(bandpowers)
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


