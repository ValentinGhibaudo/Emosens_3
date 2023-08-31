from configuration import *
from params import *
import xarray as xr
import physio
import jobtools
import ghibtools as gh
from bibliotheque import init_nan_da
from preproc import eeg_interp_artifact_job


### PSD LF
def compute_psd(run_key, **p):
    
    eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = eeg.attrs['srate']
    
    psd = None
    
    for chan in eeg.coords['chan'].values:
        sig = eeg.sel(chan=chan).values
        f , Pxx = gh.spectre(sig, srate, p['lowest_freq'])
        
        if psd is None:
            psd = init_nan_da({'chan':eeg.coords['chan'].values, 
                               'freq':f})
            
        psd.loc[chan, : ] = Pxx
                    
    psd.attrs['srate'] = srate
    psd_ds = xr.Dataset()
    psd_ds['psd'] = psd
    return psd_ds


def test_compute_psd():
    run_key = 'P02_baseline'
    psd_ds = compute_psd(run_key, **psd_params)
    print(psd_ds)
    
    
psd_eeg_job = jobtools.Job(precomputedir, 'psd_eeg', psd_params, compute_psd)
jobtools.register_job(psd_eeg_job)


### PSD BANDPOWER
def compute_psd_bandpower(run_key, **p):
    
    eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = eeg.attrs['srate']
    
    psd = None
    
    for chan in eeg.coords['chan'].values:
        sig = eeg.sel(chan=chan).values
        f , Pxx = gh.spectre(sig, srate, p['lowest_freq'])
        
        if psd is None:
            psd = init_nan_da({'chan':eeg.coords['chan'].values, 
                               'freq':f})
            
        psd.loc[chan, : ] = Pxx            
    psd.attrs['srate'] = srate
    psd_ds = xr.Dataset()
    psd_ds['psd_bandpower'] = psd
    return psd_ds

def test_compute_psd_bandpower():
    run_key = 'P02_baseline'
    psd_ds = compute_psd_bandpower(run_key, **psd_bandpower_params)
    print(psd_ds)
    
psd_bandpower_job = jobtools.Job(precomputedir, 'psd_bandpower', psd_bandpower_params, compute_psd_bandpower)
jobtools.register_job(psd_bandpower_job)

# psd_baselined
def psd_baselined(run_key, **p):
    sub, ses = run_key.split('_')
    psd_stim = psd_bandpower_job.get(run_key)['psd_bandpower']
    psd_baseline = psd_bandpower_job.get(f'{sub}_baseline')['psd_bandpower']
    
    psd_stim_baselined = psd_stim.copy()
    psd_stim_baselined[:] = psd_stim.values / psd_baseline.values
    ds = xr.Dataset()
    ds['psd_baselined'] = psd_stim_baselined
    return ds

def test_psd_baselined():
    run_key = 'P02_odor'
    psd_ds = psd_baselined(run_key, **psd_baselined_params)
    print(psd_ds)
    
psd_baselined_job = jobtools.Job(precomputedir, 'psd_baselined', psd_baselined_params, psd_baselined)
jobtools.register_job(psd_baselined_job)



def compute_all():
    # jobtools.compute_job_list(psd_eeg_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(psd_bandpower_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(psd_baselined_job, stim_keys, force_recompute=False, engine='loop')
    
    
if __name__ == '__main__':
    # test_compute_psd()
    # test_compute_psd_bandpower()
    # test_psd_baselined()
    compute_all()
