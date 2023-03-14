from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
import physio
from bibliotheque import init_nan_da
from preproc import preproc_job


def compute_hilbert_envelope(run_key, **p):
    
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    
    eeg = preproc_job.get(run_key)
    eeg = eeg['eeg_clean']
    srate = eeg.attrs['srate']
    
    eeg_envelope = None
    
    for chan in eeg.coords['chan'].values:
                    
        eeg_sig = eeg.loc[ chan , :].values
                    
        for band, bornes in p['fbands'].items():

            eeg_sig_filtered = gh.iirfilt(eeg_sig, srate, lowcut=bornes[0], highcut=bornes[1], order = 4)
            eeg_sig_amp = gh.get_amp(eeg_sig_filtered)
            
            if eeg_envelope is None:
                eeg_envelope = init_nan_da({'chan':eeg.coords['chan'].values, 'band':list(p['fbands'].keys()) , 'time':eeg.coords['time'].values})
            eeg_envelope.loc[chan, band , : ] = eeg_sig_amp
    eeg_envelope.attrs['srate'] = srate
    eeg_envelope_ds = xr.Dataset()
    eeg_envelope_ds['eeg_envelope'] = eeg_envelope
    return eeg_envelope_ds

def test_compute_hilbert_envelope():
    run_key = 'P01_ses03'
    eeg_envelope_ds = compute_hilbert_envelope(run_key, **hilbert_params)
    print(eeg_envelope_ds)
    

hilbert_envelope_job = jobtools.Job(precomputedir, 'hilbert_envelope', hilbert_params, compute_hilbert_envelope)
jobtools.register_job(hilbert_envelope_job)


def compute_all():
    jobtools.compute_job_list(hilbert_envelope_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 4)
    
if __name__ == '__main__':
    # test_compute_hilbert_envelope()
    compute_all()              




