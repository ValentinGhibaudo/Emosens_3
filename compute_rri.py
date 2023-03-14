from configuration import *
from params import *

import xarray as xr


import jobtools

from preproc import convert_vhdr_job

import matplotlib.pyplot as plt

#~ import xarray as xr
#~ import ghibtools as gh
#~ import matplotlib.pyplot as plt
#~ import pandas as pd
#~ import numpy as np
#~ from bibliotheque import get_odor_from_session
#~ import physio
#~ from params import srate, subject_keys, ecg_inversion, blocs, count_trials, trial_durations

#~ concat = []

#~ for run_key in subject_keys:
    #~ print(run_key)
    
    #~ triggs = pd.read_excel(f'../Tables/timestamps/{run_key}_timestamps.xlsx', index_col = [0,1,2,3])
    
    #~ for session in ['ses02','ses03','ses04']:
        
        #~ ecg = xr.open_dataarray(f'../Preprocessing/Data_Preprocessed/raw_{run_key}_{session}.nc').sel(chan = 'ECG').data * ecg_inversion[run_key]
        #~ ecg_preproc = physio.preprocess(ecg, srate)
        #~ peaks = physio.detect_peak(ecg_preproc, srate)
        #~ clean_peaks = physio.clean_ecg_peak(peaks, srate)
        
        #~ for bloc in blocs:
            #~ for trial in count_trials[bloc]:
                #~ start = triggs.loc[(session, bloc,trial,'start'),'timestamp']
                #~ stop = start + trial_durations[bloc]
                #~ time = np.arange(start, stop , 1 / srate)
                #~ start_idx = int(start * srate)
                #~ stop_idx = int(stop * srate)

                #~ mask_peaks = (clean_peaks >= start_idx) & (clean_peaks < stop_idx)
                #~ peaks_of_the_trial = clean_peaks[mask_peaks] - start_idx
                
                #~ metrics = physio.compute_ecg_metrics(peaks_of_the_trial, srate)

                #~ metrics.insert(0, 'trial', trial)
                #~ metrics.insert(0, 'bloc', bloc)
                #~ metrics.insert(0, 'odeur', get_odor_from_session(run_key, session))
                #~ metrics.insert(0, 'session', session)
                #~ metrics.insert(0, 'participant', run_key)
                #~ concat.append(metrics)
        
#~ hrv_metrics = pd.concat(concat)
#~ hrv_metrics.to_excel('../Tables/hrv_physiotools.xlsx')



def compute_ecg(run_key, **p):
    import physio
    #~ print(run_key, p)
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    ecg_raw = raw_dataset['raw'].sel(chan='ECG').values
    srate = raw_dataset['raw'].attrs['srate']
    
    #~ print(ecg_raw.shape, srate)
    
    subject_key, ses_key = run_key.split('_')

    inv = p['ecg_inversion'][subject_key]
    ecg_raw = ecg_raw * inv
    

    ecg = physio.preprocess(ecg_raw, srate, band=[p['low'], p['high']], ftype=p['ftype'], order=p['order'], normalize=True)
    ecg_peaks = physio.detect_peak(ecg, srate, thresh=p['threshold'], exclude_sweep_ms=p['exclude_sweep_ms'])
    #~ print(ecg_peaks.shape)
    
    ecg_peaks = physio.clean_ecg_peak(ecg, srate, ecg_peaks, min_interval_ms=p['min_interval_ms'])
    #~ print(ecg_peaks.shape)
    
    #~ return clean_ecg, ecg_R_peaks

    
    
    ds = xr.Dataset()
    ds['ecg'] = ecg
    ds['ecg'].attrs['srate'] = srate
    ds['ecg_peaks'] = ecg_peaks
    
    
    return ds
    



def test_compute_ecg():
    run_key = 'P01_ses02'
    ds = compute_ecg(run_key, **ecg_params)
    print(ds)
    
    #~ ds = ecg_job.get(run_key)
    #~ print(ds)
    


def compute_rri_signal(run_key, **p):
    import physio
    
    ds_ecg = ecg_job.get(run_key)
    ecg = ds_ecg['ecg'].values
    srate = ds_ecg['ecg'].attrs['srate']
    ecg_peaks = ds_ecg['ecg_peaks'].values
    
    times = np.arange(ecg.size) / srate

    rri = physio.compute_instantaneous_rr_interval(ecg_peaks, srate, times,
                                                min_interval_ms=p['min_interval_ms'],
                                                max_interval_ms=p['max_interval_ms'],
                                                units='bpm', interpolation_kind=p['interpolation_kind'])
    

    ds = xr.Dataset()
    ds['rri'] = rri
    ds['rri'].attrs['srate'] = srate
    
    return ds


def test_compute_rri_signal():
    
    
    # run_key = 'P01_ses02'
    
    run_key = 'P10_ses02'
    
    ds = compute_rri_signal(run_key, **rri_signal_params)
    print(ds)
    
    rri = ds['rri'].values
    srate = ds['rri'].attrs['srate']
    
    times = np.arange(rri.size) / srate
    
    fig, ax = plt.subplots()
    
    ax.plot(times, rri)
    plt.show()
    

def compute_all():

    #~ jobtools.compute_job_list(ecg_job, run_keys[:4], force_recompute=False, engine='loop')
    
    # jobtools.compute_job_list(rri_signal_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(rri_signal_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 3)

 
    


ecg_job = jobtools.Job(precomputedir, 'ecg', ecg_params, compute_ecg)
jobtools.register_job(ecg_job)

rri_signal_job = jobtools.Job(precomputedir, 'rri_signal', rri_signal_params, compute_rri_signal)
jobtools.register_job(rri_signal_job)




if __name__ == '__main__':
    #~ test_compute_ecg()
    # test_compute_rri_signal()
    
    compute_all()



