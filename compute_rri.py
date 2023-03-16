from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job
import matplotlib.pyplot as plt


def compute_ecg(run_key, **p):
    import physio
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    ecg_raw = raw_dataset['raw'].sel(chan='ECG', time = slice(0, p['session_duration'])).values[:-1]
    srate = raw_dataset['raw'].attrs['srate']
    
    subject_key, ses_key = run_key.split('_')

    inv = p['ecg_inversion'][subject_key]
    ecg_raw = ecg_raw * inv
    
    ecg = physio.preprocess(ecg_raw, srate, band=[p['low'], p['high']], ftype=p['ftype'], order=p['order'], normalize=True)
    ecg_peaks = physio.detect_peak(ecg, srate, thresh=p['threshold'], exclude_sweep_ms=p['exclude_sweep_ms'])
    
    ecg_peaks = physio.clean_ecg_peak(ecg, srate, ecg_peaks, min_interval_ms=p['min_interval_ms'])

    ds = xr.Dataset()
    ds['ecg'] = ecg
    ds['ecg'].attrs['srate'] = srate
    ds['ecg_peaks'] = ecg_peaks
    
    return ds
    



def test_compute_ecg():
    run_key = 'P02_baseline'
    ds = compute_ecg(run_key, **ecg_params)
    print(ds)
    

def compute_rri_signal(run_key, **p):
    import physio
    
    ds_ecg = ecg_job.get(run_key)
    ecg = ds_ecg['ecg'].values
    srate = ds_ecg['ecg'].attrs['srate']
    ecg_peaks = ds_ecg['ecg_peaks'].values
    
    times = np.arange(ecg.size) / srate
    
    peak_times = ecg_peaks / srate
    limits = [p['min_interval'], p['max_interval']]

    rri = physio.compute_instantaneous_rate(peak_times = peak_times, 
                                            new_times = times,
                                            limits = limits,
                                            units='bpm', 
                                            interpolation_kind=p['interpolation_kind'])
    
    ds = xr.Dataset()
    ds['rri'] = rri
    ds['rri'].attrs['srate'] = srate
    
    return ds


def test_compute_rri_signal():
    
    
    run_key = 'P10_baseline'
    
    ds = compute_rri_signal(run_key, **rri_signal_params)
    print(ds)
    
    rri = ds['rri'].values
    srate = ds['rri'].attrs['srate']
    
    times = np.arange(rri.size) / srate
    
    fig, ax = plt.subplots()
    ax.plot(times, rri)
    plt.show()
    

def compute_all():

    # jobtools.compute_job_list(ecg_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(rri_signal_job, run_keys, force_recompute=False, engine='loop')


ecg_job = jobtools.Job(precomputedir, 'ecg', ecg_params, compute_ecg)
jobtools.register_job(ecg_job)

rri_signal_job = jobtools.Job(precomputedir, 'rri_signal', rri_signal_params, compute_rri_signal)
jobtools.register_job(rri_signal_job)


if __name__ == '__main__':
    # test_compute_ecg()
    # test_compute_rri_signal()
    
    compute_all()



