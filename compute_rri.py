from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job
from compute_resp_features import respiration_features_job
import matplotlib.pyplot as plt


def compute_ecg(run_key, **p):
    import physio
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    ecg_raw = raw_dataset['raw'].sel(chan='ECG', time = slice(0, p['session_duration'])).values[:-1]
    srate = raw_dataset['raw'].attrs['srate']
    
    subject_key, ses_key = run_key.split('_')

    inv = p['ecg_inversion'][subject_key]
    ecg_raw_inv = ecg_raw * inv
    
    parameters = physio.get_ecg_parameters('human_ecg')
    parameters['peak_detection']['thresh'] = p['thresh']
    ecg, ecg_peaks = physio.compute_ecg(ecg_raw_inv, srate, parameters=parameters)
    
    ds = xr.Dataset()
    ds['ecg'] = ecg
    ds.attrs['srate'] = srate
    
    return ds
    

def test_compute_ecg():
    run_key = 'P05_baseline'
    ds = compute_ecg(run_key, **ecg_params)
    print(ds)
    print(ds.attrs['srate'])
    

# ECG PEAKS    
def compute_ecg_peaks(run_key, **p):
    import physio
    raw_dataset = convert_vhdr_job.get(run_key)
    
    ecg_raw = raw_dataset['raw'].sel(chan='ECG', time = slice(0, p['session_duration'])).values[:-1]
    srate = raw_dataset['raw'].attrs['srate']

    subject_key, ses_key = run_key.split('_')

    inv = p['ecg_inversion'][subject_key]
    ecg_raw_inv = ecg_raw * inv
    
    parameters = physio.get_ecg_parameters('human_ecg')
    parameters['peak_detection']['thresh'] = p['thresh']
    ecg, ecg_peaks = physio.compute_ecg(ecg_raw_inv, srate, parameters=parameters)
    
    # fig, ax = plt.subplots()
    # ax.plot(ecg)
    # ax.plot(ecg_peaks['peak_index'], ecg[ecg_peaks['peak_index']], 'o')
    # plt.show()
    
    ds = xr.Dataset(ecg_peaks)
    return ds
    
def test_compute_ecg_peaks():
    run_key = 'P05_baseline'
    ds = compute_ecg_peaks(run_key, **ecg_params)
    print(ds.to_dataframe())
    

# RRI VIEWER
def compute_rri_signal(run_key, **p):
    import physio
    
    ds_ecg = ecg_job.get(run_key)
    ecg = ds_ecg['ecg'].values
    srate = ds_ecg.attrs['srate']
    ecg_peaks = ecg_peak_job.get(run_key).to_dataframe()
    
    times = np.arange(ecg.size) / srate

    limits = [p['min_interval'], p['max_interval']]

    rri = physio.compute_instantaneous_rate(ecg_peaks = ecg_peaks, 
                                            new_times = times,
                                            limits = limits,
                                            units='bpm', 
                                            interpolation_kind=p['interpolation_kind'])
    
    da_rri = xr.DataArray(data = rri, dims = ['time'], coords = {'time':times})
    da_rri.attrs['srate'] = srate
    ds = xr.Dataset()
    ds['rri'] = da_rri
    return ds


def test_compute_rri_signal():
    run_key = 'P05_baseline'
    
    ds = compute_rri_signal(run_key, **rri_signal_params)
    print(ds)
    
    rri = ds['rri'].values
    srate = ds['rri'].attrs['srate']
    
#     times = np.arange(rri.size) / srate
    
#     fig, ax = plt.subplots()
    
#     ax.plot(times, rri)
#     plt.show()

def ecg_peaks_coupling(run_key, **p):
    sub, ses = run_key.split('_')
    ecg_peaks = ecg_peak_job.get(run_key).to_dataframe()
    rsp_features = respiration_features_job.get(run_key).to_dataframe()
    
    ecg_peak_angles = ecg_peaks.copy()
    ecg_peak_angles['Participant'] = sub
    ecg_peak_angles['session'] = ses
    ecg_peak_angles['Resp_Angle'] = np.nan

    for i, row in rsp_features.iterrows():
        mask_r_peaks_in_cycle = (ecg_peaks['peak_time'] >= row['inspi_time']) & (ecg_peaks['peak_time'] < row['next_inspi_time'])
        r_peaks_in_cycle = ecg_peaks[mask_r_peaks_in_cycle]
        ecg_peak_angles.loc[r_peaks_in_cycle.index, 'Resp_Angle'] = ((r_peaks_in_cycle['peak_time'].values - row['inspi_time']) / row['cycle_duration'])

    return xr.Dataset(ecg_peak_angles)

def test_ecg_peaks_coupling():
    run_key = 'P05_baseline'
    ds = ecg_peaks_coupling(run_key, **ecg_params)
    res = ds.to_dataframe()
    print(res)
    
ecg_job = jobtools.Job(precomputedir, 'ecg', ecg_params, compute_ecg)
jobtools.register_job(ecg_job)

ecg_peak_job = jobtools.Job(precomputedir, 'ecg_peak', ecg_params, compute_ecg_peaks)
jobtools.register_job(ecg_peak_job)

rri_signal_job = jobtools.Job(precomputedir, 'rri_signal', rri_signal_params, compute_rri_signal)
jobtools.register_job(rri_signal_job)

ecg_peaks_coupling_job = jobtools.Job(precomputedir, 'ecg_peaks_coupling', ecg_params, ecg_peaks_coupling)
jobtools.register_job(ecg_peaks_coupling_job)

def compute_all():
    # jobtools.compute_job_list(ecg_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(ecg_peak_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(rri_signal_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(ecg_peaks_coupling_job, run_keys, force_recompute=True, engine='loop')


if __name__ == '__main__':
    # test_compute_ecg()
    # test_compute_ecg_peaks()
    # test_compute_rri_signal()
    # test_ecg_peaks_coupling()
    
    compute_all()



