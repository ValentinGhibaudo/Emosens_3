from configuration import *
from params import *

import xarray as xr


import jobtools

from preproc import convert_vhdr_job
from store_timestamps import timestamps_job


def compute_respiration_features(run_key, **p):
    #~ print(run_key)
    import physio 
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    resp_raw = raw_dataset['raw'].sel(chan='RespiNasale').values
    srate = raw_dataset['raw'].attrs['srate']
    
    if p['inspiration_sign'] == '+':
        resp_raw = -resp_raw

    # preprocessing
    resp = physio.preprocess(resp_raw, srate, band=p['low_pass_freq'], btype='lowpass', ftype='bessel',
                                order=p['filter_order'], normalize=False)
    resp = physio.smooth_signal(resp, srate, win_shape='gaussian', sigma_ms=p['smooth_sigma_ms'])

    baseline = physio.get_empirical_mode(resp)
    espilon = (np.quantile(resp, 0.75) - np.quantile(resp, 0.25)) / 100.
    baseline_detect = baseline - espilon * 5.

    cycles = physio.detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=baseline_detect,
                                              inspiration_adjust_on_derivative=False)
    cycle_features = physio.compute_respiration_cycle_features(resp, srate, cycles, baseline=baseline)

    cycle_features_clean = physio.clean_respiration_cycles(resp, srate, cycle_features, baseline,
                                                           low_limit_log_ratio=3)

    
    ds = xr.Dataset(cycle_features_clean)
    
    return ds
    



def test_compute_respiration_features():

    ds = compute_respiration_features(run_key, **respiration_features_params)
    print(ds)
    
    # ds = respiration_features_job.get(run_key)
    # print(ds)
    # print(ds.to_dataframe())s
    
    
    
    
    
    
def label_respiration_features(run_key, **p):
    ts = timestamps_job.get(run_key).to_dataframe().set_index(['bloc','trial'])
    
    cycle_features = respiration_features_job.get(run_key).to_dataframe()
    
    cycle_features_labelized = cycle_features.copy()
    
    for bloc in p['blocs']:
        for trial in ts.loc[bloc,:].index:
            t_start = ts.loc[(bloc, trial), 'timestamp']
            t_stop = t_start + ts.loc[(bloc, trial), 'duration']
            mask = (cycle_features['inspi_time'] > t_start) & (cycle_features['next_inspi_time'] < t_stop)
            cycle_features_labelized.loc[mask, 'bloc'] = bloc
    
    ds = xr.Dataset(cycle_features_labelized)
    
    return ds

def test_label_respiration_features():
    run_key = 'P02_ses02'
    ds = label_respiration_features(run_key, **label_resp_features_params)
    # ds = label_respiration_features_job.get(run_key)
    print(ds.to_dataframe())
    

    
    
    


def compute_all():

    # jobtools.compute_job_list(compute_respiration_features_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(label_respiration_features_job, run_keys, force_recompute=False, engine='loop')

 
    


respiration_features_job = jobtools.Job(precomputedir, 'respiration_features', respiration_features_params, compute_respiration_features)
jobtools.register_job(respiration_features_job)

label_respiration_features_job = jobtools.Job(precomputedir, 'label_respiration_features', label_resp_features_params, label_respiration_features)
jobtools.register_job(label_respiration_features_job)




if __name__ == '__main__':
    # test_compute_respiration_features()
    # test_label_respiration_features()
    
    compute_all()