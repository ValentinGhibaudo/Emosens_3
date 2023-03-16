from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job

def compute_respiration_features(run_key, **p):
    import physio 
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    resp_raw = raw_dataset['raw'].sel(chan='RespiNasale', time = slice(0, p['session_duration'])).values[:-1]
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
    run_key = 'P02_baseline'
    ds = compute_respiration_features(run_key, **respiration_features_params)
    print(ds)
    

    
    
def compute_all():
    jobtools.compute_job_list(respiration_features_job, run_keys, force_recompute=False, engine='loop')

    
respiration_features_job = jobtools.Job(precomputedir, 'respiration_features', respiration_features_params, compute_respiration_features)
jobtools.register_job(respiration_features_job)



if __name__ == '__main__':
    # test_compute_respiration_features()
    compute_all()