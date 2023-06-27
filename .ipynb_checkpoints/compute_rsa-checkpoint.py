import numpy as np
import xarray as xr
from compute_rri import rri_signal_job, ecg_peak_job
from compute_resp_features import respiration_features_job
from bibliotheque import init_nan_da
import physio
import jobtools
from params import *
from configuration import *

def compute_rsa_phase(run_key, **p):
    resp_cycles = respiration_features_job.get(run_key).to_dataframe()
    ecg_peaks = ecg_peak_job.get(run_key).to_dataframe()

    rsa_cycles, cyclic_cardiac_rate = physio.compute_rsa(resp_cycles,
                                                         ecg_peaks,
                                                         srate=10.,
                                                         two_segment=True,
                                                         points_per_cycle=p['n_phase_bins'],
                                                        )
    
    rsa_da = xr.DataArray(data = cyclic_cardiac_rate,
                          dims = ['cycle','phase'],
                          coords = {'cycle':np.arange(rsa_cycles.shape[0]), 'phase':np.linspace(0, 1 , p['n_phase_bins'])})

    ds_rsa = xr.Dataset()
    ds_rsa['rsa'] = rsa_da
    return ds_rsa

def test_compute_rsa_phase():
    run_key = 'P05_baseline'
    ds = compute_rsa_phase(run_key, **rsa_params)
    print(ds)
    
rsa_phase_job = jobtools.Job(precomputedir, 'rsa_phase', rsa_params, compute_rsa_phase)
jobtools.register_job(rsa_phase_job)


def compute_rsa_features(run_key, **p):
    sub, ses = run_key.split('_')
    resp_cycles = respiration_features_job.get(run_key).to_dataframe()
    ecg_peaks = ecg_peak_job.get(run_key).to_dataframe()

    rsa_cycles, cyclic_cardiac_rate = physio.compute_rsa(resp_cycles,
                                                         ecg_peaks,
                                                         srate=10.,
                                                         two_segment=True,
                                                         points_per_cycle=p['n_phase_bins'],
                                                        )
    
    rsa_cycles['participant'] = sub
    rsa_cycles['session'] = ses
    
    ds_rsa_features = xr.Dataset(rsa_cycles)
    return ds_rsa_features

def test_compute_rsa_features():
    run_key = 'P05_baseline'
    ds = compute_rsa_features(run_key, **rsa_params)
    print(ds.to_dataframe())

rsa_features_job = jobtools.Job(precomputedir, 'rsa_features', rsa_params, compute_rsa_features)
jobtools.register_job(rsa_features_job)
   
   
    

def compute_all():
    # jobtools.compute_job_list(rsa_phase_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(rsa_features_job, run_keys, force_recompute=False, engine='loop')


if __name__ == '__main__':
    # test_compute_rsa_phase()
    # test_compute_rsa_features()

    compute_all()
    