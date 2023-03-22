import numpy as np
import xarray as xr
from compute_rri import rri_signal_job
from compute_resp_features import respiration_features_job
from bibliotheque import init_nan_da
import physio
import jobtools
from params import *
from configuration import *

def compute_rsa(run_key, **p):

    rri = rri_signal_job.get(run_key)['rri']
    resp = respiration_features_job.get(run_key).to_dataframe()
    
    times = np.arange(rri.size) / rri.attrs['srate']
    cycle_times = resp[['inspi_time','expi_time','next_inspi_time']].values
    
    rsa_cycles = physio.deform_traces_to_cycle_template(data = rri.values, 
                                                       times = times, 
                                                       cycle_times = cycle_times,
                                                       points_per_cycle = p['n_phase_bins'],
                                                       segment_ratios = p['segment_ratios'],
        
                                                       )
    
    rsa_da = xr.DataArray(data = rsa_cycles,
                          dims = ['cycle','phase'],
                          coords = {'cycle':np.arange(rsa_cycles.shape[0]), 'phase':np.linspace(0, 1 , p['n_phase_bins'])})
    
    ds_rsa = xr.Dataset()
    ds_rsa['rsa'] = rsa_da
    return ds_rsa

def test_compute_rsa():
    run_key = 'P01_baseline'
    ds = compute_rsa(run_key, **rsa_params)
    print(ds)
    
rsa_job = jobtools.Job(precomputedir, 'rsa', rsa_params, compute_rsa)
jobtools.register_job(rsa_job)
   
    

def compute_all():
    jobtools.compute_job_list(rsa_job, run_keys, force_recompute=False, engine='loop')


if __name__ == '__main__':
    # test_compute_rsa()

    compute_all()