from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job, artifact_job

def compute_respiration_features(run_key, **p):
    import physio 
    
    sub, ses = run_key.split('_')
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    resp_raw = raw_dataset['raw'].sel(chan='RespiNasale', time = slice(0, p['session_duration'])).values[:-1]
    srate = raw_dataset['raw'].attrs['srate']
    
    if p['inspiration_sign'] == '+':
        resp_raw = -resp_raw

    resp, resp_cycles = physio.compute_respiration(resp_raw, srate, parameter_preset='human_airflow')
    resp_cycles['participant'] = sub
    resp_cycles['session'] = ses
    
    artifacts = artifact_job.get(run_key).to_dataframe()
    
    resp_artifacted = resp_cycles.copy()
    resp_artifacted['artifact'] = 0
    
    for i, cycle_respi in resp_artifacted.iterrows():
        window_cycle_respi = np.arange(cycle_respi['inspi_index'], cycle_respi['next_inspi_index'])
        
        for j, artifact in artifacts.iterrows():
            window_artifact = np.arange(artifact['start_ind'], artifact['stop_ind'])
            
            if sum(np.in1d(window_cycle_respi, window_artifact)) != 0:
                resp_artifacted.loc[i, 'artifact'] = 1
    
    ds = xr.Dataset(resp_artifacted)
    
    return ds
    

def test_compute_respiration_features():
    run_key = 'P02_baseline'
    ds = compute_respiration_features(run_key, **respiration_features_params)
    print(ds.to_dataframe())
     
    
def compute_all():
    jobtools.compute_job_list(respiration_features_job, run_keys, force_recompute=False, engine='loop')

    
respiration_features_job = jobtools.Job(precomputedir, 'respiration_features', respiration_features_params, compute_respiration_features)
jobtools.register_job(respiration_features_job)


if __name__ == '__main__':
    # test_compute_respiration_features()
    compute_all()