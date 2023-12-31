from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job, artifact_job

def compute_respiration_features(run_key, **p):
    """
    Compute respiration features from raw resp signal and 
    annotate cycles according to cooccuring artifacting of EEG signals
    """
    import physio 
    
    sub, ses = run_key.split('_')
    
    raw_dataset = convert_vhdr_job.get(run_key) # load raw data
    
    resp_raw = raw_dataset['raw'].sel(chan='RespiNasale', time = slice(0, p['session_duration'])).values[:-1] # get raw resp
    srate = raw_dataset['raw'].attrs['srate']
    
    if p['inspiration_sign'] == '+': # invert signal to have inhalation below 0
        resp_raw = -resp_raw

    resp, resp_cycles = physio.compute_respiration(resp_raw, srate, parameter_preset='human_airflow') # preproc resp and compute resp cycle features
    resp_cycles['participant'] = sub
    resp_cycles['session'] = ses
    
    artifacts = artifact_job.get(run_key).to_dataframe() # load timestamps of EEG artifacts
    
    resp_artifacted = resp_cycles.copy()
    resp_artifacted['artifact'] = 0
    
    for i, cycle_respi in resp_artifacted.iterrows(): # loop over resp cycles
        window_cycle_respi = np.arange(cycle_respi['inspi_index'], cycle_respi['next_inspi_index']) # construct an index time vector of the current resp cycle
        
        for j, artifact in artifacts.iterrows(): # loop over artifacts
            window_artifact = np.arange(artifact['start_ind'], artifact['stop_ind']) # construct an index time vector of the current resp cycle 
            
            if sum(np.in1d(window_cycle_respi, window_artifact)) != 0: # see if index vector of current resp cycle in current artifact are overlapping or not
                resp_artifacted.loc[i, 'artifact'] = 1 # resp cycle is marked by 1 value if an artifacted overlap itself
    
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