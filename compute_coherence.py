from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from bibliotheque import init_nan_da
from preproc import eeg_interp_artifact_job, convert_vhdr_job
import physio


def compute_coherence(run_key, **p):

    eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = eeg.attrs['srate']
    
    resp_raw = convert_vhdr_job.get(run_key)['raw'].sel(chan = p['resp_chan'], time = slice(0, p['session_duration'])).values[:-1]
    resp_sig, resp_cycles = physio.compute_respiration(resp_raw, srate, parameter_preset='human_airflow')
    
    da_cxy = None
    
    for chan in eeg.coords['chan'].values:
        sig = eeg.loc[chan,:].values  
        
        f, Cxy = gh.coherence(sig, resp_sig, srate, p['lowest_freq_coherence'], nfft_factor = p['nfft_factor'], n_cycles = p['n_cycles'])

        if da_cxy is None:
            da_cxy = init_nan_da({'chan':eeg.coords['chan'].values, 'freq':f})
                                  
        da_cxy.loc[chan, : ] = Cxy

    ds_coherence = xr.Dataset()
    ds_coherence['coherence'] = da_cxy

    return ds_coherence


def coherence_at_resp(run_key, **p):
    participant, session = run_key.split('_')
    
    coherence_params = p['coherence_params']
    
    coherence = coherence_job.get(run_key)
    coherence = coherence['coherence']
    
    resp_sig = convert_vhdr_job.get(run_key)['raw'].sel(chan = coherence_params['resp_chan'], time = slice(0, coherence_params['session_duration'])).values[:-1]
    resp_sig, resp_cycles = physio.compute_respiration(resp_sig, srate, parameter_preset='human_airflow')
    
    rows = []

    f_resp, Pxx_resp = gh.spectre(resp_sig, 
                                  srate, 
                                  coherence_params['lowest_freq_psd_resp'], 
                                  nfft_factor = coherence_params['nfft_factor'], 
                                  n_cycles = coherence_params['n_cycles'])

    Pxx_resp_sel = Pxx_resp[f_resp<1]
    max_resp = np.max(Pxx_resp_sel)
    argmax_resp = np.argmax(Pxx_resp_sel)
    fmax_resp = f_resp[argmax_resp]

    for chan in coherence.coords['chan'].values:

        Cxy_at_resp = coherence.loc[chan, fmax_resp].values

        row = [participant, session, chan, fmax_resp, max_resp,  Cxy_at_resp]
        rows.append(row)
    
    df_coherence_at_resp = pd.DataFrame(rows, columns = ['participant','session','chan', 'fmax_resp','max_resp', 'max_coherence'])
    ds_coherence_at_resp = xr.Dataset(df_coherence_at_resp)
    return ds_coherence_at_resp


def test_compute_coherence():
    run_key = 'P02_baseline'
    
    ds_coherence = compute_coherence(run_key, **coherence_params)
    print(ds_coherence)
    
    
def test_compute_coherence_at_resp():
    run_key = 'P02_baseline'
    ds_coherence_at_resp = coherence_at_resp(run_key, **coherence_at_resp_params)
    print(ds_coherence_at_resp.to_dataframe())
    

coherence_job = jobtools.Job(precomputedir, 'coherence', coherence_params, compute_coherence)
jobtools.register_job(coherence_job)

coherence_at_resp_job = jobtools.Job(precomputedir, 'coherence_at_resp', coherence_at_resp_params, coherence_at_resp)
jobtools.register_job(coherence_at_resp_job)


def compute_all():
    # jobtools.compute_job_list(coherence_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 3)
    # jobtools.compute_job_list(coherence_job, run_keys, force_recompute=False, engine='loop')
    
    # jobtools.compute_job_list(coherence_at_resp_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(coherence_at_resp_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 6)
    
if __name__ == '__main__':
    # test_compute_coherence()
    # test_compute_coherence_at_resp()
    compute_all()           
                    
                    
                
                
                
                
                
    