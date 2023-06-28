from configuration import *
from params import *
from bibliotheque import init_nan_da
import ghibtools as gh

import xarray as xr
import pandas as pd

import physio
import jobtools

from preproc import convert_vhdr_job, eeg_interp_artifact_job
from compute_rri import rri_signal_job
from compute_resp_features import respiration_features_job




def norm(a):
    return (a - np.mean(a)) / np.std(a)

def center(a):
    return a - np.mean(a)


# JOB CYCLE SIGNAL

def cycle_signal(run_key, **p):
    
    chans = p['chans']
    chans = chans + ['resp_nose','resp_mouth','heart']
    
    eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = eeg.attrs['srate']
    
    rri = rri_signal_job.get(run_key)['rri']
    resp_sig = convert_vhdr_job.get(run_key)['raw']
    
    resp_features = respiration_features_job.get(run_key).to_dataframe()

    times = eeg.coords['time'].values

    cycle_times = resp_features[['inspi_time','expi_time','next_inspi_time']].values
    
    da_cycle_signals = None
    
    for chan in chans:
        if chan == 'resp_nose':
            data = resp_sig.sel(chan='RespiNasale', time = slice(0, p['session_duration'])).values[:-1]
        elif chan == 'resp_mouth':
            data = -resp_sig.sel(chan='RespiVentrale', time = slice(0, p['session_duration'])).values[:-1]
        elif chan == 'heart':
            data = rri.values
        else:
            data = eeg.sel(chan = chan).values
            
        cycle_signals = physio.deform_traces_to_cycle_template(data = data, 
                                                           times = times, 
                                                           cycle_times = cycle_times,
                                                           points_per_cycle = p['n_phase_bins'],
                                                           segment_ratios = p['segment_ratios'],
                                                           )
        # if not chan == 'heart':
        #     cycle_signals = np.apply_along_axis(norm , 1 , cycle_signals)
        
        if not chan == 'heart':
            cycle_signals = np.apply_along_axis(center , 1 , cycle_signals)
        elif chan in ['resp_nose','resp_mouth']:
            cycle_signals = np.apply_along_axis(norm , 1 , cycle_signals)
        


        mask_cycles = (resp_features['artifact'] == 0)
        keep_cycles = resp_features[mask_cycles].index

        cycle_signal = cycle_signals[keep_cycles,:]

        m = np.mean(cycle_signal, axis = 0)

        if da_cycle_signals is None:
            da_cycle_signals = init_nan_da({'chan':chans, 'phase':np.linspace(0,1,p['n_phase_bins'])})

        da_cycle_signals.loc[chan , : ] = m
            
    ds = xr.Dataset()
    ds['cycle_signal'] = da_cycle_signals    
                                                        
    return ds
    
def test_cycle_signal():
    run_key = 'P01_baseline'
    ds = cycle_signal(run_key, **cycle_signal_params)
    print(ds)
    
cycle_signal_job = jobtools.Job(precomputedir, 'cycle_signal', cycle_signal_params, cycle_signal)
jobtools.register_job(cycle_signal_job)





# JOB MODULATION DATAFRAME

def modulation_cycle_signal(run_key, **p):
    da_cycle_signals = cycle_signal_job.get(run_key)['cycle_signal']
    participant,session = run_key.split('_')
    
    rows = []
    for chan in da_cycle_signals.coords['chan'].values:
        if chan in ['resp_nose','resp_mouth','heart']:
            continue
        cycle_sig = da_cycle_signals.sel(chan = chan).values
        row = [participant, session, chan, np.ptp(cycle_sig)]
        rows.append(row)
    
    df = pd.DataFrame(rows, columns = ['participant','session','chan', 'amplitude'])
    
    return xr.Dataset(df)

def test_modulation_cycle_signal():
    run_key = 'P01_baseline'
    ds = modulation_cycle_signal(run_key, **cycle_signal_modulation_params).to_dataframe()
    print(ds)
    
modulation_cycle_signal_job = jobtools.Job(precomputedir, 'modulation_cycle_signal', cycle_signal_modulation_params, modulation_cycle_signal)
jobtools.register_job(modulation_cycle_signal_job)
    
    
    
    
    
# COMPUTE ALL 
def compute_all():
    # jobtools.compute_job_list(cycle_signal_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 6)
    jobtools.compute_job_list(modulation_cycle_signal_job, run_keys, force_recompute=False, engine='loop')

if __name__ == '__main__':
    # test_cycle_signal()
    # test_modulation_cycle_signal()

    compute_all()
   
    
