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
    """
    Cyclically deform EEG (and respi) signals according to respiratory timestamps 
    of each respiratory cycle and compute average evoked potential
    """
    
    chans = p['chans'] # load computing chans
    chans = chans + ['resp_nose','resp_mouth','heart'] # add physio channels
    
    eeg = eeg_interp_artifact_job.get(run_key)['interp'] # load EEG preprocessed
    srate = eeg.attrs['srate']
    
    rri = rri_signal_job.get(run_key)['rri'] # load RRI signal
    resp_sig = convert_vhdr_job.get(run_key)['raw'] # load raw resp signal
    
    resp_features = respiration_features_job.get(run_key).to_dataframe() # load resp features

    times = eeg.coords['time'].values

    cycle_times = resp_features[['inspi_time','expi_time','next_inspi_time']].values # extract resp cycle times to deform signals
    
    da_cycle_signals = None 
    
    for chan in chans:
        if chan == 'resp_nose':
            data = resp_sig.sel(chan='RespiNasale', time = slice(0, p['session_duration'])).values[:-1] # crop to 10 mins resp nose signal
        elif chan == 'resp_mouth':
            data = -resp_sig.sel(chan='RespiVentrale', time = slice(0, p['session_duration'])).values[:-1] # crop to 10 min and reverse resp belt signal
        elif chan == 'heart':
            data = rri.values
        else:
            data = eeg.sel(chan = chan).values
            
        # Cyclical deformation of signal
        cycle_signals = physio.deform_traces_to_cycle_template(data = data, 
                                                           times = times, 
                                                           cycle_times = cycle_times,
                                                           points_per_cycle = p['n_phase_bins'],
                                                           segment_ratios = p['segment_ratios'],
                                                           )
        # if not chan == 'heart':
        #     cycle_signals = np.apply_along_axis(norm , 1 , cycle_signals)
        
        if not chan == 'heart':
            cycle_signals = np.apply_along_axis(center , 1 , cycle_signals) # subtract mean if heart rate along phase axis
        elif chan in ['resp_nose','resp_mouth']:
            cycle_signals = np.apply_along_axis(norm , 1 , cycle_signals) # subtract mean and divide by SD along phase axis
        


        mask_cycles = (resp_features['artifact'] == 0) # mask resp cycles without co-occuring EEG artifacting
        keep_cycles = resp_features[mask_cycles].index # apply mask and select their indices

        cycle_signal = cycle_signals[keep_cycles,:] # select resp cycles according to mask

        m = np.mean(cycle_signal, axis = 0) # compute average deformed signal along cycles axis

        if da_cycle_signals is None:
            da_cycle_signals = init_nan_da({'chan':chans, 'phase':np.linspace(0,1,p['n_phase_bins'])}) # if first iteration, initialize a dataarray chan * phase

        da_cycle_signals.loc[chan , : ] = m # store the average dynamic at the right chan location
            
    ds = xr.Dataset()
    ds['cycle_signal'] = da_cycle_signals # store datarray in dataset
                                                        
    return ds
    
def test_cycle_signal():
    run_key = 'P01_baseline'
    ds = cycle_signal(run_key, **cycle_signal_params)
    print(ds)
    
cycle_signal_job = jobtools.Job(precomputedir, 'cycle_signal', cycle_signal_params, cycle_signal)
jobtools.register_job(cycle_signal_job)





# JOB MODULATION DATAFRAME

def modulation_cycle_signal(run_key, **p):
    """
    Compute amplitude of average evoked potential of EEG respi epochs, a marker of modulation
    """
    da_cycle_signals = cycle_signal_job.get(run_key)['cycle_signal'] # load deformed signals
    participant,session = run_key.split('_')
    
    rows = []
    for chan in da_cycle_signals.coords['chan'].values: # loop over chans
        if chan in ['resp_nose','resp_mouth','heart']: 
            continue # do not compute if chan is not EEG
        cycle_sig = da_cycle_signals.sel(chan = chan).values # select data of the chan
        row = [participant, session, chan, np.ptp(cycle_sig)] # compute peak to peak amplitude of the deformed signal and store it in a list (row of dataframe)
        rows.append(row) # list of lists (array)
    
    df = pd.DataFrame(rows, columns = ['participant','session','chan', 'amplitude']) # construct dataframe from list of rows
    
    return xr.Dataset(df) # store in a dataset

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
   
    
