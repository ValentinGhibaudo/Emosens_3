from configuration import *
from params import *

import xarray as xr
from scipy import signal
import numpy as np

import jobtools

from preproc import preproc_job
from compute_resp_features import respiration_features_job

from params import *
from bibliotheque import init_nan_da, complex_mw, define_morlet_family, mad
import physio

def sig_to_tf(sig, p, srate):
    srate_down = srate / p['decimate_factor']
    sig_down = signal.decimate(sig, p['decimate_factor'])
    time_vector_down = np.arange(sig_down.size) / srate_down

    freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
    cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)

    mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate_down)

    sigs = np.tile(sig_down, (p['n_freqs'],1))
    tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
    return {'t':time_vector_down, 'f':freqs, 'tf':tf}


def compute_baseline_power(run_key, **p):
    
    eeg = preproc_job.get(run_key)['eeg_clean']
    srate = eeg.attrs['srate']
    
    baselines = None 
    
    for chan in p['chans']:
        
        sig = eeg.sel(chan=chan).values

        tf_dict = sig_to_tf(sig, p, srate)
        
        power = np.abs(tf_dict['tf']) ** p['amplitude_exponent']
        
        if baselines is None:
            baselines = init_nan_da({'mode':['mean','med','sd','mad'], 
                                     'chan':p['chans'], 
                                     'freq':tf_dict['f']})
            
        baselines.loc['mean',chan , :] = np.mean(power, axis = 1)
        baselines.loc['med',chan , :] = np.median(power, axis = 1)
        baselines.loc['sd',chan , :] = np.std(power, axis = 1)
        baselines.loc['mad',chan , :] = mad(power.T) # time must be on 0 axis
        
    ds = xr.Dataset()
    ds['baseline'] = baselines
    
    return ds


def test_compute_baseline_power():
    run_key = 'P02_baseline'
    ds = compute_baseline_power(run_key, **time_freq_params)
    print(ds)
    

baseline_power_job = jobtools.Job(precomputedir, 'baseline_power',time_freq_params, compute_baseline_power)
jobtools.register_job(baseline_power_job)


def apply_baseline_normalization(power, baseline, mode):   
    if mode == 'z_score':
        power_norm = (power - baseline['mean']) / baseline['sd']
            
    elif mode == 'rz_score':
        power_norm = (power - baseline['med']) / baseline['mad']     
    return power_norm


    
def compute_phase_frequency(run_key, **p):
    
    tf_params = p['time_freq_params']
    
    participant, session = run_key.split('_')
    eeg = preproc_job.get(run_key)['eeg_clean']
    srate = eeg.attrs['srate']
    baselines = baseline_power_job.get(f'{participant}_baseline')['baseline']
    cycle_features = respiration_features_job.get(run_key).to_dataframe()
    cycle_times = cycle_features[['inspi_time','expi_time','next_inspi_time']].values
    
    srate_down = srate / tf_params['decimate_factor']
    
    baseline_modes = ['z_score','rz_score']
    
    phase_freq_power = None 
    phase_freq_itpc = None
    
    for chan in tf_params['chans']:
        
        sig = eeg.sel(chan=chan).values
        
        tf_dict = sig_to_tf(sig, tf_params, srate)
        
        power = np.abs(tf_dict['tf']) ** tf_params['amplitude_exponent']
        angles = np.angle(tf_dict['tf'])
        
                      
        baseline = {'mean':baselines.loc['mean',chan].values,
                    'med':baselines.loc['med',chan].values,
                    'sd':baselines.loc['sd',chan].values,
                    'mad':baselines.loc['mad',chan].values
                   }
        
        for mode in baseline_modes:
        
            power_norm = apply_baseline_normalization(power = power.T, baseline = baseline, mode = mode)
        
            
            deformed_data_stacked = physio.deform_traces_to_cycle_template(data = power_norm, 
                                                                           times = tf_dict['t'], 
                                                                           cycle_times=cycle_times, 
                                                                           segment_ratios = p['segment_ratios'], 
                                                                           points_per_cycle = p['n_phase_bins'])
        
            
            if phase_freq_power is None: 
                phase_freq_power = init_nan_da({'baseline_mode':baseline_modes, 
                                          'compress_cycle_mode':['mean_cycle','med_cycle','q75_cycle'],
                                          'chan':tf_params['chans'],
                                          'freq':tf_dict['f'], 
                                          'phase':np.linspace(0,1,p['n_phase_bins'])})
                
            phase_freq_power.loc[mode, 'mean_cycle', chan, :,:] = np.mean(deformed_data_stacked, axis = 0).T
            phase_freq_power.loc[mode, 'med_cycle', chan, :,:] = np.median(deformed_data_stacked, axis = 0).T
            phase_freq_power.loc[mode, 'q75_cycle', chan, :,:] = np.quantile(deformed_data_stacked, q = 0.75, axis = 0).T
            
        if phase_freq_itpc is None:
            phase_freq_itpc = init_nan_da({
                          'chan':tf_params['chans'],
                          'freq':tf_dict['f'], 
                          'phase':np.linspace(0,1,p['n_phase_bins'])
            })
            
        deformed_data_stacked = physio.deform_traces_to_cycle_template(data = angles.T, 
                                                                           times = tf_dict['t'], 
                                                                           cycle_times=cycle_times, 
                                                                           segment_ratios = p['segment_ratios'], 
                                                                           points_per_cycle = p['n_phase_bins'])
        
        itpc = np.abs(np.mean(np.exp(np.angle(deformed_data_stacked)* 1j), axis = 0))
        phase_freq_itpc.loc[chan, :,:] = itpc.T
        
    ds = xr.Dataset()
    ds['power'] = phase_freq_power
    ds['itpc'] = phase_freq_itpc
    
    return ds


def test_compute_phase_frequency():
    run_key = 'P02_music'
    ds = compute_phase_frequency(run_key, **phase_freq_params)
    print(ds)
    

phase_freq_job = jobtools.Job(precomputedir, 'phase_freq', phase_freq_params, compute_phase_frequency)
jobtools.register_job(phase_freq_job)




def compute_all():
    # jobtools.compute_job_list(baseline_power_job, baseline_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(baseline_power_job, baseline_keys, force_recompute=False, engine='joblib', n_jobs = 5)
    
    # jobtools.compute_job_list(phase_freq_job, stim_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(phase_freq_job, stim_keys, force_recompute=False, engine='joblib', n_jobs = 3)


if __name__ == '__main__':
    # test_compute_baseline_power()
    # test_compute_phase_frequency()
    compute_all()
        
        
            
        
    
