from configuration import *
from params import *

import xarray as xr
from scipy import signal
import numpy as np

import jobtools

from preproc import eeg_interp_artifact_job
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

def sig_to_tf_bis(sig, p, srate):
    freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
    cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)

    mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate)

    sigs = np.tile(sig, (p['n_freqs'],1))
    tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
    return {'f':freqs, 'tf':tf}

def compute_power(run_key, **p):
    eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = eeg.attrs['srate']
    down_srate = srate /  p['decimate_factor']

    powers = None

    for chan in p['chans']:
        sig = eeg.sel(chan = chan).values
        tf_dict = sig_to_tf_bis(sig, p, srate)
        power = np.abs(tf_dict['tf']) ** p['amplitude_exponent']
        power = signal.decimate(x = power, q = p['decimate_factor'], axis = 1)
        t_down = np.arange(0, power.shape[1] / down_srate, 1 / down_srate)

        if powers is None:
            powers = init_nan_da({'chan':p['chans'],
                                'freq':tf_dict['f'],
                                'time':t_down})
        powers.loc[chan, :,:] = power

    powers.attrs['down_srate'] = down_srate
    
    ds = xr.Dataset()
    ds['power'] = powers
    return ds

def test_compute_power():
    run_key = 'P02_baseline'
    ds = compute_power(run_key, **power_params)
    print(ds)
    

power_job = jobtools.Job(precomputedir, 'power', power_params, compute_power)
jobtools.register_job(power_job)



def compute_baseline_power(run_key, **p):
    
    eeg = eeg_interp_artifact_job.get(run_key)['interp']
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
    eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = eeg.attrs['srate']
    baselines = baseline_power_job.get(f'{participant}_baseline')['baseline']
    cycle_features = respiration_features_job.get(run_key).to_dataframe()
    cycle_times = cycle_features[['inspi_time','expi_time','next_inspi_time']].values
    
    mask_artifact = cycle_features['artifact'] == 0
    inds_resp_cycle_sel = cycle_features[mask_artifact].index
    
    baseline_modes = ['z_score','rz_score']
    
    phase_freq_power = None 
    
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
    

            
            deformed_data_stacked = deformed_data_stacked[inds_resp_cycle_sel,:,:]
        
            
            if phase_freq_power is None: 
                phase_freq_power = init_nan_da({'baseline_mode':baseline_modes, 
                                          'compress_cycle_mode':['mean_cycle','med_cycle','q75_cycle'],
                                          'chan':tf_params['chans'],
                                          'freq':tf_dict['f'], 
                                          'phase':np.linspace(0,1,p['n_phase_bins'])})
                
            phase_freq_power.loc[mode, 'mean_cycle', chan, :,:] = np.mean(deformed_data_stacked, axis = 0).T
            phase_freq_power.loc[mode, 'med_cycle', chan, :,:] = np.median(deformed_data_stacked, axis = 0).T
            phase_freq_power.loc[mode, 'q75_cycle', chan, :,:] = np.quantile(deformed_data_stacked, q = 0.75, axis = 0).T
            
        
    ds = xr.Dataset()
    ds['power'] = phase_freq_power
    
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
    # jobtools.compute_job_list(phase_freq_job, stim_keys, force_recompute=False, engine='joblib', n_jobs = 3)

    jobtools.compute_job_list(power_job, run_keys, force_recompute=False, engine='loop')


if __name__ == '__main__':
    # test_compute_baseline_power()
    # test_compute_phase_frequency()
    # test_compute_power()
    compute_all()
        
        
            
        
    
