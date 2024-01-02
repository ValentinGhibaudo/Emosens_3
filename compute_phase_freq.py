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


#----------------------#
#------- POWER --------#
#----------------------#

def sig_to_tf(sig, p, srate):
    """
    Tool to compute time frequency from signal with complex morlet wavelets according to predefined params 
    aiming to define the wavelets features
    """
    freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
    cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)

    mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate)

    sigs = np.tile(sig, (p['n_freqs'],1))
    tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
    return {'f':freqs, 'tf':tf} # return dict with freq vector and time-frequency map in complex domain

def compute_power(sub, ses, chan, **p):
    """
    Compute time frequency power maps of EEG signals by convolution with morlet wavelets
    """
    eeg = eeg_interp_artifact_job.get(sub,ses)['interp'] # load eeg
    chans = eeg['chan'].values
    srate = eeg.attrs['srate']
    down_srate = srate /  p['decimate_factor'] # down sampling to reduce computation time

    powers = None

    sig = eeg.sel(chan = chan).values # select eeg signal
    tf_dict = sig_to_tf(sig, p, srate) # compute time frequency complex map (with morlet wavelets)
    power = np.abs(tf_dict['tf']) ** p['amplitude_exponent'] # extract power from time frequency complex map
    power = signal.decimate(x = power, q = p['decimate_factor'], axis = 1) # down sampling power map in time axis to reduce storage size
    t_down = np.arange(0, power.shape[1] / down_srate, 1 / down_srate) # compute the time vector of the down sampled power map

    if powers is None:
        powers = init_nan_da({'freq':tf_dict['f'], # initialize a dataarray freq * time
                            'time':t_down})
                            
    powers.loc[:,:] = power # store power map in dataarray

    powers.attrs['down_srate'] = down_srate # store sampling rate of the down sampled power map as a attribute
    
    ds = xr.Dataset()
    ds['power'] = powers # store in dataset
    return ds

def test_compute_power():
    sub, ses, chan = 'P30' , 'odor', 'Cz'
    ds = compute_power(sub, ses, chan, **power_params)
    print(ds)
    

power_job = jobtools.Job(precomputedir, 'power', power_params, compute_power)
jobtools.register_job(power_job)



#----------------------#
#------ BASELINE ------#
#----------------------#

def compute_baseline(sub, chan, **p):
    """
    Compute mean/median/sd/mad of EEG time frequency power 
    for each frequency bin (so on time axis) from baseline session
    """
    baseline_power = power_job.get(sub, 'baseline', chan)['power'] # load power map of baseline session
    baseline_power[:] = baseline_power.values # load into memory
    
    baselines = None 
    
    if baselines is None: # initialize datarray with baseline features
        baselines = init_nan_da({'mode':['mean','med','sd','mad'], 
                                    'freq':baseline_power['freq'].values})
        
    baselines.loc['mean' , :] = baseline_power.mean('time') # compute mean power of baseline over time axis
    baselines.loc['med' , :] = baseline_power.median('time') # compute median power of baseline over time axis
    baselines.loc['sd' , :] = baseline_power.std('time') # compute std power of baseline over time axis
    baselines.loc['mad' , :] = mad(baseline_power.values.T) # compute MAD power of baseline over time axis. Time must be on 0 axis
        
    ds = xr.Dataset()
    ds['baseline'] = baselines # store datarray in dataset
    return ds

def test_compute_baseline():
    sub, chan = 'P06','Fp1'
    ds = compute_baseline(sub, chan, **baseline_params)
    print(ds)
    
baseline_job = jobtools.Job(precomputedir, 'baseline',baseline_params, compute_baseline)
jobtools.register_job(baseline_job)


#----------------------#
#----- PHASE FREQ -----#
#----------------------#

def apply_baseline_normalization(power, baseline, mode): 
    """
    Tool to normalize raw power map according to baseline features
    by z-scoring (power - mean_baseline) / sd_baseline
    or robust z-scoring (power - median_baseline) / mad_baseline
    """  
    if mode == 'z_score':
        power_norm = (power - baseline['mean']) / baseline['sd']
            
    elif mode == 'rz_score':
        power_norm = (power - baseline['med']) / baseline['mad']     
    return power_norm
    
def compute_phase_frequency(sub, ses, chan, **p):
    """
    Normalize raw time frequency power maps by baseline 
    + cyclically deform it by respiratory epochs/timestamps to get phase frequency power maps
    """
    powers = power_job.get(sub, ses, chan)['power'] # load raw power map
    freqs = powers['freq'].values
    times = powers['time'].values

    baselines = baseline_job.get(sub, chan)['baseline'] # load baseline features

    cycle_features = respiration_features_job.get(sub, ses).to_dataframe() # load resp features
    cycle_times = cycle_features[['inspi_time','expi_time','next_inspi_time']].values # get respi times for deformation of the map
    
    mask_artifact = cycle_features['artifact'] == 0
    inds_resp_cycle_sel = cycle_features[mask_artifact].index # select inds of resp cycles without cooccuring EEG artifacting
    
    baseline_modes = ['z_score','rz_score']
    
    phase_freq_power = None 
    
        
    power = powers.values
                    
    baseline = {'mean':baselines.loc['mean',:].values,
                'med':baselines.loc['med',:].values,
                'sd':baselines.loc['sd',:].values,
                'mad':baselines.loc['mad',:].values
                }
    
    for mode in baseline_modes: # loop over baseline modes (z-score or robust z-score)
    
        power_norm = apply_baseline_normalization(power = power.T, baseline = baseline, mode = mode) # normalize power according to the mode
        
        # deform the normalized power map according to respiratory timestamps to get a phase representation of it
        deformed_data_stacked = physio.deform_traces_to_cycle_template(data = power_norm, 
                                                                        times = times, 
                                                                        cycle_times=cycle_times, 
                                                                        segment_ratios = p['segment_ratios'], 
                                                                        points_per_cycle = p['n_phase_bins'])


        
        deformed_data_stacked = deformed_data_stacked[inds_resp_cycle_sel,:,:] # keep only non artifacted cycles
    
        
        if phase_freq_power is None: # initalize phase frequency dataarray
            phase_freq_power = init_nan_da({'baseline_mode':baseline_modes, 
                                        'compress_cycle_mode':p['compress_cycle_modes'], # different cycle axis compression methods 
                                        'freq':freqs, 
                                        'phase':np.linspace(0,1,p['n_phase_bins'])})

        for compress in p['compress_cycle_modes']: # loop of cycle axis compression methods and store the output at the right location in dataarray
            if compress == 10:
                phase_freq_power.loc[mode, compress, :,:] = np.mean(deformed_data_stacked, axis = 0).T # mean over cycle axis
            else:
                phase_freq_power.loc[mode, compress, :,:] = np.quantile(deformed_data_stacked, q = compress, axis = 0).T # quantile computing over cycle axis
            
    ds = xr.Dataset()
    ds['phase_freq'] = phase_freq_power
    return ds


def test_compute_phase_frequency():
    sub, ses, chan = 'P01','baseline', 'Fz'
    ds = compute_phase_frequency(sub, ses, chan,**phase_freq_params)
    print(ds)
    

phase_freq_job = jobtools.Job(precomputedir, 'phase_freq', phase_freq_params, compute_phase_frequency)
jobtools.register_job(phase_freq_job)


def phase_freq_concat(chan, **p):
    """
    Concatenate phase-frequency power maps from sub,ses into one Dataset by channel
    """
    all_phase_freq = None

    for sub in p['sub_keys']: # loop over subjects
        for ses in p['ses_keys']:  # loop over sessions
            ds_phase_freq = phase_freq_job.get(sub,ses,chan) # load phase freq map

            power = ds_phase_freq['phase_freq']

        
            if all_phase_freq is None: # initialize right shaped datarray
                all_phase_freq = init_nan_da({'participant':p['sub_keys'], 
                                            'session':p['ses_keys'], 
                                            'compress_cycle_mode':p['compress_cycle_modes'],
                                            'freq':power.coords['freq'].values,
                                            'phase':power.coords['phase'].values
                                            })

            all_phase_freq.loc[sub, ses, :,:,:] = power.loc[p['baseline_mode'] ,:,:,:].values # select a baselining method store datarray at the right location

    all_phase_freq = all_phase_freq.loc[:,:,:,:p['max_freq'],:] # zoom on a useful frequency band
    ds = xr.Dataset()
    ds['phase_freq_concat'] = all_phase_freq
    return ds

def test_phase_freq_concat():
    chan = 'F3'
    ds = phase_freq_concat(chan, **phase_freq_concat_params)
    print(ds['phase_freq_concat'])
    
phase_freq_concat_job = jobtools.Job(precomputedir, 'phase_freq_concat', phase_freq_concat_params, phase_freq_concat)
jobtools.register_job(phase_freq_concat_job)



#----------------------#
#------- ERP TF -------#
#----------------------#

def compute_erp_time_freq(sub, ses, chan, **p):
    """
    Same process than phase freq but without cyclically deforming EEG data, keeping a time basis, 
    and average time-frequency power dynamic centered on a respiratory time point
    """
    half_window_duration = p['half_window_duration']

    power_all = power_job.get(sub, ses, chan)['power'] # load power
    power_all[:] = power_all.values

    down_srate = power_all.attrs['down_srate']

    cycle_features = respiration_features_job.get(sub, ses).to_dataframe() # load resp features
    resp_sel = cycle_features[cycle_features['artifact'] == 0] # keep EEG non-artifacted resp cycles
    resp_sel = resp_sel.reset_index(drop = True)
    resp_sel = resp_sel.iloc[:-2,:] # remove last two cycles that could be overlapping the end of session

    # print(resp_sel.iloc[-1,:]['inspi_time'] * down_srate + half_window_duration * down_srate)

    baselines = baseline_job.get(sub, chan)['baseline'] # load baseline features
    baselines[:] = baselines.values
    baseline_modes = ['z_score','rz_score']

    centers_slice = ['inspi_time','expi_time']
    win_size_points = int(half_window_duration * 2 * down_srate) # prepare window size
    
    erp_power = None

    power = power_all.values

    baseline_dict = {
                'mean':baselines.loc['mean',:].values,
                'med':baselines.loc['med',:].values,
                'sd':baselines.loc['sd',:].values,
                'mad':baselines.loc['mad',:].values
                }

    for mode in baseline_modes: # loop over normalization methods
        power_norm = apply_baseline_normalization(power = power.T, baseline = baseline_dict, mode = mode).T # normalize

        erp_power_chan = None

        for center_slice in centers_slice: # loop over respi transitions (inspi-expi and expi-inspi)

            for ind_c, c in resp_sel.iterrows(): # loop over resp cycles
                win_center = c[center_slice] # select resp timestamp
                win_start = win_center - half_window_duration # compute start time of window
                win_start_point = int(win_start * down_srate)  # compute start ind of window
                win_stop_point = win_start_point + win_size_points  # compute stop ind of window
                win = np.arange(win_start_point, win_stop_point) # compute window ind vector

                if erp_power_chan is None: # initialize right shaped datarray
                    erp_power_chan = init_nan_da({'cycle':resp_sel.index,
                                                'freq':power_all['freq'].values,
                                                'time':np.arange(-half_window_duration, half_window_duration , 1 / down_srate)})
                
                erp_power_chan.loc[ind_c, : ,:] = power_norm[:,win] # select power of the window and store in a the right location (cycle)

            if erp_power is None:
                erp_power = init_nan_da({'baseline_mode':baseline_modes,
                                    'center':centers_slice,
                                    'freq':power_all['freq'].values,
                                    'time':erp_power_chan.coords['time'].values})
            
            if type(p['compress_cycle_mode']) is str:
                erp_power.loc[mode, center_slice, :,:] = erp_power_chan.mean(dim = 'cycle').values # average over resp cycles axis
            elif type(p['compress_cycle_mode']) is float:
                erp_power.loc[mode, center_slice, :,:] = erp_power_chan.quantile(dim = 'cycle', q = p['compress_cycle_mode']).values # quantile computing over resp cycles axis
     
    erp_power.attrs['down_srate'] = down_srate
    ds = xr.Dataset()
    ds['erp_time_freq'] = erp_power
    return ds  

def test_compute_erp_time_freq():
    sub, ses, chan = 'P27','baseline','Fp1'
    ds = compute_erp_time_freq(sub, ses,chan, **erp_time_freq_params)
    print(ds['erp_time_freq'])

erp_time_freq_job = jobtools.Job(precomputedir, 'erp_time_freq', erp_time_freq_params, compute_erp_time_freq)
jobtools.register_job(erp_time_freq_job)

def erp_time_freq_concat(chan, **p):
    """
    Concatenate erp power maps from sub,ses into one Dataset by channel
    """
    erp_concat = None

    for sub in p['sub_keys']: # loop over subjects
        for ses in p['ses_keys']: # loop over sessions
            ds = erp_time_freq_job.get(sub,ses,chan) # load power

            erp = ds['erp_time_freq']

            if erp_concat is None: # initialize the right shaped datarray
                erp_concat = init_nan_da({'participant':p['sub_keys'], 
                                            'session':p['ses_keys'], 
                                            'freq':erp.coords['freq'].values,
                                            'time':erp.coords['time'].values
                                            })

            erp_concat.loc[sub, ses,:,:] = erp.loc[p['baseline_mode'],p['center'],:,:].values # select baselining method and resp timestamp and store it at the right location

    erp_concat = erp_concat.loc[:,:,:p['max_freq'],:]
    ds = xr.Dataset()
    ds['erp_concat'] = erp_concat
    return ds

def test_erp_time_freq_concat():
    chan = 'Fz'
    ds = erp_time_freq_concat(chan, **erp_time_freq_concat_params)
    print(ds['erp_concat'])

erp_concat_job = jobtools.Job(precomputedir, 'erp_time_freq_concat', erp_time_freq_concat_params, erp_time_freq_concat)
jobtools.register_job(erp_concat_job)
 



#----------------------#
#---- COMPUTE ALL -----#
#----------------------#

def compute_all():
    # run_keys = [(sub, ses, chan) for sub in subject_keys for ses in session_keys for chan in eeg_chans]
    # jobtools.compute_job_list(power_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(power_job, run_keys, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'20', 'mem':'20G', },
    #                         module_name='compute_phase_freq')

    # run_keys_bl = [(sub, chan) for sub in subject_keys for chan in eeg_chans]
    # jobtools.compute_job_list(baseline_job, run_keys_bl, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(baseline_job, run_keys_bl, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'20', 'mem':'20G', },
    #                         module_name='compute_phase_freq')

    
    # run_keys_pf = [(sub, ses, chan) for sub in subject_keys for ses in session_keys for chan in eeg_chans]
    # jobtools.compute_job_list(phase_freq_job, run_keys_pf, force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(phase_freq_job, run_keys_pf, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'20', 'mem':'20G', },
    #                         module_name='compute_phase_freq')

    # run_keys_erp = [(sub, ses, chan) for sub in subject_keys for ses in session_keys for chan in eeg_chans]
    # jobtools.compute_job_list(erp_time_freq_job, run_keys_erp, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(erp_time_freq_job, run_keys_erp, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'1', 'mem':'1G', },
    #                         module_name='compute_phase_freq')
    
    # keys = [(chan,) for chan in eeg_chans]
    # jobtools.compute_job_list(phase_freq_concat_job, keys, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'1', 'mem':'1G', },
    #                         module_name='compute_phase_freq')

    keys = [(chan,) for chan in eeg_chans]
    jobtools.compute_job_list(erp_concat_job, keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(erp_concat_job, keys, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'1', 'mem':'1G', },
    #                         module_name='compute_phase_freq')




#----------------------#
#-------- RUN ---------#
#----------------------#

if __name__ == '__main__':
    # test_compute_power()
    # test_compute_baseline()
    
    # test_compute_phase_frequency()
    # test_phase_freq_concat()

    # test_compute_erp_time_freq()
    # test_erp_time_freq_concat()
    
    compute_all()
        
        
            
        
    
