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
    freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
    cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)

    mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate)

    sigs = np.tile(sig, (p['n_freqs'],1))
    tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
    return {'f':freqs, 'tf':tf}

def compute_power(sub, ses, chan, **p):

    eeg = eeg_interp_artifact_job.get(sub,ses)['interp']
    chans = eeg['chan'].values
    srate = eeg.attrs['srate']
    down_srate = srate /  p['decimate_factor']

    powers = None

    sig = eeg.sel(chan = chan).values
    tf_dict = sig_to_tf(sig, p, srate)
    power = np.abs(tf_dict['tf']) ** p['amplitude_exponent']
    power = signal.decimate(x = power, q = p['decimate_factor'], axis = 1)
    t_down = np.arange(0, power.shape[1] / down_srate, 1 / down_srate)

    if powers is None:
        powers = init_nan_da({'freq':tf_dict['f'],
                            'time':t_down})
                            
    powers.loc[:,:] = power

    powers.attrs['down_srate'] = down_srate
    
    ds = xr.Dataset()
    ds['power'] = powers
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

    baseline_power = power_job.get(sub, 'baseline', chan)['power']
    baseline_power[:] = baseline_power.values
    
    baselines = None 
    
    if baselines is None:
        baselines = init_nan_da({'mode':['mean','med','sd','mad'], 
                                    'freq':baseline_power['freq'].values})
        
    baselines.loc['mean' , :] = baseline_power.mean('time')
    baselines.loc['med' , :] = baseline_power.median('time')
    baselines.loc['sd' , :] = baseline_power.std('time')
    baselines.loc['mad' , :] = mad(baseline_power.values.T) # time must be on 0 axis
        
    ds = xr.Dataset()
    ds['baseline'] = baselines
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
    if mode == 'z_score':
        power_norm = (power - baseline['mean']) / baseline['sd']
            
    elif mode == 'rz_score':
        power_norm = (power - baseline['med']) / baseline['mad']     
    return power_norm
    
def compute_phase_frequency(sub, ses, chan, **p):
    
    powers = power_job.get(sub, ses, chan)['power']
    freqs = powers['freq'].values
    times = powers['time'].values

    baselines = baseline_job.get(sub, chan)['baseline']

    cycle_features = respiration_features_job.get(sub, ses).to_dataframe()
    cycle_times = cycle_features[['inspi_time','expi_time','next_inspi_time']].values
    
    mask_artifact = cycle_features['artifact'] == 0
    inds_resp_cycle_sel = cycle_features[mask_artifact].index
    
    baseline_modes = ['z_score','rz_score']
    
    phase_freq_power = None 
    
        
    power = powers.values
                    
    baseline = {'mean':baselines.loc['mean',:].values,
                'med':baselines.loc['med',:].values,
                'sd':baselines.loc['sd',:].values,
                'mad':baselines.loc['mad',:].values
                }
    
    for mode in baseline_modes:
    
        power_norm = apply_baseline_normalization(power = power.T, baseline = baseline, mode = mode)
        
        deformed_data_stacked = physio.deform_traces_to_cycle_template(data = power_norm, 
                                                                        times = times, 
                                                                        cycle_times=cycle_times, 
                                                                        segment_ratios = p['segment_ratios'], 
                                                                        points_per_cycle = p['n_phase_bins'])


        
        deformed_data_stacked = deformed_data_stacked[inds_resp_cycle_sel,:,:]
    
        
        if phase_freq_power is None: 
            phase_freq_power = init_nan_da({'baseline_mode':baseline_modes, 
                                        'compress_cycle_mode':p['compress_cycle_modes'],
                                        'freq':freqs, 
                                        'phase':np.linspace(0,1,p['n_phase_bins'])})

        for compress in p['compress_cycle_modes']:
            if compress == 10:
                phase_freq_power.loc[mode, compress, :,:] = np.mean(deformed_data_stacked, axis = 0).T
            else:
                phase_freq_power.loc[mode, compress, :,:] = np.quantile(deformed_data_stacked, q = compress, axis = 0).T
            
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
    all_phase_freq = None

    for sub in p['sub_keys']:
        for ses in p['ses_keys']:
            ds_phase_freq = phase_freq_job.get(sub,ses,chan)

            power = ds_phase_freq['phase_freq']

        
            if all_phase_freq is None:
                all_phase_freq = init_nan_da({'participant':p['sub_keys'], 
                                            'session':p['ses_keys'], 
                                            'compress_cycle_mode':p['compress_cycle_modes'],
                                            'freq':power.coords['freq'].values,
                                            'phase':power.coords['phase'].values
                                            })

            all_phase_freq.loc[sub, ses, :,:,:] = power.loc[p['baseline_mode'] ,:,:,:].values

    all_phase_freq = all_phase_freq.loc[:,:,:,:p['max_freq'],:]
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
    
    half_window_duration = p['half_window_duration']

    power_all = power_job.get(sub, ses, chan)['power']
    power_all[:] = power_all.values

    down_srate = power_all.attrs['down_srate']

    cycle_features = respiration_features_job.get(sub, ses).to_dataframe() 
    resp_sel = cycle_features[cycle_features['artifact'] == 0]
    resp_sel = resp_sel.reset_index(drop = True)
    resp_sel = resp_sel.iloc[:-2,:] # remove last two cycles that could be overlapping the end of session

    # print(resp_sel.iloc[-1,:]['inspi_time'] * down_srate + half_window_duration * down_srate)

    baselines = baseline_job.get(sub, chan)['baseline']
    baselines[:] = baselines.values
    baseline_modes = ['z_score','rz_score']

    centers_slice = ['inspi_time','expi_time']
    win_size_points = int(half_window_duration * 2 * down_srate)
    
    erp_power = None

    power = power_all.values

    baseline_dict = {
                'mean':baselines.loc['mean',:].values,
                'med':baselines.loc['med',:].values,
                'sd':baselines.loc['sd',:].values,
                'mad':baselines.loc['mad',:].values
                }

    for mode in baseline_modes:
        power_norm = apply_baseline_normalization(power = power.T, baseline = baseline_dict, mode = mode).T

        erp_power_chan = None

        for center_slice in centers_slice:

            for ind_c, c in resp_sel.iterrows():
                win_center = c[center_slice]
                win_start = win_center - half_window_duration
                win_start_point = int(win_start * down_srate)
                win_stop_point = win_start_point + win_size_points 
                win = np.arange(win_start_point, win_stop_point)

                if erp_power_chan is None:
                    erp_power_chan = init_nan_da({'cycle':resp_sel.index,
                                                'freq':power_all['freq'].values,
                                                'time':np.arange(-half_window_duration, half_window_duration , 1 / down_srate)})
                
                erp_power_chan.loc[ind_c, : ,:] = power_norm[:,win]

            if erp_power is None:
                erp_power = init_nan_da({'baseline_mode':baseline_modes,
                                    'center':centers_slice,
                                    'freq':power_all['freq'].values,
                                    'time':erp_power_chan.coords['time'].values})
            
            if type(p['compress_cycle_mode']) is str:
                erp_power.loc[mode, center_slice, :,:] = erp_power_chan.mean(dim = 'cycle').values
            elif type(p['compress_cycle_mode']) is float:
                erp_power.loc[mode, center_slice, :,:] = erp_power_chan.quantile(dim = 'cycle', q = p['compress_cycle_mode']).values
     
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
    erp_concat = None

    for sub in p['sub_keys']:
        for ses in p['ses_keys']:
            ds = erp_time_freq_job.get(sub,ses,chan)

            erp = ds['erp_time_freq']

            if erp_concat is None:
                erp_concat = init_nan_da({'participant':p['sub_keys'], 
                                            'session':p['ses_keys'], 
                                            'freq':erp.coords['freq'].values,
                                            'time':erp.coords['time'].values
                                            })

            erp_concat.loc[sub, ses,:,:] = erp.loc[p['baseline_mode'],p['center'],:,:].values

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
        
        
            
        
    
