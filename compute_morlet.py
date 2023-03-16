from configuration import *
from params import *

import xarray as xr
from scipy import signal
import numpy as np

import jobtools
from preproc import preproc_job
from params import *
from bibliotheque import init_nan_da
import physio

def complex_mw(time, n_cycles , freq, a = 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ------------------------------
    a = amplitude of the wavelet
    time = time vector of the wavelet
    n_cycles = number of cycles in the wavelet
    freq = frequency of the wavelet
    m = 
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def define_morlet_family(freqs, cycles , srate, return_time = False):
    tmw = np.arange(-10,10,1/srate)
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        n_cycles = cycles[i]
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles, freq = fi)
        
    if return_time:
        return tmw, mw_family
    else:
        return mw_family

    
def compute_power(run_key, **p):
    eeg = preproc_job.get(run_key)['eeg_clean']

    srate_down = srate / p['decimate_factor']
    
    time_freq = None 
    
    for chan in eeg.coords['chan'].values:
        
        sig = eeg.sel(chan=chan).values
        sig_down = signal.decimate(sig, p['decimate_factor'])
        time_vector_down = np.arange(sig_down.size) / srate_down
                
        freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
        cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)
        
        mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate_down)
        
        sigs = np.tile(sig_down, (p['n_freqs'],1))
        tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
        
        power = np.abs(tf) ** p['amplitude_exponent']
        
        if time_freq is None:
            time_freq = init_nan_da({'chan':eeg.coords['chan'].values, 'freq':freqs, 'time':time_vector_down})
            
        time_freq.loc[chan, : , : ] = power
           
    ds = xr.Dataset()
    ds['power'] = time_freq
    
    return ds

def test_compute_power():
    run_key = 'P02_baseline'
    ds = compute_power(run_key, **time_freq_params)
    print(ds)
    
power_job = jobtools.Job(precomputedir, 'morlet_power', time_freq_params, compute_power)
jobtools.register_job(power_job)




    
def compute_angle(run_key, **p):
    eeg = preproc_job.get(run_key)['eeg_clean']

    srate_down = srate / p['decimate_factor']
    
    time_freq = None 
    
    for chan in eeg.coords['chan'].values:
        
        sig = eeg.sel(chan=chan).values
        sig_down = signal.decimate(sig, p['decimate_factor'])
        time_vector_down = np.arange(sig_down.size) / srate_down
                
        freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
        cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)
        
        mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate_down)
        
        sigs = np.tile(sig_down, (p['n_freqs'],1))
        tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
        
        angles = np.angle(tf)
        
        if time_freq is None:
            time_freq = init_nan_da({'chan':eeg.coords['chan'].values, 'freq':freqs, 'time':time_vector_down})

        time_freq.loc[chan, : , : ] = angles
           
    ds = xr.Dataset()
    ds['angle'] = time_freq
    
    return ds


def test_compute_angle():
    run_key = 'P02_baseline'
    ds = compute_angle(run_key, **time_freq_params)
    print(ds)
    
angle_job = jobtools.Job(precomputedir, 'morlet_angle', time_freq_params, compute_angle)
jobtools.register_job(angle_job)   
    

def compute_all():
    jobtools.compute_job_list(power_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(power_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 2)
    
    # jobtools.compute_job_list(angle_job, run_keys, force_recompute=False, engine='loop')






if __name__ == '__main__':
    # test_compute_angle()
    # test_compute_power()
    
    compute_all()
        
        
            
        
    
