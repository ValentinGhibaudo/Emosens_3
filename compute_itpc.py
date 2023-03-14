from configuration import *
from params import *

import xarray as xr
from scipy import signal
import numpy as np

import jobtools
from preproc import preproc_job, convert_vhdr_job
from store_timestamps import timestamps_job
from compute_resp_features import label_respiration_features_job
from params import *
from bibliotheque import init_nan_da
import physio

def complex_mw(time, n_cycles , freq, a= 1, m = 0): 
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

    
def compute_itpc(run_key, **p):
    ds_eeg = preproc_job.get(run_key)
    eeg = ds_eeg['eeg_clean']
    
    cycle_features = label_respiration_features_job.get(run_key).to_dataframe()
    srate_down = srate / p['decimate_factor']
    
    phase_freq = None 
    
    for chan in eeg.coords['chan'].values:
        # print(chan)
        
        sig = eeg.sel(chan=chan).values
        sig_down = signal.decimate(sig, p['decimate_factor'])
        time_vector_down = np.arange(sig_down.size) / srate_down

        timestamps = timestamps_job.get(run_key).to_dataframe().set_index(['bloc','trial'])
        
        freqs = np.logspace(np.log10(p['f_start']), np.log10(p['f_stop']), num = p['n_freqs'], base = 10)
        cycles = np.logspace(np.log10(p['c_start']), np.log10(p['c_stop']), num = p['n_freqs'], base = 10)
        
        mw_family = define_morlet_family(freqs = freqs , cycles = cycles, srate=srate_down)
        
        sigs = np.tile(sig_down, (p['n_freqs'],1))
        tf = signal.fftconvolve(sigs, mw_family, mode = 'same', axes = 1)
        
        cycle_times = cycle_features[['inspi_time','expi_time','next_inspi_time']].values
        deformed_data_stacked = physio.deform_traces_to_cycle_template(data = tf.T, 
                                                                       times = time_vector_down, 
                                                                       cycle_times=cycle_times, 
                                                                       segment_ratios = p['segment_ratios'], 
                                                                       points_per_cycle = p['n_phase_bins'])
        
        
        for bloc in p['blocs']:
            inds_cycle_bloc = cycle_features[cycle_features['bloc'] == bloc].index
            deformed_data_stacked_bloc = deformed_data_stacked[inds_cycle_bloc,:,:]
            
            deformed_data_itpc = np.abs(np.mean(np.exp(np.angle(deformed_data_stacked_bloc)* 1j), axis = 0))
            
            if phase_freq is None: 
                phase_freq = init_nan_da({'chan':eeg.coords['chan'].values, 'bloc':p['blocs'], 'freq':freqs, 'phase':np.linspace(0,1,p['n_phase_bins'])})
                
            phase_freq.loc[chan, bloc, :,:] = deformed_data_itpc.T
            
    ds = xr.Dataset()
    ds['phase_freq_itpc'] = phase_freq
    
    return ds

def test_compute_iptc():
    run_key = 'P13_ses02'
    # ds = compute_itpc(run_key, **itpc_params)
    ds = itpc_job.get(run_key)
    print(ds)
    

def compute_all():
    # jobtools.compute_job_list(itpc_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(itpc_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 2)


itpc_job = jobtools.Job(precomputedir, 'itpc',itpc_params, compute_itpc)
jobtools.register_job(itpc_job)


if __name__ == '__main__':
    # test_compute_iptc()
    compute_all()
        
        
            
        
    
