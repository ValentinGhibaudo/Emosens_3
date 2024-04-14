from configuration import *
from params import *

import xarray as xr
from scipy import signal
import numpy as np

import jobtools

from params import *
from bibliotheque import init_nan_da, compute_spectrum_log_slope
import physio
from compute_phase_freq import power_job

def slope(sub, ses, chan, **p):
    """
    Compute time frequency power maps of EEG signals by convolution with morlet wavelets
    """
    power = power_job.get(sub,ses,chan)['power'].values # load power
    print(power.shape)
    freqs = power['freq'].values # load freqs
    down_srate = power.attrs['down_srate'] # down sampling to reduce computation time

    slopes = np.apply_along_axis(compute_spectrum_log_slope, axis = 0, arr=power, freqs=freqs) # apply compute slope function along freq axis
    ds = xr.Dataset()
    ds['slope'] = xr.DataArray(data = slopes,
                               dims = ['time'],
                               coords = {'time':power['time'].values},
                               attrs = {'down_srate':down_srate}) # store datarray into dataset
    return ds

def test_slope():
    sub, ses, chan = 'P30' , 'odor', 'Cz'
    ds = slope(sub, ses, chan, **slope_params)
    print(ds)
    

slope_job = jobtools.Job(precomputedir, 'slope', slope_params, slope)
jobtools.register_job(slope_job)



#----------------------#
#---- COMPUTE ALL -----#
#----------------------#

def compute_all():
    run_keys = [(sub, ses, chan) for sub in subject_keys for ses in session_keys for chan in eeg_chans]
    jobtools.compute_job_list(slope_job, run_keys, force_recompute=False, engine='loop')


#----------------------#
#-------- RUN ---------#
#----------------------#

if __name__ == '__main__':
    test_slope()
    
    # compute_all()