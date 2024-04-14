from configuration import *
from params import *

import xarray as xr
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jobtools

from params import *
from bibliotheque import init_nan_da, compute_spectrum_log_slope
import physio
from compute_phase_freq import power_job
from compute_resp_features import respiration_features_job

import ghibtools as gh


# SLOPE JOB

def slope(sub, ses, chan, **p):
    """
    Compute spectrum slopes on each time bin
    """
    power = power_job.get(sub,ses,chan)['power'] # load power
    power[:] = power.values # load into memory
    freqs = power['freq'].values # load freqs
    down_srate = power.attrs['down_srate'] # down sampling to reduce computation time

    slopes = np.apply_along_axis(compute_spectrum_log_slope, axis = 0, arr=power.values, freqs=freqs) # apply compute slope function along freq axis
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

# DEFORM SLOPE JOB

def deform_slope(sub, ses, chan, **p):
    """
    Deform slope trace by resp
    """
    slopes = slope_job.get(sub, ses, chan)['slope'] # load slopes
    slopes[:] = pd.Series(slopes.values).fillna(method = 'bfill').values
    slopes[:] = pd.Series(slopes.values).fillna(method = 'ffill').values
    times = slopes['time'].values # load time vector
    resp_features = respiration_features_job.get(sub, ses).to_dataframe() # load resp features

    cycle_times = resp_features[['inspi_time','expi_time','next_inspi_time']].values # extract resp cycle times to deform signals
 
    slopes_deformed = physio.deform_traces_to_cycle_template(data = slopes.values.T, 
                                                        times = times, 
                                                        cycle_times = cycle_times,
                                                        points_per_cycle = p['n_phase_bins'],
                                                        segment_ratios = p['segment_ratios'],
                                                        )
 
    mask_cycles = (resp_features['artifact'] == 0) # mask resp cycles without co-occuring EEG artifacting
    keep_cycles = resp_features[mask_cycles].index # apply mask and select their indices

    cycle_signal = slopes_deformed[keep_cycles,:] # select resp cycles according to mask

    m = np.mean(cycle_signal, axis = 0) # compute average deformed slope signal along cycles axis

    ds = xr.Dataset()
    ds['deform_slope'] = xr.DataArray(data = m,
                                      dims = ['phase'],
                                      coords = {'phase':np.linspace(0,1,p['n_phase_bins'])})

    return ds

def test_deform_slope():
    sub, ses, chan = 'P02' , 'odor', 'Cz'
    ds = deform_slope(sub, ses, chan, **deform_slope_params)
    print(ds)
    

deform_slope_job = jobtools.Job(precomputedir, 'deform_slope', deform_slope_params, deform_slope)
jobtools.register_job(deform_slope_job)


# DEFORM SLOPE JOB

def individual_slope_fig(sub, **p):
    nrows = 8
    ncols = 4
    poss = gh.attribute_subplots(eeg_chans, nrows, ncols)

    fig, axs = plt.subplots(nrows, ncols , constrained_layout = True, figsize = (10, 18))
    fig.suptitle(sub, fontsize = 20, y = 1.01)

    for chan, pos in poss.items():
        ax = axs[pos[0], pos[1]]

        for ses in session_keys:
            deform_slope = deform_slope_job.get(sub, ses, chan)['deform_slope']
            phase = deform_slope['phase'].values
            ax.plot(phase, deform_slope.values, label = ses)
        ax.set_title(chan)
        ax.set_ylim(-3, - 0.3)
        ax.axvline(0.4, color = 'r')
        ax.legend(fontsize = 7, loc = 2)

    
    fig.savefig(base_folder / 'Figures' / 'slopes' / 'by_subject' / f'{sub}.png', dpi = 500, bbox_inches = 'tight')
    plt.close('all')
    return xr.Dataset()

def test_individual_slope_fig():
    sub = 'P02'
    ds = individual_slope_fig(sub, **deform_slope_params)
    print(ds)
    

individual_slope_fig_job = jobtools.Job(precomputedir, 'individual_slope_fig', individual_slope_fig_params, individual_slope_fig)
jobtools.register_job(individual_slope_fig_job)



#----------------------#
#---- COMPUTE ALL -----#
#----------------------#

def compute_all():
    # run_keys = [(sub, ses, chan) for sub in subject_keys for ses in session_keys for chan in eeg_chans]
    # jobtools.compute_job_list(slope_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(slope_job, run_keys, force_recompute=False, engine='slurm',
    #                         slurm_params={'cpus-per-task':'1', 'mem':'1G', },
    #                         module_name='compute_slopes')

    # jobtools.compute_job_list(deform_slope_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 30)

    jobtools.compute_job_list(individual_slope_fig_job, subject_keys, force_recompute=True, engine='joblib',n_jobs = 5)


#----------------------#
#-------- RUN ---------#
#----------------------#

if __name__ == '__main__':
    # test_slope()
    # test_deform_slope()
    # test_individual_slope_fig()
    compute_all()