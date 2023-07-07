import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from compute_cycle_signal import cycle_signal_job
from compute_resp_features import respiration_features_job
from compute_global_dataframes import oas_concat_job, bmrq_concat_job
from bibliotheque import get_pos, init_nan_da
from params import subject_keys, eeg_chans, run_keys 
from configuration import base_folder
from params import *
import os
import mne

p = cycle_signal_params

scale_factor = 30

def get_N_resp_cycles(run_keys):
    concat = []
    for run_key in run_keys:
        participant, session = run_key.split('_')
        resp = respiration_features_job.get(run_key).to_dataframe()
        concat.append(resp)

    all_resp = pd.concat(concat)

    N_cycles = all_resp.value_counts(subset = ['participant','session']).to_frame().reset_index().rename(columns ={'count':'N'}).set_index(['participant','session'])
    return N_cycles

N_cycles = get_N_resp_cycles(run_keys)
N_cycles_pooled = N_cycles.groupby(['session']).sum(numeric_only = True)

fig_folder = base_folder / 'Figures' / 'Cycle_Signal' / 'whole_signal'


# CONCAT
all_cycle_signal = None

for run_key in run_keys:
    
    participant, session = run_key.split('_')
    ds = cycle_signal_job.get(run_key)
    
    cycle_signal = ds['cycle_signal']
    
    if all_cycle_signal is None:
        all_cycle_signal = init_nan_da({'participant':subject_keys, 
                                        'session':session_keys,
                                      'chan':cycle_signal.coords['chan'].values,
                                      'phase':cycle_signal.coords['phase'].values
                                     })
        
    all_cycle_signal.loc[participant,session,:,:] = cycle_signal.values



### FIG 1 = GLOBAL
phase = all_cycle_signal.coords['phase'].values 


print('FIG 1')

global_cycle_signal = all_cycle_signal.sel(participant = [sub for sub in all_cycle_signal.coords['participant'].values if not sub == 'P02']).mean('participant')
# global_cycle_signal = all_cycle_signal.mean('participant')

for chan in p['chans']:

    fig, axs = plt.subplots(ncols = len(session_keys), figsize = (15,5), constrained_layout = True)
    fig.suptitle(f'Mean EEG waveform along respiration phase across {len(subject_keys)} subjects at electrode {chan}', fontsize = 20, y = 1.05) 
    
    vmin_eeg = global_cycle_signal.sel(chan=chan).min()
    vmax_eeg = global_cycle_signal.sel(chan=chan).max()

    
    for c, session in enumerate(session_keys):
        
        ax = axs[c]

        chan_sig = global_cycle_signal.loc[session, chan , :].values

        ax.plot(phase , chan_sig  , lw = 1, color = 'k', label = 'eeg')
        ax.set_ylim(vmin_eeg, vmax_eeg)

        ax2 = ax.twinx()
        ax2.plot(phase , global_cycle_signal.loc[session, 'heart' , :].values  , lw = 1, color = 'r', label = 'heart')
        ax2.set_ylim(60,82)

        ax3 = ax.twinx()
        ax3.plot(phase , global_cycle_signal.loc[session, 'resp_nose' , :].values , lw = 1, color = None, label = 'resp_nose')
        ax3.plot(phase , global_cycle_signal.loc[session, 'resp_mouth' , :].values , lw = 1, color = 'darkorange', label = 'resp_mouth')
        ax3.set_yticks([])

        if c == len(session_keys) - 1:
            ax2.set_ylabel('Heart rate [bpm]') 
        else:
            ax2.set_yticks([])

        ax.legend(fontsize = 'x-small', loc = 'upper left')
        ax2.legend(fontsize = 'x-small', loc = 'upper right')
        ax3.legend(fontsize = 'x-small', loc = 'lower left')

        if c == 0:
            ax.set_ylabel('Amplitude [AU]')
        else:
            ax.set_yticks([])

        ax.set_xlabel('Phase')

        ax.axvline(x = p['segment_ratios'], color = 'g')

        N = N_cycles_pooled.loc[session, 'N']
        ax.set_title(f'{session} - N : {N}')

    file = fig_folder / 'global' / f'{chan}.tif'
    fig.savefig(file, bbox_inches = 'tight', dpi = 200)
    plt.close()
    
    
## FIG 2 = BY SUBJECT 

print('FIG 2')

oas = oas_concat_job.get(global_key).to_dataframe().set_index('participant')
bmrq = bmrq_concat_job.get(global_key).to_dataframe().set_index('participant')

for sub in subject_keys:
    
    bmrq_sub = bmrq.loc[sub, 'BMRQ'].round(3)
    oas_sub = oas.loc[sub, 'OAS'].round(3)
    
    for chan in p['chans']:
        
        vmin_eeg = all_cycle_signal.sel(participant = sub, chan=chan).min()
        vmax_eeg = all_cycle_signal.sel(participant = sub, chan=chan).max()
        
        fig, axs = plt.subplots(ncols = len(session_keys), figsize = (15,5), constrained_layout = True)
        fig.suptitle(f'Mean EEG waveform along respiration phase in {sub} at electrode {chan} \n OAS : {oas_sub} - BMRQ : {bmrq_sub}', fontsize = 20, y = 1.05) 

        for c, session in enumerate(session_keys):

            ax = axs[c]

            chan_sig = all_cycle_signal.loc[sub, session, chan , :].values

            ax.plot(phase , chan_sig  , lw = 1, color = 'k', label = 'eeg')
            ax.set_ylim(vmin_eeg, vmax_eeg)

            ax2 = ax.twinx()
            ax2.plot(phase , all_cycle_signal.loc[sub, session, 'heart', :].values  , lw = 1, color = 'r', label = 'heart')
            ax2.set_ylim(40,120)

            ax3 = ax.twinx()
            ax3.plot(phase , all_cycle_signal.loc[sub, session, 'resp_nose' , :].values , lw = 1, color = None, label = 'resp_nose')
            ax3.plot(phase , all_cycle_signal.loc[sub, session, 'resp_mouth' , :].values , lw = 1, color = 'darkorange', label = 'resp_mouth')
            ax3.set_yticks([])

            if c == len(session_keys) - 1:
                ax2.set_ylabel('Heart rate [bpm]') 
            else:
                ax2.set_yticks([])

            ax.legend(fontsize = 'x-small', loc = 'upper left')
            ax2.legend(fontsize = 'x-small', loc = 'upper right')
            ax3.legend(fontsize = 'x-small', loc = 'lower left')

            if c == 0:
                ax.set_ylabel('Amplitude [AU]')
            else:
                ax.set_yticks([])

            ax.set_xlabel('Phase')

            ax.axvline(x = p['segment_ratios'], color = 'g')
            N = N_cycles.loc[(participant,session), 'N']
            ax.set_title(f'{session} - N : {N}')

        folder = fig_folder / 'by_subject' / f'{chan}.tif'
        if not os.path.exists(folder):
            os.mkdir(folder)
        fig.savefig(folder / f'{sub}.png', bbox_inches = 'tight', dpi = 200) 
        plt.close()
