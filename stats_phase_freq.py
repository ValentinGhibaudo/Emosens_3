import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from compute_phase_freq import phase_freq_job
from compute_resp_features import label_respiration_features_job
from bibliotheque import get_odor_from_session, init_nan_da
from params import subject_keys, blocs, odeurs, eeg_chans, run_keys 
from configuration import base_folder
import os

def get_N_resp_cycles(run_keys):
    concat = []
    for run_key in run_keys:
        participant, session = run_key.split('_')
        resp = label_respiration_features_job.get(run_key).to_dataframe()
        resp.insert(0, 'odor', get_odor_from_session(run_key))
        resp.insert(0, 'session', session)
        resp.insert(0, 'participant', participant)
        concat.append(resp)

    all_resp = pd.concat(concat)
    
    N_cycles = all_resp.value_counts(subset = ['participant','session','odor','bloc']).to_frame().reset_index().rename(columns ={0:'N'}).set_index(['participant','odor','bloc'])
    return N_cycles

N_cycles = get_N_resp_cycles(run_keys)
N_cycles_pooled = N_cycles.groupby(['odor','bloc']).sum(numeric_only = True)


fig_folder = base_folder / 'Figures' / 'Phase_freq'

# COLORBAR POS
ax_x_start = 1.05
ax_x_width = 0.02
ax_y_start = 0
ax_y_height = 1

# QUANTILE COLORLIM
delta_colorlim = 0.

# CONCAT
all_phase_freq = None
for run_key in run_keys:
    participant = run_key.split('_')[0]
    odeur = get_odor_from_session(run_key)

    phase_freq = phase_freq_job.get(run_key)['phase_freq']
    
    if all_phase_freq is None:
        all_phase_freq = init_nan_da({'participant':subject_keys, 
                                      'odor':odeurs, 
                                      'chan':phase_freq.coords['chan'].values,
                                      'bloc':phase_freq.coords['bloc'].values, 
                                      'freq':phase_freq.coords['freq'].values,
                                      'phase':phase_freq.coords['phase'].values
                                     })
        
    all_phase_freq.loc[participant, odeur, :,:,:,:] = phase_freq.values

    
### FIG 1 = GLOBAL
print('FIG 1')

mean_phase_freq = all_phase_freq.mean('participant')

for chan in eeg_chans:
    
    fig, axs = plt.subplots(nrows = len(odeurs), ncols = len(blocs), figsize = (15,7), constrained_layout = True)
    fig.suptitle(f'Mean phase-frequency power map across {len(subject_keys)} subjects in electrode {chan}', fontsize = 20, y = 1.05) 
    
    vmin = mean_phase_freq.sel(chan = chan).quantile(delta_colorlim)
    vmax = mean_phase_freq.sel(chan = chan).quantile(1 - delta_colorlim)
    vlim = abs(vmin) if abs(vmin) > abs(vmax) else abs(vmax)

    for r, odeur in enumerate(odeurs):
        for c, bloc in enumerate(blocs):

            ax = axs[r,c]
            
            im = ax.pcolormesh(phase_freq.coords['phase'].values, 
                               phase_freq.coords['freq'].values,  
                               mean_phase_freq.loc[odeur, chan, bloc , : ,:].values,
                               cmap = 'seismic',
                               norm = 'linear',
                               vmin = -vlim,
                               vmax = vlim)
            ax.set_yscale('log')

            
            if c == 0:
                ax.set_yticks([4,8,12, 30, 65, 100 , 150])
                ax.set_yticklabels([4,8,12, 30, 65, 100 , 150])
                ax.set_ylabel('Freq [Hz]')
            else:
                ax.set_yticks([])

            if r == len(odeurs) - 1:
                ax.set_xlabel('Phase')
            else:
                ax.set_xticklabels([])

            ax.axvline(x = 0.4, color = 'r')
            N = N_cycles_pooled.loc[(odeur,bloc), 'N']
            ax.set_title(f'{bloc} - {odeur} - N : {N}')
            

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('Power (z-normalized)',fontsize=10)        

    file = fig_folder / 'global' / f'{chan}'
    fig.savefig(file, bbox_inches = 'tight') 
    plt.close()
    
    
### FIG 2 = BY SUBJECT 

print('FIG 2')

for chan in eeg_chans:

    folder_path = fig_folder / 'by_subject' / f'{chan}' 

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    for participant in subject_keys:

        fig, axs = plt.subplots(nrows = len(odeurs), ncols = len(blocs), figsize = (15,7), constrained_layout = True)
        fig.suptitle(f'Mean phase-frequency power map in {participant} in electrode {chan}', fontsize = 20, y = 1.05) 

        vmin = all_phase_freq.sel(chan = chan, participant = participant).quantile(delta_colorlim)
        vmax = all_phase_freq.sel(chan = chan, participant = participant).quantile(1 - delta_colorlim)
        vlim = abs(vmin) if abs(vmin) > abs(vmax) else abs(vmax)

        for r, odeur in enumerate(odeurs):
            for c, bloc in enumerate(blocs):

                ax = axs[r,c]

                im = ax.pcolormesh(phase_freq.coords['phase'].values, 
                                   phase_freq.coords['freq'].values,  
                                   all_phase_freq.loc[participant, odeur, chan, bloc , : ,:].values,
                                   cmap = 'seismic',
                                   norm = 'linear',
                                   vmin = -vlim,
                                   vmax = vlim)
                ax.set_yscale('log')
                if c == 0:
                    ax.set_yticks([4,8,12, 30, 65, 100 , 150])
                    ax.set_yticklabels([4,8,12, 30, 65, 100 , 150])
                    ax.set_ylabel('Freq [Hz]')
                else:
                    ax.set_yticks([])

                if r == len(odeurs) - 1:
                    ax.set_xlabel('Phase')
                else:
                    ax.set_xticklabels([])

                ax.axvline(x = 0.4, color = 'r')  
                N = N_cycles.loc[(participant,odeur,bloc), 'N']
                ax.set_title(f'{bloc} - {odeur} - N : {N}')


        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title('Power (z-normalized)',fontsize=10)        

        
        file = folder_path / f'{participant}_{chan}'
        fig.savefig(file, bbox_inches = 'tight') 
        plt.close()
