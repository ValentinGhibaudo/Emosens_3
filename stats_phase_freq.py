import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from compute_phase_freq import phase_freq_job
from compute_resp_features import respiration_features_job
from bibliotheque import init_nan_da
from params import *
from configuration import base_folder
import os

p = phase_freq_fig_params
baseline_mode = p['baseline_mode']
compress_cycle_mode = p['compress_cycle_mode']

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

fig_folder = base_folder / 'Figures' / 'phase_freq'

# COLORBAR POS
ax_x_start = 1.05
ax_x_width = 0.02
ax_y_start = 0
ax_y_height = 1


# CONCAT
all_phase_freq = None

for run_key in stim_keys:
    
    participant, session = run_key.split('_')
    phase_freq = phase_freq_job.get(run_key)
    
    power = phase_freq['power']
    itpc = phase_freq['itpc']
    
    if all_phase_freq is None:
        all_phase_freq = init_nan_da({'participant':subject_keys, 
                                      'feature_type':['power','itpc'],
                                      'session':p['stim_sessions'], 
                                      'chan':power.coords['chan'].values,
                                      'freq':power.coords['freq'].values,
                                      'phase':power.coords['phase'].values
                                     })
        
    all_phase_freq.loc[participant, 'power', session, :,:,:] = power.loc[baseline_mode , compress_cycle_mode,:,:].values
    all_phase_freq.loc[participant, 'itpc', session, :,:,:] = itpc.values
       
### FIG 1 = GLOBAL
print('FIG 1')

global_phase_freq = all_phase_freq.mean('participant')

for feature in ['power','itpc']:

    for chan in global_phase_freq.coords['chan'].values:

        fig, axs = plt.subplots(ncols = len(p['stim_sessions']), figsize = (15,5), constrained_layout = True)

        fig.suptitle(f'Mean phase-frequency {feature} map across {len(subject_keys)} subjects in electrode {chan}', fontsize = 20, y = 1.05) 

        vmin = global_phase_freq.sel(chan=chan).quantile(p['delta_colorlim'])
        vmax = global_phase_freq.sel(chan=chan).quantile(1 - p['delta_colorlim'])
        
        if feature == 'power':
            if vmax > 0 and vmin < 0:
                vmin = vmin if abs(vmin) > abs(vmax) else -vmax 
                vmax = vmax if abs(vmax) > abs(vmin) else abs(vmin)
                cmap = 'seismic'
            else:
                vmin = vmin
                vmax = vmax
                cmap = 'viridis'
            
        elif feature == 'itpc':
            cmap = 'viridis'
            vmin = vmin
            vmax = vmax
            
        
        
        for c, session in enumerate(p['stim_sessions']):
            ax = axs[c]

            im = ax.pcolormesh(global_phase_freq.coords['phase'].values, 
                               global_phase_freq.coords['freq'].values,  
                               global_phase_freq.loc[feature, session, chan, : ,:].values,
                               cmap = cmap,
                               norm = 'linear',
                               vmin = vmin,
                               vmax = vmax)
            ax.set_yscale('log')

            if c == 0:
                ax.set_yticks([4,8,12, 30, 65, 100 , 150])
                ax.set_yticklabels([4,8,12, 30, 65, 100 , 150])
                ax.set_ylabel('Freq [Hz]')
            else:
                ax.set_yticks([])

            ax.set_xlabel('Phase')
            
            ax.axvline(x = 0.4, color = 'r')
            N = N_cycles_pooled.loc[session, 'N']
            ax.set_title(f'{session} - N : {N}')

        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        
        clb_title = f'Power ({baseline_mode})' if feature == 'power' else 'Phase-Clustering' 
        clb.ax.set_title(clb_title,fontsize=10)        

        file = fig_folder / feature / 'global' / f'{chan}.png'
        fig.savefig(file, bbox_inches = 'tight') 
        plt.close()
    
    
### FIG 2 = BY SUBJECT 

print('FIG 2')

for feature in ['power','itpc']:
    for chan in all_phase_freq.coords['chan'].values:

        folder_path = fig_folder / feature / 'by_subject' / chan

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        for participant in subject_keys:

            fig, axs = plt.subplots(ncols = len(p['stim_sessions']), figsize = (15,5), constrained_layout = True)

            fig.suptitle(f'Mean phase-frequency {feature} in electrode {chan} in participant {participant}', fontsize = 20, y = 1.05) 

            vmin = all_phase_freq.sel(participant = participant, chan=chan).quantile(p['delta_colorlim'])
            vmax = all_phase_freq.sel(participant = participant, chan=chan).quantile(1 - p['delta_colorlim'])

            if feature == 'power':
                if vmax > 0 and vmin < 0:
                    vmin = vmin if abs(vmin) > abs(vmax) else -vmax 
                    vmax = vmax if abs(vmax) > abs(vmin) else abs(vmin)
                    cmap = 'seismic'
                else:
                    vmin = vmin
                    vmax = vmax
                    cmap = 'viridis'

            elif feature == 'itpc':
                cmap = 'viridis'
                vmin = vmin
                vmax = vmax

            for c, session in enumerate(p['stim_sessions']):
                ax = axs[c]

                im = ax.pcolormesh(all_phase_freq.coords['phase'].values, 
                                   all_phase_freq.coords['freq'].values,  
                                   all_phase_freq.loc[participant, feature, session, chan, : ,:].values,
                                   cmap = cmap,
                                   norm = 'linear',
                                   vmin = vmin,
                                   vmax = vmax)
                ax.set_yscale('log')

                if c == 0:
                    ax.set_yticks([4,8,12, 30, 65, 100 , 150])
                    ax.set_yticklabels([4,8,12, 30, 65, 100 , 150])
                    ax.set_ylabel('Freq [Hz]')
                else:
                    ax.set_yticks([])

                ax.set_xlabel('Phase')

                ax.axvline(x = 0.4, color = 'r')
                N = N_cycles.loc[(participant,session), 'N']
                ax.set_title(f'{session} - N : {N}')

            cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
            clb = fig.colorbar(im, cax=cbar_ax)

            clb_title = f'Power ({baseline_mode})' if feature == 'power' else 'Phase-Clustering' 
            clb.ax.set_title(clb_title,fontsize=10)        

            file = folder_path / f'{participant}.png'
            fig.savefig(file, bbox_inches = 'tight') 
            plt.close()
