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
import ghibtools as gh

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

# FIGS TEXT PRELOAD VARIABLES
subject_compress_mode = p['compress_subject']
cycle_compress_mode = p['compress_cycle_mode']
sup_fontsize=20
sup_pos=1.05
yticks = [4,8,12, p['max_freq']]

baseline_mode = p["baseline_mode"]
low_q_clim = p['delta_colorlim']
high_q_clim = 1  - p['delta_colorlim']
delta_clim= f'{low_q_clim} - {high_q_clim}'
clb_fontsize = 10
clb_title =f'Power\n({baseline_mode} vs baseline)\nDelta clim : {delta_clim}'   

x_axvline = 0.4
figsize = (15,5)

vmin = 0
vmax = 3
cmap = 'viridis'

# CONCAT
all_phase_freq = None

for run_key in stim_keys:
    
    participant, session = run_key.split('_')
    phase_freq = phase_freq_job.get(run_key)
    
    power = phase_freq['power']

    if all_phase_freq is None:
        all_phase_freq = init_nan_da({'participant':subject_keys, 
                                      'session':p['stim_sessions'], 
                                      'chan':power.coords['chan'].values,
                                      'freq':power.coords['freq'].values,
                                      'phase':power.coords['phase'].values
                                     })
        
    all_phase_freq.loc[participant, session, :,:,:] = power.loc[baseline_mode , compress_cycle_mode,:,:].values
  
all_phase_freq = all_phase_freq.loc[:,:,:,:p['max_freq'],:]

### FIG 1 = GLOBAL
print('FIG 1')

if p['compress_subject'] == 'Mean':
    global_phase_freq = all_phase_freq.mean('participant')
elif p['compress_subject'] == 'Median':
    global_phase_freq = all_phase_freq.median('participant')


for chan in global_phase_freq.coords['chan'].values:

    fig, axs = plt.subplots(ncols = len(p['stim_sessions']), figsize = figsize, constrained_layout = True)
    suptitle = f'{subject_compress_mode} phase-frequency power map across {len(subject_keys)} subjects in electrode {chan} ({cycle_compress_mode})'
    fig.suptitle(suptitle, fontsize = sup_fontsize, y = sup_pos) 

    # vmin = global_phase_freq.sel(chan=chan).quantile(low_q_clim)
    # vmax = global_phase_freq.sel(chan=chan).quantile(high_q_clim)
    

    # if vmax > 0 and vmin < 0:
    #     vmin = vmin if abs(vmin) > abs(vmax) else -vmax 
    #     vmax = vmax if abs(vmax) > abs(vmin) else abs(vmin)
    #     cmap = 'seismic'
    # else:
    #     vmin = vmin
    #     vmax = vmax
    #     cmap = 'viridis'
        
    
    for c, session in enumerate(p['stim_sessions']):
        ax = axs[c]

        im = ax.pcolormesh(global_phase_freq.coords['phase'].values, 
                            global_phase_freq.coords['freq'].values,  
                            global_phase_freq.loc[session, chan, : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax)
        
        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Phase')
        
        if c == 0:
            ax.set_ylabel('Freq [Hz]')

        ax.axvline(x = x_axvline, color = 'r')
        N = N_cycles_pooled.loc[session, 'N']
        ax.set_title(f'{session} - N : {N}')

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    # cbar_ax.set_ylim(-2,3)

    clb.ax.set_title(clb_title,fontsize=clb_fontsize)        

    file = fig_folder / 'power' / 'global' / f'{chan}.png'
    fig.savefig(file, bbox_inches = 'tight') 
    plt.close()
    
    
### FIG 2 = BY SUBJECT 

print('FIG 2')


for chan in all_phase_freq.coords['chan'].values:

    folder_path = fig_folder / 'power' / 'by_subject' / chan

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    for participant in subject_keys:

        fig, axs = plt.subplots(ncols = len(p['stim_sessions']), figsize = figsize, constrained_layout = True)

        fig.suptitle(f'Mean phase-frequency power in electrode {chan} in participant {participant}', fontsize = sup_fontsize, y = sup_pos) 

        # vmin = all_phase_freq.sel(participant = participant, chan=chan).quantile(p['delta_colorlim'])
        # vmax = all_phase_freq.sel(participant = participant, chan=chan).quantile(1 - p['delta_colorlim'])


        # if vmax > 0 and vmin < 0:
        #     vmin = vmin if abs(vmin) > abs(vmax) else -vmax 
        #     vmax = vmax if abs(vmax) > abs(vmin) else abs(vmin)
        #     cmap = 'seismic'
        # else:
        #     vmin = vmin
        #     vmax = vmax
        #     cmap = 'viridis'


        for c, session in enumerate(p['stim_sessions']):
            ax = axs[c]

            im = ax.pcolormesh(all_phase_freq.coords['phase'].values, 
                                all_phase_freq.coords['freq'].values,  
                                all_phase_freq.loc[participant, session, chan, : ,:].values,
                                cmap = cmap,
                                norm = 'linear',
                                vmin = vmin,
                                vmax = vmax)
            ax.set_yscale('log')
            ax.set_yticks(ticks = yticks, labels = yticks)
            ax.minorticks_off()

            if c == 0:
                ax.set_ylabel('Freq [Hz]')
            ax.set_xlabel('Phase')

            ax.axvline(x =x_axvline, color = 'r')
            N = N_cycles.loc[(participant,session), 'N']
            ax.set_title(f'{session} - N : {N}')

        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)

        clb.ax.set_title(clb_title,fontsize=clb_fontsize)        

        file = folder_path / f'{participant}.png'
        fig.savefig(file, bbox_inches = 'tight') 
        plt.close()
