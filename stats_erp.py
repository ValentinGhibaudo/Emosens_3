import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from compute_phase_freq import erp_concat_job
from compute_resp_features import respiration_features_job
from bibliotheque import init_nan_da
from params import *
from configuration import *
import os
import ghibtools as gh
import jobtools



def get_N_resp_cycles(run_keys):
    concat = []
    for run_key in run_keys:
        participant, session = run_key.split('_')
        resp = respiration_features_job.get(run_key).to_dataframe()
        concat.append(resp)

    all_resp = pd.concat(concat)

    N_cycles = all_resp.value_counts(subset = ['participant','session']).to_frame().reset_index().rename(columns ={'count':'N'}).set_index(['participant','session'])
    return N_cycles


# GLOBAL
def global_erp_fig(chan, center, **p):

    N_cycles = get_N_resp_cycles(run_keys)
    N_cycles_pooled = N_cycles.groupby(['session']).sum(numeric_only = True)

    all_erp = erp_concat_job.get(global_key)['erp_concat']

    sessions = all_erp['session'].values

    # COLORBAR POS
    ax_x_start = 1.05
    ax_x_width = 0.02
    ax_y_start = 0
    ax_y_height = 1


    # FIGS TEXT PRELOAD VARIABLES
    sup_fontsize=20
    sup_pos=1.05
    yticks = [4,8,12, p['max_freq']]

    baseline_mode = p["baseline_mode"]
    low_q_clim = p['delta_colorlim']
    high_q_clim = 1  - p['delta_colorlim']
    delta_clim= f'{low_q_clim} - {high_q_clim}'
    clim_fontsize = 10
    clim_title = f'Power\n({baseline_mode} vs baseline)\nDelta clim : {delta_clim}'

    figsize = (15,7)

    cmap = p['cmap']

    global_erp = all_erp.mean('participant')

    vmin = global_erp.quantile(low_q_clim)
    vmax = global_erp.quantile(high_q_clim)

    fig, axs = plt.subplots(ncols = len(sessions), figsize = figsize, constrained_layout = True)
    suptitle = f'Mean ERP-like time-frequency map across {len(subject_keys)} subjects in electrode {chan} ({center})'
    fig.suptitle(suptitle, fontsize = sup_fontsize, y = sup_pos) 

    for c, ses in enumerate(sessions):

        ax = axs[c]

        im = ax.pcolormesh(global_erp.coords['time'].values, 
                            global_erp.coords['freq'].values,  
                            global_erp.loc[ses, chan , center, : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )

        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Time [sec]')

        ax.axvline(x = 0, color = 'k', ls = '--', lw = 0.5)  
        N = N_cycles_pooled.loc[ses, 'N']
        ax.set_title(f'{ses} - N : {N}')


    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title(clim_title,fontsize=clim_fontsize)

    folder = base_folder / 'Figures' / 'ERP' / 'global' / center

    fig.savefig(folder / f'{chan}.png', bbox_inches = 'tight', dpi = 100) 
    plt.close()

    return xr.Dataset()

def test_global_fig_erp():
    chan, center = ('P7','inspi_time')
    ds = global_erp_fig(chan, center, **erp_fig_params)
    print(ds)

global_erp_fig_job = jobtools.Job(precomputedir, 'global_erp_fig', erp_fig_params, global_erp_fig)
jobtools.register_job(global_erp_fig_job)
 


# BY SUBJECT
def subject_erp_fig(participant, chan, center, **p):

    N_cycles = get_N_resp_cycles(run_keys)
    N_cycles_pooled = N_cycles.groupby(['session']).sum(numeric_only = True)
    
    all_erp = erp_concat_job.get(global_key)['erp_concat']

    sessions = all_erp['session'].values

    # COLORBAR POS
    ax_x_start = 1.05
    ax_x_width = 0.02
    ax_y_start = 0
    ax_y_height = 1


    # FIGS TEXT PRELOAD VARIABLES
    sup_fontsize=20
    sup_pos=1.05
    yticks = [4,8,12, p['max_freq']]

    baseline_mode = p["baseline_mode"]
    low_q_clim = p['delta_colorlim']
    high_q_clim = 1  - p['delta_colorlim']
    delta_clim= f'{low_q_clim} - {high_q_clim}'
    clim_fontsize = 10
    clim_title = f'Power\n({baseline_mode} vs baseline)\nDelta clim : {delta_clim}'

    figsize = (15,7)

    cmap = p['cmap']

    # vmin = all_erp.quantile(low_q_clim)
    # vmax = all_erp.quantile(high_q_clim)

    vmin = all_erp.loc[participant,:,chan,:,:,:].quantile(low_q_clim)
    vmax = all_erp.loc[participant,:,chan,:,:,:].quantile(high_q_clim)

    fig, axs = plt.subplots(ncols = len(sessions), figsize = figsize, constrained_layout = True)
    fig.suptitle(f'ERP-like time-frequency power map in {participant} in electrode {chan} ({center})', fontsize = sup_fontsize, y = sup_pos) 

    for c, ses in enumerate(sessions):

        ax = axs[c]

        im = ax.pcolormesh(all_erp.coords['time'].values, 
                            all_erp.coords['freq'].values,  
                            all_erp.loc[participant, ses, chan , center, : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )

        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Time [sec]')

        ax.axvline(x = 0, color = 'k', ls = '--', lw = 0.5)  
        N = N_cycles.loc[(participant,ses), 'N']
        ax.set_title(f'{ses} - N : {N}')


    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)

    clb.ax.set_title(clim_title,fontsize=clim_fontsize)      

    folder_path = base_folder / 'Figures' / 'ERP' / 'by_subject' / center / chan

    fig.savefig(folder_path / f'{participant}.png', bbox_inches = 'tight') 
    plt.close()

    return xr.Dataset()

def test_subject_erp_fig():
    participant, chan , center = ('P01','P7','inspi_time')
    ds = subject_erp_fig(participant, chan , center , **erp_fig_params)
    print(ds)

subject_erp_fig_job = jobtools.Job(precomputedir, 'subject_erp_fig', erp_fig_params, subject_erp_fig)
jobtools.register_job(subject_erp_fig_job)


# COMPUTE
def compute_all():

    chan_keys = power_params['chans']
    centers = ['inspi_time','expi_time']

    global_erp_keys = [(chan, center) for chan in chan_keys for center in centers]

    jobtools.compute_job_list(global_erp_fig_job, global_erp_keys, force_recompute=True, engine='slurm',
                              slurm_params={'cpus-per-task':'20', 'mem':'30G', },
                              module_name='stats_erp',
                              )

    subject_erp_keys = [(sub_key, chan_key, center) for center in centers for chan_key in chan_keys for sub_key in subject_keys]
    
    jobtools.compute_job_list(subject_erp_fig_job, subject_erp_keys, force_recompute=True, engine='slurm',
                              slurm_params={'cpus-per-task':'20', 'mem':'30G', },
                              module_name='stats_erp',
                              )

if __name__ == '__main__':
    # test_global_fig_erp()
    # test_subject_erp_fig()
    compute_all()
        
        
            
        
