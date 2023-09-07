import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from compute_phase_freq import phase_freq_concat_job
from compute_resp_features import respiration_features_job
from bibliotheque import init_nan_da
from params import *
from configuration import *
import os
import ghibtools as gh
import jobtools
import mne

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
def global_phase_freq_fig(chan, cycle_compress_mode, **p):

    N_cycles = get_N_resp_cycles(run_keys)
    N_cycles_pooled = N_cycles.groupby(['session']).sum(numeric_only = True)

    cycle_compress_mode = float(cycle_compress_mode)

    all_phase_freq = phase_freq_concat_job.get(global_key)['phase_freq_concat']
    


    sessions = all_phase_freq['session'].values

    # COLORBAR POS
    ax_x_start = 1.05
    ax_x_width = 0.02
    ax_y_start = 0
    ax_y_height = 1


    # FIGS TEXT PRELOAD VARIABLES
    subject_compress_mode = p['compress_subject']

    sup_fontsize=20
    sup_pos=1.05
    yticks = [4,8,12, p['max_freq']]

    baseline_mode = p["baseline_mode"]
    low_q_clim = p['delta_colorlim']
    high_q_clim = 1  - p['delta_colorlim']
    delta_clim= f'{low_q_clim} - {high_q_clim}'
    clim_fontsize = 10
    clim_title = f'Power\n({baseline_mode} vs baseline)\nDelta clim : {delta_clim}'

    x_axvline = 0.4
    figsize = (15,7)

    cmap = p['cmap']
    
    x1 = all_phase_freq.loc[:,'odor',cycle_compress_mode,chan,:,:].values
    x2 = all_phase_freq.loc[:,'music',cycle_compress_mode,chan,:,:].values
    t_obs, clusters, cluster_pv,H0 = mne.stats.permutation_cluster_1samp_test(x1 - x2, out_type = 'mask', tail =0, verbose = False)

    if p['compress_subject'] == 'Mean':
        global_phase_freq = all_phase_freq.mean('participant')
    elif p['compress_subject'] == 'Median':
        global_phase_freq = all_phase_freq.median('participant')
    elif p['compress_subject'] == 'q75':
        global_phase_freq = all_phase_freq.quantile(q=0.75, dim='participant')

    vmin = global_phase_freq.loc[:,cycle_compress_mode,:,:,:].quantile(low_q_clim)
    vmax = global_phase_freq.loc[:,cycle_compress_mode,:,:,:].quantile(high_q_clim)

    fig, axs = plt.subplots(ncols = len(sessions), figsize = figsize, constrained_layout = True)
    suptitle = f'{subject_compress_mode} phase-frequency map across {len(subject_keys)} subjects in electrode {chan} ({cycle_compress_mode})'
    fig.suptitle(suptitle, fontsize = sup_fontsize, y = sup_pos) 

    for c, ses in enumerate(sessions):

        ax = axs[c]

        im = ax.pcolormesh(global_phase_freq.coords['phase'].values, 
                            global_phase_freq.coords['freq'].values,  
                            global_phase_freq.loc[ses, cycle_compress_mode, chan , : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )
        
        for cluster, pval in zip(clusters,cluster_pv):
            if pval < p['cluster_based_pval']:
                ax.contour(global_phase_freq.coords['phase'].values,
                           global_phase_freq.coords['freq'].values,
                           cluster, 
                           levels = 0, 
                           colors = 'k', 
                           corner_mask = True)   

        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()

        ax.set_xlabel('Phase', fontsize = 15)
        ax.set_ylabel('Freq [Hz]', fontsize = 15)

        ax.axvline(x = x_axvline, color = 'r')
        N = N_cycles_pooled.loc[ses, 'N']
        ax.set_title(f'{ses} - N : {N}', fontsize = 15)


    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title(clim_title,fontsize=clim_fontsize)

    folder = base_folder / 'Figures' / 'phase_freq' / 'power' / 'global' / chan

    fig.savefig(folder / f'{cycle_compress_mode}.png', bbox_inches = 'tight', dpi = 100) 
    plt.close()

    return xr.Dataset()

def test_global_fig_phase_freq():
    chan, cycle_compress_mode = ('F3','0.75')
    ds = global_phase_freq_fig(chan,cycle_compress_mode, **phase_freq_fig_params)
    print(ds)

global_phase_freq_fig_job = jobtools.Job(precomputedir, 'global_phase_freq_fig', phase_freq_fig_params, global_phase_freq_fig)
jobtools.register_job(global_phase_freq_fig_job)
 


# BY SUBJECT
def subject_phase_freq_fig(participant, chan, **p):

    N_cycles = get_N_resp_cycles(run_keys)

    cycle_compress_mode = p['quantile_by_subject_fig']

    all_phase_freq = phase_freq_concat_job.get(global_key)['phase_freq_concat']

    sessions = all_phase_freq['session'].values

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

    x_axvline = 0.4
    figsize = (15,7)

    cmap = p['cmap']

    folder_path = base_folder / 'Figures' / 'phase_freq' / 'power' / 'by_subject' / chan 


    # vmin = all_phase_freq.quantile(low_q_clim)
    # vmax = all_phase_freq.quantile(high_q_clim)

    vmin = all_phase_freq.loc[participant,:,cycle_compress_mode,chan,:,:].quantile(low_q_clim)
    vmax = all_phase_freq.loc[participant,:,cycle_compress_mode,chan,:,:].quantile(high_q_clim)


    fig, axs = plt.subplots(ncols = len(sessions), figsize = figsize, constrained_layout = True)
    fig.suptitle(f'Phase-frequency power map in {participant} in electrode {chan} ({cycle_compress_mode})', fontsize = sup_fontsize, y = sup_pos) 

    for c,ses in enumerate(sessions):

        ax = axs[c]

        im = ax.pcolormesh(all_phase_freq.coords['phase'].values, 
                            all_phase_freq.coords['freq'].values,  
                            all_phase_freq.loc[participant, ses, cycle_compress_mode, chan , : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )

        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()

        ax.set_xlabel('Phase')
        ax.set_ylabel('Freq [Hz]')


        ax.axvline(x = x_axvline, color = 'r')  
        N = N_cycles.loc[(participant,ses), 'N']
        ax.set_title(f'{ses} - N : {N}')


    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)

    clb.ax.set_title(clim_title,fontsize=clim_fontsize)      

    fig.savefig(folder_path / f'{participant}.png', bbox_inches = 'tight') 
    plt.close()

    return xr.Dataset()

def test_subject_phase_freq_fig():
    participant, chan = ('P01','P7')
    ds = subject_phase_freq_fig(participant, chan, **phase_freq_fig_params)
    print(ds)

subject_phase_freq_fig_job = jobtools.Job(precomputedir, 'subject_phase_freq_fig', phase_freq_fig_params, subject_phase_freq_fig)
jobtools.register_job(subject_phase_freq_fig_job)


# COMPUTE
def compute_all():
    chan_keys = power_params['chans']
    quantile_keys = [ str(e) for e in phase_freq_params['compress_cycle_modes']]


    global_phase_freq_fig_keys = [(chan_key, quantile_key) for quantile_key in quantile_keys for chan_key in chan_keys]

    jobtools.compute_job_list(global_phase_freq_fig_job, global_phase_freq_fig_keys, force_recompute=True, engine='slurm',
                              slurm_params={'cpus-per-task':'10', 'mem':'30G', },
                              module_name='stats_phase_freq',
                              )



#     subject_phase_freq_fig_keys = [(sub_key, chan_key) for chan_key in chan_keys for sub_key in subject_keys]

#     jobtools.compute_job_list(subject_phase_freq_fig_job, subject_phase_freq_fig_keys, force_recompute=True, engine='slurm',
#                               slurm_params={'cpus-per-task':'10', 'mem':'30G', },
#                               module_name='stats_phase_freq',
#                               )

if __name__ == '__main__':
    # test_global_fig_phase_freq()
    # test_subject_phase_freq_fig()
    compute_all()
        
        
            
        
