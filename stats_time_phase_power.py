import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from compute_phase_freq import phase_freq_concat_job, erp_concat_job
from compute_global_dataframes import resp_features_concat_job
from compute_psycho import oas_job, bmrq_job
from bibliotheque import init_nan_da
from params import *
from configuration import *
import os
import ghibtools as gh
import jobtools
import mne

def get_N_resp_cycles():
    all_resp = resp_features_concat_job.get(global_key).to_dataframe()
    N_cycles = all_resp.value_counts(subset = ['participant','session']).to_frame().reset_index().rename(columns ={'count':'N'}).set_index(['participant','session'])
    return N_cycles

def get_freq_resp():
    all_resp = resp_features_concat_job.get(global_key).to_dataframe()
    all_resp = all_resp.groupby(['participant','session']).mean(True).reset_index()
    return all_resp.loc[:,['participant','session','cycle_duration','inspi_duration','expi_duration']].set_index(['participant','session'])  

def get_oas_and_bmrq(sub):
    oas = oas_job.get(sub).to_dataframe().round(2)['OAS'].values[0]
    bmrq = bmrq_job.get(sub).to_dataframe().round(2)['BMRQ'].values[0]
    return oas, bmrq


# GLOBAL
def global_time_phase_fig(chan, **p):

    N_cycles = get_N_resp_cycles()
    N_cycles_pooled = N_cycles.groupby(['session']).sum(numeric_only = True)

    all_phase = phase_freq_concat_job.get(global_key)['phase_freq_concat']
    global_phase = all_phase.mean('participant')

    all_time = erp_concat_job.get(global_key)['erp_concat']
    global_time = all_time.mean('participant')

    sessions = all_phase['session'].values

    # COLORBAR POS
    ax_x_start, ax_x_width, ax_y_height, ax_y_start = 1.05, 0.02, 1, 0

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
    q = 0.75

    x_axvline = 0.4
    figsize = (15,10)

    cmap = p['cmap']
    
    x1_phase = all_phase.loc[:,'odor',q,chan,:,:].values
    x2_phase = all_phase.loc[:,'music',q,chan,:,:].values
    t_obs, clusters_phase, cluster_pv_phase,H0 = mne.stats.permutation_cluster_1samp_test(x1_phase - x2_phase, out_type = 'mask', tail =0, verbose = False)
    
    x1_time = all_time.loc[:,'odor',chan,'expi_time',:,:].values
    x2_time = all_time.loc[:,'music',chan,'expi_time',:,:].values
    t_obs, clusters_time, cluster_pv_time,H0 = mne.stats.permutation_cluster_1samp_test(x1_time - x2_time, out_type = 'mask', tail =0, verbose = False)

    vmin_phase = global_phase.loc[:,q,:,:,:].quantile(low_q_clim)
    vmax_phase = global_phase.loc[:,q,:,:,:].quantile(high_q_clim)

    vmin_time = global_time.loc[:,:,'expi_time',:,:].quantile(low_q_clim)
    vmax_time = global_time.loc[:,:,'expi_time',:,:].quantile(high_q_clim)

    vmin = vmin_time if vmin_time < vmin_phase else vmin_phase
    vmax = vmax_time if vmax_time > vmax_phase else vmax_phase

    fig, axs = plt.subplots(nrows = 2, ncols = len(sessions), figsize = figsize, constrained_layout = True)
    suptitle = f'Mean power map across {len(subject_keys)} subjects in electrode {chan}'
    fig.suptitle(suptitle, fontsize = sup_fontsize, y = sup_pos) 

    for c, ses in enumerate(sessions):

        ax = axs[0,c]
        im_phase = ax.pcolormesh(global_phase.coords['phase'].values, 
                            global_phase.coords['freq'].values,  
                            global_phase.loc[ses, q, chan , : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )
        
        for cluster, pval in zip(clusters_phase,cluster_pv_phase):
            if pval < p['cluster_based_pval']:
                ax.contour(global_phase.coords['phase'].values,
                           global_phase.coords['freq'].values,
                           cluster, 
                           levels = 0, 
                           colors = 'k', 
                           corner_mask = True)   
                
        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Phase (proportion)')
        ax.set_ylabel('Freq [Hz]')
        ax.axvline(x = x_axvline, color = 'r')
        N = N_cycles_pooled.loc[ses, 'N']
        ax.set_title(f'Phase map - {ses} - N : {N}')

        ax = axs[1,c]
        im_time = ax.pcolormesh(global_time.coords['time'].values, 
                            global_time.coords['freq'].values,  
                            global_time.loc[ses, chan , 'expi_time', : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )
        for cluster, pval in zip(clusters_time,cluster_pv_time):
            if pval < p['cluster_based_pval']:
                ax.contour(global_time.coords['time'].values,
                           global_time.coords['freq'].values,
                           cluster, 
                           levels = 0, 
                           colors = 'k', 
                           corner_mask = True)   

        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Freq [Hz]')
        ax.axvline(x = 0, color = 'r', ls = '--', lw = 1) 
        N = N_cycles_pooled.loc[ses, 'N']
        ax.set_title(f'Time map - {ses} - N : {N}')

    cbar_ax_phase = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb_phase = fig.colorbar(im_phase, cax=cbar_ax_phase)
    clb_phase.ax.set_title(clim_title,fontsize=clim_fontsize)

    folder = base_folder / 'Figures' / 'erp_phase' / 'global' 

    fig.savefig(folder / f'{chan}.png', bbox_inches = 'tight', dpi = 100) 
    plt.close()

    return xr.Dataset()

def test_global_time_phase_fig():
    chan = 'P7'
    ds = global_time_phase_fig(chan, **time_phase_fig_params)
    print(ds)

global_time_phase_fig_job = jobtools.Job(precomputedir, 'global_time_phase_fig', time_phase_fig_params, global_time_phase_fig)
jobtools.register_job(global_time_phase_fig_job)
 


# BY SUBJECT
def subject_time_phase_fig(participant, chan, **p):

    oas, bmrq = get_oas_and_bmrq(participant)

    N_cycles = get_N_resp_cycles()

    freq_resp = get_freq_resp()

    all_phase = phase_freq_concat_job.get(global_key)['phase_freq_concat']
    all_time = erp_concat_job.get(global_key)['erp_concat']

    sessions = all_phase['session'].values

    # COLORBAR POS
    ax_x_start, ax_x_width, ax_y_height, ax_y_start = 1.05, 0.02, 1, 0

    # FIGS TEXT PRELOAD VARIABLES
    sup_fontsize=20
    sup_pos=1.05
    yticks = [4,8,12, p['max_freq']]

    baseline_mode = p["baseline_mode"]
    low_q_clim = p['delta_colorlim']
    high_q_clim = 1  - p['delta_colorlim']
    delta_clim= f'{low_q_clim} - {high_q_clim}'
    clim_fontsize = 10
    q = 0.75
    clim_title = f'Power\n({baseline_mode} vs baseline)\nDelta clim : {delta_clim}'

    x_axvline = 0.4
    figsize = (15,10)

    cmap = p['cmap']

    vmin_phase = all_phase.loc[participant, :, q, chan,:,:].quantile(low_q_clim)
    vmax_phase = all_phase.loc[participant, :, q, chan,:,:].quantile(high_q_clim)
    vmin_time = all_time.loc[participant, :, chan , 'expi_time', : ,:].quantile(low_q_clim)
    vmax_time = all_time.loc[participant, :, chan , 'expi_time', : ,:].quantile(high_q_clim)

    vmin = vmin_time if vmin_time < vmin_phase else vmin_phase
    vmax = vmax_time if vmax_time > vmax_phase else vmax_phase

    fig, axs = plt.subplots(nrows = 2, ncols = len(sessions), figsize = figsize, constrained_layout = True)
    suptitle = f'Power map in {participant} in electrode {chan} \n OAS : {oas} - BMRQ {bmrq} '
    fig.suptitle(suptitle, fontsize = sup_fontsize, y = sup_pos) 

    for c, ses in enumerate(sessions):

        ax = axs[0,c]

        im_phase = ax.pcolormesh(all_phase.coords['phase'].values, 
                            all_phase.coords['freq'].values,  
                            all_phase.loc[participant, ses, q, chan , : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )
        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Phase (proportion)')
        ax.set_ylabel('Freq [Hz]')
        ax.axvline(x = x_axvline, color = 'r')
        N = N_cycles.loc[(participant,ses), 'N']
        ax.set_title(f'Phase map - {ses} - N : {N}')

        ax = axs[1,c]
        im_time = ax.pcolormesh(all_time.coords['time'].values, 
                            all_time.coords['freq'].values,  
                            all_time.loc[participant, ses, chan , 'expi_time', : ,:].values,
                            cmap = cmap,
                            norm = 'linear',
                            vmin = vmin,
                            vmax = vmax
                            )
        ax.set_yscale('log')
        ax.set_yticks(ticks = yticks, labels = yticks)
        ax.minorticks_off()
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Freq [Hz]')
        ax.axvline(x = 0, color = 'r', ls = '--', lw = 1) 
        N = N_cycles.loc[(participant,ses), 'N']
        sub_cycle = freq_resp.loc[(participant,ses),'cycle_duration'].round(2)
        sub_expi = freq_resp.loc[(participant,ses),'inspi_duration'].round(2)
        sub_inspi = freq_resp.loc[(participant,ses),'expi_duration'].round(2)
        ax.set_title(f'Time map - {ses} - N : {N} - Respi duration : {sub_cycle} ({sub_expi}+{sub_inspi}) ')

    cbar_ax_phase = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb_phase = fig.colorbar(im_phase, cax=cbar_ax_phase)
    clb_phase.ax.set_title(clim_title,fontsize=clim_fontsize)

    folder = base_folder / 'Figures' / 'erp_phase' / 'by_subject' / chan

    fig.savefig(folder / f'{participant}.png', bbox_inches = 'tight', dpi = 100) 
    plt.close()

    return xr.Dataset()

def test_subject_time_phase_fig():
    participant, chan = ('P01','P7')
    ds = subject_time_phase_fig(participant, chan, **time_phase_fig_params)
    print(ds)

subject_time_phase_fig_job = jobtools.Job(precomputedir, 'subject_time_phase_fig', time_phase_fig_params, subject_time_phase_fig)
jobtools.register_job(subject_time_phase_fig_job)


# COMPUTE
def compute_all():
    chan_keys = power_params['chans']

    global_time_phase_figs_keys = [(chan,) for chan in chan_keys]
    jobtools.compute_job_list(global_time_phase_fig_job, global_time_phase_figs_keys, force_recompute=True, engine='slurm',
                              slurm_params={'cpus-per-task':'10', 'mem':'30G', },
                              module_name='stats_time_phase_power',
                              )
    
    # jobtools.compute_job_list(global_time_phase_fig_job, global_time_phase_figs_keys, force_recompute=True, engine='loop')


#     subject_time_phase_fig_keys = [(sub_key, chan_key) for chan_key in chan_keys for sub_key in subject_keys]

#     jobtools.compute_job_list(subject_time_phase_fig_job, subject_time_phase_fig_keys, force_recompute=True, engine='slurm',
#                               slurm_params={'cpus-per-task':'10', 'mem':'30G', },
#                               module_name='stats_time_phase_power',
#                               )

if __name__ == '__main__':
    # test_global_time_phase_fig()
    # test_subject_time_phase_fig()
    compute_all()
        
        
            
        
