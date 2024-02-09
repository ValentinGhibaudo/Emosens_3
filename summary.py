import ghibtools as gh
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pingouin
from compute_global_dataframes import resp_features_concat_job, rsa_concat_job, hrv_concat_job, relaxation_concat_job
from params import *
import physio
from configuration import base_folder
import os
from bibliotheque import df_baseline

# save_folder = base_folder / 'Figures' / 'pour_manuscrit'
save_folder = base_folder / 'Figures' / 'pour_article'

rename_col = {'session':'experimental block'}

indexes = ['participant','experimental block']

letters = gh.get_plot_letters()

# OBJECTIVE RELAXATION
resp = resp_features_concat_job.get(global_key).to_dataframe()
resp = resp.rename(columns = rename_col)
resp = resp.groupby(indexes).median(True).reset_index()


rsa = rsa_concat_job.get(global_key).to_dataframe()
rsa = rsa.rename(columns = rename_col)
rsa = rsa.groupby(indexes).median(True).reset_index()

hrv = hrv_concat_job.get(global_key).to_dataframe()
hrv = hrv.rename(columns = rename_col)

df_loop = [resp,rsa,hrv]
metrics = ['cycle_duration','decay_amplitude','HRV_Mad']
metrics_clean = ['Cycle Duration','RSA Amplitude', 'HRV MAD']
units = ['sec','bpm','ms']

nrows = len(metrics)

fig, axs = plt.subplots(nrows = nrows, figsize = (10,17), constrained_layout = True)

for r in range(nrows):
    ax = axs[r]
    
    df = df_loop[r]
    metric = metrics[r]
    metric_clean = metrics_clean[r]
    unit = units[r]
    
    gh.auto_stats(df = df, 
                  predictor = 'experimental block', 
                  outcome = metric, 
                  design = 'within',
                  subject = 'participant', 
                  ax=ax,
                 outcome_clean_label = metric_clean,
                 outcome_unit = unit,
                 strip = True,
                 lines = True,
                 xtick_info = True,
                 fontsize= 20
                 )
    
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_title('{}'.format(letters[r]), loc = 'left', fontsize = 30)

                
file =  save_folder / 'summary_physio.png'
fig.savefig(file, bbox_inches = 'tight', dpi = 500)
plt.show()

# SUBJECTIVE RELAXATION
metrics = ['Arousal','Relaxation','Relaxation_intensity','Perceived_duration']
df = relaxation_concat_job.get(global_key).to_dataframe()
df = df.rename(columns = rename_col)
df = df.drop(columns = ['stim_name'])
df[metrics] = df[metrics].astype(float)

# metrics =  ['Relaxation_intensity','Arousal']
# metrics_clean = ['Relaxation Intensity', 'Arousal']

df = df.rename(columns = {'Relaxation_intensity':'Subjective relaxation'})
metrics =  ['Subjective relaxation','Arousal']
metrics_clean = ['Subjective relaxation', 'Arousal']


nrows = 2

fig , axs = plt.subplots(nrows = nrows, figsize = (10,12), constrained_layout = True)

for i, metric in enumerate(metrics):
    ax = axs[i]
    gh.auto_stats(df = df, 
                  predictor = 'experimental block', 
                  outcome = metric, 
                  design = 'within', 
                  subject = 'participant', 
                  ax=ax, 
                  outcome_clean_label = metrics_clean[i], 
                  outcome_unit = '/100',
                 strip = True,
                 lines = True,
                 xtick_info = True,
                 fontsize = 20,
                  force_post_hoc = False
                 )
    
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_title('{}'.format(letters[i]), loc = 'left', fontsize = 30)
    
fig.savefig(save_folder / 'summary_relaxation_subjective.png', bbox_inches = 'tight', dpi = 500)

