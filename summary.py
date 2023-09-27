import ghibtools as gh
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pingouin
from compute_global_dataframes import resp_features_concat_job
from params import *
import physio
from configuration import base_folder
import os
from bibliotheque import df_baseline


# OBJECTIVE RELAXATION
resp =resp_features_concat_job.get(global_key).to_dataframe()


fig, ax = plt.subplots()
        
gh.auto_stats(df = resp, 
              predictor = 'session', 
              outcome = 'cycle_duration', 
              design = 'within',
              subject = 'participant', 
              ax=ax,
             outcome_clean_label = 'Cycle Duration',
             outcome_unit = 'sec',
             strip = True,
             lines = True,
             xtick_info = False)

                
file = base_folder / 'Figures' / 'resp_features' / f'cycle_duration_clean.png'
fig.savefig(file, bbox_inches = 'tight', dpi = 300)
plt.show()