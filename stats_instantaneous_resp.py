from compute_resp_features import respiration_features_job

import numpy as np
from configuration import *
from params import *
from bibliotheque import *
import matplotlib.pyplot as plt

metrics = ['cycle_duration','inspi_duration','expi_duration',
           'total_amplitude','inspi_amplitude','expi_amplitude',
          'total_volume','inspi_volume','expi_volume']

ncols = 3
nrows = 3

metrics_array = np.array(metrics).reshape(nrows, ncols)

for sub in subject_keys:
    
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (15,12) , constrained_layout = True)
    fig.suptitle(f'Instaneous respiration features in {sub}', fontsize = 20 , y = 1.02)
    
    for ses in session_keys:
        
        run_key = f'{sub}_{ses}'
        
        resp = respiration_features_job.get(run_key).to_dataframe()
        
        for r in range(nrows):
            for c in range(ncols):
            
                ax =axs[r,c]
                metric = metrics_array[r,c]
                ax.plot(resp[metric], label = ses)
                ax.set_xlabel('N resp cycle')
                ax.set_title(metric)
                ax.legend()
            
    fig.savefig(base_folder / 'Figures' / 'Instantaneous_resp' / f'{sub}.png', bbox_inches = 'tight')
    plt.close()