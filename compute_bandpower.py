# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import ghibtools as gh
# import pandas as pd
# from params import *
# from bibliotheque import *

# rows = []
# for participant in subject_keys:
#     print(participant)
#     psds = xr.open_dataarray(f'../Analyses/PSD/{participant}_psds.nc')
#     for odor in odeurs:
#         session = get_session_from_odor(participant, odor)
#         state = get_anxiety_state_from_session(participant, session)
#         for bloc in blocs:
#             for trial in count_trials[bloc]:
#                 for chan in eeg_chans:
#                     Pxx = psds.loc[odor, bloc, trial, chan, :]
#                     for band in fbands.keys():
#                         lowcut = fbands[band][0]
#                         highcut = fbands[band][1]
#                         power = float(Pxx.loc[lowcut:highcut].mean('freq'))
#                         row = [participant, session, odor, state, bloc, trial, chan, band, power]
#                         rows.append(row)
# df = pd.DataFrame(rows, columns = ['participant','session','odor','state','bloc','trial','chan','band','power'])      
# df.to_excel('../Tables/bandpower.xlsx')


from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from bibliotheque import get_odor_from_session
from psd import psd_eeg_job


def compute_bandpower(run_key, **p):
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    odor = get_odor_from_session(run_key)
    psd_eeg = psd_eeg_job.get(run_key)
    psd_eeg = psd_eeg['psd']
    srate = psd_eeg.attrs['srate']
    
    rows = []
    
    for bloc in psd_eeg.coords['bloc'].values:
        for trial in psd_eeg.coords['trial'].values:
            for chan in psd_eeg.coords['chan'].values:
                
                Pxx = psd_eeg.loc[bloc, trial , chan ,:]
                
                for band, bornes in p['fbands'].items():
                    
                    power_mean = float(Pxx.loc[bornes[0]:bornes[1]].mean('freq'))
                    power_median = float(Pxx.loc[bornes[0]:bornes[1]].median('freq'))
                    power_integral = float(Pxx.loc[bornes[0]:bornes[1]].integrate('freq'))
                    
                    row = [participant, session, odor, bloc, trial, chan, band, power_mean, power_median, power_integral]
                    rows.append(row)
    
    bandpowers = pd.DataFrame(rows, columns = ['participant','session','odor','bloc','trial','chan','band','power_mean','power_median','power_integral'])
    ds_bandpower = xr.Dataset(bandpowers)
    return ds_bandpower

def test_compute_bandpower():
    run_key = 'P01_ses03'
    ds_bandpower = compute_bandpower(run_key, **bandpower_params)
    

bandpower_job = jobtools.Job(precomputedir, 'bandpower', bandpower_params, compute_bandpower)
jobtools.register_job(bandpower_job)


def compute_all():
    # jobtools.compute_job_list(epoch_eeg_job, run_keys[:8], force_recompute=False, engine='joblib', n_jobs = 10)
    jobtools.compute_job_list(bandpower_job, run_keys[:8], force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    # test_compute_bandpower()
    compute_all()


