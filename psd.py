# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import ghibtools as gh
# import pandas as pd
# from params import subject_keys, odeurs , blocs, count_trials, all_chans, lowest_freq, srate

# psds = None

# for participant in subject_keys:
#     print(participant)
#     data = xr.open_dataarray(f'../Preprocessing/Data_Epoched/{participant}_epoched.nc')
#     psds = None
#     for odor in odeurs:
#         for bloc in blocs:
#             for trial in count_trials[bloc]:
#                 for chan in all_chans:
#                     sig = data.loc[odor,bloc,trial,chan,:].dropna('time').values  
#                     f , Pxx = gh.spectre(sig, srate, lowest_freq)
#                     if Pxx.size == 0:
#                         print(sig.mean())
#                         print(odor, bloc, trial, chan)
#                     if psds is None:
#                         psds = gh.init_da({'odor':odeurs,'bloc':blocs,'trial':[1,2,3],'chan':all_chans, 'freq':f})
#                     psds.loc[odor,bloc,trial,chan,:] = Pxx
#     psds.to_netcdf(f'../Analyses/PSD/{participant}_psds.nc')



from configuration import *
from params import *
import xarray as xr
import physio
import jobtools
import ghibtools as gh
from bibliotheque import init_nan_da
from epoching import epoch_eeg_job

def compute_psd(run_key, **p):
    
    epoched_eeg = epoch_eeg_job.get(run_key)
    epoched_eeg = epoched_eeg['epoched_eeg']
    srate = epoched_eeg.attrs['srate']
    
    psd = None
    
    
    for bloc in epoched_eeg.coords['bloc'].values:
        for trial in epoched_eeg.coords['trial'].values:
            for chan in epoched_eeg.coords['chan'].values:
                sig = epoched_eeg.loc[bloc, trial , chan ,:].dropna('time').values
                f , Pxx = gh.spectre(sig, srate, p['lowest_freq'])
                if psd is None:
                    psd = init_nan_da({'bloc':epoched_eeg.coords['bloc'].values , 
                                       'trial':epoched_eeg.coords['trial'].values , 'chan':epoched_eeg.coords['chan'].values, 'freq':f})
                if not sig.size == 0:
                    psd.loc[bloc, trial , chan, : ] = Pxx
                    
    psd.attrs['srate'] = srate
    psd_ds = xr.Dataset()
    psd_ds['psd'] = psd
    return psd_ds


def test_compute_psd():
    run_key = 'P01_ses03'
    psd_ds = compute_psd(run_key, **psd_params)
    
    
psd_eeg_job = jobtools.Job(precomputedir, 'psd_eeg', psd_params, compute_psd)
jobtools.register_job(psd_eeg_job)


def compute_all():
    # jobtools.compute_job_list(epoch_eeg_job, run_keys[:8], force_recompute=False, engine='joblib', n_jobs = 10)
    jobtools.compute_job_list(psd_eeg_job, run_keys[:8], force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    # test_compute_psd()
    compute_all()
