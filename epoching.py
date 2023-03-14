# import numpy as np
# import mne
# import xarray as xr
# import ghibtools as gh
# import pandas as pd
# import seaborn as sns
# from params import *
# from bibliotheque import *

# def epoching(da, timestamps, session, blocs, count_trials, trial_durations):
#     da_concat = []
#     for bloc in blocs:
#         da_bloc = gh.init_da({'trial':count_trials[bloc],'chan':da.coords['chan'].values, 'time':np.arange(0,trial_durations[bloc], 1/srate)})
#         for i, trial in enumerate(count_trials[bloc]):
#             start = int(timestamps.loc[(session,bloc,trial,'start'),'timestamp'] * srate)
#             stop = start + int(trial_durations[bloc]*srate)
#             da_bloc[i,:,:] = da[:,start:stop].values
#         da_concat.append(da_bloc)
#     da_sliced = xr.concat(da_concat, dim = 'bloc').assign_coords({'bloc':blocs})
#     return da_sliced

# for participant in subject_keys:
#     print(participant)
#     timestamps = pd.read_excel(f'../Tables/timestamps/{participant}_timestamps.xlsx', index_col = [0,1,2,3])
#     concat = []
#     odors = []
#     for session in recording_sessions:
#         odors.append(get_odor_from_session(participant ,session))
#         da = xr.load_dataarray(f'../Preprocessing/Data_Preprocessed/clean_{participant}_{session}.nc')
#         da_sliced_session = epoching(da, timestamps, session, blocs, count_trials, trial_durations)
#         concat.append(da_sliced_session)        
#     da_epoched = xr.concat(concat, dim = 'odor').assign_coords({'odor':odors})
#     da_epoched.to_netcdf(f'../Preprocessing/Data_Epoched/{participant}_epoched.nc')

import xarray as xr


import jobtools
from params import *
from configuration import *
from bibliotheque import init_nan_da

from store_timestamps import timestamps_job
from preproc import preproc_job, convert_vhdr_job


def epoching(da, srate, timestamps, blocs):
    srate = da.attrs['srate']

    trials = np.arange(1,timestamps['trial'].max()+1,1)
    time = np.arange(0 , timestamps['duration'].max(), 1 / srate)
    
    epoched_da = init_nan_da({'bloc':blocs, 'trial':trials, 'chan':da.coords['chan'].values, 'time':time})
    
    for bi, bloc in enumerate(blocs):
        timestamps_bloc = timestamps[timestamps['bloc'] == bloc]
        for ti, trial in enumerate(timestamps_bloc['trial'].unique()):
            start = timestamps_bloc.set_index('trial').loc[trial, 'timestamp']
            start_ind = int(start * srate)
            duration_inds = int(timestamps_bloc.set_index('trial').loc[trial, 'duration'] * srate)
            stop_ind = start_ind + duration_inds
            
            data_slice = da[:,start_ind:stop_ind].values
            epoched_da[bi, ti, : , :duration_inds] = data_slice
            
    epoched_da.attrs['srate'] = srate
    return epoched_da




def epoch_eeg(run_key, **p):
    ds = timestamps_job.get(run_key)
    timestamps = ds.to_dataframe()
    
    clean_eeg = preproc_job.get(run_key)
    clean_eeg = clean_eeg['eeg_clean']
    srate = clean_eeg.attrs['srate']
    
    epoched_eeg = epoching(clean_eeg, srate, timestamps, p['blocs'])
            
    epoched_eeg_ds = xr.Dataset()
    epoched_eeg_ds['epoched_eeg'] = epoched_eeg
    return epoched_eeg_ds

    
def test_epoch_eeg():
    run_key = 'P02_ses02'
    epoched_eeg_ds = epoch_eeg(run_key, **epoching_eeg_params)
    print(epoched_eeg_ds)
    
epoch_eeg_job = jobtools.Job(precomputedir, 'epoch_eeg', epoching_eeg_params, epoch_eeg)
jobtools.register_job(epoch_eeg_job)








def epoch_bio(run_key, **p):
    ds = timestamps_job.get(run_key)
    timestamps = ds.to_dataframe()
    
    bio = convert_vhdr_job.get(run_key)
    bio = bio['raw'].sel(chan = p['bio_chans'])
    srate = bio.attrs['srate']
    
    epoched_bio = epoching(bio, srate, timestamps, p['blocs'])
            
    epoched_bio_ds = xr.Dataset()
    epoched_bio_ds['epoched_bio'] = epoched_bio
    return epoched_bio_ds

def test_epoch_bio():
    run_key = 'P02_ses02'
    epoched_bio_ds = epoch_bio(run_key, **epoching_bio_params)
    print(epoched_bio_ds)
    
epoch_bio_job = jobtools.Job(precomputedir, 'epoch_bio', epoching_bio_params, epoch_bio)
jobtools.register_job(epoch_bio_job)





def compute_all():
    jobtools.compute_job_list(epoch_eeg_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 10)
    # jobtools.compute_job_list(epoch_eeg_job, run_keys[:8], force_recompute=False, engine='loop')
    # jobtools.compute_job_list(epoch_bio_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 4)
    
    
    
    
    
if __name__ == '__main__':
    # test_epoch_eeg()
    # test_epoch_bio()
    compute_all()
