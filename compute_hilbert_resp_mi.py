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
#     da_sliced = xr.open_dataarray(f'../Preprocessing/Data_Epoched/{participant}_epoched.nc')
#     for odor in odeurs:
#         session = get_session_from_odor(participant, odor)
#         state = get_anxiety_state_from_session(participant, session)
#         for bloc in blocs:
#             for trial in count_trials[bloc]:
                
#                 rsp_preprocessed = da_sliced.loc[odor,bloc,trial,'RespiNasale',:].dropna('time').values
#                 inspi_mask = rsp_preprocessed > 0
#                 expi_mask = rsp_preprocessed < 0

#                 for chan in eeg_chans:
                    
#                     eeg_sig = da_sliced.loc[odor,bloc,trial,chan,:].dropna('time').values
                    
#                     for band, bornes in fbands.items():

#                         eeg_sig_filtered = gh.iirfilt(eeg_sig, srate, lowcut=bornes[0], highcut=bornes[1], order = 4)
#                         eeg_sig_amp = gh.get_amp(eeg_sig_filtered)
#                         inspi_amp = np.mean(eeg_sig_amp[inspi_mask])
#                         expi_amp = np.mean(eeg_sig_amp[expi_mask])
#                         mi_abs = np.abs((inspi_amp - expi_amp) / inspi_amp)
#                         mi_abs_sum = np.abs((inspi_amp - expi_amp) / (inspi_amp+expi_amp))
#                         mi_abs_mean = np.abs((inspi_amp - expi_amp) / np.mean(eeg_sig_amp))
#                         mi_real_sum = (inspi_amp - expi_amp) / (inspi_amp+expi_amp)
#                         mi_real_mean = (inspi_amp - expi_amp) / np.mean(eeg_sig_amp)
#                         row = [participant, odor, session, state, bloc, trial , chan, band , mi_real_sum, mi_real_mean, mi_abs, mi_abs_sum, mi_abs_mean]
#                         rows.append(row)
                    
# df_pac_rsp = pd.DataFrame(rows, columns = ['participant','odor','session','state','bloc','trial','chan','band','mi_real_sum','mi_real_mean','mi_abs','mi_abs_sum','mi_abs_mean'])
# df_pac_rsp.to_excel('../Tables/hilbert_resp_mi.xlsx')



from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
import physio
from bibliotheque import get_odor_from_session
from store_timestamps import timestamps_job
from compute_hilbert_envelope import hilbert_envelope_job
from preproc import convert_vhdr_job


def compute_hilbert_resp_mi(run_key, **p):
    
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    odor = get_odor_from_session(run_key)
    
    hilbert_envelope = hilbert_envelope_job.get(run_key)
    hilbert_envelope = hilbert_envelope['eeg_envelope']
    srate = hilbert_envelope.attrs['srate']
    
    resp_raw = convert_vhdr_job.get(run_key)
    resp_raw = resp_raw['raw'].sel(chan = p['resp_chan'])
    resp_clean = physio.preprocess(resp_raw, srate, band=p['low_pass_freq'], btype='lowpass', ftype='bessel',
                                order=p['filter_order'], normalize=False)
    resp_clean = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=p['smooth_sigma_ms'])


    timestamps = timestamps_job.get(run_key)
    timestamps = timestamps.to_dataframe()
    
    rows = []
    
    for bi, bloc in enumerate(timestamps['bloc'].unique()):
        for ti, trial in enumerate(timestamps['trial'].unique()):

            if bloc == 'Free' and trial == 3:
                continue
            
            start_ind = int(timestamps.set_index(['bloc','trial']).loc[(bloc,trial), 'timestamp'] * srate)
            duration_inds = int(timestamps.set_index(['bloc','trial']).loc[(bloc,trial), 'duration'] * srate)
            stop_ind = start_ind + duration_inds

            resp_slice = resp_clean[start_ind:stop_ind]

            baseline = physio.get_empirical_mode(resp_slice)
            espilon = (np.quantile(resp_slice, 0.75) - np.quantile(resp_slice, 0.25)) / 100.
            baseline_detect = baseline - espilon * 5.
            
            inspi_mask = resp_slice > baseline_detect
            expi_mask = resp_slice < baseline_detect

            hilbert_envelope_slice = hilbert_envelope[:,:,start_ind:stop_ind]

            for chan in hilbert_envelope_slice.coords['chan'].values:
                
                for band in hilbert_envelope_slice.coords['band'].values:

                    eeg_sig_amp = hilbert_envelope_slice.loc[chan, band, :].values
                    inspi_amp = np.mean(eeg_sig_amp[inspi_mask])
                    expi_amp = np.mean(eeg_sig_amp[expi_mask])
                    mi_abs = np.abs((inspi_amp - expi_amp) / inspi_amp)
                    mi_abs_sum = np.abs((inspi_amp - expi_amp) / (inspi_amp+expi_amp))
                    mi_abs_mean = np.abs((inspi_amp - expi_amp) / np.mean(eeg_sig_amp))
                    mi_real_sum = (inspi_amp - expi_amp) / (inspi_amp+expi_amp)
                    mi_real_mean = (inspi_amp - expi_amp) / np.mean(eeg_sig_amp)
                    
                    row = [participant, odor, session, bloc, trial , chan, band , mi_real_sum, mi_real_mean, mi_abs, mi_abs_sum, mi_abs_mean]
                    rows.append(row)

    hilbert_resp_mi = pd.DataFrame(rows, columns = ['participant','odor','session','bloc','trial','chan',
                                                    'band','mi_real_sum','mi_real_mean','mi_abs','mi_abs_sum','mi_abs_mean'])
    hilbert_resp_mi_ds = xr.Dataset(hilbert_resp_mi)
    return hilbert_resp_mi_ds

def test_compute_hilbert_resp_mi():
    run_key = 'P01_ses03'
    hilbert_resp_mi_ds = compute_hilbert_resp_mi(run_key, **hilbert_at_resp_mi)
    print(hilbert_resp_mi_ds.to_dataframe())
    

hilbert_resp_mi_job = jobtools.Job(precomputedir, 'hilbert_resp_mi', hilbert_at_resp_mi, compute_hilbert_resp_mi)
jobtools.register_job(hilbert_resp_mi_job)


def compute_all():
    jobtools.compute_job_list(hilbert_resp_mi_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 4)
    # jobtools.compute_job_list(hilbert_resp_mi_job, run_keys, force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    # test_compute_hilbert_resp_mi()
    compute_all()