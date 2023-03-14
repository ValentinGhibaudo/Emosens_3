# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import ghibtools as gh
# import pandas as pd
# from params import subject_keys, odeurs , blocs, count_trials, all_chans, lowest_freq, srate


# seeds = ['RespiNasale','FCI']

# for participant in subject_keys:
#     print(participant)
#     data = xr.open_dataarray(f'../Preprocessing/Data_Epoched/{participant}_epoched.nc')
#     coherences = None
#     for odor in odeurs:
#         for bloc in blocs:
#             for trial in count_trials[bloc]:
#                 for seed in seeds:
#                     sig_seed = data.loc[odor,bloc,trial,seed,:].dropna('time').values  
#                     for chan in all_chans:
#                         sig = data.loc[odor,bloc,trial,chan,:].dropna('time').values  
#                         f, Cxy = gh.coherence(sig, sig_seed, srate, lowest_freq, nfft_factor = 1)
#                         if coherences is None:
#                             coherences = gh.init_da({'odor':odeurs,'bloc':blocs,'trial':[1,2,3],'seed':seeds,'chan':all_chans, 'freq':f})
#                         coherences.loc[odor,bloc,trial,seed,chan,:] = Cxy
#     coherences.to_netcdf(f'../Analyses/Coherence/{participant}_coherences.nc')



# import numpy as np
# import xarray as xr
# import pandas as pd
# from params import subject_keys, all_chans, blocs, odeurs, count_trials, srate, lowest_freq


# rows = []
# for participant in subject_keys:
#     print(participant)
#     psds = xr.open_dataarray(f'../Analyses/PSD/{participant}_psds.nc')
#     for odeur in odeurs:
#         for bloc in blocs:
#             for trial in count_trials[bloc]:

#                 respi_power = psds.loc[odeur, bloc , trial,  'RespiNasale', lowest_freq:0.7]  
#                 f_resp = float(respi_power.idxmax(dim ='freq'))

#                 for chan in all_chans : 
#                     power_at_resp = float(psds.loc[odeur, bloc , trial,  chan, lowest_freq:0.7].max())

#                     rows.append([participant, odeur, bloc , trial , chan, f_resp, power_at_resp])
                        
# df_power_at_resp = pd.DataFrame(rows, columns = ['participant','odor','bloc','trial','chan','f_resp','power_at_resp'])
# df_power_at_resp.to_excel('../Tables/power_at_resp.xlsx')

from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from bibliotheque import get_odor_from_session, init_nan_da
from epoching import epoch_bio_job, epoch_eeg_job


def compute_coherence(run_key, **p):

    eeg = epoch_eeg_job.get(run_key)
    eeg = eeg['epoched_eeg']
    srate = eeg.attrs['srate']
    
    resp = epoch_bio_job.get(run_key)
    resp = resp['epoched_bio']
    resp = resp.sel(chan = p['resp_chan'])
    
    da_cxy = None
    
    for bloc in eeg.coords['bloc'].values:
        for trial in eeg.coords['trial'].values:
                
            resp_sig = resp.loc[bloc, trial, :].dropna('time').values
            
            if resp_sig.size == 0:
                continue 
                
            f_resp, Pxx_resp = gh.spectre(resp_sig, srate, p['lowest_freq_psd_resp'], nfft_factor = p['nfft_factor'], n_cycles = p['n_cycles'])

            Pxx_resp_sel = Pxx_resp[f_resp<1]
            max_resp = np.max(Pxx_resp_sel)
            argmax_resp = np.argmax(Pxx_resp_sel)
            fmax_resp = f_resp[argmax_resp]
            
            for chan in eeg.coords['chan'].values:
                
                sig = eeg.loc[bloc,trial,chan,:].dropna('time').values  
                f, Cxy = gh.coherence(sig, resp_sig, srate, p['lowest_freq_coherence'], nfft_factor = p['nfft_factor'], n_cycles = p['n_cycles'])
                
                if da_cxy is None:
                    da_cxy = init_nan_da({'bloc':eeg.coords['bloc'].values , 'trial':eeg.coords['trial'].values, 'chan':eeg.coords['chan'].values, 'freq':f})
                da_cxy.loc[bloc, trial , chan, : ] = Cxy
                
                max_cxy = float(Cxy[argmax_resp])  

    
    
    ds_coherence = xr.Dataset()
    ds_coherence['coherence'] = da_cxy

    return ds_coherence

def coherence_at_resp(run_key, **p):
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    odeur = get_odor_from_session(run_key)
    
    coherence = coherence_job.get(run_key)
    coherence = coherence['coherence']
    
    resp = epoch_bio_job.get(run_key)
    resp = resp['epoched_bio']
    resp = resp.sel(chan = p['resp_chan'])
    
    rows = []
    
    for bloc in coherence.coords['bloc'].values:
        for trial in coherence.coords['trial'].values:
            
            resp_sig = resp.loc[bloc, trial, :].dropna('time').values
            
            if resp_sig.size == 0:
                continue 
                
            f_resp, Pxx_resp = gh.spectre(resp_sig, srate, p['lowest_freq_psd_resp'], nfft_factor = p['nfft_factor'], n_cycles = p['n_cycles'])

            Pxx_resp_sel = Pxx_resp[f_resp<1]
            max_resp = np.max(Pxx_resp_sel)
            argmax_resp = np.argmax(Pxx_resp_sel)
            fmax_resp = f_resp[argmax_resp]
            
            for chan in coherence.coords['chan'].values:
                
                Cxy_at_resp = coherence.loc[bloc, trial, chan, fmax_resp].values

                row = [participant, session, odeur, bloc, trial, chan, fmax_resp, max_resp,  Cxy_at_resp]
                rows.append(row)
    
    
    df_coherence_at_resp = pd.DataFrame(rows, columns = ['participant','session','odor','bloc','trial','chan', 'fmax_resp','max_resp', 'max_coherence'])
    ds_coherence_at_resp = xr.Dataset(df_coherence_at_resp)
    return ds_coherence_at_resp


def test_compute_coherence():
    run_key = 'P02_ses02'
    # run_key = 'P02_ses03'
    
    # ds_coherence = compute_coherence(run_key, **coherence_params)
    # print(ds_coherence)
    
    ds_coherence = coherence_job.get(run_key)
    print(ds_coherence)
    
    
def test_compute_coherence_at_resp():
    run_key = 'P02_ses02'
    ds_coherence_at_resp = coherence_at_resp_job.get(run_key)
    print(ds_coherence_at_resp.to_dataframe())
    

    

coherence_job = jobtools.Job(precomputedir, 'coherence', coherence_params, compute_coherence)
jobtools.register_job(coherence_job)

coherence_at_resp_job = jobtools.Job(precomputedir, 'coherence_at_resp', coherence_at_resp_params, coherence_at_resp)
jobtools.register_job(coherence_at_resp_job)


def compute_all():
    # jobtools.compute_job_list(coherence_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 3)
    # jobtools.compute_job_list(coherence_job, run_keys, force_recompute=False, engine='loop')
    
    # jobtools.compute_job_list(coherence_at_resp_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(coherence_at_resp_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 3)
    
if __name__ == '__main__':
    # test_compute_coherence()
    # test_compute_coherence_at_resp()
    compute_all()           
                    
                    
                
                
                
                
                
    