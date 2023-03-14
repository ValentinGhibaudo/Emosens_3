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
from bibliotheque import get_odor_from_session
from psd import psd_eeg_job
from epoching import epoch_bio_job


def compute_power_at_resp(run_key, **p):
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    odor = get_odor_from_session(run_key)
    psd_eeg = psd_eeg_job.get(run_key)
    psd_eeg = psd_eeg['psd']
    srate = psd_eeg.attrs['srate']
    
    resp = epoch_bio_job.get(run_key)
    resp = resp['epoched_bio']
    resp = resp.sel(chan = p['resp_chan'])
    
    rows = []
    
    for bloc in psd_eeg.coords['bloc'].values:
        for trial in psd_eeg.coords['trial'].values:
            
            resp_sig = resp.loc[bloc, trial, :].dropna('time').values
            
            if not resp_sig.size == 0:
                f_resp, Pxx_resp = gh.spectre(resp_sig, srate, p['lowest_freq_psd_resp'])
                Pxx_resp_sel = Pxx_resp[f_resp<1]
                max_resp = np.max(Pxx_resp_sel)
                argmax_resp = np.argmax(Pxx_resp_sel)
                fmax_resp = f_resp[argmax_resp]

                for chan in psd_eeg.coords['chan'].values:

                    max_eeg = float(psd_eeg.loc[bloc, trial , chan , fmax_resp])   
                    row = [participant, session, odor, bloc, trial, chan,  fmax_resp, max_resp, max_eeg]
                    rows.append(row)

    power_at_resp = pd.DataFrame(rows, columns = ['participant','session','odor','bloc','trial','chan','fmax_resp','max_resp','max_eeg'])
    ds_power_at_resp = xr.Dataset(power_at_resp)
    return ds_power_at_resp

def test_compute_power_at_resp():
    run_key = 'P01_ses03'
    ds_power_at_resp = compute_power_at_resp(run_key, **power_at_resp_params)
    print(ds_power_at_resp.to_dataframe())
    

power_at_resp_job = jobtools.Job(precomputedir, 'power_at_resp', power_at_resp_params, compute_power_at_resp)
jobtools.register_job(power_at_resp_job)


def compute_all():
    jobtools.compute_job_list(power_at_resp_job, run_keys, force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    # test_compute_power_at_resp()
    compute_all()              
                    
                    
                
                
                
                
                
    