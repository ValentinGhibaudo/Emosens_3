from configuration import *
from params import *
import xarray as xr
import pandas as pd
import jobtools
import ghibtools as gh
from compute_psd import psd_eeg_job
from preproc import convert_vhdr_job

def compute_power_at_resp(run_key, **p):
    participant, session = run_key.split('_')
    
    psd_eeg = psd_eeg_job.get(run_key)['psd']
    srate = psd_eeg.attrs['srate']
    
    resp_sig = convert_vhdr_job.get(run_key)['raw'].sel(chan = p['resp_chan'], time = slice(0,p['session_duration'])).values[:-1]

    f_resp, Pxx_resp = gh.spectre(resp_sig, srate, p['lowest_freq_psd_resp'])
    Pxx_resp_sel = Pxx_resp[f_resp<1]
    max_resp = np.max(Pxx_resp_sel)
    argmax_resp = np.argmax(Pxx_resp_sel)
    fmax_resp = f_resp[argmax_resp]

    rows = []
    
    for chan in psd_eeg.coords['chan'].values:

        max_eeg = float(psd_eeg.loc[chan , fmax_resp])   
        row = [participant, session, chan,  fmax_resp, max_resp, max_eeg]
        rows.append(row)

    power_at_resp = pd.DataFrame(rows, columns = ['participant','session','chan','fmax_resp','max_resp','max_eeg'])
    ds_power_at_resp = xr.Dataset(power_at_resp)
    return ds_power_at_resp


def test_compute_power_at_resp():
    run_key = 'P02_baseline'
    ds_power_at_resp = compute_power_at_resp(run_key, **power_at_resp_params)
    print(ds_power_at_resp.to_dataframe())
    

power_at_resp_job = jobtools.Job(precomputedir, 'power_at_resp', power_at_resp_params, compute_power_at_resp)
jobtools.register_job(power_at_resp_job)


def compute_all():
    jobtools.compute_job_list(power_at_resp_job, run_keys, force_recompute=False, engine='loop')
    
if __name__ == '__main__':
    # test_compute_power_at_resp()
    compute_all()              