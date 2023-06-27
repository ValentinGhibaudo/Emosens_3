from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job
import matplotlib.pyplot as plt
import pandas as pd

def get_eda_metrics(eda_signal, srate, show = False):
    import neurokit2 as nk
    df, info = nk.eda_process(eda_signal, sampling_rate=srate, method='neurokit')
    tonic = df['EDA_Tonic'].mean()
    info_df = pd.DataFrame.from_dict(info, orient = 'columns')
    n_scr = info_df.shape[0]
    metrics = pd.DataFrame.from_dict(info, orient = 'columns').drop(columns = ['sampling_rate','SCR_Onsets','SCR_Peaks','SCR_Recovery']).mean().to_frame().T
    metrics.insert(0, 'N_SCR', n_scr)
    metrics.insert(0, 'Tonic', tonic)

    if show:
        plt.figure()
        nk.eda_plot(df)
        plt.show()

    return metrics


def compute_eda_metrics(run_key, **p):
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    eda_raw = raw_dataset['raw'].sel(chan='GSR', time = slice(0, p['session_duration'])).values[:-1]
    srate = raw_dataset['raw'].attrs['srate']
    
    subject_key, ses_key = run_key.split('_')

    eda_metrics = get_eda_metrics(eda_raw, srate)
    eda_metrics.insert(0, 'session', ses_key)
    eda_metrics.insert(0, 'participant', subject_key)
  
    ds = xr.Dataset(eda_metrics)

    return ds
    


def test_compute_eda():
    run_key = 'P01_baseline'
    ds = compute_eda_metrics(run_key, **eda_params)
    print(ds)
    

def compute_all():
    jobtools.compute_job_list(eda_job, run_keys, force_recompute=False, engine='loop')


eda_job = jobtools.Job(precomputedir, 'eda', eda_params, compute_eda_metrics)
jobtools.register_job(eda_job)


if __name__ == '__main__':
    # test_compute_eda()
    compute_all()



