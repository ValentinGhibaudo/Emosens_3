from configuration import *
from params import *
import xarray as xr
import jobtools
from preproc import convert_vhdr_job
import matplotlib.pyplot as plt
import pandas as pd

def get_eda_metrics(eda_signal, srate, show = False):
    """
    Compute electrodermal activity metrics with neurokit2
    """
    import neurokit2 as nk
    df, info = nk.eda_process(eda_signal, sampling_rate=srate, method='neurokit') # compute eda metrics with neurokit2 function
    tonic = df['EDA_Tonic'].mean() # compute average value of tonic component during the 10 minutes
    info_df = pd.DataFrame.from_dict(info, orient = 'columns') # dict to df
    n_scr = info_df.shape[0] # count SCR during the 10 mins of session
    metrics = pd.DataFrame.from_dict(info, orient = 'columns').drop(columns = ['sampling_rate','SCR_Onsets','SCR_Peaks','SCR_Recovery']).mean().to_frame().T # drop non-used columns
    metrics.insert(0, 'N_SCR', n_scr) # add number of SCR metric to the dataframe
    metrics.insert(0, 'Tonic', tonic) # add average tonic value to the dataframe

    if show:
        plt.figure()
        nk.eda_plot(df)
        plt.show()

    return metrics


def compute_eda_metrics(run_key, **p):
    """
    Compute electrodermal activity metrics
    """
    raw_dataset = convert_vhdr_job.get(run_key) # load raw data
    
    eda_raw = raw_dataset['raw'].sel(chan='GSR', time = slice(0, p['session_duration'])).values[:-1] # select EDA signal and crop it to 10 mins
    srate = raw_dataset['raw'].attrs['srate']
    
    subject_key, ses_key = run_key.split('_')

    eda_metrics = get_eda_metrics(eda_raw, srate) # compute EDA metrics
    eda_metrics.insert(0, 'session', ses_key)
    eda_metrics.insert(0, 'participant', subject_key)
  
    ds = xr.Dataset(eda_metrics) # store dataframe into dataset

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



