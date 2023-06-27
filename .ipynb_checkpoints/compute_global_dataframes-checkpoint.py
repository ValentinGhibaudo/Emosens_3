import numpy as np
import physio
import pandas as pd
import xarray as xr
from params import *
from bibliotheque import get_odor_from_session, get_metadata
from configuration import *
import os
import jobtools

from preproc import count_artifact_job
from compute_bandpower import bandpower_job
from compute_coherence import coherence_at_resp_job
from compute_eda import eda_job
from compute_coherence import coherence_at_resp_job
from compute_eda import eda_job
from compute_power_at_resp import power_at_resp_job
from compute_rri import ecg_peak_job
from compute_psycho import stai_shortform_job, maia_job, stai_longform_job
from compute_resp_features import respiration_features_job
from compute_rsa import rsa_features_job


#### FUNCTIONS

# def get_bad_chan_mapper(run_key):
#     participant, session = run_key.split('_')
#     mapper_keep_chans = {}
    
#     for chan in eeg_chans:
#         keep_code = 1 if chan not in bad_channels[participant][session] else 0
#         mapper_keep_chans[chan] = keep_code
#     return mapper_keep_chans

# def get_gender_mapper():
#     metadata = get_metadata().reset_index()
#     mapper_gender = {row['participant'] : row['gender'] for i, row in metadata.iterrows()}
#     return mapper_gender

def get_stai_long_mapper(run_key):
    sub, ses = run_key.split('_')
    df = stai_longform_job.get(run_key).to_dataframe().set_index(['participant','session'])
    return df.loc[(sub,ses),'etat']

def get_stai_long_trait_mapper():
    mapper = {}
    for sub in subject_keys:
        mapper[sub] = stai_longform_job.get(f'{sub}_ses02').to_dataframe().set_index(['participant','session']).loc[(sub,'ses02'), 'trait']
    return mapper
    
def get_maia_mapper():
    maia = maia_concat_job.get(global_key).to_dataframe()
    mapper_maia = {row['participant'] : row['Maia_Mean'] for i, row in maia.iterrows()} 
    return mapper_maia

def map_artifacted_sessions(df, artifacts):
    df['keep_trial'] = 1
    artifacts_remove = artifacts[artifacts['remove'] == 1]
    if artifacts_remove.shape[0] == 0:
        mapped_df = df.copy()   
    else:
        df_set = df.set_index(['bloc','trial'])
        
        for i, row in artifacts_remove.iterrows():
            df_set.loc[(row['bloc'],row['trial']),'keep_trial'] = 0
            
        mapped_df = df_set.reset_index()
    return mapped_df
    

#### JOBS

# STAI LONG 
def stai_long_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        concat.append(stai_longform_job.get(run_key).to_dataframe())
    df_return = pd.concat(concat).reset_index(drop = True)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return)

def test_stai_long_concat():
    ds = stai_long_concat(global_key, **stai_long_concat_params)
    print(ds.to_dataframe())
    
stai_long_concat_job = jobtools.Job(precomputedir, 'stai_long_concat', stai_long_concat_params, stai_long_concat)
jobtools.register_job(stai_long_concat_job)

# MAIA
def maia_concat(global_key, **p):
    concat = []
    for sub_key in p['subject_keys']:
        concat.append(maia_job.get(sub_key).to_dataframe())
    df_return = pd.concat(concat).reset_index(drop = True)
    outcomes = list(p['maia_params']['items'].keys())
    df_return['Maia_Mean'] = df_return.loc[:,outcomes].mean(axis = 1)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Long_Stai_Trait'] = df_return['participant'].map(get_stai_long_trait_mapper())
    return xr.Dataset(df_return)

def test_maia_concat():
    ds = maia_concat(global_key, **maia_concat_params)
    print(ds.to_dataframe())
    
maia_concat_job = jobtools.Job(precomputedir, 'maia_concat', maia_concat_params, maia_concat)
jobtools.register_job(maia_concat_job)

# ODOR

def odor_rating(global_key, **p):
    
    concat = []
    for participant in p['subject_keys']:
        df_participant = pd.read_excel(data_path / 'raw_data' / participant / 'questionnaires' / f'cotations_odeurs_relatives_{participant}.xlsx')
        concat.append(df_participant)
    df_return = pd.concat(concat)
    
    keep_cols = ['participant', 'odeur_label', 'odeur_name', 'appréciation_absolue_normalisée',
                 'appréciation_relative_normalisée','intensité_émotionnelle_relative_normalisée',
                 'familiarité_relative_normalisée','intensité_relative_normalisée','evocation_relative_normalisée']
    
    rename_cols = p['rename_cols']
    
    neg_odors = ['DiA','HeA','FM']
    pos_odors = [odor for odor in df_return['odeur_label'].unique() if not odor in neg_odors]
    supposed_hedo = ['O+' if odor in pos_odors else 'O-' for odor in df_return['odeur_label']]
    df_return = df_return.loc[:,keep_cols].rename(columns = rename_cols)
    df_return.insert(2, 'Supposed Hedonicity', supposed_hedo)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_odor_rating():
    ds = odor_rating(global_key, **odor_rating_params)
    print(ds.to_dataframe())
    
odor_rating_job = jobtools.Job(precomputedir, 'odor_rating', odor_rating_params, odor_rating)
jobtools.register_job(odor_rating_job)


# EDA

def eda_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = eda_job.get(run_key).to_dataframe()
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_eda_concat():
    ds = eda_concat(global_key, **eda_concat_params)
    print(ds.to_dataframe())
    
eda_concat_job = jobtools.Job(precomputedir, 'eda_concat', eda_concat_params, eda_concat)
jobtools.register_job(eda_concat_job)

# HRV

def hrv_concat(global_key, **p):
    
    run_keys = [key for key in p['run_keys'] if not key in p['exception_keys']]
    
    concat = []
    
    for run_key in run_keys:
        participant, session = run_key.split('_')
        
        odor = get_odor_from_session(run_key)
        
        ecg_peaks = ecg_peak_job.get(run_key).to_dataframe()

        ts = timestamps_job.get(run_key).to_dataframe().set_index(['bloc','trial'])

        concat_blocs = []
        for bloc in blocs:
            for trial in ts.loc[bloc,:].index:
                t_start = ts.loc[(bloc, trial), 'timestamp']
                t_stop = t_start + ts.loc[(bloc, trial), 'duration']
                mask = (ecg_peaks['peak_time'] > t_start) & (ecg_peaks['peak_time'] < t_stop)
                ecg_peaks_mask = ecg_peaks[mask]
                metrics = physio.compute_ecg_metrics(ecg_peaks_mask)
                df = metrics.to_frame().T
                df.insert(0 , 'trial', trial)
                df.insert(0 , 'bloc', bloc)
                df.insert(0 , 'odor', odor)
                df.insert(0 , 'session', session)
                df.insert(0 , 'participant', participant)
                concat_blocs.append(df)
        df_run_key = pd.concat(concat_blocs)
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))
        
def test_hrv_concat():
    ds = hrv_concat(global_key, **hrv_concat_params)
    print(ds.to_dataframe())
    
hrv_concat_job = jobtools.Job(precomputedir, 'hrv_concat', hrv_concat_params, hrv_concat)
jobtools.register_job(hrv_concat_job)
    
    
    
# RSA 

def rsa_concat(global_key, **p):   
    run_keys = [key for key in p['run_keys'] if not key in p['exception_keys']]
    
    concat = []
    for run_key in p['run_keys']:
        df_run_key = rsa_features_job.get(run_key).to_dataframe()
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))    
        
def test_rsa_concat():
    ds = rsa_concat(global_key, **rsa_concat_params)
    print(ds.to_dataframe())
    
rsa_concat_job = jobtools.Job(precomputedir, 'rsa_concat', rsa_concat_params, rsa_concat)
jobtools.register_job(rsa_concat_job)

# BANDPOWER
def bandpower_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = bandpower_job.get(run_key).to_dataframe()
        df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        artifacts = count_artifact_job.get(run_key).to_dataframe()
        df_run_key = map_artifacted_trials(df = df_run_key, artifacts= artifacts)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_bandpower_concat():
    ds = bandpower_concat(global_key, **eda_concat_params)
    print(ds.to_dataframe())
    
bandpower_concat_job = jobtools.Job(precomputedir, 'bandpower_concat', bandpower_concat_params, bandpower_concat)
jobtools.register_job(bandpower_concat_job)

# COHERENCE AT RESP
def coherence_at_resp_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = coherence_at_resp_job.get(run_key).to_dataframe()
        df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        artifacts = count_artifact_job.get(run_key).to_dataframe()
        df_run_key = map_artifacted_trials(df = df_run_key, artifacts= artifacts)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_coherence_at_resp_concat():
    ds = coherence_at_resp_concat(global_key, **eda_concat_params)
    print(ds.to_dataframe())
    
coherence_at_resp_concat_job = jobtools.Job(precomputedir, 'coherence_at_resp_concat', coherence_at_resp_concat_params, coherence_at_resp_concat)
jobtools.register_job(coherence_at_resp_concat_job)


# HILBERT MI
def hilbert_mi_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = hilbert_resp_mi_job.get(run_key).to_dataframe()
        df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        artifacts = count_artifact_job.get(run_key).to_dataframe()
        df_run_key = map_artifacted_trials(df = df_run_key, artifacts= artifacts)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_hilbert_mi_concat_concat():
    ds = hilbert_mi_concat(global_key, **hilbert_mi_concat_params)
    print(ds.to_dataframe())
    
hilbert_mi_concat_job = jobtools.Job(precomputedir, 'hilbert_mi_concat', hilbert_mi_concat_params, hilbert_mi_concat)
jobtools.register_job(hilbert_mi_concat_job)

# POWER AT RESP
def power_at_resp_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = power_at_resp_job.get(run_key).to_dataframe()
        df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        artifacts = count_artifact_job.get(run_key).to_dataframe()
        df_run_key = map_artifacted_trials(df = df_run_key, artifacts= artifacts)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_power_at_resp_concat():
    ds = power_at_resp_concat(global_key, **power_at_resp_concat_params)
    print(ds.to_dataframe())
    
power_at_resp_concat_job = jobtools.Job(precomputedir, 'power_at_resp_concat', power_at_resp_concat_params, power_at_resp_concat)
jobtools.register_job(power_at_resp_concat_job)

# CYCLE SIGNAL MODULATION
def modulation_cycle_signal_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = modulation_cycle_signal_job.get(run_key).to_dataframe()
        df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_modulation_cycle_signal_concat():
    ds = modulation_cycle_signal_concat(global_key, **modulation_cycle_signal_concat_params)
    print(ds.to_dataframe())
    
modulation_cycle_signal_concat_job = jobtools.Job(precomputedir, 'modulation_cycle_signal_concat', modulation_cycle_signal_concat_params, modulation_cycle_signal_concat)
jobtools.register_job(modulation_cycle_signal_concat_job)


# RESP FEATURES
def resp_features_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = label_respiration_artifact_job.get(run_key).to_dataframe()
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_resp_features_concat():
    ds = resp_features_concat(global_key, **resp_features_concat_params)
    print(ds.to_dataframe())
    
resp_features_concat_job = jobtools.Job(precomputedir, 'resp_features_concat', resp_features_concat_params, resp_features_concat)
jobtools.register_job(resp_features_concat_job)


# STAI SHORT FORM
def stai_short_job_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = stai_shortform_job.get(run_key).to_dataframe()
        df_run_key['Long_Stai_State'] = get_stai_long_mapper(run_key)
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_stai_short_job_concat():
    ds = stai_short_job_concat(global_key, **stai_short_concat_params)
    print(ds.to_dataframe())
    
stai_short_concat_job = jobtools.Job(precomputedir, 'stai_short_job_concat', stai_short_concat_params, stai_short_job_concat)
jobtools.register_job(stai_short_concat_job)


def compute_and_save_all():
    jobs = ['stai_long','maia','odor_ratings','bandpower','coherence_at_resp',
            'eda','hilbert_mi','hrv','power_at_resp','stai_short',
            'resp_features','rsa','cycle_signal_modulation']
    for job in jobs:
        file = base_folder / 'Tables' / f'{job}.xlsx'
        print(job)
        
        if job == 'odor_ratings':
            odor_rating_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'maia':
            maia_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'stai_long':
            stai_long_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'bandpower':
            bandpower_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'coherence_at_resp':
            coherence_at_resp_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'eda':
            eda_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'hilbert_mi':
            hilbert_mi_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'hrv':
            hrv_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'power_at_resp':
            power_at_resp_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'stai_short':
            stai_short_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'resp_features':
            resp_features_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'rsa':
            rsa_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'cycle_signal_modulation':
            modulation_cycle_signal_concat_job.get(global_key).to_dataframe().to_excel(file)
            
    
if __name__ == '__main__':
    # test_stai_long_concat()
    # test_maia_concat()
    # test_odor_rating()
    # test_eda_concat()
    # test_hrv_concat()
    # test_rsa_concat()
    # test_bandpower_concat()
    # test_coherence_at_resp_concat()
    # test_hilbert_mi_concat_concat()
    # test_power_at_resp_concat()
    # test_modulation_cycle_signal_concat()
    # test_resp_features_concat()
    # test_stai_short_job_concat()
    
    # get_stai_long_trait_mapper()
    
    
    compute_and_save_all()
    
    
