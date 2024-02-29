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
from compute_power_at_resp import power_at_resp_job
from compute_rri import ecg_peak_job
from compute_psycho import relaxation_job, maia_job, stai_longform_job, oas_job, bmrq_job
from compute_resp_features import respiration_features_job
from compute_rsa import rsa_features_job
from compute_cycle_signal import modulation_cycle_signal_job


#### FUNCTIONS
def get_gender_mapper():
    file = base_folder / 'Raw_Data' / 'metadata.xlsx'
    metadata = pd.read_excel(file)
    mapper_gender = {row['participant'] : row['gender'] for i, row in metadata.iterrows()}
    return mapper_gender

def get_stai_long_mapper(sub_key):
    run_key = f'{sub_key}_ses02'
    df = stai_longform_job.get(run_key).to_dataframe().set_index(['participant','session'])
    return df.loc[(sub_key,'ses02'),'etat'], df.loc[(sub_key,'ses02'),'trait']
    
def get_maia_mapper():
    maia = maia_concat_job.get(global_key).to_dataframe()
    mapper_maia = {row['participant'] : row['Maia_Mean'] for i, row in maia.iterrows()} 
    return mapper_maia

def is_session_clean_of_artifact(run_key):
    sub, ses = run_key.split('_')
    encoder_artifacts = count_artifact_job.get(sub).to_dataframe()
    encoder_artifacts = encoder_artifacts.set_index(['session'])
    artifacted = encoder_artifacts.loc[ses,'remove']
    return 1 if artifacted == 0 else 0
    
def get_oas_mapper():
    oas = oas_concat_job.get(global_key).to_dataframe()
    mapper_oas = {row['participant'] : row['OAS'] for i, row in oas.iterrows()} 
    return mapper_oas

def get_bmrq_mapper():
    bmrq = bmrq_concat_job.get(global_key).to_dataframe()
    mapper_bmrq = {row['participant'] : row['BMRQ'] for i, row in bmrq.iterrows()} 
    return mapper_bmrq

#### JOBS
# Next jobs aim to concanenate outputs from pre defined jobs in order to store it in one dataframe by job for all subjects and sessions
# They loop over subjects and sessions and add some co-variable like state / trait axiety + OAS + BMRQ + gender + MAIA results

# MAIA
def maia_concat(global_key, **p):
    concat = []
    for sub_key in p['subject_keys']:
        df_sub_key = maia_job.get(sub_key).to_dataframe()
        stai_long = stai_longform_job.get(f'{sub_key}_ses01').to_dataframe().set_index(['participant','session'])
        df_sub_key['Stai_Trait'] = stai_long.loc[(sub_key, 'ses01'),'trait']
        df_sub_key['Stai_State'] = stai_long.loc[(sub_key, 'ses01'),'etat']
        concat.append(df_sub_key)
    df_return = pd.concat(concat).reset_index(drop = True)
    outcomes = list(p['maia_params']['items'].keys())
    df_return['Maia_Mean'] = df_return.loc[:,outcomes].mean(axis = 1)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    return xr.Dataset(df_return)

def test_maia_concat():
    ds = maia_concat(global_key, **maia_concat_params)
    print(ds.to_dataframe())
    
maia_concat_job = jobtools.Job(precomputedir, 'maia_concat', maia_concat_params, maia_concat)
jobtools.register_job(maia_concat_job)

# ODOR

# def odor_rating(global_key, **p):
    
#     concat = []
#     for participant in p['subject_keys']:
#         df_participant = pd.read_excel(data_path / 'raw_data' / participant / 'questionnaires' / f'cotations_odeurs_relatives_{participant}.xlsx')
#         concat.append(df_participant)
#     df_return = pd.concat(concat)
    
#     keep_cols = ['participant', 'odeur_label', 'odeur_name', 'appréciation_absolue_normalisée',
#                  'appréciation_relative_normalisée','intensité_émotionnelle_relative_normalisée',
#                  'familiarité_relative_normalisée','intensité_relative_normalisée','evocation_relative_normalisée']
    
#     rename_cols = p['rename_cols']
    
#     neg_odors = ['DiA','HeA','FM']
#     pos_odors = [odor for odor in df_return['odeur_label'].unique() if not odor in neg_odors]
#     supposed_hedo = ['O+' if odor in pos_odors else 'O-' for odor in df_return['odeur_label']]
#     df_return = df_return.loc[:,keep_cols].rename(columns = rename_cols)
#     df_return.insert(2, 'Supposed Hedonicity', supposed_hedo)
#     df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
#     df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
#     return xr.Dataset(df_return.reset_index(drop = True))

# def test_odor_rating():
#     ds = odor_rating(global_key, **odor_rating_params)
#     print(ds.to_dataframe())
    
# odor_rating_job = jobtools.Job(precomputedir, 'odor_rating', odor_rating_params, odor_rating)
# jobtools.register_job(odor_rating_job)


# EDA

def eda_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = eda_job.get(run_key).to_dataframe()
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_eda_concat():
    ds = eda_concat(global_key, **eda_concat_params)
    print(ds.to_dataframe())
    
eda_concat_job = jobtools.Job(precomputedir, 'eda_concat', eda_concat_params, eda_concat)
jobtools.register_job(eda_concat_job)

# HRV

def hrv_concat(global_key, **p):
    
    concat = []
    
    for run_key in p['run_keys']:
        participant, session = run_key.split('_')
        
        ecg_peaks = ecg_peak_job.get(run_key).to_dataframe()

        metrics = physio.compute_ecg_metrics(ecg_peaks)
        df_run_key = metrics.to_frame().T
        df_run_key.insert(0 , 'session', session)
        df_run_key.insert(0 , 'participant', participant)
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait

        concat.append(df_run_key)

    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))
        
def test_hrv_concat():
    ds = hrv_concat(global_key, **hrv_concat_params)
    print(ds.to_dataframe())
    
hrv_concat_job = jobtools.Job(precomputedir, 'hrv_concat', hrv_concat_params, hrv_concat)
jobtools.register_job(hrv_concat_job)
    
    
    
# RSA 

def rsa_concat(global_key, **p):   
    concat = []
    for run_key in p['run_keys']:
        df_run_key = rsa_features_job.get(run_key).to_dataframe()
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
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
        # mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = bandpower_job.get(run_key).to_dataframe()
        # df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        keep = is_session_clean_of_artifact(run_key)
        df_run_key['keep_session'] = keep
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_bandpower_concat():
    ds = bandpower_concat(global_key, **bandpower_concat_params)
    print(ds.to_dataframe())
    
bandpower_concat_job = jobtools.Job(precomputedir, 'bandpower_concat', bandpower_concat_params, bandpower_concat)
jobtools.register_job(bandpower_concat_job)

# COHERENCE AT RESP
def coherence_at_resp_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        # mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = coherence_at_resp_job.get(run_key).to_dataframe()
        # df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        keep = is_session_clean_of_artifact(run_key)
        df_run_key['keep_session'] = keep
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_coherence_at_resp_concat():
    ds = coherence_at_resp_concat(global_key, **coherence_at_resp_concat_params)
    print(ds.to_dataframe())
    
coherence_at_resp_concat_job = jobtools.Job(precomputedir, 'coherence_at_resp_concat', coherence_at_resp_concat_params, coherence_at_resp_concat)
jobtools.register_job(coherence_at_resp_concat_job)


# POWER AT RESP
def power_at_resp_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        # mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = power_at_resp_job.get(run_key).to_dataframe()
        # df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        keep = is_session_clean_of_artifact(run_key)
        df_run_key['keep_session'] = keep
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_power_at_resp_concat():
    ds = power_at_resp_concat(global_key, **power_at_resp_concat_params)
    print(ds.to_dataframe())
    
power_at_resp_concat_job = jobtools.Job(precomputedir, 'power_at_resp_concat', power_at_resp_concat_params, power_at_resp_concat)
jobtools.register_job(power_at_resp_concat_job)


# RESP FEATURES
def resp_features_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = respiration_features_job.get(run_key).to_dataframe()
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_resp_features_concat():
    ds = resp_features_concat(global_key, **resp_features_concat_params)
    print(ds.to_dataframe())
    
resp_features_concat_job = jobtools.Job(precomputedir, 'resp_features_concat', resp_features_concat_params, resp_features_concat)
jobtools.register_job(resp_features_concat_job)


# RELAXATION
def relaxation_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = relaxation_job.get(run_key).to_dataframe()
        state, trait = get_stai_long_mapper(run_key)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_relaxation_concat():
    ds = relaxation_concat(global_key, **relaxation_concat_params)
    print(ds.to_dataframe())

relaxation_concat_job = jobtools.Job(precomputedir, 'relaxation_concat', relaxation_concat_params, relaxation_concat)
jobtools.register_job(relaxation_concat_job)

# CYCLE SIGNAL MODULATION
def modulation_cycle_signal_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        # mapper_keep_chans = get_bad_chan_mapper(run_key)
        df_run_key = modulation_cycle_signal_job.get(run_key).to_dataframe()
        # df_run_key['keep_chan'] = df_run_key['chan'].map(mapper_keep_chans)
        sub, ses = run_key.split('_')
        state, trait = get_stai_long_mapper(sub)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        keep = is_session_clean_of_artifact(run_key)
        df_run_key['keep_session'] = keep
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    df_return['OAS'] = df_return['participant'].map(get_oas_mapper())
    df_return['BMRQ'] = df_return['participant'].map(get_bmrq_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_modulation_cycle_signal_concat():
    ds = modulation_cycle_signal_concat(global_key, **modulation_cycle_signal_concat_params)
    print(ds.to_dataframe())
    
modulation_cycle_signal_concat_job = jobtools.Job(precomputedir, 'modulation_cycle_signal_concat', modulation_cycle_signal_concat_params, modulation_cycle_signal_concat)
jobtools.register_job(modulation_cycle_signal_concat_job)

# OAS
def oas_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = oas_job.get(run_key).to_dataframe()
        state, trait = get_stai_long_mapper(run_key)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_oas_concat():
    ds = oas_concat(global_key, **oas_concat_params)
    print(ds.to_dataframe())

oas_concat_job = jobtools.Job(precomputedir, 'oas_concat', oas_concat_params, oas_concat)
jobtools.register_job(oas_concat_job)


# BMRQ
def bmrq_concat(global_key, **p):
    concat = []
    for run_key in p['run_keys']:
        df_run_key = bmrq_job.get(run_key).to_dataframe()
        state, trait = get_stai_long_mapper(run_key)
        df_run_key['stai_state'] = state
        df_run_key['stai_trait'] = trait
        concat.append(df_run_key)
    df_return = pd.concat(concat)
    df_return['Gender'] = df_return['participant'].map(get_gender_mapper())
    df_return['Maia_Mean'] = df_return['participant'].map(get_maia_mapper())
    return xr.Dataset(df_return.reset_index(drop = True))

def test_bmrq_concat():
    ds = bmrq_concat(global_key, **bmrq_concat_params)
    print(ds.to_dataframe())

bmrq_concat_job = jobtools.Job(precomputedir, 'bmrq_concat', bmrq_concat_params, bmrq_concat)
jobtools.register_job(bmrq_concat_job)




def compute_and_save_all():
    jobs = ['maia','bandpower','coherence_at_resp',
            'eda','hrv','power_at_resp','relaxation',
            'resp_features','rsa','cycle_signal_modulation','oas','bmrq']

    for job in jobs:
        file = base_folder / 'Tables' / f'{job}.xlsx'
        print(job)
        
        if job == 'maia':
            maia_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'bandpower':
            bandpower_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'coherence_at_resp':
            coherence_at_resp_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'eda':
            eda_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'hrv':
            hrv_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'power_at_resp':
            power_at_resp_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'relaxation':
            relaxation_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'resp_features':
            resp_features_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'rsa':
            rsa_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'cycle_signal_modulation':
            modulation_cycle_signal_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'oas':
            oas_concat_job.get(global_key).to_dataframe().to_excel(file)
        elif job == 'bmrq':
            bmrq_concat_job.get(global_key).to_dataframe().to_excel(file)
    
if __name__ == '__main__':

    # print(is_session_clean_of_artifact('P02_baseline'))
    # test_maia_concat()
    # test_eda_concat()
    # test_hrv_concat()
    test_rsa_concat()
    # test_bandpower_concat()
    # test_coherence_at_resp_concat()
    # test_power_at_resp_concat()
    # test_resp_features_concat()
    # test_relaxation_concat()
    # test_modulation_cycle_signal_concat()
    # test_oas_concat()
    # test_bmrq_concat()
    # compute_and_save_all()
    
    
