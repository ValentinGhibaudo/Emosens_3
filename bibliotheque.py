import xarray as xr
import pandas as pd
import numpy as np

from configuration import base_folder, data_path
from params import *

def init_nan_da(coords, name = None):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.full(shape, np.nan)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da

def get_pos(eeg_chans=eeg_chans):
    
    import mne
    ch_types = ['eeg'] * len(eeg_chans)
    pos = mne.create_info(eeg_chans, ch_types=ch_types, sfreq=srate)
    pos.set_montage('standard_1020')
        
    return pos


def get_metadata():
    return pd.read_excel('/crnldata/cmo/multisite/DATA_MANIP/EEG_Lyon_VJ/Data/raw_data/metadata.xlsx', index_col = 0)


def get_anxiety_state_from_session(participant, session):
    path_stai = f'/crnldata/cmo/multisite/DATA_MANIP/EEG_Lyon_VJ/Data/raw_data/{participant}/questionnaires/stai_long_form_{session}_{participant}.xlsx'
    raw_stai = pd.read_excel(path_stai)
    list_scores = list(raw_stai['score'].values)
    list_corrections = list(raw_stai['correction'].values)
    score_corrected = []
    for score, correction in zip(list_scores, list_corrections):
        if correction == '-':
            score_c = 5-score
        else : 
            score_c = score
            
        score_corrected.append(score_c)
    etat = np.sum(score_corrected[0:20])
    trait = np.sum(score_corrected[20:None])
    return etat

def preproc_bio(sig, sig_type, srate, bio_filters):
    import ghibtools as gh
    low = bio_filters[sig_type]['low']
    high = bio_filters[sig_type]['high']
    ftype = bio_filters[sig_type]['ftype']
    order = bio_filters[sig_type]['order']
    return gh.iirfilt(sig=sig, srate=srate, lowcut=low, highcut=high, ftype=ftype, order=order)

def mne_to_xarray(raw):
    import ghibtools as gh
    data = raw.get_data()
    srate = raw.info['sfreq']
    da = xr.DataArray(data=data, dims = ['chan','time'], coords = {'chan':raw.info['ch_names'], 'time':gh.time_vector(data[0,:], srate)}, attrs={'srate':srate})
    return da

def get_triggs(raw, blocs, code_trigg):
    raw_triggs = pd.DataFrame(raw.annotations)
    rows = []
    for type_stim in raw_triggs['description'].unique():
        for bloc in blocs:
            if type_stim in code_trigg[bloc]:
                onsets = raw_triggs[raw_triggs['description'] == type_stim]['onset']
                for i in range(onsets.size):
                    onset = onsets.reset_index().loc[i,'onset']
                    trial = i+1
                    if '1' in type_stim:
                        timing = 'start'
                    if '2' in type_stim:
                        timing = 'stop'
                    row = [bloc, trial, timing, onset]
                    rows.append(row)
    df_triggs = pd.DataFrame(rows, columns = ['bloc','trial','timing','timestamp']).set_index(['bloc','trial','timing'])
    return df_triggs

def get_odor_from_session(run_key):
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    file = data_path / 'raw_data' / 'metadata.xlsx'
    df = pd.read_excel(file, index_col = 0)
    return df.loc[participant,session]

def processing_raw_maia(participant):
    path_maia = f'/crnldata/cmo/multisite/DATA_MANIP/EEG_Lyon_VJ/Data/raw_data/{participant}/questionnaires/maia_{participant}.xlsx'
    raw_maia = pd.read_excel(path_maia)

    labels = ['participant','noticing','not_distracting','not_worrying','attention_regulation','emotional_awareness','self_regulation','body_listening','trusting','awareness_of_body_sensations','emotional_reaction','capicity_regulation_attention','awareness_of_mind_body','trusting_body_sensations','global_mean']
    sujet = raw_maia['participant'][0]
    idx_labels = [(0,4),(4,7),(7,10),(10,17),(17,22),(22,26),(26,29),(29,None),(0,4),(4,10),(10,17),(17,22),(22,29),(29,None),(None,None)]

    dict_means = {}
    for label, idxs in zip(labels, idx_labels):
        if label == 'participant':
            dict_means[label] = sujet
        else:
            dict_means[label] = np.mean(raw_maia['score'][idxs[0]:idxs[1]])

    return pd.DataFrame.from_dict(dict_means, orient = 'index').T.set_index('participant').astype(float).reset_index()

def processing_stai_longform(participant, session):
    path_stai = f'/crnldata/cmo/multisite/DATA_MANIP/EEG_Lyon_VJ/Data/raw_data/{participant}/questionnaires/stai_long_form_{session}_{participant}.xlsx'
    raw_stai = pd.read_excel(path_stai)
    sujet = raw_stai['participant'][0]
    list_scores = list(raw_stai['score'].values)
    list_corrections = list(raw_stai['correction'].values)
    score_corrected = []
    for score, correction in zip(list_scores, list_corrections):
        if correction == '-':
            score_c = 5-score
        else : 
            score_c = score
            
        score_corrected.append(score_c)
    etat = np.sum(score_corrected[0:20])
    trait = np.sum(score_corrected[20:None])


    mean_etat = 35.4
    mean_trait = 24.8
    std_etat = 10.5
    std_trait = 9.2

    if etat > (mean_etat+1.96*std_etat):
        interpretation_etat = 'Etat anxieux'
    elif etat < (mean_etat-1.96*std_etat):
        interpretation_etat = 'Etat moins que anxieux'
    else :
        interpretation_etat = 'Etat dans les normes'
        
    if trait > (mean_trait+1.96*std_trait): 
        interpretation_trait = 'Trait anxieux'
    elif trait < (mean_trait-1.96*std_trait):
        interpretation_trait = 'Trait moins que anxieux'
    else : 
        interpretation_trait = 'Trait dans les normes'
    
    dict_results = {'participant':sujet, 'etat': etat, 'trait':trait, 'interpretation_etat':interpretation_etat, 'interpretation_trait':interpretation_trait}
    return pd.DataFrame.from_dict(dict_results, orient = 'index').T

        
def processing_short_stai(participant, session):  
    import glob  
    odeur = get_odor_from_session(participant, session)
    logtrigg = glob.glob(f'/crnldata/cmo/multisite/DATA_MANIP/EEG_Lyon_VJ/Data/raw_data/{participant}/questionnaires/sub{participants_label[participant]}_{session}_LogTrigger*')
    random_block = glob.glob(f'/crnldata/cmo/multisite/DATA_MANIP/EEG_Lyon_VJ/Data/raw_data/{participant}/signaux/sub{participants_label[participant]}_{session}_bloc_random_order_*')
    
    logtrigg = pd.read_fwf(logtrigg[0], colspecs = [(0,1000000)]).values
    random_block = pd.read_fwf(random_block[0], colspecs = [(0,1000000)]).values
    
    logtrigg = list(logtrigg.reshape(logtrigg.shape[0],))
    bloc_order = random_block.reshape(random_block.shape[0],)[0].rsplit(sep = " ")
    
    bloc_order.insert(0, 'entrainement')
    bloc_order.insert(0, 'free')
    bloc_order.insert(len(bloc_order), 'free')
    
    bloc_types = list(set(bloc_order))
    bloc_pos = {}
    for bloc in bloc_types:
        bloc_pos[bloc] = np.where(np.array(bloc_order) == bloc)[0]
        
    bloc_nums = bloc_order.copy()
    for bloc in bloc_order:
        for i, idx in enumerate(list(bloc_pos[bloc])):
            bloc_nums[idx] = f'{bloc}{i+1}'
            
    items = [
        'calme',
        'crispé',
        'ému',
        'décontracté',
        'satisfait',
        'inquiet',
        'attention',
        'relaxé'
    ]

    df = pd.DataFrame(columns = items)
    
    for item in items:
        value_item = []
        for line in logtrigg:
            if item in line:
                value = int(line[len(line) - 2 :])
                value_item.append(value)
        for i,value in enumerate(value_item):
            df.loc[i+1, item] = value
    
    df = df.astype(int)
    df.index = bloc_nums
    
    etats = []
    for index in bloc_nums:
        etat = 100 - df.loc[index,'calme'] + df.loc[index,'crispé'] + df.loc[index,'ému'] + 100 - df.loc[index,'décontracté'] + 100 - df.loc[index,'satisfait'] + df.loc[index,'inquiet']
        etats.append(etat)
        
    df = df.reset_index().rename(columns = {"index":"trial"})
    
    
    df.insert(df.shape[1], 'état', etats)
    df.insert(0, 'bloc', bloc_order)
    df.insert(0, 'odeur', odeur)
    df.insert(0, 'session', session)
    df.insert(0, 'participant', participant)
    
    return df[df['bloc'] != 'entrainement']


def get_raw_mne(run_key, participants_label, preload=False):
    import mne
    participant, session = run_key.split('_')
    file = data_path / f'{participant}' / 'signaux' / f'sub{participants_label[participant]}_{session}.vhdr'
    raw = mne.io.read_raw_brainvision(file, preload = preload, verbose = 'CRITICAL')
    return raw

