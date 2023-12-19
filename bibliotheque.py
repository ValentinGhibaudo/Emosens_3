# TOOLS FOR VARIOUS UTILITIES

import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
from configuration import base_folder, data_path
from params import *

def keep_clean(df_raw, metrics_to_clean, fill_method = 'ffill'):
    """
    Cleaning of dataframes with a keep session columns that indicates to keep or remove the session

    ----------
    Parameters
    ----------
    - df_raw : pd.DataFrame
        pd.DataFrame with a keep_session column that indicates to keep remove or not the session (the row)
    - metrics_to_clean : list
        List of name of metrics that have to undergo the the cleaning
    - fill_method : str
        pandas fill method to replace the Nan

    -------
    Returns
    -------
    - df_clean : pd.DataFrame
        Cleaned dataframe
    """
    df_raw_nan = df_raw.reset_index()
    # mask_clean = (df_raw_nan['keep_chan'] == 1) & (df_raw_nan['keep_trial'] == 1)
    mask_clean = (df_raw_nan['keep_session'] == 1) 
    mask_bad = ~mask_clean
    df_raw_nan.loc[mask_bad,metrics_to_clean] = np.nan
    df_clean = df_raw_nan.fillna(method = fill_method)
    return df_clean

def mad(data, axis=0):
    """
    Compute median absolute deviation along defined axis

    ----------
    Parameters
    ----------
    - data : np.array
    - axis : int

    -------
    Returns
    -------
    - float or np.array
    """
    return np.median(np.abs(data - np.median(data, axis = axis)), axis = axis) * 1.4826

def complex_mw(time, n_cycles , freq, a = 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ----------
    Parameters
    ----------
    time : np.array
        Time vector of the wavelet
    n_cycles : int
        Number of cycles in the wavelet
    freq : float 
        frequency of the wavelet
    a : float
        amplitude of the wavelet
    m : float
        center of the time window
    -------
    Returns
    -------
    - np.array
        Complex morlet wavelet
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def define_morlet_family(freqs, cycles , srate, return_time = False):
    """
    Create a complex morlet wavelet family
    
    ----------
    Parameters
    ----------
    freqs : np.array
        Frequency vector 
    cycles : int
        Number of cycles in the wavelets
    srate : float 
        Frequency of the wavelet
    return_time : bool
        Return time if True added to morlet family
    -------
    Returns
    -------
    - 2D np.array
        Complex morlet wavelet family with 0-axis = frequency and 1-axis = time
    """
    tmw = np.arange(-10,10,1/srate)
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        n_cycles = cycles[i]
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles, freq = fi)
        
    if return_time:
        return tmw, mw_family
    else:
        return mw_family

def df_baseline(df, indexes, metrics, mode = 'ratio'):
    """
    Normalize dataframe data according to baseline
    
    ----------
    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame tidy containg odor, music, and baseline data
    indexes : list
        Column names to be set as indexes
    metrics : list 
        Column names whose data will undergo normalization
    mode : str
        Could be 'ratio' or 'substract' to normalize by division of subtraction 
    -------
    Returns
    -------
    - pd.DataFrame
    """
    odor = df[df['session'] == 'odor'].set_index(indexes)
    music = df[df['session'] == 'music'].set_index(indexes)
    baseline = df[df['session'] == 'baseline'].set_index(indexes)
    
    if mode == 'ratio':
        data_odor = odor[metrics].values / baseline[metrics].values
        data_music = music[metrics].values / baseline[metrics].values
    elif mode == 'substract':
        data_odor = odor[metrics].values - baseline[metrics].values
        data_music = music[metrics].values - baseline[metrics].values
    
    df_odor = pd.DataFrame(data = data_odor, columns = metrics, index = odor.index)
    df_music = pd.DataFrame(data = data_music, columns = metrics, index = music.index)
    
    return pd.concat([df_odor, df_music]).reset_index()

def init_nan_da(coords, name = None):
    """
    Initialize an DataArray (xarray) filled with Nan
    
    ----------
    Parameters
    ----------
    coords : dict
        Dictionnary whose keys will be dimension names of the DataArray and values will be the coordinates of the corresponding dims 
    name : str
        Name of the DataArray

    -------
    Returns
    -------
    - xr.DataArray
    """
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
    """
    Get MNE position info object according to 10_20 standard
    
    ----------
    Parameters
    ----------
    eeg_chans : list
        Channel names list

    -------
    Returns
    -------
    - pos : mne.info
    """   
    import mne
    ch_types = ['eeg'] * len(eeg_chans)
    pos = mne.create_info(eeg_chans, ch_types=ch_types, sfreq=srate)
    pos.set_montage('standard_1020')
        
    return pos


def get_metadata():
    """
    Load metadata
    
    ----------
    Parameters
    ----------

    -------
    Returns
    -------
    - pd.DataFrame
    """   
    return pd.read_excel(base_folder / 'Data' / 'order_stims.xlsx', index_col = 0)


def get_anxiety_state_from_session(participant, session):
    """
    Load STAI state
    
    ----------
    Parameters
    ----------
    - participant : str
    - session : str
    -------
    Returns
    -------
    - pd.DataFrame
    """  
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
    """
    Preproc bio signals
    
    ----------
    Parameters
    ----------
    - sig : np.array
    - sig_type : str
        'ECG' or 'RespiNasale' or 'RespiVentrale' or 'GSR'
    - srate : float
        Sampling rate
    - bio_filters : dict
        Dictionnary of filter params according to type of bio signal

    -------
    Returns
    -------
    - np.array : bio signal filtered
    """  
    import ghibtools as gh
    low = bio_filters[sig_type]['low']
    high = bio_filters[sig_type]['high']
    ftype = bio_filters[sig_type]['ftype']
    order = bio_filters[sig_type]['order']
    return gh.iirfilt(sig=sig, srate=srate, lowcut=low, highcut=high, ftype=ftype, order=order)

def mne_to_xarray(raw):
    """
    Convert raw mne object to xarray
    
    ----------
    Parameters
    ----------
    - raw : mne.raw object
        
    -------
    Returns
    -------
    - xr.DataArray 
    """  
    import ghibtools as gh
    data = raw.get_data()
    srate = raw.info['sfreq']
    da = xr.DataArray(data=data, dims = ['chan','time'], coords = {'chan':raw.info['ch_names'], 'time':gh.time_vector(data[0,:], srate)}, attrs={'srate':srate})
    return da

def get_triggs(raw, blocs, code_trigg):
    """
    Load timestamps of triggers from mne.raw object
    
    ----------
    Parameters
    ----------
    - raw : mne.raw object
    - blocs : list
        List of str of blocks
    - code_trigg : dict
        Dictionnary keys = blocs and values = code trigg
    -------
    Returns
    -------
    - pd.DataFrame
    """  
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
    """
    Get odor name from (sub_ses)
    
    ----------
    Parameters
    ----------
    - run_key : str
        sub_ses

    -------
    Returns
    -------
    - str
    """  
    participant, session = run_key.split('_')[0], run_key.split('_')[1]
    file = data_path / 'raw_data' / 'metadata.xlsx'
    df = pd.read_excel(file, index_col = 0)
    return df.loc[participant,session]

def processing_raw_maia(participant):
    """
    Process MAIA questionnaire
    
    ----------
    Parameters
    ----------
    - participant : str

    -------
    Returns
    -------
    - pd.DataFrame
    """  
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
    """
    Process STAI Longform
    
    ----------
    Parameters
    ----------
    - participant : str
    - session : str

    -------
    Returns
    -------
    - pd.DataFrame
    """  
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
    """
    Process STAI Shortform
    
    ----------
    Parameters
    ----------
    - participant : str
    - session : str

    -------
    Returns
    -------
    - pd.DataFrame
    """  
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
    """
    Load Brainvision into mne.raw object
    
    ----------
    Parameters
    ----------
    - run_key : str
    - participants_label : dict
        Dictionnary of conversion participant label to participant num
    - preload : bool
        Lazy load or not

    -------
    Returns
    -------
    - mne.raw object
    """  
    import mne
    participant, session = run_key.split('_')
    file = data_path / f'{participant}' / 'signaux' / f'sub{participants_label[participant]}_{session}.vhdr'
    raw = mne.io.read_raw_brainvision(file, preload = preload, verbose = 'CRITICAL')
    return raw

def permutation_test_homemade(x,y, design = 'within', n_resamples=999, diff = 'mean'):
    """
    Permutation test
    
    ----------
    Parameters
    ----------
    - x : np.array
    - y : np.array
    - design : str
        'within' or 'between' according to the design
    - n_resamples : int
        Number of iterations to create null distribution
    - diff : str
        'mean' or 'median' to compute mean difference of median difference between the two groups

    -------
    Returns
    -------
    - pvalue : float
    """  
    def statistic(x, y):
        if diff == 'mean':
            return np.mean(x) - np.mean(y)
        elif diff == 'median':
            return np.median(x) - np.median(y)
        
    if design == 'within':
        permutation_type = 'samples'
    elif design == 'between':
        permutation_type = 'independent'
    res = stats.permutation_test(data=[x,y], statistic=statistic, permutation_type=permutation_type, n_resamples=n_resamples, batch=None, alternative='two-sided', axis=0, random_state=None)
    return res.pvalue

def get_pval(df, predictor, outcome, subject=None, design='within', verbose = False):
    import ghibtools as gh
    parametricity = gh.parametric(df, predictor, outcome, subject)
    tests = gh.guidelines(df, predictor, design, parametricity)
    pre_test = tests['pre']
    post_test = tests['post']
    if verbose:
        print(f'Pre : {pre_test}')
        print(f'Post : {post_test}')
    results = gh.pg_compute_pre(df, predictor, outcome, pre_test, subject)
    return results['p']

def get_df_mask_chan_signif(df, chans, predictor, outcome, subject, design = 'within', multicomp_method = 'bonf', stats_type = 'permutations', diff = 'mean'):

    def statistic(x, y):
        if diff == 'mean':
            return np.mean(x) - np.mean(y)
        elif diff == 'median':
            return np.median(x) - np.median(y)
    
    import pingouin as pg
    rows = []
    for chan in chans:
        if stats_type == 'classic':
            p = get_pval(df = df[df['chan'] == chan], predictor = 'session', outcome = outcome, subject = subject,verbose = False, design= design)
        elif stats_type == 'permutations':
            levels = df[predictor].unique()
            x_df = df[(df[predictor] == levels[0]) & (df['chan'] == chan)]
            y_df = df[(df[predictor] == levels[1]) & (df['chan'] == chan)]
            x = x_df[outcome].values
            y = y_df[outcome].values
            res = stats.permutation_test(data=[x,y], statistic=statistic, permutation_type='samples' if design == 'within' else 'independent', n_resamples=1000)
            p = res.pvalue 
        signif = True if p > 0.05 else False
        rows.append([chan, p, signif])
    chan_signif = pd.DataFrame(rows, columns = ['chan','p','mask'])
    mask_corr, p_corr = pg.multicomp(chan_signif['p'], method = multicomp_method)
    chan_signif['p_corr'] = p_corr
    chan_signif['mask_corr'] = mask_corr
    chan_signif = chan_signif.set_index('chan').reindex(eeg_chans)
    return chan_signif

def cluster_stats(x1,x2, chans, verbose = False):
    """
    Cluster based statistics on repeated measures
    
    ----------
    Parameters
    ----------
    - x1 : np.array
        Size = len(chans)
    - x2 : np.array
        Size = len(chans)
    - chans : list
        List of channel names in the same order than the corresponding value in x1/x2
    - verbose : bool
        Verbosity of the process

    -------
    Returns
    -------
    - np.array of bool
    """  
    import mne
    
    X = x2 - x1
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, out_type = 'indices', verbose = False)
    
    if verbose:
        print(clusters, cluster_p_values)

    mask_signif_chans = np.full(len(chans), False)
    mask_non_signif_chans = np.full(len(chans), False)

    chan_inds_signif = []
    chan_inds_non_signif = []
    for cluster, p  in zip(clusters, cluster_p_values):
        if p < 0.05:
            chan_inds_signif.extend(list(cluster[0]))
        else:
            chan_inds_non_signif.extend(list(cluster[0]))
    mask_signif_chans[chan_inds_signif] = True
    mask_non_signif_chans[chan_inds_non_signif] = True
    return mask_signif_chans
    # return  mask_signif_chans, mask_non_signif_chans
