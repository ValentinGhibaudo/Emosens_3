# RUN KEYS

from configuration import data_path

# subject_keys = ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P10',
#                 'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',
#                 'P21','P23','P24','P25','P26','P27','P28','P29','P30','P31']

subject_keys = ['P02']

session_keys = ['baseline','music','odor']

run_keys = [f'{sub_key}_{ses_key}' for sub_key in subject_keys for ses_key in session_keys]


# USEFUL LISTS & DICTS
eeg_chans = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
 'CP5','CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6','CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']
bio_chans = ['ECG','RespiNasale','RespiVentrale','GSR']
all_chans = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
 'CP5','CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6','CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2','ECG','RespiNasale','RespiVentrale','GSR','FCI']

participants_label = {'P01':'CM01','P02':'FB02','P03':'CA03','P04':'BC04','P05':'ML05',
                      'P06':'PE06','P07':'MA07','P08':'OK08','P09':'SE09','P10':'WD10',
                      'P11':'GC11','P12':'FM12','P13':'RA13','P14':'DI14','P15':'RS15',
                      'P16':'KC16','P17':'LC17','P18':'GM18','P19':'SA19','P20':'DM20',
                      'P21':'FF21','P22':'NR22','P23':'MT23','P24':'GC24','P25':'CF25',
                      'P26':'CM26','P27':'RS27','P28':'AY28','P29':'WD29','P30':'AL30',
                      'P31':'VT31'}


session_duration = 600.


#### PROCESSING PARAMS
srate = 1000

fbands = {'delta':[1,4],'theta':[4,8], 'alpha':[8,12], 'beta':[12,30], 'low_gamma':[30,45], 'high_gamma':[55,100], '100_200':[100,200], '200_300':[200,300], '300_400':[300,400], '400_500':[400,499]}

ecg_inversion = {'P01':-1,
'P02':-1,
'P03':-1,
'P04':-1,
'P05':-1, 
'P06':-1,
'P07':-1,
'P08':-1,
'P09':-1,
'P10':-1,
'P11':-1,
'P12':-1,
'P13':-1,
'P14':-1,
'P15':-1,
'P16':-1,
'P17':-1,
'P18':-1,
'P19':-1,
'P20':-1, 
'P21':-1,
'P22':-1,
'P23':-1,
'P24':-1,
'P25':-1,
'P26':-1,
'P27':-1, 
'P28':-1,
'P29':-1,
'P30':-1,
'P31':-1,
'P32':-1         
}



## ICA : 
n_components_decomposition = 10

# components exclusion
ica_excluded_component = {'P01':{'baseline':[0],'music':[0],'odor':[0]}, # 
'P02':{'baseline':[0,1],'music':[0,3],'odor':[0,2]}, # OK
'P03':{'baseline':[0,2],'music':[0,4],'odor':[1,3]}, #
'P04':{'baseline':[0,1],'music':[0,1],'odor':[0,2]}, #
'P05':{'baseline':[0,1],'music':[0,3],'odor':[0,1]}, #
'P06':{'baseline':[2,4],'music':[0,2],'odor':[0,1]}, #
'P07':{'baseline':[0,2],'music':[5,7],'odor':[2,4]}, #
'P08':{'baseline':[0,1],'music':[0,1],'odor':[0,1]}, #
'P09':{'baseline':[0,3],'music':[0],'odor':[2,3]}, #
'P10':{'baseline':[2,3],'music':[0,2],'odor':[0,1,2]}, #
'P11':{'baseline':[0,1],'music':[0,1],'odor':[3]}, #
'P12':{'baseline':[0,2],'music':[0,2],'odor':[4,6]}, # 
'P13':{'baseline':[0,9],'music':[0],'odor':[3]}, # 
'P14':{'baseline':[0,2],'music':[0,1,2,3],'odor':[0,1]}, #
'P15':{'baseline':[0,1],'music':[0,1],'odor':[1,2]}, # 
'P16':{'baseline':[0],'music':[0],'odor':[0,3]}, #
'P17':{'baseline':[3,6],'music':[0,4,6],'odor':[0,1,5]}, #
'P18':{'baseline':[0,2],'music':[0,2],'odor':[0,2]}, #
'P19':{'baseline':[0,1],'music':[4,6],'odor':[0,1]}, # 
'P20':{'baseline':[0,2],'music':[0,2],'odor':[0,2]}, #
'P21':{'baseline':[0,2],'music':[0,2],'odor':[0,2]}, # 
'P22':{'baseline':[0],'music':[0],'odor':[0]}, # not recorded
'P23':{'baseline':[0],'music':[0],'odor':[0,5]}, # 
'P24':{'baseline':[1,7],'music':[2,7],'odor':[0,8]}, # 
'P25':{'baseline':[0,2],'music':[0,2],'odor':[0,3]}, # 
'P26':{'baseline':[0,1],'music':[0,1],'odor':[1,2,4]}, #
'P27':{'baseline':[0,1],'music':[0,2],'odor':[0,1]}, #
'P28':{'baseline':[0,2],'music':[0,2],'odor':[0,2]}, #
'P29':{'baseline':[3],'music':[1],'odor':[0]}, # 
'P30':{'baseline':[0],'music':[1],'odor':[1,4]}, #
'P31':{'baseline':[1,2],'music':[0],'odor':[0]}
}

bio_filters = {
    'ECG':{'low':5,'high':45, 'ftype':'bessel', 'order':5} , 
    'RespiNasale':{'low':0.02, 'high':1.5, 'ftype':'butter', 'order':5} , 
    'RespiVentrale':{'low':0.02, 'high':1.5, 'ftype':'butter', 'order':5} , 
    'GSR':{'low':None, 'high':3, 'ftype':'butter', 'order':5}
}


## ANALYSES

## job params

import numpy as np

convert_vhdr_params = {
    'participants_label': participants_label,  
}

random_state = 27
notch_freqs = np.arange(50, 500, 50).tolist()
n_components_decomposition = 10

ica_figure_params = {
     'random_state' : random_state,
     'notch_freqs' : notch_freqs,
     'n_components_decomposition' : n_components_decomposition,
}

preproc_params = {
    'participants_label': participants_label,
    'notch_freqs' : notch_freqs,
    'save_ica_fig': True,
    'random_state' : random_state,
    'ica_excluded_component': ica_excluded_component,
    'eeg_chans': eeg_chans ,
    'n_components_decomposition' :n_components_decomposition ,
    'session_duration':session_duration
}

respiration_features_params = {
    'inspiration_sign' : '+',
    'low_pass_freq': 15.,
    'filter_order' : 5,
    'smooth_sigma_ms': 100.0,
    
}


ecg_params = {
    'ecg_inversion' : ecg_inversion,
    'low': 5.,
    'high':45.,
    'ftype':'bessel',
    'order':5,
    'threshold' : 7,
    'exclude_sweep_ms': 4.0,
    'min_interval_ms': 400.
}

rri_signal_params = {
    'max_interval_ms' : 2000.,
    'min_interval_ms': 400.,
    'interpolation_kind': 'cubic',
}



psd_params = {
    'lowest_freq':0.05,
}

bandpower_params = {
    'fbands':fbands,
}

hilbert_params = {
    'fbands':fbands,
}

power_at_resp_params = {
    'resp_chan':'RespiNasale',
    'lowest_freq_psd_resp':0.05,
}

coherence_params = {
    'resp_chan':'RespiNasale',
    'lowest_freq_psd_resp':0.1,
    'lowest_freq_coherence':0.1,
    'nfft_factor':2,
    'n_cycles':4
}

coherence_at_resp_params = {
    'coherence_params': coherence_params,
}

hilbert_at_resp_mi = {
    'resp_chan':'RespiNasale',
    'low_pass_freq': 15.,
    'filter_order' : 5,
    'smooth_sigma_ms': 100.0,
    'fbands':fbands,
}

phase_freq_params = {
    'decimate_factor':2,
    'n_freqs':150,
    'f_start':4,
    'f_stop':150,
    'c_start':10,
    'c_stop':30,
    'amplitude_exponent':2,
    'mode_normalization':'z_transform',
    'n_phase_bins':200,
    'segment_ratios':0.4,

}

itpc_params = {
    'decimate_factor':2,
    'n_freqs':150,
    'f_start':4,
    'f_stop':150,
    'c_start':10,
    'c_stop':30,
    'n_phase_bins':200,
    'segment_ratios':0.4,

}

