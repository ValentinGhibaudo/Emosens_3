# RUN KEYS

from configuration import data_path

subject_keys = ['P01','P02','P03','P04','P05',
                'P06','P07','P08','P09','P10',
                'P11','P12','P13','P14','P15',
                'P16','P17','P18','P19','P20',
                'P21','P23','P24','P25', # P22 not in list because artifacted
                'P26','P27','P28','P29','P30','P31'] 

session_keys = ['baseline','music','odor']

run_keys = [f'{sub_key}_{ses_key}' for sub_key in subject_keys for ses_key in session_keys]

baseline_keys = [f'{sub_key}_baseline' for sub_key in subject_keys]
stim_keys = [f'{sub_key}_{stim_key}' for sub_key in subject_keys for stim_key in ['music','odor']]

run_keys_stai = [f'{sub_key}_{ses_key}' for sub_key in subject_keys for ses_key in ['ses01','ses02']]

global_key = 'all'


# USEFUL LISTS & DICTS
eeg_chans = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
             'TP9','CP5','CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 
             'P8', 'TP10', 'CP6','CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

bio_chans = ['ECG','RespiNasale','RespiVentrale','GSR']

all_chans = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
             'CP5','CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10',
             'CP6','CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2',
             'ECG','RespiNasale','RespiVentrale','GSR','FCI']

participants_label = {
    'P01':'DB01', # OK
    'P02':'FB02', # OK
    'P03':'ZB03', # OK
    'P04':'EM04', # OK
    'P05':'TM05', # OK
    'P06':'AC06', # OK
    'P07':'CB07', # OK
    'P08':'ZB08', # OK
    'P09':'MA09', # OK
    'P10':'AA10', # OK
    'P11':'MB11', # OK
    'P12':'AP12', # OK
    'P13':'ZC13', # OK
    'P14':'FC14', # OK
    'P15':'AP15', # OK
    'P16':'EP16', # OK
    'P17':'AG17', # OK
    'P18':'MP18', # OK
    'P19':'SR19', # OK
    'P20':'MV20', # OK
    'P21':'GA21', # OK
    'P22':'SB22', # OK
    'P23':'PB23', # OK
    'P24':'MB24', # OK
    'P25':'MB25', # OK
    'P26':'EZ26', # OK
    'P27':'AM27', # OK
    'P28':'MC28', # OK
    'P29':'ML29', # OK
    'P30':'EG30', # OK
    'P31':'MG31' # OK
    }


session_duration = 600.


#### PROCESSING PARAMS
srate = 1000

fbands = {
    'delta':[1,4],
    'theta':[4,8],
    'alpha':[8,12],
    'beta':[12,30],
    'low_gamma':[30,45],
    'high_gamma':[55,100], 
    'very_high_gamma':[100,200],
    }

ecg_inversion = { 
'P01':1, # OK
'P02':1, # OK
'P03':-1, # OK
'P04':-1, # OK
'P05':-1,  # OK
'P06':-1, # OK
'P07':1, # OK
'P08':-1, # OK
'P09':-1, # OK
'P10':-1, # OK
'P11':1, # OK
'P12':-1, # OK
'P13':-1, # OK
'P14':-1, # OK
'P15':1, # OK
'P16':-1, # OK
'P17':1, # OK
'P18':-1, # OK
'P19':-1, # OK
'P20':-1, # OK 
'P21':-1, # OK
'P22':-1, # OK
'P23':1, # OK
'P24':-1, # OK
'P25':-1, # Ok
'P26':-1, # OK
'P27':-1, # OK
'P28':-1, # OK
'P29':-1, # OK
'P30':-1, # OK
'P31':-1 # OK        
}



## ICA : 
n_components_decomposition = 10

# components exclusion
ica_excluded_component = {
'P01':{'baseline':[0,1],'music':[0,1],'odor':[0,1]}, # OK
'P02':{'baseline':[0,1],'music':[0,3],'odor':[0,2]}, # OK
'P03':{'baseline':[0,2],'music':[0,1],'odor':[1,3]}, # OK
'P04':{'baseline':[5],'music':[5],'odor':[0,4]}, # OK
'P05':{'baseline':[0],'music':[0],'odor':[0]}, # OK
'P06':{'baseline':[1,3],'music':[1,3],'odor':[2,4]}, # OK
'P07':{'baseline':[1],'music':[3],'odor':[3]}, # OK
'P08':{'baseline':[1,3],'music':[0,2],'odor':[0,2]}, # OK
'P09':{'baseline':[1,2],'music':[1,3],'odor':[1,2]}, # OK
'P10':{'baseline':[0,1],'music':[0,1],'odor':[0,1]}, # OK
'P11':{'baseline':[0,4],'music':[0,4],'odor':[0,6]}, # OK
'P12':{'baseline':[0,1],'music':[0,1],'odor':[0,1]}, # OK 
'P13':{'baseline':[0,2],'music':[0,1],'odor':[0,1]}, # OK
'P14':{'baseline':[0,5],'music':[0,5],'odor':[0,2]}, # OK
'P15':{'baseline':[0,4],'music':[0,1,4],'odor':[0,2]}, # OK
'P16':{'baseline':[0,2],'music':[0,1],'odor':[0,1]}, # OK
'P17':{'baseline':[3,4],'music':[3,4],'odor':[1,2]}, # OK
'P18':{'baseline':[0,1,2],'music':[0,2],'odor':[0,2]}, # OK
'P19':{'baseline':[0,2],'music':[0,2],'odor':[0,2]}, # OK
'P20':{'baseline':[0,3],'music':[0,2],'odor':[0,4]}, # OK
'P21':{'baseline':[0,1],'music':[0,2],'odor':[0,1]}, # OK
'P22':{'baseline':[0,6],'music':[0,2],'odor':[0,4]}, # OK
'P23':{'baseline':[0,1],'music':[0,1],'odor':[0,1]}, # OK
'P24':{'baseline':[1,3],'music':[1,2],'odor':[2,3]}, # OK
'P25':{'baseline':[0,2],'music':[0,3],'odor':[0,2]}, # OK
'P26':{'baseline':[0,2],'music':[0,2],'odor':[0,2]}, # OK
'P27':{'baseline':[0,4],'music':[0,6],'odor':[0,2]}, # OK
'P28':{'baseline':[0],'music':[0,2],'odor':[0,2]}, # OK
'P29':{'baseline':[2],'music':[2],'odor':[1]}, # OK
'P30':{'baseline':[0,1],'music':[0,1],'odor':[0,1]}, # OK
'P31':{'baseline':[0,2],'music':[0,2],'odor':[0,2]} # OK
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
notch_freqs = np.arange(50, 200, 50).tolist()
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
    'session_duration':session_duration,
    'lowcut':0.05, # lowcut frequency in Hz
    'highcut':200, # highcut frequency in Hz
    'ftype':'butter',
    'order':6
}

artifact_params = {
    'preproc_params':preproc_params,
    'n_deviations':3, # number of deviations (MAD) to the median
    'window_size':1.5, # size in seconds of th moving window of rms
    'step': 0.5, # step time in second to compute rms windows
    'lf':30, # low cutoff frequency to filter signal on which artifacts are detected
    'hf':150, # high cutoff frequency to filter signal on which artifacts are detected
    'n_chan_artifacted':5 # when this number of channels are bugging at the same time, an artifact is detected
}

artifact_by_chan_params = {
    'preproc_params':preproc_params,
    'n_deviations':4.5, # number of deviations (MAD) to the median
    'window_size':1.5, # size in seconds of th moving window of rms
    'step': 0.5, # step time in second to compute rms windows
    'lf':30, # low cutoff frequency to filter signal on which artifacts are detected
    'hf':150, # high cutoff frequency to filter signal on which artifacts are detected
}

interp_artifact_params = {
    'artifact_params':artifact_params,
    'freq_min':30.,
    'margin_s':0.2, 
    'seed':None
}

count_artifact_params = {
    'artifact_params':artifact_params,
    'session_duration':session_duration,
    'thresh_prop_time_artifacted':0.3 # trials with more than this proportion of time artifacted will be marked as removable
}

respiration_features_params = {
    'inspiration_sign' : '+',
    'session_duration':session_duration
}


ecg_params = {
    'session_duration':session_duration,
    'ecg_inversion' : ecg_inversion,
    'thresh':5 # N mads to detect ECG peaks
}

rri_signal_params = {
    'ecg_params' : ecg_params,
    'max_interval' : 120, # in bpm
    'min_interval': 30., # in bpm
    'interpolation_kind': 'cubic',
}


psd_params = {
    'interp_artifact_params':interp_artifact_params,
    'lowest_freq':0.1,
}

bandpower_params = {
    'psd_params':psd_params,
    'fbands':fbands,
    'total_band':[psd_params['lowest_freq'] , 200] # keep clean freq band (> 200 Hz = noisy)
}

power_at_resp_params = {
    'psd_params':psd_params,
    'session_duration':session_duration,
    'resp_chan':'RespiNasale',
    'lowest_freq_psd_resp':0.1,
}

coherence_params = {
    'interp_artifact_params':interp_artifact_params,
    'resp_chan':'RespiNasale',
    'lowest_freq_psd_resp':0.15,
    'lowest_freq_coherence':0.15,
    'nfft_factor':2,
    'n_cycles':4,
    'session_duration':session_duration
}

coherence_at_resp_params = {
    'coherence_params': coherence_params,
}

time_freq_params = {
    'interp_artifact_params':interp_artifact_params,
    'chans':['F3','F4','C3','C4','T7','T8','P7','P8','O1','O2'],
    'decimate_factor':2,
    'n_freqs':150,
    'f_start':4,
    'f_stop':150,
    'c_start':10,
    'c_stop':30,
    'amplitude_exponent':2
}

power_params = {
    'interp_artifact_params':interp_artifact_params,
    'chans':['F3','F4'],
    'decimate_factor':10,
    'n_freqs':150,
    'f_start':4,
    'f_stop':150,
    'c_start':10,
    'c_stop':30,
    'amplitude_exponent':2
}

phase_freq_params = {
    'respiration_features_params':respiration_features_params,
    'time_freq_params':time_freq_params,
    'n_phase_bins':200,
    'segment_ratios':0.4,
}

phase_freq_fig_params = {
    'phase_freq_params':phase_freq_params,
    'compress_subject':'Mean',
    'baseline_mode':'rz_score',
    # 'compress_cycle_mode':'med_cycle',
    # 'baseline_mode':'z_score',
    # 'compress_cycle_mode':'mean_cycle',
    'compress_cycle_mode':'q75_cycle',
    'stim_sessions':['music','odor'],
    'delta_colorlim':0.,
    'max_freq':20
}

eda_params = {
    'session_duration':session_duration
}

rsa_params = {
    'respiration_features_params':respiration_features_params,
    'ecg_params':ecg_params,
    'n_phase_bins':100,
}

stai_longform_params = {
    'mean_etat':35.4,
    'mean_trait':34.8,
    'sd_etat':10.5,
    'sd_trait':9.2
}

maia_params = {
    'reverse':{1:'+',2:'+',3:'+',4:'+',5:'-',6:'-',7:'-',8:'-',9:'-',10 :'+',
               11:'+', 12:'+', 13:'+', 14:'+',15:'+',16:'+', 17:'+', 18:'+', 19:'+',20:'+',
               21:'+', 22:'+', 23:'+', 24:'+',25:'+',26:'+', 27:'+', 28:'+', 29:'+',30:'+',
               31:'+', 32:'+'},
    'items':{
        'noticing':[1,2,3,4],
        'not_distracting':[5,6,7],
        'not_worrying':[8,9,10],
        'attention_regulation':[11,12,13,14,15,16,17],
        'emotional_awareness':[18,19,20,21,22],
        'self_regulation':[23,24,25,26],
        'body_listening':[27,28,29],
        'trusting':[30,31,32],
    }
    }

relaxation_params = {}

emotions_params = {}

cycle_signal_params = {
    'interp_artifact_params':interp_artifact_params,
    'respiration_features_params':respiration_features_params,
    'n_phase_bins':1000,
    'segment_ratios':0.4,
    'chans':eeg_chans,
    'session_duration':session_duration
}

cycle_signal_modulation_params = {
    'cycle_signal_params':cycle_signal_params,
}


# Global concat of dataframes jobs params
# odor_rating_params = {'subject_keys':subject_keys,
#                      'date_of_data_generation':'06-06-2023',
#                     'rename_cols':{
#                         'odeur_name':'Odor_Name',
#                         'odeur_label':'Odor_Label',
#                         'appréciation_absolue_normalisée':'Absolute Hedonicity',
#                          'appréciation_relative_normalisée':'Hedonicity',
#                        'intensité_émotionnelle_relative_normalisée':'Emotional Intensity',
#                          'familiarité_relative_normalisée':'Familiarity',
#                            'intensité_relative_normalisée':'Stimulus Intensity',
#                            'evocation_relative_normalisée':'Memory Evocation'}
#                      }

eda_concat_params = {'run_keys':run_keys,
                    'eda_params':eda_params
                    }

hrv_concat_params = {'run_keys':run_keys,
                    'ecg_params':ecg_params,
                    }


rsa_concat_params = {'run_keys':run_keys,
                    'rsa_params':rsa_params,
                    }

bandpower_concat_params = {'run_keys':run_keys,
                           'bandpower_params':bandpower_params
                          }

coherence_at_resp_concat_params = {'run_keys':run_keys,
                           'coherence_params':coherence_params
                          }
         
                           
power_at_resp_concat_params = {'run_keys':run_keys,
                           'power_at_resp_params':power_at_resp_params
                          }


resp_features_concat_params = {'run_keys':run_keys,
                           'respiration_features_params':respiration_features_params
                          }


relaxation_concat_params = {'run_keys':subject_keys,
                           'relaxation_params':relaxation_params
                          }

maia_concat_params = {'subject_keys':subject_keys,
                     'maia_params':maia_params
                     }

# stai_long_concat_params = {'run_keys':run_keys,
#                            'stai_longform_params':stai_longform_params
#                           }


modulation_cycle_signal_concat_params = {'run_keys':run_keys,
                           'cycle_signal_modulation_params':cycle_signal_modulation_params
                          }