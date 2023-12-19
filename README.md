# Analysis pipeline for Emosens 3 data

This repository is organized into 5 types of python scripts : 

- 1) Scripts containing useful tools (functions) and variables used in several scripts : 
    - bibliotheque_artifact_detection.py
    - bibliotheque.py
    - circulat_stats.py
    - configuration.py
    - jobtools.py

- 2) Scripts encoding entangled jobs dedicated to compute outputs used for statistics, going from raw data to dataframes usable for stats : 
    - preproc.py
    - compute_psd.py
    - compute_bandpower.py
    - compute_coherence.py
    - compute_power_at_resp.py
    - compute_resp_features.py
    - compute_rri.py
    - compute_rsa.py
    - compute_cycle_signal.py
    - compute_phase_freq.py
    - compute_eda.py
    - compute_psycho.py
    - compute_global_dataframes.py 

- 3) Scripts allowing visualisation of preprocessed data, loading outputs from jobs : 
    - myqt.py
    - mainwindow.py
    - mainviewer.py

- 4) Notebooks and scripts used for generating figures and associated statistics by loading outputs from jobs : 
    - stats_bandpower.ipynb
    - stats_coherence_at_resp.ipynb
    - stats_correlation.ipynb
    - stats_cycle_signal.py
    - stats_ecg_peaks_coupling.ipynb
    - stats_eda.ipynb
    - stats_erp.py
    - stats_hrv.ipynb
    - stats_instantaneous_resp.py
    - stats_modulation_cycle_signal.ipynb
    - stats_phase_freq.py
    - stats_power_at_resp.ipynb
    - stats_psycho.ipynb
    - stats_resp_features.ipynb
    - stats_rsa.ipynb
    - stats_stim_scoring.ipynb
    - stats_time_phase_power.py
    - stats_with_tempo.ipynb

- 5) Other scripts/notebooks for explorations or debugging : 
    - video_cycle_signal.py
    - verif_ecg.ipynb
    - summary.py
    - sandbox.ipynb
    - spectro_artifacts.py



The second type of scripts (computing scripts) are coded thanks to entangled jobs (output = xarray.Dataset()) corresponding to the following : 
- preproc.py
    * convert_vhdr_job
        - function : Convert raw data from brainvision to an xarray format
        - recruit : --
        - run keys : sub, ses
    * ica_figure_job
        - function : Save ICA figures to manually select (in a dictionnary in params.py) for each sub/ses the EOG components to remove
        - recruit : --
        - run keys : sub, ses
    * preproc_job
        - function : Preproc raw EEG (Notch + ICA + detrend + bandpass filter)
        - recruit : --
        - run keys : sub, ses
    * artifact_job
        - function : Detect movement artifacts based on sharp cooccuring burst of gamma power on all channels
        - recruit : preproc_job
        - run keys : sub, ses
    * artifact_by_chan_job
        - function : Detect movement artifacts based on sharp burst of gamma power channel by channel
        - recruit : preproc_job
        - run keys : sub, ses
    * eeg_interp_artifact_job
        - function : Replace movement artifacts times by interpolation of patches of signal containing the average frequency content of the whole signal of the channel
        - recruit : preproc_job + artifact_by_chan_job
        - run keys : sub, ses
    * count_artifact_job
        - function : Count time duration (absolute and relative) of movement artifacting 
        - recruit : artifact_job
        - run keys : sub

- compute_psd.py
    * psd_eeg_job
        - function : Compute power spectrum of EEG (lowest freq = 0.1 Hz)
        - recruit : eeg_interp_artifact_job
        - run keys : sub, ses
    * psd_bandpower_job
        - function : Compute power spectrum of EEG (lowest freq = 1 Hz) ready to extract bandpower
        - recruit : eeg_interp_artifact_job
        - run keys : sub, ses
    * psd_baselined_job
        - function : Normalize power spectrum of music and odor session by baseline session power spectrum
        - recruit : psd_bandpower_job
        - run keys : sub, stim (music,odor)

- compute_bandpower.py
    * psd_eeg_job
        - function : Compute bandpower for each frequency band (EEG)
        - recruit : psd_baselined_job
        - run keys : sub, stim (music,odor)

- compute_coherence.py
    * coherence_job
        - function : Compute magnitude squated coherence between EEG and Resp signals
        - recruit : eeg_interp_artifact_job + convert_vhdr_job (to get resp signal)
        - run keys : sub, ses
    * coherence_at_resp_job
        - function : Extract coherence value between EEG and RESP signal at the dominant respiratory frequency
        - recruit : coherence_job + convert_vhdr_job (to get resp signal to get its dominant frequency)
        - run keys : sub, ses

- compute_power_at_resp.py
    * power_at_resp_job
        - function : Compute power spectrum value of EEG at respiratory dominant frequency
        - recruit : psd_eeg_job + convert_vhdr_job (to get resp signal)
        - run keys : sub, ses

- compute_resp_features.py
    * respiration_features_job
        - function : Compute respiration features from raw resp signal and annotate cycles according to cooccuring artifacting of EEG signals
        - recruit : convert_vhdr_job + artifact_job
        - run keys : sub, ses

- compute_rri.py
    * ecg_job
        - function : Preproc ECG
        - recruit : convert_vhdr_job
        - run keys : sub, ses
    * ecg_peak_job
        - function : Detect ECG R peaks
        - recruit : convert_vhdr_job
        - run keys : sub, ses
    * rri_signal_job
        - function : Compute heart rate continuous signal from ECG R peaks
        - recruit : ecg_job + ecg_peak_job
        - run keys : sub, ses
    * ecg_peaks_coupling_job
        - function : Compute phase angles of ECG R peaks according to their relative position during cooccuring respiratory cycle
        - recruit : ecg_peak_job + respiration_features_job
        - run keys : sub, ses

- compute_rsa.py
    * rsa_phase_job
        - function : Cyclically deform heart rate signal according to respiratory timestamps of each respiratory cycle
        - recruit : ecg_peak_job + respiration_features_job
        - run keys : sub, ses
    * rsa_features_job
        - function : Extract Respiratory Sinus Arrhythmia features respiratory cycle by respiratory cycle
        - recruit :  ecg_peak_job + respiration_features_job
        - run keys : sub, ses

- compute_cycle_signal.py
    * cycle_signal_job
        - function : Cyclically deform EEG (and respi) signals according to respiratory timestamps of each respiratory cycle and compute average evoked potential
        - recruit : eeg_interp_artifact_job + convert_vhdr_job + respiration_features_job + rri_signal_job (to deform heart rate signal at the same time)
        - run keys : sub, ses
    * modulation_cycle_signal_job
        - function : Compute amplitude of average evoked potential of EEG respi epochs, a marker of modulation
        - recruit : cycle_signal_job
        - run keys : sub, ses

- compute_phase_freq.py
    * power_job
        - function : Compute time frequency power maps of EEG signals by convolution with morlet wavelets
        - recruit : eeg_interp_artifact_job
        - run keys : sub, ses, chan
    * baseline_job
        - function : Compute mean/median/sd/mad of EEG time frequency power for each frequency bin (so on time axis) from baseline session
        - recruit : power_job
        - run keys : sub, chan
    * phase_freq_job
        - function : Normalize raw time frequency power maps by baseline + cyclically deform it by respiratory epochs/timestamps to get phase frequency power maps
        - recruit : power_job + baseline_job + respiration_features_job 
        - run keys : sub, ses, chan
    * phase_freq_concat_job
        - function : Concatenate phase-frequency power maps from sub,ses into one Dataset by channel
        - recruit : phase_freq_job
        - run keys : chan
    * erp_time_freq_job
        - function : Same process than phase freq but without cyclically deforming EEG data, keeping a time basis, and average time-frequency power dynamic centered on a respiratory time point
        - recruit : power_job + baseline_job + respiration_features_job
        - run keys : sub, ses, chan
    * erp_concat_job
        - function : Concatenate erp power maps from sub,ses into one Dataset by channel
        - recruit : convert_vhdr_job
        - run keys : chan

- compute_eda.py
    * eda_job
        - function : Compute electrodermal activity metrics
        - recruit : convert_vhdr_job
        - run keys : sub, ses

- compute_psycho.py
    * maia_job
        - function : Compute MAIA metrics (multidimensional assessment of Interoceptive Awareness)
        - recruit : --
        - run keys : sub
    * stai_longform_job
        - function : Compute State Trait Anxiety metrics from long-form STAI questionnaire
        - recruit : --
        - run keys : sub, ses
    * relaxation_job
        - function : Compute relaxation metrics from relaxation questionnaire
        - recruit : --
        - run keys : sub
    * emotions_job
        - function : Process evaluation of emotions induced by stimuli assessed on questionnaires
        - recruit : --
        - run keys : sub
    * oas_job
        - function : Process OAS (Odor Awareness Scale) questionnaires
        - recruit : --
        - run keys : sub
    * bmrq_job
        - function : Process BMRQ (Barcelona Music Reward Questionnaire) questionnaires
        - recruit : --
        - run keys : sub

- compute_global_dataframes.py
    * defines jobs just to concatenate metric outputs of previous jobs into dataset (dataframes into dataset)
    * maia_concat_job
        - recruit : maia_job
    * bandpower_concat_job
        - recruit : bandpower_job
    * coherence_at_resp_concat_job
        - recruit : coherence_at_resp_job
    * eda_concat_job
        - recruit : eda_job
    * power_at_resp_concat_job
        - recruit : power_at_resp_job
    * relaxation_concat_job
        - recruit : relaxation_job
    * resp_features_concat_job
        - recruit : respiration_features_job
    * rsa_concat_job
        - recruit : rsa_features_job
    * modulation_cycle_signal_concat_job
        - recruit : modulation_cycle_signal_job
    * oas_concat_job
        - recruit : oas_job
    * bmrq_concat_job
        - recruit : bmrq_job
