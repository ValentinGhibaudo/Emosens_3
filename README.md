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
    - compute_bandpower.py
    - compute_coherence.py
    - compute_cycle_signal.py
    - compute_eda.py
    - compute_global_dataframes.py 
    - compute_phase_freq.py
    - compute_power_at_resp.py
    - compute_psd.py
    - compute_psycho.py
    - compute_resp_features.py
    - compute_rri.py
    - compute_rsa.py

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
        - function : Replace movement artifacts times by patches of signal containing the average frequency content of the whole signal of the channel
        - recruit : preproc_job + artifact_by_chan_job
        - run keys : sub, ses
    * count_artifact_job
        - function : Count time duration (absolute and relative) of movement artifacting 
        - recruit : artifact_job
        - run keys : sub



