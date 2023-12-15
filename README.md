# Analysis pipeline for Emosens 3 data

This folder is organized into 6 types of python scripts : 

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




