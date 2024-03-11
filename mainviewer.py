import netCDF4
from configuration import *
from params import *

import xarray as xr
import pandas as pd
import numpy as np

import ephyviewer as ev
from ephyviewer import MainViewer, TraceViewer, TimeFreqViewer, EpochViewer, EventList, VideoViewer, DataFrameView, InMemoryAnalogSignalSource

from params import eeg_chans, session_duration
from preproc import convert_vhdr_job, preproc_job, artifact_job, eeg_interp_artifact_job
from compute_resp_features import respiration_features_job
from compute_rri import ecg_job, rri_signal_job, ecg_peak_job



def get_viewer_from_run_key(run_key, parent=None):
    
    print('get_viewer_from_run_key', run_key)
    
    
    resp_features = respiration_features_job.get(run_key).to_dataframe()
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    raw_data = raw_dataset['raw'].sel(time = slice(0, session_duration))[:,:-1]
    srate = raw_dataset['raw'].attrs['srate']

    win = MainViewer(show_label_datetime=False, parent=parent, show_global_xsize=True, show_auto_scale=True)



    t_start = 0

    #respi = viewer1 
    inspi_index = resp_features['inspi_index'].values
    expi_index = resp_features['expi_index'].values
    
    scatter_indexes_resp = {0: inspi_index, 1:expi_index}
    scatter_channels_resp = {0: [0], 1: [0]}
    scatter_colors_resp = {0: '#FF0000', 1: '#00FF00'}

    sig_resp = raw_data.sel(chan='RespiNasale').values[:, None] * -1
    # print(sig_resp.shape)
    
    view1 = TraceViewer.from_numpy( sig_resp, srate, t_start, 'resp', channel_names=['resp'],
                scatter_indexes=scatter_indexes_resp, scatter_channels=scatter_channels_resp, scatter_colors=scatter_colors_resp)
    win.add_view(view1)
    view1.params['scale_mode'] = 'by_channel'
    view1.params['display_labels'] = False
    view1.params['display_offset'] = False
    view1.params['antialias'] = True
    view1.by_channel_params[ 'ch0' ,'color'] = '#ffc83c'
    
    
    # ecg
    ecg_ds = ecg_job.get(run_key)
    srate = ecg_ds.attrs['srate']
    sig_ecg = ecg_ds['ecg'].values[:, None]
    
    ecg_peak = ecg_peak_job.get(run_key).to_dataframe()
    ecg_peak = ecg_peak['peak_index'].values
    
    scatter_indexes_resp = {0: ecg_peak}
    scatter_channels_resp = {0: [0],}
    scatter_colors_resp = {0: '#FF0000'}
    
    
    view_ecg = TraceViewer.from_numpy(sig_ecg, srate, t_start, 'ecg', channel_names=['ecg'], 
                scatter_indexes=scatter_indexes_resp, scatter_channels=scatter_channels_resp, scatter_colors=scatter_colors_resp)
    win.add_view(view_ecg)

    

    ###### viewer2 = bio
    #~ channel_names = ['RRI']
    
    da_rri = rri_signal_job.get(run_key)['rri']
    srate = da_rri.attrs['srate']
    rri_sig = da_rri.values[:, None]

    view_rri = TraceViewer.from_numpy(rri_sig,  srate, t_start, 'RRI', channel_names=['RRI'])
    win.add_view(view_rri)
    view_rri.params['display_labels'] = True
    view_rri.params['scale_mode'] = 'real_scale'
    view_rri.by_channel_params[ 'ch0' ,'color'] = '#FF773C'


     ######################################### VIEW ARTIFACTS
    # VIEWER RESPI CYCLES REMOVED
    resp_features_removed = resp_features[resp_features['artifact'] == 1]
    periods = []
    d = {
        'time' : resp_features_removed['inspi_time'].values,
        'duration' : resp_features_removed['cycle_duration'].values,
        'label': np.full(shape = resp_features_removed.shape[0], fill_value='removed'),
        'name': 'Removed resp cycle',
    }
    periods.append(d)
    

    # VIEWER ARTIFACT EPOCHS
    artifacts = artifact_job.get(run_key).to_dataframe()
    d = {
        'time' : artifacts['start_t'].values,
        'duration' : artifacts['duration'].values,
        'label': np.full(shape = artifacts.shape[0], fill_value='artifact'),
        'name': 'Artifact eeg epoch',
    }
    periods.append(d)

    view_artifacts = EpochViewer.from_numpy(periods, 'Artifact')
    view_artifacts.by_channel_params['ch0', 'color'] = '#ffc83c'
    view_artifacts.by_channel_params['ch1', 'color'] = '#B9B9B9'
    win.add_view(view_artifacts)



    # # VIEWER EEG
    # artifacts
    da_eeg = eeg_interp_artifact_job.get(run_key)['interp']
    srate = da_eeg.attrs['srate']

    eeg_clean = da_eeg.values.T
    channel_names = da_eeg.coords['chan'].values

    view_eeg = TraceViewer.from_numpy(eeg_clean,  srate, 0, 'eeg', channel_names=channel_names)
    win.add_view(view_eeg)
    view_eeg.params['display_labels'] = True
    view_eeg.params['scale_mode'] = 'by_channel'
    for c, chan_name in enumerate(channel_names):
        view_eeg.by_channel_params[ f'ch{c}' ,'visible'] = c < 5

    # VIEWER EEG 2
    # artifacts
    da_eeg = preproc_job.get(run_key)['eeg_clean']
    srate = da_eeg.attrs['srate']

    eeg_clean = da_eeg.values.T
    channel_names = da_eeg.coords['chan'].values

    view_eeg = TraceViewer.from_numpy(eeg_clean,  srate, 0, 'eeg2', channel_names=channel_names)
    win.add_view(view_eeg, tabify_with='eeg')
    # win.add_view(view_eeg)
    view_eeg.params['display_labels'] = True
    view_eeg.params['scale_mode'] = 'by_channel'
    for c, chan_name in enumerate(channel_names):
        view_eeg.by_channel_params[ f'ch{c}' ,'visible'] = c < 5




    # VIEWER TIME-FREQUENCY
    source = InMemoryAnalogSignalSource(eeg_clean, srate, t_start, channel_names=channel_names)
    # create a time freq viewer connected to the same source
    view_tf = TimeFreqViewer(source=source, name='tfr')
    win.add_view(view_tf)
    view_tf.params['show_axis'] = True
    view_tf.params['timefreq', 'deltafreq'] = 1
    view_tf.params['timefreq', 'f0'] = 3.
    view_tf.params['timefreq', 'f_start'] = 1.
    view_tf.params['timefreq', 'f_stop'] = 100.
    for c, chan_name in enumerate(channel_names):
        view_tf.by_channel_params[ f'ch{c}' ,'visible'] = c == 2

    
    

    win.set_xsize(60.)

    win.auto_scale()
    
    return win

def test_get_viewer():
    
    run_key = 'P09_odor' # choose run key (subject_session) to display # baseline, music, odor

    # ds = respiration_features_job.get(run_key)
    # print(ds)
    # print(ds.to_dataframe())

    
    app = ev.mkQApp()
    win = get_viewer_from_run_key(run_key)
    
    
    win.show()
    app.exec_()


if __name__ == '__main__':
    test_get_viewer()
    
    
    
