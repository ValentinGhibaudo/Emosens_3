import netCDF4
from configuration import *
from params import *

import xarray as xr
import pandas as pd
import numpy as np

import ephyviewer as ev
from ephyviewer import MainViewer, TraceViewer, TimeFreqViewer, EpochViewer, EventList, VideoViewer, DataFrameView, InMemoryAnalogSignalSource

#~ from params import subject_keys, srate, eeg_chans, trial_durations
#~ from configuration import base_folder

#~ from myqt import QT, DebugDecorator

from params import eeg_chans, session_duration
from preproc import convert_vhdr_job, preproc_job, eeg_viewer_job
from compute_resp_features import respiration_features_job
from compute_rri import ecg_job, rri_signal_job



def get_viewer_from_run_key(run_key, parent=None):
    
    print('get_viewer_from_run_key', run_key)
    
    
    #~ session = get_session_from_odor(run_key, odeur)
    
    #~ raw_data = xr.open_dataarray(base_folder / 'Preprocessing' / 'Data_Preprocessed' / f'raw_{run_key}_{session}.nc')
    #~ clean_data = xr.open_dataarray(base_folder / 'Preprocessing' / 'Data_Preprocessed' / f'clean_{run_key}_{session}.nc')
    #~ resp_features = pd.read_excel(base_folder / 'Tables' / 'resp_features' / f'{run_key}_{session}_resp_features.xlsx')
    
    resp_features = respiration_features_job.get(run_key).to_dataframe()
    
    raw_dataset = convert_vhdr_job.get(run_key)
    
    raw_data = raw_dataset['raw'].sel(time = slice(0, session_duration))[:,:-1]
    srate = raw_dataset['raw'].attrs['srate']
    # print(raw_data)


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
    #~ sig_ecg = raw_data.sel(chan='ECG').values[:, None]
    
    ecg_ds = ecg_job.get(run_key)
    sig_ecg = ecg_ds['ecg'].values[:, None]
    
    ecg_peak = ecg_ds['ecg_peaks'].values
    
    scatter_indexes_resp = {0: ecg_peak}
    scatter_channels_resp = {0: [0],}
    scatter_colors_resp = {0: '#FF0000'}
    
    
    view_ecg = TraceViewer.from_numpy(sig_ecg, srate, t_start, 'ecg', channel_names=['ecg'], 
                scatter_indexes=scatter_indexes_resp, scatter_channels=scatter_channels_resp, scatter_colors=scatter_colors_resp)
    win.add_view(view_ecg)

    

    ###### viewer2 = bio
    #~ channel_names = ['RRI']
    
    ds_rri = rri_signal_job.get(run_key)
    rri_sig = ds_rri['rri'].values[:, None]
    srate = ds_rri['rri'].attrs['srate']

    view_rri = TraceViewer.from_numpy(rri_sig,  srate, t_start, 'RRI', channel_names=['RRI'])
    win.add_view(view_rri)
    view_rri.params['display_labels'] = True
    view_rri.params['scale_mode'] = 'real_scale'
    view_rri.by_channel_params[ 'ch0' ,'color'] = '#FF773C'


    
    
    
    # # ###### viewer3
    # # channel_names = eeg_chans
    
    # ds_raw_eeg = convert_vhdr_job.get(run_key)
    # eeg_raw = ds_raw_eeg['raw'].sel(chan=eeg_chans).values.T
    # srate = ds_raw_eeg['raw'].attrs['srate']

    # view3 = TraceViewer.from_numpy(eeg_raw,  srate, t_start, 'Raw eeg', channel_names=eeg_chans)
    # win.add_view(view3)
    # view3.params['display_labels'] = True
    # view3.params['scale_mode'] = 'by_channel'
    # for c, chan_name in enumerate(eeg_chans):
    #     view3.by_channel_params[ f'ch{c}' ,'visible'] = c < 3
    
    ###### viewer4
    #~ channel_names = eeg_chans
    
    ds_eeg = eeg_viewer_job.get(run_key)
    eeg_clean = ds_eeg['eeg_viewer'].values.T
    srate = ds_eeg['eeg_viewer'].attrs['srate']
    channel_names = ds_eeg['eeg_viewer'].coords['chan'].values
    
    source = InMemoryAnalogSignalSource(eeg_clean, srate, 0, channel_names=channel_names)

    view4 = TraceViewer(source = source, name='Clean eeg')
    win.add_view(view4)
    view4.params['display_labels'] = True
    view4.params['scale_mode'] = 'by_channel'
    for c, chan_name in enumerate(channel_names):
        view4.by_channel_params[ f'ch{c}' ,'visible'] = c < 3


    #### viewer 5
    #create a time freq viewer conencted to the same source
    view5 = TimeFreqViewer(source=source, name='tfr')
    win.add_view(view5)
    view5.params['show_axis'] = True
    view5.params['timefreq', 'deltafreq'] = 1
    view5.params['timefreq', 'f0'] = 3.
    view5.params['timefreq', 'f_start'] = 1.
    view5.params['timefreq', 'f_stop'] = 100.
    for c, chan_name in enumerate(channel_names):
        view5.by_channel_params[ f'ch{c}' ,'visible'] = c < 1

    
    


    win.set_xsize(60.)

    win.auto_scale()
    
        
    


    return win


def test_get_viewer():
    
    run_key = 'P10_music'

    # ds = respiration_features_job.get(run_key)
    # print(ds)
    # print(ds.to_dataframe())

    
    app = ev.mkQApp()
    win = get_viewer_from_run_key(run_key)
    
    
    win.show()
    app.exec_()


if __name__ == '__main__':
    test_get_viewer()
    
    
    
