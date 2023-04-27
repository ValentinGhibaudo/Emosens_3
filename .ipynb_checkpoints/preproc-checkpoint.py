from configuration import *
from params import *
import xarray as xr
from bibliotheque import mne_to_xarray
import pandas as pd
import jobtools
from bibliotheque import get_raw_mne
import matplotlib.pyplot as plt

def convert_vhdr(run_key, **p):
    raw = get_raw_mne(run_key, participants_label, preload=True)
    ds = xr.Dataset()
    ds['raw'] = mne_to_xarray(raw)
    return ds


def test_convert_vhdr():
    run_key = 'P03_music'
    ds = convert_vhdr(run_key, **convert_vhdr_params)
    print(ds)



def apply_ica(raw_eeg, exclude, participant, session, n_components, save_figures=False):
    import mne
    import ghibtools as gh
    raw_eeg_for_ica_filtered = raw_eeg.copy()
    raw_eeg_for_ica_filtered = raw_eeg_for_ica_filtered.filter(1, None, fir_design='firwin', verbose = 'CRITICAL') # filter with highpass 1 Hz for better working of ICA

    random_state = 27
    method = 'fastica'
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, method=method, verbose = 'CRITICAL') # compute ICA
    ica.fit(raw_eeg_for_ica_filtered, verbose = 'CRITICAL') # run ICA on provided raw data

    if save_figures:
        sources_signals = ica.get_sources(raw_eeg_for_ica_filtered).get_data() # get source * time np array
        duration = 60000 # duration of time vector of figures = 60 secondes at 1000 Hz
        
        fig, axs = plt.subplots(nrows = 2, figsize = (15,13), constrained_layout = True)
        ax = axs[0]
        ax.set_title(f'{participant} - {session} - 60 seconds sample')
        ax.set_xlabel('Time points')
        start = int(300 * srate) # start = during a fast trial after 60 seconds 
        stop = start + duration
        for i in range(sources_signals.shape[0]):
            ax.plot(sources_signals[i,start:stop] + i * -10, linewidth = 0.5, label = f'ICA00{i}') # plot sources during one minute of fast trials
        ax.legend()

        ax = axs[1]
        lowest = 0.5
        rows = []
        for i in range(sources_signals.shape[0]):
            f, Pxx = gh.spectre(sources_signals[i], srate, lowest_freq = lowest)
            slow_power = np.trapz(Pxx[(f > 0.7) & (f < 1.4)])
            rows.append([participant, session, i, slow_power])
            mask = (f > lowest)
            ax.loglog(f[mask], Pxx[mask], label = i)
        for j in range(50,550,50):
            ax.axvline(x=j, color='red', alpha = 0.2)
            
        powers = pd.DataFrame(rows, columns = ['participant','session','component', 'slow_power']).sort_values(by = 'slow_power', ascending = False)
        order_compo = list(powers['component'].values)
        
        ax.grid(which = 'minor', alpha = 0.3)
        ax.set_ylabel('Power [µV**2]')
        ax.set_xlabel('Freq [Hz]')
        ax.set_title(f'{participant} - {session} - PSD - slow power order : {order_compo}')
        ax.legend()
        plt.savefig(base_folder / 'Figures' / 'ICA' / f'{participant}_{session}_Time_PSD', bbox_inches = 'tight')
        plt.close()
        

        
        plt.figure()
        ica.plot_components(title=f'{participant} - {session} - topography components - slow power order : {order_compo}')
        plt.savefig(base_folder / 'Figures' / 'ICA' / f'{participant}_{session}_Topo')
        plt.close()
        
        plt.close('all')
        
            
            


    raw_without_filter_but_with_ica = raw_eeg.copy()
    ica.apply(raw_without_filter_but_with_ica, exclude = exclude, verbose = 'CRITICAL') # applique sur le raw originel (pas filtré) ce qui a été calculé sur le raw_ica filtré 1 Hz lowcut

    return raw_without_filter_but_with_ica
    

def compute_ica_figure(run_key, **p):

    participant, session = run_key.split('_')

    raw = get_raw_mne(run_key, participants_label, preload=True)
    raw_eeg = raw.copy()
    raw_eeg = raw_eeg.pick_types(eeg = True)

    apply_ica(raw_eeg, [0], participant, session, p['n_components_decomposition'], save_figures=True)

    return None

def test_compute_ica_figure():

    sub = 'P22' 
    
    for run_key in [f'{sub}_baseline',f'{sub}_music',f'{sub}_odor']:
        compute_ica_figure(run_key, **ica_figure_params)





def compute_preproc(run_key, **p):
    import mne
    #~ print(run_key)
    #~ print(p['notch_freqs'])

    participant, session = run_key.split('_')

    raw = get_raw_mne(run_key, participants_label, preload=True)
    raw.crop(tmin = 0, tmax = p['session_duration'], include_tmax = False)

    # NOTCH
    raw_notched = raw.copy()
    raw_notched.notch_filter(p['notch_freqs'], verbose = False)

    # PREPROC NEURO (ICA)
    ica_excluded_component = p['ica_excluded_component']
    exclude = ica_excluded_component[participant][session]
    raw_eeg = raw.copy()
    raw_eeg = raw_eeg.pick_types(eeg = True)
    raw_clean_from_eog = apply_ica(raw_eeg, exclude, participant, session, p['n_components_decomposition'], save_figures= p['save_ica_fig'])
    
    data = raw_clean_from_eog.get_data()
    
    times = np.arange(data.shape[1]) / srate

    # CONCAT DATA
    ds = xr.Dataset()
    ds['eeg_clean'] = xr.DataArray(data = data, dims = ['chan','time'],
                                   coords = {'chan':p['eeg_chans'], 'time':times}, 
                                   attrs = {'srate':srate})

    return ds

def test_compute_preproc():
    run_key = 'P02_baseline'
    ds = compute_preproc(run_key, **preproc_params)
    print(ds)



def preproc_viewer(run_key, **p):
    import ghibtools as gh
    eeg = preproc_job.get(run_key)['eeg_clean']
    srate = eeg.attrs['srate']

    eeg_viewer = eeg.copy()
    eeg_viewer[:] = gh.iirfilt(eeg.values, srate, p['lf'], p['hf'] , axis = 1)
    ds = xr.Dataset()
    ds['eeg_viewer'] = eeg_viewer
    return ds

def test_preproc_viewer():
    run_key = 'P02_baseline'
    ds = preproc_viewer(run_key, **eeg_viewer_params)
    print(ds)


def compute_all():
    jobtools.compute_job_list(preproc_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(ica_figure_job, run_keys, force_recompute=False, engine='loop')
        
    # jobtools.compute_job_list(preproc_job, run_keys, force_recompute=False, engine='joblib', n_jobs=3)

    # jobtools.compute_job_list(convert_vhdr_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(eeg_viewer_job, run_keys, force_recompute=False, engine='loop')


convert_vhdr_job = jobtools.Job(precomputedir, 'convert_vhdr',convert_vhdr_params, convert_vhdr)
jobtools.register_job(convert_vhdr_job)

ica_figure_job = jobtools.Job(precomputedir, 'ica_figure', ica_figure_params, compute_ica_figure)
jobtools.register_job(ica_figure_job)

preproc_job = jobtools.Job(precomputedir, 'preproc',preproc_params, compute_preproc)
jobtools.register_job(preproc_job)

eeg_viewer_job = jobtools.Job(precomputedir, 'viewer', eeg_viewer_params, preproc_viewer)
jobtools.register_job(eeg_viewer_job)



if __name__ == '__main__':
    # test_convert_vhdr()
    # test_compute_ica_figure()
    # test_compute_preproc()
    # test_preproc_viewer()
    compute_all()
