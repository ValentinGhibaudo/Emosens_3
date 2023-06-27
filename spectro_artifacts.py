import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from params import *
from bibliotheque import *
from preproc import convert_vhdr_job

meta = get_metadata()

p = {
    'wsize_sec':1, # welch window size in seconds
    'overlap_prop':0.75, # proportion of overlap between windows
    'nfft_factor':2, # zero-padding : nfft = nperseg * nfft_factor
    'lowcut':10,
    'highcut':500,
    'chans':['Fz','Pz','Oz','FC5','FC6','ECG','RespiNasale']
}


for sub in subject_keys:
    
    sess = ['baseline'] + list(meta.loc[sub,:])
    
    fig, axs = plt.subplots(nrows = len(p['chans']), ncols = len(sess), figsize = (20,25), constrained_layout = True)
    fig.suptitle(sub, fontsize = 20)
    
    for c, ses in enumerate(sess):
        
        run_key = f'{sub}_{ses}'
        raw = convert_vhdr_job.get(run_key)['raw']
        srate = raw.attrs['srate']
        
        nperseg = int(srate * p['wsize_sec'])
        overlap_prop = p['overlap_prop']
        noverlap = nperseg // (1 / overlap_prop)
        nfft = nperseg * p['nfft_factor']
        
        for r, chan in enumerate(p['chans']):
            
            ax = axs[r,c]

            sig = raw.sel(chan = chan).values 
            f, t , Sxx = signal.spectrogram(sig, fs = srate , nperseg = nperseg , noverlap = noverlap, nfft = nfft, scaling = 'spectrum')

            ax.pcolormesh(t,f,np.log(Sxx))
            
            ax.set_title(f'{ses} - {chan}')
            
            if c == 0:
                ax.set_ylabel('Freq [Hz]')
                
            if r == len(p['chans']) -1:
                ax.set_xlabel('Time [s]')
            
    fig.savefig(base_folder / 'Figures' / 'Autres' / 'artefact_mouche' / f'{sub}.png', dpi = 300)
    plt.close()