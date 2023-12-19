# TOOLS FOR ARTIFACT DETECTION AND REPLACEMENT

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal

def detect_cross(sig, threshold):
    """
    Detect crossings
    ------
    inputs =
    - sig : numpy 1D array
    - show : plot figure showing rising zerox in red and decaying zerox in green (default = False)
    output =
    - pandas dataframe with index of rises and decays
    """
    rises, = np.where((sig[:-1] <=threshold) & (sig[1:] >threshold)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=threshold) & (sig[1:] <threshold)) # detect where sign inversion from + to -

    if rises.size != 0:

        if rises[0] > decays[0]: # first point detected has to be a rise
            decays = decays[1:] # so remove the first decay if is before first rise
        if rises[-1] > decays[-1]: # last point detected has to be a decay
            rises = rises[:-1] # so remove the last rise if is after last decay

        return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T
    else:
        return None

def compute_rms(x):
    """Fast root mean square."""
    n = x.size
    ms = 0
    for i in range(n):
        ms += x[i] ** 2
    ms /= n
    return np.sqrt(ms)

def sliding_rms(x, sf, window=0.5, step=0.2, interp=True):
    """
    Compute a sliding root mean square 

    ----------
    Parameters
    ----------
    - x : np.array
        Signal
    - sf : float
        Sampling frequency
    - window : float
        Duration of the sliding window in seconds
    - step : float
        Time duration between steps
    - interp : bool
        Interpolate to return a vector of same size that input signal

    -------
    Returns
    -------
    - t : np.array
        Time vector of the returned trace
    - out : np.array
        Signal (trace) smoothed by RMS
    """

    halfdur = window / 2
    n = x.size
    total_dur = n / sf
    last = n - 1
    idx = np.arange(0, total_dur, step)
    out = np.zeros(idx.size)

    # Define beginning, end and time (centered) vector
    beg = ((idx - halfdur) * sf).astype(int)
    end = ((idx + halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end[end > last] = last
    # Alternatively, to cut off incomplete windows (comment the 2 lines above)
    # mask = ~((beg < 0) | (end > last))
    # beg, end = beg[mask], end[mask]
    t = np.column_stack((beg, end)).mean(1) / sf



    # Now loop over successive epochs
    for i in range(idx.size):
        out[i] = compute_rms(x[beg[i] : end[i]])

    # Finally interpolate
    if interp and step != 1 / sf:
        f = interpolate.interp1d(t, out, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        t = np.arange(n) / sf
        out = f(t)

    return t, out

def compute_artifact_features(inds, srate):
    artifacts = pd.DataFrame()
    artifacts['start_ind'] = inds['rises'].astype(int)
    artifacts['stop_ind'] = inds['decays'].astype(int)
    artifacts['start_t'] = artifacts['start_ind'] / srate
    artifacts['stop_t'] = artifacts['stop_ind'] / srate
    artifacts['duration'] = artifacts['stop_t'] - artifacts['start_t']
    return artifacts


def detect_artifacts(sig, srate, n_deviations = 5, low_freq = 40 , high_freq = 150, wsize = 1, step = 0.2):
    
    import ghibtools as gh
    
    sig_filtered = gh.iirfilt(sig, srate, low_freq, high_freq, ftype = 'bessel', order = 2)
    t, sig_cross = sliding_rms(sig_filtered, sf=srate, window = wsize, step = step)   
    pos = np.median(sig_cross)
    dev = gh.mad(sig_cross)
    detect_threshold = pos + n_deviations * dev
    times = detect_cross(sig_cross, detect_threshold)
    if not times is None:
        artifacts = compute_artifact_features(times, sig_cross, srate)
            
        return artifacts 
    else:
        return None
    
    
def insert_noise(sig, srate, chan_artifacts, freq_min=30., margin_s=0.2, seed=None):
    sig_corrected = sig.copy()

    margin = int(srate * margin_s)
    up = np.linspace(0, 1, margin)
    down = np.linspace(1, 0, margin)
    
    noise_size = np.sum(chan_artifacts['stop_ind'].values - chan_artifacts['start_ind'].values) + 2 * margin * chan_artifacts.shape[0]
    
    # estimate psd sig
    freqs, spectrum = signal.welch(sig, nperseg=noise_size, nfft=noise_size, noverlap=0, scaling='spectrum', window='box', return_onesided=False, average='median')
    
    spectrum = np.sqrt(spectrum)
    
    # pregenerate long noise piece
    rng = np.random.RandomState(seed=seed)
    
    long_noise = rng.randn(noise_size)
    noise_F = np.fft.fft(long_noise)
    #long_noise = np.fft.ifft(np.abs(noise_F) * spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = np.fft.ifft(spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = long_noise.astype(sig.dtype)
    sos = signal.iirfilter(2, freq_min / (srate / 2), analog=False, btype='highpass', ftype='bessel', output='sos')
    long_noise = signal.sosfiltfilt(sos, long_noise, axis=0)
    
    
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=0)
    rms_sig = np.median(filtered_sig**2)
    rms_noise = np.median(long_noise**2)
    factor = np.sqrt(rms_sig) / np.sqrt(rms_noise)
    long_noise *= factor
    
    
    noise_ind = 0
    for _, artifact in chan_artifacts.iterrows():
        ind0, ind1 = artifact['start_ind'], artifact['stop_ind']
        
        n_samples = ind1 - ind0 + 2 * margin
        
        sig_corrected[ind0:ind1] = 0
        sig_corrected[ind0-margin:ind0] *= down
        sig_corrected[ind1:ind1+margin] *= up
        
        noise = long_noise[noise_ind: noise_ind + n_samples]
        noise_ind += n_samples
        
        noise += np.linspace(sig[ind0-1-margin], sig[ind1+1+margin], n_samples)
        noise[:margin] *= up
        noise[-margin:] *= down
        
        sig_corrected[ind0-margin:ind1+margin] += noise
        
    
    return sig_corrected


