import numpy as np
import os
import mne
import xarray as xr
import matplotlib.pyplot as plt
import ghibtools as gh
from params import *
from bibliotheque import *
from configuration import *
import physio
from compute_cycle_signal import cycle_signal_job
import jobtools

# CYCLE SIGNAL FRAME JOB

def cycle_signal_frames(sub, **p):
    
    sub_folder = 'Cycle_Signal_Video_2'
    
    pos = get_pos()
    
    sess = ['baseline','music','odor']
    
    das = None
    for ses in sess:
        da = cycle_signal_job.get(sub, ses)['cycle_signal']
        if das is None:
            das = gh.init_da({'ses':sess, 'chan':da.coords['chan'].values, 'phase':da.coords['phase'].values})
        das.loc[ses,:,:] = da.values
        
    eeg = das.sel(chan = eeg_chans)
    
    phases = np.arange(0, eeg['phase'].size, 1)

    vmin = eeg.min()
    vmax = eeg.max()

    chan_line = p['chan_line_signal']
    chan_vmin = das.loc[:,chan_line,:].min()
    chan_vmax = das.loc[:,chan_line,:].max()

    resp_chan = p['resp_chan']
    resp_vmin = das.loc[:,resp_chan,:].min()
    resp_vmax = das.loc[:,resp_chan,:].max()
    
    resp_mouth_chan = 'resp_mouth'
    resp_mouth_vmin = das.loc[:,resp_mouth_chan,:].min()
    resp_mouth_vmax = das.loc[:,resp_mouth_chan,:].max()

    fontsize_titles = 15
    
    xvline = int(p['cycle_signal_params']['segment_ratios'] * phases.size)

    folder = base_folder / 'Figures' / sub_folder / sub
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    for phase in phases:
        # fig, axs = plt.subplots(nrows = 5, ncols = len(sess), figsize = (30,20))
        fig, axs = plt.subplots(nrows = 3, ncols = len(sess), figsize = (15,10))
        fig.suptitle(f'Participant : {sub}', fontsize = 25, y = 1.02)

        for c, ses in enumerate(sess):
            ax = axs[0,c]
            im, cn = mne.viz.plot_topomap(data = eeg[c,:,int(phase)].values , pos = pos, names = eeg_chans, axes = ax, show = False, vlim = (vmin,vmax))
            ax.set_title(ses, fontsize = fontsize_titles + 5)

            ax = axs[1,c]
            chan_sig = eeg.loc[ses,chan_line,:].values
            ax.plot(chan_sig, color = 'k')
            ax.scatter(phase, chan_sig[int(phase)], color = 'r', lw=3)
            ax.axvline(xvline, color = 'g')
            ax.set_title(f'EEG signal from {chan_line}', fontsize = fontsize_titles)
            ax.axis('off')
            ax.set_ylim(chan_vmin, chan_vmax)

            ax = axs[2,c]
            resp_sig = das.loc[ses,resp_chan,:].values
            ax.plot(resp_sig, color = None, lw = 2)
            ax.scatter(phase, resp_sig[int(phase)], color = 'r', lw=3)
            ax.axvline(xvline, color = 'g')
            ax.axis('off')
            ax.set_ylim(resp_vmin, resp_vmax)
            ax.set_title(f'Respiratory signal', fontsize = fontsize_titles)
            
            # ax = axs[3,c]
            # mouth_sig = das.loc[ses,resp_mouth_chan,:].values
            # ax.plot(mouth_sig, color = 'm', lw = 2)
            # ax.scatter(phase, mouth_sig[int(phase)], color = 'r', lw=3)
            # ax.axvline(xvline, color = 'g')
            # ax.axis('off')
            # ax.set_ylim(resp_mouth_vmin, resp_mouth_vmax)
            
            # ax = axs[4,c]
            # heart_sig = das.loc[ses,'heart',:].values
            # ax.plot(heart_sig, color = 'r', lw = 2)
            # ax.scatter(phase, heart_sig[int(phase)], color = 'r', lw=3)
            # ax.axvline(xvline, color = 'g')
            # ax.axis('off')

        file =  folder / f'im_{phase}.png'
        fig.savefig(file, bbox_inches = 'tight')
        plt.close('all')
    return xr.Dataset()

def test_cycle_signal_frames():
    sub = 'P11'
    ds = cycle_signal_frames(sub, **cycle_signal_frames_params)
    print(ds)

cycle_signal_frames_job = jobtools.Job(precomputedir, 'cycle_signal_frames', cycle_signal_frames_params, cycle_signal_frames)
jobtools.register_job(cycle_signal_frames_job)

# MAKE VIDEOS
def make_video_cycle_signal(sub):
    
    import cv2
    import glob
    
    sub_folder = 'Cycle_Signal_Video_2'
    
    step = video_params['step']
    video_duration = video_params['video_duration']
    
    folder = base_folder / 'Figures' / sub_folder / sub
    folder = str(folder)
    
    n_images_generated = len(glob.glob(f'{folder}/im*.png'))
    
    images = [f'{str(folder)}/im_{i}.png' for i in np.arange(0, n_images_generated, step)]
    
    n_images = len(images)
    fps = int(n_images / video_duration)
    
    folder_video = base_folder / 'Figures' / sub_folder / 'videos'
    folder_video = str(folder_video)
    video_name = f'{folder_video}/video_{sub}.avi'
    video_name = str(video_name)

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

# COMPUTE ALL
def compute_all():
    run_keys = [(sub,) for sub in subject_keys]
    # jobtools.compute_job_list(cycle_signal_frames_job, run_keys, force_recompute=True, engine='loop')
    jobtools.compute_job_list(cycle_signal_frames_job, run_keys, force_recompute=True, engine='joblib', n_jobs = 31)
    # jobtools.compute_job_list(cycle_signal_frames_job, run_keys, force_recompute=True, engine='slurm',
    #                           slurm_params={'cpus-per-task':'1', 'mem':'1G', },
    #                           module_name='video_cycle_signal',
    #                           )
    
def make_all_videos():
    for sub in subject_keys:
        print(sub)
        make_video_cycle_signal(sub) 
    

if __name__ == '__main__':
    # test_cycle_signal_frames()
    # compute_all()
    
    # make_video_cycle_signal('P02')
    make_all_videos()
    
        