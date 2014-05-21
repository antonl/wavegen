from __future__ import division, absolute_import, print_function
import numpy as np

def sigmoidal_500Hz_1off_1on(phase=4, plot=False):
    smoothing = 10./5.  # smooth out the ends at the expense of larger derivative
    trigger_rate = 500 # Hz
    segment_period = 1./trigger_rate - 1e-6 # A bit less than expected period
    sample_rate = 5e6 # 10 MHz

    n_samples = int(segment_period*sample_rate)

    x = np.linspace(0, 20, int(2*n_samples))# ms
    gd1 = lambda x: 2/np.pi*np.arctan2(np.exp(smoothing*x),1)
    gd2 = lambda x: 1 - 2/np.pi*np.arctan2(np.exp(smoothing*x),1)
    sq = lambda x, width: 0.5 * (np.sign(x) - np.sign(x - width))

    fn = lambda x: gd1(x - phase)*sq(x + 5 - phase, 10) + gd2(x - 10 - phase)*sq(x - 5 - phase, 10)
    fnx = fn(x)

    if plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1)
        t = trigger_rate*x/sample_rate * 2*500 # ms
        ax1.plot(t, fnx)
        ax1.plot(t[:-2], np.diff(fnx[:-1])*1000)
        ax1.grid()
        ax1.set_xlabel('time/ms')
        ax1.set_ylabel('amplitude/fullscale')
        ax1.set_ylim(-0.5, 1.1)

        n = np.arange(2*n_samples)
        ax2.plot(n[:n_samples], fnx[:n_samples], linewidth=3)
        ax2.plot(n[n_samples:2*n_samples], fnx[n_samples:2*n_samples], linewidth=3)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid()
        #fig.show()
        
    max_int = 1<<15 - 1 # signed integers

    return ((np.array(fnx[:n_samples]*max_int, dtype='>i2'), 
            np.array(fnx[n_samples:]*max_int, dtype='>i2')), 
            sample_rate)

def sigmoidal_6phase_6off_6on(plot=False):
    smoothing = 10./5.  # smooth out the ends at the expense of larger derivative
    segment_period = 12e-3 - 1e-6 # s, a bit less than expected period
    sample_rate = 5e6 # 5 MHz

    n_samples = int(segment_period*sample_rate)

    x = np.linspace(0, 2, int(2*n_samples))
    
    sin = 0.5*np.sin(2*np.pi*6*x[:n_samples//6] - (2*np.pi*0.25)) + 0.5 # halfwave of sin
    
    waveform = np.ones_like(x)
    waveform[:n_samples//12] = sin[:n_samples//12]
    waveform[n_samples:n_samples+sin[n_samples//12:].shape[0]] = sin[n_samples//12:]
    waveform[n_samples+n_samples//12:] = 0

    max_int = 1<<15 - 1 # signed integers

    return np.array(waveform[:n_samples]*max_int, dtype='>i2'), \
            np.array(waveform[n_samples:]*max_int, dtype='>i2'), \
            sample_rate


waveforms = {
    'sigmoidal 500Hz 1 off 1 on': sigmoidal_500Hz_1off_1on,
    'sigmoidal 500Hz 6 off 6 on': sigmoidal_6phase_6off_6on,
    }

def generate_waveform(waveform):
    if not waveform in waveforms.keys():
        raise ValueError('incorrect waveform chosen')

    return waveforms[waveform]()

