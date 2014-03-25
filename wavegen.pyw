from __future__ import division, print_function
import sys
from PySide import QtGui, QtCore
from wavegen_ui import Ui_MainWindow
import logging
import logging.handlers

import visa
import numpy as np
from functools import partial
#import scipy.signal

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure

from matplotlib.pyplot import *

logger = logging.getLogger('wavegen')
logger.setLevel(logging.DEBUG)
mem = logging.handlers.MemoryHandler(1)
logger.addHandler(mem)

sample_rate = 10.e6 # 5 MSa/s rate
length = 499e-6 # length of gaussian in ms

def gen_waveform():
    smoothing = 10./5.  # smooth out the ends at the expense of larger derivative
    trigger_rate = 1000 # Hz
    segment_period = 1./trigger_rate - 1e-6 # A bit less than 1 ms (Trigger rate 1 kHz)

    n_samples = segment_period*sample_rate

    print(n_samples//2)
    x = np.linspace(-7.5, 32.5, int(4*n_samples))# ms
    gd1 = lambda x: 2/np.pi*np.arctan2(np.exp(smoothing*x),1)
    gd2 = lambda x: 1 - 2/np.pi*np.arctan2(np.exp(smoothing*x),1)
    sq = lambda x, width: 0.5 * (np.sign(x) - np.sign(x - width))

    fn = lambda x: gd1(x)*sq(x + 5, 10) + gd2(x - 10)*sq(x - 5, 10) + gd1(x - 20)*sq(x - 15, 10) + gd2(x - 30)*sq(x - 25, 10)

    t = trigger_rate*x/sample_rate*2 * 1000 # ms
    fnx = fn(x)
    plot(t, fnx)
    plot(t[:-2], np.diff(fnx[:-1])*1000)
    vlines([x + 1.5  for x in range(-1, 5, 2)], -0.5, 1, color='g', linewidth=3)
    vlines([x + 0.5  for x in range(-1, 5, 1)], -0.5, 1.2, color='r')

    grid()
    show()
    xlabel('time/ms')
    ylabel('amplitude/fullscale')

    n = np.arange(2*n_samples)
    print(n)
    n_samplesd2 = int(n_samples//2)
    plot(n[:n_samplesd2], fnx[:n_samplesd2], linewidth=3)
    plot(n[n_samplesd2:n_samples], fnx[n_samplesd2:n_samples], linewidth=3)
    plot(n[n_samples:n_samples + n_samplesd2], fnx[n_samples:n_samples + n_samplesd2], linewidth=3)
    plot(n[n_samples + n_samplesd2:2*n_samples], fnx[n_samples + n_samplesd2:2*n_samples], linewidth=3)
    grid()
    show()

    max_uint = 2<<15 - 1

    # return parts of the sequence
    return (np.array(fnx[:n_samplesd2]*max_uint, dtype='>i2'),
            np.array(fnx[n_samplesd2:n_samples]*max_uint, dtype='>i2'),
            np.array(fnx[n_samples:n_samples + n_samplesd2]*max_uint, dtype='>i2'),
            np.array(fnx[n_samples + n_samplesd2:2*n_samples]*max_uint, dtype='>i2'))

'''
def gen_trap(rate=rate, length=length):
    logger.debug('making trapezoidal window')
    n = np.arange(int(rate*length))
    buf = len(n)/20
    q = (len(n) - 2*buf)/4

    wave = np.zeros((len(n), ))
    wave[buf:buf+q] = np.linspace(0, 1, num=q)
    wave[buf+q:buf+3*q] = 1
    wave[buf+3*q:buf+4*q] = np.linspace(1, 0, num=q)
    
    gaussian = np.exp(-(n - len(n)/2)**2/(len(n)/25)**2)
    gaus = partial(np.convolve, gaussian, mode='same')
    swave = gaus(wave)
    # normalize window
    swave *= 1./swave.max()
'''
class Ui_Wavegen(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Ui_Wavegen, self).__init__()
        self.setupUi(self)
        self.textLog.setFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        mem.setTarget(self.textLog)

        self.instrument = None
        self.devID.currentIndexChanged.connect(self.on_instrument_change)
        self.init_list()

        self.sendWaveform.clicked.connect(self.on_send_waveform_clicked)
        self.outputEnabled.stateChanged.connect(self.on_click_enabled)
        self.outputVoltage.valueChanged.connect(self.on_output_voltage_changed)
        self.delay.valueChanged.connect(self.on_send_delay)

    def on_instrument_change(self):
        logger.debug('instrument choice changed')
        logger.debug('>   ' + self.devID.currentText())

        self.sendWaveform.setEnabled(True)

        try:
            self.instrument = visa.instrument(self.devID.currentText(),
                    timeout=1)
            inst = self.instrument
            logger.debug('resetting instrument')
            inst.write('*rst; *cls'); inst.write('outp 0')
            logger.debug('response: ' + inst.ask('syst:err?'))
        except visa.VisaIOError as e:
            logger.error('timeout on instrument at ' +
                    self.devID.currentText())
            logger.error(e.message)
        except Exception as e:
            logger.exception(e)

    def on_output_voltage_changed(self):
        logger.info('changed output voltage')
        val = float(self.outputVoltage.value())
        self.instrument.write('volt:high {:2.4f}'.format(val))
        logger.info('response: ' + self.instrument.ask('syst:err?'))

    def on_send_waveform_clicked(self):
        if(getattr(self, 'waveform', None) is not None):
            logger.info('reusing precalculated waveform')
        else:
            logger.info('creating waveform')
            self.generate_waveform()

        try:
            inst = self.instrument
            logger.debug('sending smoothtrap waveform')

            for i,w in enumerate(self.waveform):
                nbytes = 2*len(w)
                ndigits = len(str(nbytes))

                logger.debug('data:arb:dac part{which:d}, #{n:d}{nbytes:d}'.format(
                    which=i, n=ndigits, nbytes=nbytes, data=w.tostring()))

                inst.write('data:arb:dac part{which:d}, #{n:d}{nbytes:d}{data:s}'.format(
                    which=i, n=ndigits, nbytes=nbytes, data=w.tostring()))
                logger.debug('response: ' + inst.ask('syst:err?'))

            logger.info('available waveforms: (data:vol:cat?)')
            logger.info('>   ' + inst.ask('data:vol:cat?'))

            logger.debug('constructing sequence stark')

            binblock = \
                'stark,part0,1,onceWaitTrig,lowAtStart,4,' + \
                'part1,1,onceWaitTrig,lowAtStart,4,' + \
                'part2,1,onceWaitTrig,lowAtStart,4,' + \
                'part3,1,onceWaitTrig,lowAtStart,4'

            nbytes = len(binblock)
            ndigits = len(str(nbytes))

            inst.write('data:seq #{n:d}{nbytes:d}{binblock:s}'.format(
                n=ndigits, nbytes=nbytes, binblock=binblock))

            logger.info('should be ' + str(np.sum(self.waveform)) + 
                ' points in sequence')
            logger.info('got ' + inst.ask('data:attr:poin? stark')) 

            logger.info('selecting sequence')
            inst.write('func:arb stark')
            inst.write('func arb')
            logger.debug('response: ' + inst.ask('syst:err?'))

            logger.debug('setting output parameters')
            inst.write('outp:load 2e3')
            inst.write('volt:unit vpp')
            inst.write('func:arb:srate {rate:2.3f}'.format(rate=sample_rate))
            inst.write('volt:offs 0')
            inst.write('volt:high 1')
            inst.write('volt:low 0')
            inst.write('trig:sour ext')
            inst.write('trig:del 0.770e-6')

            logger.debug('volts ptp: ' + inst.ask('func:arb:ptp?'))

        except Exception as e:
            logger.debug('Error code: ' + inst.ask('syst:err?'))
            logger.exception(e)

        # disable button
        self.sendWaveform.setEnabled(False)

    def on_send_delay(self):
        logger.info('changed trigger delay')
        val = float(self.delay.value())*1e-6
        self.instrument.write('trig:del {:.4e}'.format(val))
        logger.info('current delay: ' + self.instrument.ask('trig:del?'))

    def on_click_enabled(self):
        if self.outputEnabled.isChecked() is True:
            self.instrument.write('outp 1')
        else:
            self.instrument.write('outp 0')

    def generate_waveform(self):
        self.waveform = gen_waveform()

    def init_list(self):
        insts = visa.get_instruments_list()

        logger.info('Found ' + str(len(insts)) + ' addresses')
        for i in insts:
            logger.info('>  ' + i)

        self.devID.addItems(insts)

app = QtGui.QApplication(sys.argv)
window = Ui_Wavegen()
window.show()

sys.exit(app.exec_())
