import sys
from PySide import QtGui, QtCore
from wavegen_ui import Ui_MainWindow
import logging
import logging.handlers

import visa
import numpy as np
from functools import partial
#import scipy.signal

'''
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
'''

logger = logging.getLogger('wavegen')
logger.setLevel(logging.DEBUG)
mem = logging.handlers.MemoryHandler(1)
logger.addHandler(mem)

rate = 5.e6 # 5 MSa/s rate
length = 499e-6 # length of gaussian in ms

def gen_waveform(rate=rate, length=length):
    '''generates a on-period waveform that has maximum value of 1.0 and
    rescales to the unsigned int range
    '''
    n = np.arange(int(rate*length))
    swave = np.sin(pi/(rate*length) * n)
    return np.asarray(swave*(2**15-1), dtype='>i2')

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

            nbytes = 2*len(self.waveform)
            ndigits = len(str(nbytes))
            
            logger.debug('data:arb:dac smoothtrap, #{n:d}{nbytes:d}'.format(
                n=ndigits, nbytes=nbytes))

            inst.write('data:arb:dac smoothtrap, #{n:d}{nbytes:d}{data:s}'.format(
                n=ndigits, nbytes=nbytes, data=self.waveform.tostring()))
            logger.debug('response: ' + inst.ask('syst:err?'))

            logger.debug('sending dc waveform')

            dc = np.zeros(int(len(self.waveform)/4), dtype='>i2')
            nbytes = 2*len(dc)
            ndigits = len(str(nbytes))

            logger.debug('data:arb:dac dc, #{n:d}{nbytes:d}'.format(
                n=ndigits, nbytes=nbytes))

            inst.write('data:arb:dac dc, #{n:d}{nbytes:d}{data:s}'.format(
                n=ndigits, nbytes=nbytes, data=dc.tostring()))
            logger.debug('response: ' + inst.ask('syst:err?'))
            
            logger.info('available waveforms: (data:vol:cat?)')
            logger.info('>   ' + inst.ask('data:vol:cat?'))

            logger.debug('constructing sequence stark')

            binblock = \
                'stark,smoothtrap,1,onceWaitTrig,lowAtStart,4,' + \
                'dc,10,onceWaitTrig,highAtStart,4'

            nbytes = len(binblock)
            ndigits = len(str(nbytes))

            inst.write('data:seq #{n:d}{nbytes:d}{binblock:s}'.format(
                n=ndigits, nbytes=nbytes, binblock=binblock))
            logger.info('should be ' + str(len(dc) + len(self.waveform)) + 
                ' points in sequence')
            logger.info('got ' + inst.ask('data:attr:poin? stark')) 

            logger.info('selecting sequence')
            inst.write('func:arb stark')
            inst.write('func arb')
            logger.debug('response: ' + inst.ask('syst:err?'))

            logger.debug('setting output parameters')
            inst.write('outp:load 2e3')
            inst.write('volt:unit vpp')
            inst.write('func:arb:srate {rate:2.3f}'.format(rate=rate))
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
        self.waveform = gen_waveform(rate, length)

    '''
    def verify_plot(self, waveform):
        # generate the plot
        fig = Figure(figsize=(400,400), dpi=72, facecolor=(1,1,1), 
                edgecolor=(0,0,0))
        ax = fig.add_subplot(111)
        #ax.set_ylim([0, 1.1])
        ax.set_xlim([0, len(waveform)])
        ax.plot(waveform)
        ax.set_xlabel('samples')
        ax.set_ylabel('amplitude')
        # generate the canvas to display the plot
        canvas = FigureCanvas(fig)

        self.win = QtGui.QMainWindow()
        # add the plot canvas to a window
        self.win.setCentralWidget(canvas)
        self.win.show()
    '''

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
