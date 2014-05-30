from __future__ import division, print_function, absolute_import
import statemachine as fsm
import logging
import numpy as np
import sys
from PySide import QtGui
from waveforms import generate_waveform, waveforms

log = logging.getLogger('wavegen')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

try:
    import visa
except ImportError as e:
    log.error('please install pyvisa package')
    raise e
except OSError:
    log.error('could not find visa library; is a visa implementation installed?')

class Agilent332xxError(Exception): pass

class Agilent332xx(object):
    def __init__(self, instrument, **kwargs):
        self.me = instrument

        try:
            self.ask('*idn?')
        except Exception as e:
            log.error(e)
            raise e

    def write(self, msg):
        self.me.write(msg)
        log.log(logging.DEBUG-1, '<< %s', msg)

    def ask(self, msg):
        log.log(logging.DEBUG-1, '<< %s', msg)
        resp = self.me.ask(msg)
        log.log(logging.DEBUG-1, '>> %s', resp)
        return resp
        
    def send_waveform_macro(self, waveform):
        log.info('generating waveform')
        self.waveform, sample_rate = generate_waveform(waveform)
	
        log.info('clearing volatile memory')
        self.write('data:vol:cle')

        log.info('sending waveform')

        for i,segment in enumerate(self.waveform):
            nbytes = 2*len(segment)
            ndigits = len(str(nbytes))

            log.debug('data:arb:dac part{nsegment:02d}, #{n:d}{nbytes:d}'.format(
                nsegment=i, n=ndigits, nbytes=nbytes, data=segment.tostring()))

            self.write('data:arb:dac part{nsegment:02d}, #{n:d}{nbytes:d}{data:s}'.format(
                nsegment=i, n=ndigits, nbytes=nbytes, data=segment.tostring()))
            self.check_error()

        log.info('available waveforms: (data:vol:cat?)')
        log.info(self.ask('data:vol:cat?'))

        log.info('constructing sequence stark')

        binblock = 'stark'

        for i in xrange(len(self.waveform)):
            binblock += ',part{:02d},1,onceWaitTrig,lowAtStart,4'.format(i)

        nbytes = len(binblock)
        ndigits = len(str(nbytes))

        self.write('data:seq #{n:d}{nbytes:d}{binblock:s}'.format(
            n=ndigits, nbytes=nbytes, binblock=binblock))

        log.info('should be ' + str(np.sum([x.shape[0] for x in self.waveform])) + 
            ' points in sequence')
        log.info('got ' + self.ask('data:attr:poin? stark')) 

        log.info('asking free points: ' + self.ask('data:vol:free?'))

        log.info('selecting sequence')
        self.write('func:arb stark')
        self.write('func arb')

        self.check_error()

        self.write('outp 0')
        self.write('outp:load 2e3')
        self.write('volt:unit vpp')
        self.write('volt:offs 0')
        self.write('volt:high 1')
        self.write('volt:low 0')
        self.write('trig:sour ext')
        self.write('trig:del 0.0')

        self.write('func:arb:srate {rate:2.3f}'.format(rate=sample_rate))
        self.write('func:arb:filt norm')
        self.check_error()

    def set_delay(self, delay):
        val = float(delay)*1e-6
        self.write('trig:del {:.4e}'.format(val))

    def get_delay(self):
        return self.write('trig:del?')

    delay = property(get_delay, set_delay)
    del set_delay; del get_delay

    def set_voltage(self, voltage):
        val = float(voltage)
        log.info('changing output voltage to {:2.3f}'.format(val))
        self.write('volt:high {:2.4f}'.format(val))

    def get_voltage(self):
        return self.ask('volt:high?')

    voltage = property(get_voltage, set_voltage)
    del get_voltage; del set_voltage

    def send_reset_macro(self):
        self.write('*rst; *cls') 

    def enable(self):
        self.write('outp 1')

    def disable(self):
        self.write('outp 0')

    def check_error(self):
        resp = self.ask('syst:err?')
        if resp[:2] != u'+0': raise Agilent332xxError(resp)

class WavegenController(fsm.Machine):
    initial_state = 'preinit'

    @fsm.event
    def init(self):
        self.reason = ''
        self.ResourceManager = visa.ResourceManager()
        self.instruments = visa.get_instruments_list()
        
        # Create application
        self.app = QtGui.QApplication(sys.argv)

        # Create GUI
        self.gui = WavegenGui()
        # Connect log
        #self.log_handler = GuiLoggingHandler(self.gui.log)
        #self.log_handler.setFormatter(HtmlFormatter())
        #self.log_handler.setLevel(logging.DEBUG)
        #log.addHandler(self.log_handler)
        # Set instruments
        self.gui.device_chooser.addItems(self.instruments)
        self.gui.waveform_box.addItems(waveforms.keys())
        # Connect events
        self.connect_events()
        
        # Show dialog
        self.gui.show()

        if len(self.instruments) < 1: 
            success = False
            self.reason = 'did not get any instruments from visa'
        else:
            success = True

        # find device ids
        if success:
            yield 'preinit', 'wait1'
        else:
            yield 'preinit', 'fail_init'

    @fsm.after_transition('preinit', 'wait1')
    def invert_control(self):
        self.app.exec_()

    def choose_waveform(self, id):
        waveform_names = waveforms.keys()
        self.waveform = self.gui.waveform_box[waveform_names[id]]

    def connect_events(self):
        g = self.gui
        g.device_chooser.currentIndexChanged.connect(self.set_device)
        g.waveform_box.currentIndexChanged.connect(self.choose_waveform)
        g.load_waveform_btn.clicked.connect(self.load_waveform)
        g.delay_box.valueChanged.connect(self.set_delay)
        g.voltage_box.valueChanged.connect(self.set_output_voltage)
        g.enable_box.stateChanged.connect(self.handle_enable)

        g.close_callback = self.close
    
    def handle_enable(self):
        if self.gui.enable_box.isChecked():
            self.enable()
        else:
            self.disable()

    @fsm.event
    def set_device(self, id):
        try:
            self.instrument = Agilent332xx(visa.instrument(self.instruments[id]))
            self.instrument.send_reset_macro()
            success = True
        except visa.VisaIOError as e:
            log.error('timeout on instrument at ' +
                    self.instruments[id])
            log.error(e.message)
            self.reason = 'couldn\'t open that device'
            success = False

        if success: 
            yield ['wait1', 'device_chosen'], 'device_chosen'
        else:
            yield ['wait1', 'device_chosen'], 'wait1'

    @fsm.event
    def load_waveform(self):
        try:
            self.instrument.send_waveform_macro(self.waveform)
            success = True
        except Exception as e:
            self.reason = 'failed sending a waveform, message:' + e.message
            success = False

        if success:
            yield 'device_chosen', 'wait2'
        else:
            yield 'device_chosen', 'error'

    @fsm.event
    def set_delay(self, delay):
        try:
            self.instrument.delay = delay
            success = True
        except:
            success = False
        if not success:
            yield ['wait2', 'error'], 'error'

    @fsm.event
    def set_output_voltage(self, voltage):
        try:
            self.instrument.voltage = voltage
            success = True
        except: 
            success = False
        if not success:
            yield ['wait2', 'error'], 'error'

    @fsm.event
    def enable(self):
        try:
            self.instrument.enable()
            success = True
        except:
            success = False

        if success:
            yield ['wait2', 'running'], 'running'
        else:
            yield 'wait2', 'error'

    @fsm.event
    def disable(self):
        self.instrument.disable()
        success = True

        if success:
            yield 'running', 'wait2'
        else:
            yield 'error'

    @fsm.transition_to('error')
    def log_reason(self):
        log.error(self.reason)

    @fsm.event
    def close(self):
        log.info('closing')

        if self.state not in ['preinit']:
            del self.ResourceManager

            if self.state not in ['wait1']:
                del self.instrument

        yield '*', 'terminated'

class HtmlFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(record):
        return "this is a record!"
        formatted = logging.Formatter.format(record)

        if record.level <= logging.DEBUG:
            color = "#fee5d9"
        elif record.level <= logging.INFO:
            color = "#fcae91"
        elif record.level <= logging.WARNING:
            color = "#fb6a4a"
        elif record.level <= logging.CRITICAL:
            color = "#cb181d"
        else:
            color = "#ffffff"

        res = "<p style=\"color: {color:s}\">{msg:s}</p>".format(
            color, formatted)
        print(res)
        return res

class GuiLoggingHandler(logging.Handler):
    def __init__(self, target, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.target = target

    def emit(self, record):
        formatted = logging.Handler.format(self, record)
        print(formatted)
        self.target.append(formatted)

class WavegenGui(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        self.close_callback = None
        self.make_main_frame()
    
    def make_main_frame(self):
        self.setWindowTitle("Stark Waveform Controller")
        self.main_frame = QtGui.QWidget()
        
        hgroup = QtGui.QGroupBox("Instructions")
        hlayout = QtGui.QHBoxLayout()

        #self.log = QtGui.QTextEdit()
        #for w in [hgroup, self.log]:
        #    hlayout.addWidget(w)
        hlayout.addWidget(hgroup)

        form = QtGui.QFormLayout()
        
        self.device_chooser = QtGui.QComboBox()
        self.waveform_box = Qt.QComboBox()
        self.load_waveform_btn = QtGui.QPushButton("Send!")
        self.delay_box = QtGui.QDoubleSpinBox()
        self.voltage_box = QtGui.QDoubleSpinBox()
        self.enable_box = QtGui.QCheckBox("Enabled")

        self.voltage_box.setRange(0.01, 10.0)
        self.voltage_box.setSingleStep(0.1)
        self.delay_box.setRange(0.0, 20000.)
        self.delay_box.setSingleStep(100)

        form.addRow("1. Choose device", self.device_chooser)
        form.addRow("2. Choose waveform", self.waveform_box)
        form.addRow("3. Send waveform", self.load_waveform_btn)
        form.addRow("4. Set delay (us)", self.delay_box)
        form.addRow("5. Set output volage", self.voltage_box)
        form.addRow("6. Enable output", self.enable_box)
        
        hgroup.setLayout(form)
        self.main_frame.setLayout(hlayout)
        self.setCentralWidget(self.main_frame)

    def closeEvent(self, ev):
        if self.close_callback:
            self.close_callback()
        
        super(QtGui.QMainWindow, self).closeEvent(ev)

if __name__ == '__main__':
    ctrl = WavegenController()
    ctrl.init()
