from __future__ import division, print_function
import statemachine as fsm
import logging
import numpy as np

log = logging.getLogger('wavegen')
log.addHandler(logging.NullHandler())

try:
    import visa
    _rm = visa.ResourceManager()
except ImportError as e:
    log.error('please install pyvisa package')
    raise e
except OSError:
    log.error('could not find visa library; is a visa implementation installed?')

class Agilent332xx(object):
    def __init__(self, name, **kwargs):
        self.me = _rm.get_instrument(name, **kwargs)

        try:
            self.ask('*idn?')
        except Exception as e:
            log.error(e)
            raise e

    def write(self, *args, **kwargs):
        self.me.write(*args, **kwargs)

    def ask(self, *args, **kwargs):
        return self.me.ask(*args, **kwargs)
        
    def send_load_waveform_macro(self):
        waveform = generate_waveform()

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
        self.write('outp 0')
        self.write('outp:load 2e3')
        self.write('volt:unit vpp')
        self.write('volt:offs 0')
        self.write('volt:high 1')
        self.write('volt:low 0')
        self.write('trig:sour ext')
        self.write('trig:del 0.440e-6')

    def enable(self):
        self.write('outp 1')

    def disable(self):
        self.write('outp 0')


class WavegenController(fsm.Machine):
    initial_state = 'preinit'

    @fsm.event
    def init(self):
        reason = ''
        self.instruments = visa.get_instruments_list()
        
        if len(self.instruments) < 1: 
            success = False
            reason = 'did not get any instruments from visa'
        else:
            success = True

        # find device ids
        if success:
            yield 'preinit', 'wait1'
        else:
            yield 'preinit', 'fail_init'

    @fsm.event
    def set_device(self, id):
        try:
            self.instrument = Agilent332xx(self.instruments[id])
            self.instrument.send_reset_macro()
            success = True
        except visa.VisaIOError as e:
            log.error('timeout on instrument at ' +
                    self.instruments[id])
            log.error(e.message)
            reason = 'couldn\'t open that device'
            success = False

        if success: 
            yield ['wait1', 'device_chosen'], 'device_chosen'
        else:
            yield ['wait1', 'device_chosen'], 'wait1'

    @fsm.event
    def load_waveform(self, which):
        try:
            self.instrument.send_waveform_macro()
            success = True
        except:
            reason = 'failed sending a waveform'
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
        success = self.instrument.enable()

        if success:
            yield ['wait2', 'running'], 'running'
        else:
            yield 'wait2', 'error'

    @fsm.event
    def disable(self):
        success = self.instrument.disable()
        if success:
            yield 'running', 'wait2'
        else:
            yield 'error'

