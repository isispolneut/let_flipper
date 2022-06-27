from PyDAQmx import *  # pylint: disable=W0614
import numpy as np
from time import time, sleep
import os

"""PyDAQmx provides an OOP wrapper to the DAQmx C Library. All commands
parallel their C equivalent one to one. For documentation on the functionality
of each DAQmx command, see

    /Public Documents/National Instruments/NI-DAQ/Documentation.
"""

SAMPRATE = 1e6  # Define a sampling rate such that 1 sample = 1 microsecond
FLIP_MAX = 4.5
COMP_CONST = 2.5
COMP_MAX = 3.0
TRIGGER_CHAN = "APFI0"

#---------------------------------------------------------------------------       

def ZeroOutput(flipper_chan,comp_chan):
    """Sets all the analog output channels of the DAQ card to 0V"""

    for chan in [flipper_chan, comp_chan]:
        sleep(0.2)

        task = Task()

        wrote = int32()

        task.CreateAOVoltageChan(chan,
                                 "",
                                 0, 10,
                                 DAQmx_Val_Volts,  # pylint: disable=E0602
                                 None
                                 )

        task.CfgSampClkTiming("",
                              5e4,
                              DAQmx_Val_Rising,      # pylint: disable=E0602
                              DAQmx_Val_FiniteSamps,  # pylint: disable=E0602
                              2
                              )

        task.WriteAnalogF64(2,
                            False,
                            1e-2,
                            DAQmx_Val_GroupByChannel,  # pylint: disable=E0602
                            np.array(
                                [0.0], dtype=np.float64),  # pylint: disable=E1101
                            wrote,
                            None
                            )

        task.StartTask()
        task.ClearTask()

#---------------------------------------------------------------------------       

class FlipperTask(Task):
    """Task that drives the current to the flipping coil.

    This task is triggered by the Analog Input start trigger, which is in turn triggered
    by a timing signal supplied to the terminal "APFI0". This trick is required as two
    tasks cannot both reserve APFI0, but conveniently also syncs the write and read tasks
    for us. Otherwise, this is a bogstandard retriggerable regenerated task that writes
    the functional form we want.
    """

    def __init__(self, flip_chan, mode, flipper_params):
        Task.__init__(self)
        self.flip_chan = flip_chan
        self.mode = mode
        self.flipper_params = flipper_params
        
        self.waveform_lib = {'static': self.static,
                             'steps': self.steps,
                             'analytical': self.analytical,
                             'file': self.file}
        
   
        self.t = np.linspace(1e-6, 92.5e-3, num=int(92.5e3))      # Assuming 10^5 samples per frame
        self.padding = np.ones((int(7.5e3),))         # Prepare flipper for next pulse
        
        self.waveform_lib[mode](flipper_params)
        self.wrote = int32()

        """DAQmx procedure:

        1) Create Analog Output channel.
        2) Configure Sample Clock to time a sample every microsecond.
        3) Set the task to be regenerative (restores the buffered values after writing)
           and retriggerable (allows the write to trigger multiple times).
        4) Configure a digital edge start trigger to trigger off "ai/StartTrigger".
        5) Write the desired signal to the buffer.
        """

        self.CreateAOVoltageChan(self.flip_chan,
                                 "",
                                 0, 10,
                                 DAQmx_Val_Volts,  # pylint: disable=E0602
                                 None)

        self.CfgSampClkTiming("",
                              SAMPRATE,
                              DAQmx_Val_Rising,      # pylint: disable=E0602
                              DAQmx_Val_FiniteSamps,  # pylint: disable=E0602
                              np.size(self.waveform))

        self.SetWriteRegenMode(DAQmx_Val_AllowRegen)  # pylint: disable=E0602

        if mode != 'static':
        
            print('Trigger mode')
            self.SetStartTrigRetriggerable(1)

            self.CfgDigEdgeStartTrig("ai/StartTrigger",
                                    DAQmx_Val_RisingSlope)    # pylint: disable=E0602

        self.WriteAnalogF64(np.size(self.waveform),
                            False,
                            0,
                            DAQmx_Val_GroupByChannel,  # pylint: disable=E0602
                            self.waveform,
                            self.wrote,
                            None)

    def static(self, flipper_params):
            
        self.waveform = flipper_params * np.ones((100,))
        self.check_max_current()
    
    def steps(self, flipper_params):

        self.waveform = np.zeros_like(self.t)
        delay = 1e3
        tof = np.add(flipper_params[::2],delay)
        tof = tof/1e6
        tof = np.insert(tof,0,min(self.t))
        currents = flipper_params[1::2]
        
        for i in range(np.size(tof)-1):
            idx = np.logical_and((self.t >= tof[i]), (self.t < tof[i+1]))
            self.waveform[idx] = currents[i]
            
        self.padding = self.padding*np.amax(currents)
        self.waveform = np.concatenate((self.waveform, self.padding))
        self.check_max_current()
        
    def analytical(self, flipper_params):
        
        amp = flipper_params[0]
        offset = flipper_params[1]
        self.waveform = amp / (self.t+offset)
        self.padding = self.padding*FLIP_MAX
        self.waveform = np.concatenate((self.waveform, self.padding))
        self.check_max_current()
        
    def file(self, filename):
        
        if os.path.isfile(os.path.join(os.getcwd(), filename)):
            self.waveform = np.loadtxt(filename, unpack=True)
        else:
            print('Invalid file path')

        self.check_max_current()
        
    def check_max_current(self):
        self.waveform[self.waveform > FLIP_MAX] = FLIP_MAX

#---------------------------------------------------------------------------


class CompensationTask(Task):
    
    """Task that sets the compensation coil current to a constant value."""

    def __init__(self, comp_chan, comp_current):
        Task.__init__(self)

        # Arbitrarily n = 100 samples, can be any n > 2
        self.waveform = comp_current * np.ones((100,)) / COMP_CONST
        self.wrote = int32()
        self.comp_chan = comp_chan

        """DAQmx procedure:

        1) Create Analog Output channel.
        2) Configure Sample Clock timing.
        3) Write value to the output buffer.
        """

        self.CreateAOVoltageChan(self.comp_chan,
                                 "",
                                 0, 10,
                                 DAQmx_Val_Volts,  # pylint: disable=E0602
                                 None)

        self.CfgSampClkTiming("",
                              SAMPRATE,
                              DAQmx_Val_Rising,      # pylint: disable=E0602
                              DAQmx_Val_FiniteSamps,  # pylint: disable=E0602
                              np.size(self.waveform))

        self.WriteAnalogF64(np.size(self.waveform),
                            False,
                            1e-2,
                            DAQmx_Val_GroupByChannel,  # pylint: disable=E0602
                            self.waveform,
                            self.wrote,
                            None)


#---------------------------------------------------------------------------


class ReadbackTask(Task):
    """Task the reads back a signal supplied to the input terminal.

    This task can be used to monitor the signal being supplied to the flipper coil.
    Manipulating this data in real time must be done extremely carefully as to not
    interrupt the thread running the DAQ tasks as it will crash the program if the thread
    blocks outside the timing window of each task run. Currently the task is simply timing
    the triggering using a callback to ensure no timing pulses are being missed. To do
    this it is not even necessary to connect a signal to the input terminal.

    Currently we are only using this task to monitor for the beam turning off.
    """

    def __init__(self):
        Task.__init__(self)

        self.i = 0
        self.freq = 0
        self.missed = 0
        self.data = np.zeros(int(50))
        self.time = time()
        self.sum_delta_t = 0

        """DAQmx procedure:

        1) Create Analog Input channel.
        2) Configure Sample Clock to time N samples
        3) Register a callback function every time we have finished reading samples to
           handle the data / do any timing we want.
        4) Set the task to be retriggerable.
        5) Configure an analog edge start trigger off "APFI0".
        """

        self.CreateAIVoltageChan("Dev2/ai0",
                                 "",
                                 DAQmx_Val_Cfg_Default,  # pylint: disable=E0602
                                 0,
                                 1,
                                 DAQmx_Val_Volts,       # pylint: disable=E0602
                                 None)
        
        self.CfgSampClkTiming("",
                              SAMPRATE,
                              DAQmx_Val_Rising,       # pylint: disable=E0602
                              DAQmx_Val_FiniteSamps,  # pylint: disable=E0602
                              int(50))

        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,  # pylint: disable=E0602
                                            int(50),
                                            0)

        self.CfgAnlgEdgeStartTrig(TRIGGER_CHAN,
                                  DAQmx_Val_RisingSlope,       # pylint: disable=E0602
                                  0.5)

        self.SetStartTrigRetriggerable(1)

    def EveryNCallback(self):
        # Read the data out of the buffer
        read = int32()
        self.ReadAnalogF64(int(50),
                           10,
                           DAQmx_Val_GroupByScanNumber,  # pylint: disable=E0602
                           self.data,
                           int(50),
                           read,
                           None)

        self.time = time()
