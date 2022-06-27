#!/usr/bin/env python3
import socket
import threading
import re
import queue
import numpy as np
from time import time, sleep
import flip_daqtasks

BUFFER_SIZE = 5
q = queue.Queue(BUFFER_SIZE)

LINE_END = '\n'

#---------------------------------------------------------------------------


class Flipper():
    
    devices = {}
    
    def __init__(self,name,flip_chan,comp_chan,remote = 0):
    
        self.device_params = {'mode':'static',
                              'device_state':'on',
                              'flipper_params':0,
                              'compensation_current':0}
        
        self.command_lib={'device_state':self.set_device_state,
                          'flipper_current':self.set_flipper_current,
                          'compensation_current':self.set_compensation_current,
                          'flipper_filename':self.set_flipper_filename,
                          'flipper_steps':self.set_flipper_steps,
                          'flipper_analytical':self.set_flipper_analytical}
        
        self.remote = remote
        self.flip_chan = flip_chan
        self.comp_chan = comp_chan
        self.flip_task = 0
        self.comp_task = 0
        self.bad_command = 0
        
        if remote == 0:
            fr = Flipper_Readback(name = 'flipper_readback', args = ('flipper',))
            fr.start()
        
        self.update_device()
        
        Flipper.devices[name] = self
        
    def update_device(self):
        
        if self.device_params['device_state'] == 'on':

            if self.flip_task != 0:
                self.flip_task.ClearTask() #for initialization
    
            flip_daqtasks.ZeroOutput(self.flip_chan,self.comp_chan)
            
            self.comp_task = flip_daqtasks.CompensationTask(self.comp_chan,
                                                            self.device_params['compensation_current'])
            self.comp_task.StartTask()
            self.comp_task.ClearTask()
            
            self.flip_task = flip_daqtasks.FlipperTask(self.flip_chan,
                                                       self.device_params['mode'],
                                                       self.device_params['flipper_params'])
            self.flip_task.StartTask()
            
        elif self.device_params['device_state'] == 'off':
        
            self.flip_task.ClearTask()        
            flip_daqtasks.ZeroOutput(self.flip_chan,self.comp_chan)
    
        else:
        
            flip_daqtasks.ZeroOutput(self.flip_chan,self.comp_chan)
            print('State incorrect')

    def set_device_state(self, state):
        
        if state in ['on','off']:
            print('Setting device state to '+state)
            self.device_params['device_state'] = state
            self.update_device()
            print('Success!')
        
        else:
            print('Bad device state')
            self.bad_command = 1
                    
    def set_flipper_current(self, val):
        
        if self.remote == 1:
            val = float(val)
        
        print('Setting flipper current to '+str(val))
        self.device_params['flipper_params'] = val
        self.device_params['mode'] = 'static'
        self.update_device()
        print('Success!')

    def set_compensation_current(self, val):
    
        if self.remote == 1:
            val = float(val)

        print('Setting compensation current to '+str(val))
        self.device_params['compensation_current'] = val
        self.update_device()
        print('Success!')   
    
    def set_flipper_filename(self, filename):
        
        print('Setting filename to '+filename) 
        self.device_params['flipper_params'] = filename
        self.device_params['mode'] = 'file'
        self.update_device()
        print('Success!')        
        
    def set_flipper_steps(self, vals):
        
        if self.remote == 1:
            vals = self.astr_to_array(vals)
            
        print('Setting flipper steps to '+str(vals))
        self.device_params['flipper_params'] = vals
        self.device_params['mode'] = 'steps'
        self.update_device()
        print('Success!')     
    
    def set_flipper_analytical(self, vals):
        
        if self.remote == 1:
            vals = self.astr_to_array(vals)
            
        print('Setting analytical flipper parameters to '+str(vals))
        self.device_params['flipper_params'] = vals
        self.device_params['mode'] = 'analytical'
        self.update_device()
        print('Success!')
        
    def execute_command(self,com_string,val_string):
    
        self.bad_command = 0
        
        if com_string in self.command_lib:
            self.command_lib[com_string](val_string)
        else:
            print('Bad command')
            self.bad_command = 1
            
    def astr_to_array(self, val_string):
        vals = val_string.strip('[]').split(',')
        vals = np.array(vals, dtype=np.float)
        return vals

#---------------------------------------------------------------------------


class Flipper_Server(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        
        super().__init__()
        self.target = target
        self.name = name
        self.host = host        # : Hostname on which to listen
        self.port = port        # : Port on which to listen

    def run(self):
        
        print('TCP server running')
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(5)
                
        while True:
            
            conn, addr = s.accept()
            conn.settimeout(10)
            com_thread = threading.Thread(target=self.read_command, args=(conn,addr))
            com_thread.start()
    
    def read_command(self, conn, addr):

        print('Connected by', addr)
        
        size = 128
        com_string = []
        f = ''
        
        while True:
            
            f = conn.recv(size)
            f = f.decode('utf8')
            
            if LINE_END in f:
                
                com_string.append(f[:f.find(LINE_END)])
                break
            
            if len(com_string)>1:
                
                last_pair = com_string[-2] + com_string[-1]
                
                if LINE_END in last_pair:
                    
                    com_string[-2] = last_pair[:last_pair.find(LINE_END)]
                    com_string.pop()
                    break
                
        com_string = ''.join(com_string)
        
        if "?" in com_string:
            
            #flipper name hard coded as 'flipper', multi-flipper implementation to come            
            response = Flipper.devices['flipper'].device_params[com_string[:-1]]
            print('Getting '+ com_string[:-1] + ' = ' + str(response))
            conn.sendall(bytes(str(response) + LINE_END,'utf8'))
       
        else:
        
            q.put(com_string)
            sleep(0.5) #wait for 0.5s for command to resolve
            
            if Flipper.devices['flipper'].bad_command == 1:
                conn.sendall(bytes('Error' + LINE_END,'utf8'))
            else:
                conn.sendall(bytes(com_string + LINE_END,'utf8'))
        
        conn.shutdown(socket.SHUT_RDWR)
        conn.close()
        print('Connection closed')

#---------------------------------------------------------------------------            

class Flipper_Control(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        
        super().__init__()
        self.target = target
        self.name = name

    def run(self):
        
        print('Flipper control running')
        
        for device in Flipper.devices.values():
            device.remote = 1
            
        while True:
            if not q.empty():
                com_string = q.get()
                func_string,val_string = com_string.split('=')
                
                #flipper name hard coded as 'flipper', multi-flipper implementation to come
                Flipper.devices['flipper'].execute_command(func_string.strip(),val_string.strip())
                
#---------------------------------------------------------------------------            

class Flipper_Readback(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        
        super().__init__()
        self.target = target
        self.name = name
        self.flipper_name = args[0]
        self.interrupted = 0
        
    def run(self):
        
        print('Flipper timeout check running')
            
        self.rt = flip_daqtasks.ReadbackTask()  # Read task for diagnostics
        self.rt.StartTask()
        
        while True:
            sleep(3)
            t = time()
            ds = Flipper.devices['flipper'].device_params['device_state']
            td = np.abs(t - self.rt.time)
            
            if ds == 'on' and (td > 5):
                Flipper.devices['flipper'].device_params['device_state'] = 'off'
                Flipper.devices['flipper'].update_device()
                self.interrupted = 1
                print('ISIS pulse interrupted')
                
            if ds == 'off' and (self.interrupted == 1) and (td < 5):

                Flipper.devices['flipper'].device_params['device_state'] = 'on'
                Flipper.devices['flipper'].update_device()
                self.interrupted = 0
                print('ISIS pulse restored')
                
#---------------------------------------------------------------------------

if __name__ == '__main__':
    
    host = 'ndw1305'
    port = 65432
    
    flipper = Flipper('flipper','dev2/ao1','dev2/ao0',remote=1)
    
    fr = Flipper_Readback(name = 'flipper_readback', args = ('flipper',))
    ss = Flipper_Server(name = 'signal_server', args = (host,port))
    fc = Flipper_Control(name = 'flipper_control', args = ('flipper',))

    fr.start()
    ss.start()
    fc.start()
    
