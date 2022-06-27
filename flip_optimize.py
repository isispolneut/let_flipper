import numpy as np
from time import perf_counter
import datetime
#import genie_python as g
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit

#---------------------------------------------------------------------------


class Flipper_Discrete():

    def __init__ (self, energies, count_source, count_mode,
                  count_time = 10,
                  count_ncounts = 1000,
                  flip_coeff = 0.04,
                  polmon_distance = 25.38,
                  polmon_delay = 100,
                  polmon_spectrum = 98312,
                  sample_distance = 25.00,
                  detector_distance = 28.5,
                  detector_delay = 100,
                  int_range = 0.02,
                  plot_scans = True,
                  output_file_format = 'currents_{0}_{1}.dat', #check format later
                  output_file_dir = '' #check format later
                  ):

        self.energies = np.array(energies)        
        self.count_source = count_source
                #count_source is either 'monitor' or 'detector'
        self.count_mode = count_mode
                #count_mode is either 'counts' or 'time'
        self.count_time = count_time #in secs
        self.count_ncounts = count_ncounts
        self.flip_coeff = flip_coeff
                #flipper coefficient calculated from JPCM paper
        self.polmon_distance = polmon_distance  #in metres
        self.polmon_delay = polmon_delay        #in microsecs
        self.polmon_spectrum = polmon_spectrum
                #monitor spectrum number
        self.sample_distance = sample_distance #in metres
        self.detector_distance = detector_distance #in metres
        self.detector_delay = detector_delay #in microsecs
        self.plot_scans = plot_scans
        self.int_range = int_range
        self.output_file_format = output_file_format
        self.output_file_dir = output_file_dir
        self.function_lib = {'flipper_func':self.flipper_func,
                             'compensation_func':self.compensation_func,
                             'coil_scan_monitor_dummy':self.coil_scan_monitor_dummy,
                             'coil_scan_monitor_counts':self.coil_scan_monitor_counts,
                             'coil_scan_monitor_time':self.coil_scan_monitor_time,
                             'coil_scan_detector_counts':self.coil_scan_detector_counts,
                             'coil_scan_detector_time':self.coil_scan_detector_time}

#---------------------------------------------------------------------------

    def energy_conversion(self):
    
        self.wl = 9.045/np.sqrt(self.energies)
        self.tof = 251.9 * self.polmon_distance * self.wl
                #in microsecs
        self.tof_range = np.array([(1.0-self.int_range)*self.tof,
                                   (1.0+self.int_range)*self.tof])
                # integration range in microsecs
        self.tof_range = self.tof_range.T
        self.flip_const = 0.0232*4*np.pi*2.2*.35
                # 3.5 mm thick flipper with 2.2 windings per mm 
        
        return self

#---------------------------------------------------------------------------

    def flipper_func(self, currents, a, b, c):

        return a * np.cos(b * currents)**2 + c

#---------------------------------------------------------------------------

    def compensation_func(self, currents, a, b, c):

        return a * (currents - b)**2 + c

#---------------------------------------------------------------------------

    def add_scan(self, scan, i_scan, scan_type, currents, energy, tof_range, iteration, mps):

        scan['scan_type'][i_scan] = scan_type
        scan['currents'][i_scan] = currents
        scan['energy'][i_scan] = energy
        scan['tof_range'][i_scan] = tof_range
        scan['iteration'][i_scan] = iteration
        scan['mps'][i_scan] = mps
        
        return scan
        
#---------------------------------------------------------------------------

    def get_scan(self, i_scan, *param_names):
        
        scan_string = list(param_names)
        params = self.scan[scan_string][i_scan]
        
        return params
    
#---------------------------------------------------------------------------

    def initialize_optimization(self, currents, n_points, iterations):

        dt = [('scan_type',np.unicode_,12),
              ('currents','f8',n_points),
              ('counts','f8',n_points),
              ('errs','f8',n_points),
              ('energy','f8'),
              ('tof_range','f8',2),
              ('iteration','i4'),
              ('mps','f')]
        
        scan = np.zeros(2*iterations*self.energies.size, dtype = dt)
        i_scan = 0
        
        
        #remove this
        if type(currents) is str and currents == 'auto':
        
            flip_mps = np.pi/(2*self.flip_const*self.wl)
            flip_range = 0.5
            comp_mps = 3*np.ones(self.energies.size) #expect ~3A in comp coil
            comp_range = 0.5
            
            for c in range(self.energies.size):
            
                fc = np.linspace(flip_mps[c]*(1-flip_range),
                                 flip_mps[c]*(1+flip_range),
                                 n_points)
                self.add_scan(scan,
                              i_scan,
                              'flipper',
                              fc,
                              self.energies[c],
                              self.tof_range[c],
                              1,
                              flip_mps[c])

                cc = np.linspace(comp_mps[c]*(1-comp_range),
                                 comp_mps[c]*(1+comp_range),
                                 n_points)     
                                 
                iscan+=1

                self.add_scan(scan,
                              i_scan,
                              'compensation',
                              cc,
                              self.energies[c],
                              self.tof_range[c],
                              1,
                              comp_mps[c])
                              
                iscan+=1

        elif type(currents) is list and len(currents) == 4:
            
            for c in range(max(map(len,currents))):

                flip_mps[c] = (currents[1][c]-currents[0][c])/2
                flip_diff = flip_mps[c]-currents[0][c]
                
                fc = np.linspace(flip_mps[c]-flip_diff,
                                 flip_mps[c]+flip_diff,
                                 n_points)
                
                self.add_scan(scan,
                              i_scan,
                              'flipper',
                              fc,
                              self.energies[c],
                              self.tof_range[c],
                              1,
                              flip_mps[c])
                              
                i_scan+=1
                
                comp_mps[c] = (currents[3][c]-currents[2][c])/2
                comp_diff = comp_mps[c]-currents[2][c]
                
                cc = np.linspace(comp_mps[c]-comp_diff,
                                 comp_mps[c]+comp_diff,
                                 n_points)

                self.add_scan(scan,
                              1,
                              'compensation',
                              cc,
                              self.energies[c],
                              self.tof_range[c],
                              1,
                              comp_mps[c])
                              
                i_scan+=1

        else:

            print('Current input incorrent')

        self.scan = scan

        return self
        
#---------------------------------------------------------------------------

    def guess_params(self, i_scan):
        
        s = self.get_scan(i_scan,'scan_type','currents','counts','errs','energy','mps')
        
        if s['scan_type'] == 'flipper':
        
            a_guess = 2*np.amax(s['counts'])-np.amin(s['counts'])
            b_guess = self.flip_const*9.045/np.sqrt(s['energy'])
            c_guess = np.amin(s['counts'])
            pin = [a_guess, b_guess, c_guess]

        if s['scan_type'] == 'compensation':
            
            a_guess = (s['counts'][0] - np.amin(s['counts']))/ \
                          ((s['currents'][0] - s['mps'])**2)
            b_guess = s['mps']
            c_guess = np.amin(s['counts'])
            pin = [a_guess, b_guess, c_guess]
        
        return pin

#---------------------------------------------------------------------------

    def fit_scan(self, i_scan, pin):
        
        s = self.get_scan(i_scan,'scan_type','currents','counts','errs')
        
        func_string = s['scan_type']+'_func';
        
        popt, pcov = curve_fit(self.function_lib[func_string],
                               s['currents'],
                               s['counts'],
                               p0=pin,
                               sigma=s['errs'])

        if s['scan_type'] == 'flipper':          
            cmin = np.pi/2/popt[1]
            
        elif s['scan_type'] == 'compensation':
            cmin = popt[1]

        return cmin, popt
        
#---------------------------------------------------------------------------

    def plot_scan(self, i_scan, popt):
        
        s = self.get_scan(i_scan,'scan_type','currents','counts','errs')
        
        fig, ax = pyplot.subplots()
        fit_x = np.linspace(np.amin(s['currents']),
                            np.amax(s['currents']),
                            50)
        
        func_string = s['scan_type']+'_func'
        
        fit_y = self.function_lib[func_string](fit_x, *popt)
        
        ax.errorbar(s['currents'],
                    s['counts'],
                    s['errs'],
                    marker='o')
        
        ax.plot(fit_x, fit_y, 'r-')
        
        ax.set(xlabel='current (A)',
               ylabel='counts',
               title=s['scan_type']+' scan')
        
        pyplot.show()
        
        return

#---------------------------------------------------------------------------

    def coil_scan_monitor_dummy(self, i_scan):
        
        s = self.get_scan(i_scan,'scan_type','currents')
        
        wi = 0.1
        
        if s['scan_type'] == 'flipper':
            a = 1e3
            b = np.pi/2/s['currents'][s['currents'].size//2]
            c = 200
        elif s['scan_type'] == 'compensation':
            a = 3e3
            b = 3
            c = 200
            
        noise = np.random.normal(1,wi,s['currents'].size)
        
        print(s['scan_type'])
    
        func_string = s['scan_type']+'_func';
        dummy_counts = self.function_lib[func_string](s['currents'], a, b, c)
    
        dummy_counts = dummy_counts * noise
        dummy_errs = wi*dummy_counts
           
        return dummy_counts, dummy_errs

#---------------------------------------------------------------------------

    def coil_scan_monitor_counts(self, i_scan):

        s = self.get_scan(i_scan,'scan_type','currents','counts','errs','tof_range')
        
        r = int(0)
        total_counts = 0
        prev_counts = 0
        counts = s['counts']
        errs = s['errs']

        g.change_monitor(self.polmon_spectrum,s['tof_range'][0],s['tof_range'][1])
        g.change_period(1)
        g.begin()
        g.pause()

        for i in currents:
        
            g.cset(s['scan_type'] + '_current', i)
            g.resume()
            start = perf_counter()

            while total_counts < self.count_ncounts*(float(r)+1):
                total_counts = g.get_pv("IN:LET:DAE:MONITORCOUNTS")

            end = perf_counter()
            g.pause()
            g.change_period(r+2)
            counts[r] = float(total_counts - prev_counts) / (start - end)
            errs[r] = sqrt(counts[r])
            prev_counts = total_counts
            r+=1

        g.end()
        
        self.scan['counts'][i_scan] = counts
        self.scan['errs'][i_scan] = errs
           
        return sekf

#---------------------------------------------------------------------------

    def coil_scan_monitor_time(self, i_scan):

        r = int(0)
        total_counts = 0
        prev_counts = 0
        counts = np.zeros_like(currents)
        errs = counts
            
        g.change_monitor(self.polmon_spectrum,tof_range[0],tof_range[1])
        g.change_period(1)
        g.begin()
        g.pause()

        for i in currents:
        
            g.cset(scan_type + '_current', i)
            g.resume()
            g.waitfor_time(seconds=self.count_time)
            g.pause()
            total_counts = g.get_pv("IN:LET:DAE:MONITORCOUNTS")
            g.change_period(r+2)
            counts[r] = float(total_counts - prev_counts)
            errs[r] = sqrt(counts[r])
            prev_counts = counts[r]
            r+=1

        g.end()
            
        return counts, errs

#---------------------------------------------------------------------------

    def coil_scan_detector_counts(self, i_scan):

        r = int(0)
        total_counts = 0
        prev_counts = 0
        counts = np.zeros_like(currents)
        errs = counts

        #g.change_tcb(low=tof_range[0], high=tof_range[1], step=None, trange=1, log=False, regime=1)
        g.change_period(1)
        g.begin()
        g.pause()

        for i in currents:
        
            g.cset(scan_type + '_current', i)
            g.resume()
            start = perf_counter()

            while total_counts < self.count_ncounts*(float(r)+1):
                total_counts = g.get_events()

            end = perf_counter()
            g.pause()
            g.change_period(r+2)
            counts[r] = float(total_counts - prev_counts) / (start - end)
            errs[r] = sqrt(counts[r])
            prev_counts = total_counts
            r+=1

        g.end()
              
        return counts, errs

#---------------------------------------------------------------------------

    def coil_scan_detector_time(self, i_scan):

        r = int(0)
        total_counts = 0
        prev_counts = 0
        counts = np.zeros_like(currents)
        errs = counts
        
        #g.change_tcb(low=tof_range[0], high=tof_range[1], step=None, trange=1, log=False, regime=1)
        g.change_period(1)
        g.begin()
        g.pause()

        for i in currents:
        
            g.cset(scan_type + '_current', i)
            g.resume()
            g.waitfor_time(seconds=self.count_time)
            g.pause()
            total_counts = g.get_events()
            g.change_period(r+2)
            counts[r] = float(total_counts - prev_counts)
            errs[r] = sqrt(counts[r])
            prev_counts = counts[r]
            r+=1

        g.end()
  
        return counts, errs

#---------------------------------------------------------------------------

    def write_step_output(self, cmin):
        
        pass

#---------------------------------------------------------------------------
              
    def flipper_opt_steps(self, currents='auto', n_points=11, iterations=1):

        """
        Optimizes flipper currents assuming multiple discrete Ei. Calculates ranges for each
        scan by default. If manual ranges are required, syntax is:

        inst.flipper_opt_steps(currents=[(flip_start1, flip_start2, ...), 
                                         (flip_end1, flip_end2, ...),
                                         (comp_start1, comp_start2, ...),
                                         (comp_end1, comp_end2, ...)],
                               n_points=11, iterations=1)
        
        Where the length of each tuple corresponds to the number of energies to be optimized
        """
        if currents = "auto"
            
            
        self.initialize_optimization(currents,n_points,iterations)
        scan_width = 1/2;
       
        for i_scan in range(2*iterations*self.energies.size):
            
            if i_scan == 0:
                pass
                #g.cset(flipper_current = self.flip_mps[iscan])
                #g.cset(compensation_current = self.flip_mps[iscan+1])

            func_string = 'coil_scan_'+self.count_source+'_'+self.count_mode
            counts, errs = self.function_lib[func_string](i_scan)
            
            self.scan['counts'][i_scan] = counts
            self.scan['errs'][i_scan] = errs
            
            pin = self.guess_params(i_scan)
            
            cmin, popt = self.fit_scan(i_scan,pin)
            
            print(self.scan['scan_type'][i_scan]+' scan Ei = {0}, iteration {1}' \
                  .format(self.scan['energy'][i_scan],self.scan['iteration'][i_scan]))
            print('optimized parameters are: {0:2f}, {1:2f}, {2:2f}' \
                  .format(popt[0],popt[1],popt[2]))
                  
            self.add_scan(scan,
              i_scan+self.energies.size,
              selc.scan['scan_type'][i_scan],
              np.linspace(p),
              self.scan['energy'][i_scan],
              self.scan['tof_range'][i_scan],
              self.scan['iteration'][i_scan]+1,
              pOpt[1])
            
            if self.plot_scans == True:
                self.plot_scan(i_scan,popt)
                
            #g.cset(flipper_current = flip_cmin)
            #g.cset(compensation_current = comp_cmin)
        
        return self


