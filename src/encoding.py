from typing import Any
import numpy as np


class add_dim(object):
    def __init__(self, dim): #dim need to be a vector of at least 1 element
        
        self.dim = dim

    def __call__(self, sample):
        
        shape = sample.shape
        for i in self.dim:
            shape = np.insert(shape, i, 1)
        
        return sample.reshape(shape) 

class To_spike(object): #discretization
    def __init__(self, num_levels):
        self.num_levels = num_levels
        self.thresholds = np.linspace(-1, 1, num_levels)

    def __call__(self, sample):
        
        digitized = np.digitize(sample, self.thresholds) - 1
        out = np.eye(self.num_levels)[digitized]
        return out.reshape(sample.shape[0]*self.num_levels, sample.shape[-1])


class SendOnDelta(object):
    def __init__(self, tresholds): #treshold needs to be a value (0,1)
        # sourcery skip: raise-specific-error
        
        for treshold in tresholds:
            if(treshold<0 or treshold>1):
                raise Exception("Thresholds out of range permitted") 
        
        self.tresholds = tresholds
    def __call__(self, sample):
        # sourcery skip: raise-specific-error
        if len(self.tresholds) == 1:
            self.tresholds = self.tresholds * sample.shape[0]
        if(len(self.tresholds)!=sample.shape[0]):
            raise Exception(f"tresholds len {len(self.tresholds)} not compatible with {sample.shape[0]}")
        
        new_sample = np.zeros((sample.shape[0], 2, sample.shape[1]))
        for i in range(sample.shape[0]):
            t = 0
            t_ref = 0
            while t<sample.shape[1]:
                if sample[i,t] - sample[i,t_ref] >= self.tresholds[i]:
                    new_sample[i, 0, t] = 1
                    t_ref = t
                elif sample[i,t] - sample[i,t_ref] <= - self.tresholds[i]:
                    new_sample[i, 1, t] = 1
                    t_ref = t
                t += 1
        return new_sample.reshape(sample.shape[0]*2, sample.shape[1])

class LIF(object):
    def __init__(self, dt, V_th, V_reset, tau, g_L, V_init, E_L, tref):
        
        self.V_th = V_th
        self.V_reset = V_reset
        self.g_L = g_L
        self.V_init = V_init
        self.E_L = E_L

        self.tau = tau
        self.dt = dt
        self.tref = tref
    
    def __call__(self,sample):
        
        dim = sample.shape[0]
        if len(self.V_th) == 1:
            self.V_th = self.V_th * dim

        if len(self.V_reset) == 1:
            self.V_reset = self.V_reset * dim  

        if len(self.g_L) == 1:
           self. g_L = self.g_L * dim

        if len(self.V_init) == 1:
            self.V_init = self.V_init * dim

        if len(self.E_L) == 1:
           self.E_L = self.E_L * dim

        if(len(self.V_th) != dim):
            raise Exception(f"parameters len {self.dim} not compatible with {sample.shape[0]}")
        
        new_sample = np.zeros((sample.shape[0], sample.shape[1]))

        for i in range(dim):
            
            v = np.zeros((sample.shape[1]))
            v[0] = self.V_init[i]
            tr = 0.

            for t in range(sample.shape[1]-1):
                
                if tr > 0:
                    v[t] = self.V_reset[i]
                    tr = tr - 1
                
                elif v[t] >= self.V_th[i]:
                    new_sample[i][t] = 1
                    v[t] = self.V_reset[i]
                    tr = self.tref / self.dt
                
                dv = (-(v[t] - self.E_L[i]) + sample[i][t] / self.g_L[i]) * (self.dt / self.tau)
                v[t + 1] = v[t] + dv

        return new_sample
    
class oversample(object):
    
    def __init__(self, scaling_factor):
        
        self.scaling_factor = scaling_factor
        

    def __call__(self, signal) :
        new_sample = []
        for i in range(signal.shape[0]):
            
            
            new_signal = [signal[i,0]]
            new_len = self.scaling_factor
            for t in range(signal.shape[-1]-1):

                if signal[i,t+1] - signal[i,t] >= 0:

                    values = np.interp(
                        np.linspace(0, 1, new_len),
                        [0,1],
                        signal[i,[t,t+1]]
                    )
                
                    new_signal.extend(values[1:])
                else :
                    values = np.interp(
                    np.linspace(0, 1, new_len),
                    [0,1],
                    np.flipud(signal[i,[t,t+1]])
                    )      
                    new_signal.extend(np.flipud(values)[1:])

            new_sample.append(new_signal)


        return np.array(new_sample)