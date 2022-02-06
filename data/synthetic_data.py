import numpy as np

def euler_step(x, f, dt):
    return x + dt * f(x)

def rk4_step(x, f, dt):
    k1 = dt * f(x)
    k2 = dt * f(x + 0.5*k1)
    k3 = dt * f(x + 0.5*k2)
    k4 = dt * f(x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4)/6
    
class DynamicalSystem():
    def __init__(self):
        pass
        
    def gradient(self, state):
        pass
    
    def rescale(self, xt):
        return xt
    
    def generate_inputs(self, dims):
        return None
    
    def update(self, order=4):
        if order == 1:
            return euler_step(x= self.state, f=self.gradient, dt=self.dt)
        else:
            return rk4_step(x= self.state, f=self.gradient, dt=self.dt)
    
    def integrate(self, num_steps, inputs, burn_steps = 0):
        
        result = np.zeros((num_steps,) + self.state.shape)
        for t in range(burn_steps):
            self.state = self.update()
        for t in range(num_steps):
            self.state = self.update()
            if inputs is not None:
                self.state += inputs[t]
            result[t] = self.state
            
        result = self.rescale(result)
        self.result = result
        return result
    
class LorenzSystem(DynamicalSystem):
    def __init__(self, num_inits=100, weights=[10.0, 28.0, 8.0/3.0], dt=0.01):        
        self.state  = np.random.randn(num_inits, 3)
        self.weights = np.array(weights)
        self.num_inits = num_inits
        self.net_size = 3
        self.dt = dt
        
    def gradient(self, state):
        y1, y2, y3 = state.T
        w1, w2, w3 = self.weights
        dy1 = w1 * (y2 - y1)
        dy2 = y1 * (w2 - y3) - y2
        dy3 = y1 *  y2 - w3  * y3
        return np.array([dy1, dy2, dy3]).T
    
    def rescale(self, xt):
        xt -= xt.mean(axis=0).mean(axis=0)
        xt /= np.abs(xt).max()
        return xt
    
class EmbeddedLowDNetwork(DynamicalSystem):
    def __init__(self, low_d_system, net_size=64, base_rate=1.0, dt= 0.01):
        super(EmbeddedLowDNetwork, self).__init__()
        
        self.low_d_system = low_d_system
        self.net_size = net_size
        self.proj = (np.random.rand(self.low_d_system.net_size, self.net_size) + 1) * np.sign(np.random.randn(self.low_d_system.net_size, net_size))
        self.bias = np.log(base_rate)
        self.dt = dt
        self.num_inits = self.low_d_system.num_inits
        
    def gradient(self, state):
        return self.low_d_system.gradient(state)
    
    def rescale(self, xt):
        return np.exp(xt.dot(self.proj) + self.bias)
    
    def integrate(self, burn_steps, num_steps, inputs):
        result = self.low_d_system.integrate(burn_steps = burn_steps, num_steps = num_steps, inputs=inputs)
        result = self.rescale(result)
        self.result = result
        return result
        
class AR1Calcium(DynamicalSystem):
    
    def __init__(self, dims, tau=0.1, dt=0.01):
        self.state = np.zeros(dims)
        self.tau = tau
        self.dt = dt
        
    def gradient(self, state):
        return -state/self.tau
    
    def rescale(self, xt):
        return xt
    
class HillAR1Calcium(AR1Calcium):
    def __init__(self, dims, tau=0.1, dt=0.01, n=2, gamma=0.001, A=1.0):
        super(HillAR1Calcium, self).__init__(dims=dims, tau=tau, dt=dt)
        self.n = n
        self.gamma=gamma
        self.A = A
        
    def rescale(self, xt):
        return (self.A * xt**self.n/(1 + self.gamma*xt**self.n))

class SyntheticCalciumDataGenerator():
    def __init__(self, system, seed, trainp = 0.8,
                 burn_steps = 1000, num_trials = 100, num_steps= 100,
                 tau_cal=0.1, dt_cal= 0.01, sigma=0.2,
                 n=2.0, A=1.0, gamma=0.01, save=True):
        
        self.seed = seed
        np.random.seed(seed)
        self.trainp = trainp
        
        self.system = system
        self.burn_steps = burn_steps
        
        self.num_steps  = num_steps
        self.num_trials = num_trials
        
        self.ar1_calcium_dynamics = AR1Calcium(dims=(self.num_trials,
                                               self.system.num_inits,
                                               self.system.net_size), 
                                               tau=tau_cal, dt=dt_cal)
        
        self.hillar1_calcium_dynamics = HillAR1Calcium(dims=(self.num_trials,
                                                       self.system.num_inits,
                                                       self.system.net_size),
                                                       tau=tau_cal,
                                                       n=n,
                                                       A=A,
                                                       gamma=gamma,
                                                       dt=dt_cal)
        self.sigma = sigma
                
    def generate_dataset(self):
        inputs  = self.system.generate_inputs(dims=(self.num_steps, self.system.num_inits, self.system.net_size))
        rates   = self.system.integrate(burn_steps = self.burn_steps, num_steps = self.num_steps, inputs= inputs)
        if type(self.system) is EmbeddedLowDNetwork:
            latent = self.system.low_d_system.result
            latent = self.trials_repeat(latent)
        else:
            latent = None
        if inputs is not None:
            inputs = self.trials_repeat(inputs)
            
        rates   = self.trials_repeat(rates)
        spikes  = self.spikify(rates, self.ar1_calcium_dynamics.dt)
        calcium = self.ar1_calcium_dynamics.integrate(num_steps=self.num_steps, inputs=spikes.transpose(2, 0, 1, 3)).transpose(1, 2, 0, 3)
        fluor_ar1   = calcium + np.random.randn(*calcium.shape)*self.sigma
        
        fluor_hillar1 = self.hillar1_calcium_dynamics.integrate(num_steps=self.num_steps, inputs=spikes.transpose(2, 0, 1, 3)).transpose(1, 2, 0, 3) + np.random.randn(*calcium.shape)*self.sigma
        
        data_dict = {}
        for data, data_name in zip((inputs, rates, latent, spikes, calcium, fluor_ar1, fluor_hillar1), 
                                   ('inputs', 'rates', 'latent', 'spikes', 'calcium', 'fluor_ar1', 'fluor_hillar1')):
            if data is not None:
                data_dict['train_%s'%data_name], data_dict['valid_%s'%data_name] = self.train_test_split(data)
        
        data_dict['dt'] = self.ar1_calcium_dynamics.dt
        
        return data_dict
        
    def trials_repeat(self, data):
        data = data[..., None] * np.ones(self.num_trials)
        return data.transpose(3, 1, 0, 2)
        
    def spikify(self, rates, dt):
        return np.random.poisson(rates*dt)
    
    def calcify(self, spikes):
        return self.calcium_dynamics.integrate(num_steps=self.num_steps, inputs=spikes)
           
    def train_test_split(self, data):
        num_trials, num_inits, num_steps, num_cells = data.shape
        num_train = int(self.trainp * num_trials)
        train_data = data[:num_train].reshape(num_train*num_inits, num_steps, num_cells)
        valid_data = data[num_train:].reshape((num_trials - num_train)*num_inits, num_steps, num_cells)
        return train_data, valid_data