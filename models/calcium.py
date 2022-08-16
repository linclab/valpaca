import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.extend(['.', '..'])
from models import lfads, objective


PARAMS = {
    'gain' : {'value' : 1.0, 'learnable' : False},
    'bias' : {'value' : 0.0, 'learnable' : False},
    'tau'  : {'value' : 10., 'learnable' : False},
    'var'  : {'value' : 0.1, 'learnable' : True}
    }

PRIOR = {
    'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
            'var'  : {'value': 0.1, 'learnable' : False}},
    'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
            'var'  : {'value': 0.1, 'learnable' : False}}
    }


class Calcium_Net(nn.Module):
    def __init__(self, input_size, encoder_size=128, latent_size=64, 
                 controller_size=128, factor_size=4, pad=0, parameters=PARAMS,
                 prior=PRIOR, clip_val=5.0, dropout=0.05, device='cpu'):
        
        super(Calcium_Net, self).__init__()
        
        self.input_size      = input_size
        self.encoder_size    = encoder_size
        self.u_latent_size   = latent_size
        self.controller_size = controller_size
        self.factor_size     = factor_size
        self.clip_val        = clip_val
        self.device          = device
        self.pad             = pad

        self.encoder = Calcium_Encoder(
            input_size=self.input_size,
            encoder_size=self.encoder_size,
            clip_val=self.clip_val,
            pad=self.pad,
            dropout=dropout
            )
        
        self.controller = lfads.LFADS_ControllerCell(
            input_size=self.encoder_size * 2 + self.input_size,
            controller_size=self.controller_size,
            u_latent_size=self.u_latent_size,
            clip_val=self.clip_val,
            dropout=dropout
            )

        self.generator = Calcium_Generator(
            input_size=self.u_latent_size + self.factor_size,
            output_size=self.input_size,
            parameters=parameters,
            dropout=dropout,
            device=self.device
            )
        
        # Initialize learnable biases
        self.encoder_init    = nn.Parameter(torch.zeros(2, self.encoder_size))
        self.controller_init = nn.Parameter(torch.zeros(self.controller_size))
            
        self.u_prior_mean = torch.ones(
            self.u_latent_size, device=device
            ) * prior['u']['mean']['value']
        if prior['u']['mean']['learnable']:
            self.u_prior_mean = nn.Parameter(self.u_prior_mean)
        
        self.u_prior_logvar = torch.ones(
            self.u_latent_size, device=device
            ) * np.log(prior['u']['var']['value'])
        if prior['u']['var']['learnable']:
            self.u_prior_logvar = nn.Parameter(self.u_prior_logvar)
            
    def forward():
        pass
        
    def kl_div(self):
        kl = objective.kldiv_gaussian_gaussian(
            post_mu  = self.u_posterior_mean,
            post_lv  = self.u_posterior_logvar,
            prior_mu = self.u_prior_mean,
            prior_lv = self.u_prior_logvar
            )
        return kl
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
        
        def standard_init(weights):
            k = weights.shape[1] # dimensionality of inputs
            weights.data.normal_(std=k ** -0.5) # inplace resetting W ~ N(0, 1/sqrt(K))
        
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'weight' in name:
                    standard_init(p)
                    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    
    def sample_gaussian(self, mean, logvar):
        '''
        sample_gaussian(mean, logvar)
        
        Sample from a diagonal gaussian with given mean and log-variance
        
        Required Arguments:
            - mean (torch.Tensor)   : mean of diagional gaussian
            - logvar (torch.Tensor) : log-variance of diagonal gaussian
        '''
        # Generate noise from standard gaussian
        eps = torch.randn(mean.shape, requires_grad=False).to(self.device)
        # Scale and shift by mean and standard deviation
        return torch.exp(logvar*0.5)*eps + mean
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    
    def initialize_hidden_states(self, input):
        '''
        initialize_hidden_states()
        
        Initialize hidden states of recurrent networks
        '''
        
        self.steps_size, self.batch_size, input_size = input.shape
        if input_size != self.input_size:
            raise ValueError(
                'Input is expected to have dimensions '
                f'[{self.steps_size}, {self.batch_size}, {self.input_size}]'
            )
        
        encoder_state  = (torch.ones(
            self.batch_size, 2,  self.encoder_size, device=self.device
            ) * self.encoder_init).permute(1, 0, 2)
        controller_state = torch.ones(
            self.batch_size, self.controller_size, device=self.device
            ) * self.controller_init

        return encoder_state, controller_state
            
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class Calcium_Encoder(nn.Module):
    '''
    Calcium_Encoder
    
    Calcium Encoder Network 
    
    __init__(self, input_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0)
    
    Required Arguments:
        - input_size (int):  size of input dimensions
        - encoder_size (int):  size of generator encoder network
        
    Optional Arguments:
        - dropout (float): dropout probability
        - clip_val (float): RNN hidden state value limit
        
    '''
    def __init__(self, input_size, encoder_size, pad=0, dropout=0.0, 
                 clip_val=5.0):
        super(Calcium_Encoder, self).__init__()
        self.input_size   = input_size
        self.encoder_size = encoder_size
        self.pad          = pad
        self.clip_val     = clip_val
        self.dropout      = nn.Dropout(dropout)
        
        # encoder BiRNN
        self.gru  = nn.GRU(
            input_size=self.input_size, 
            hidden_size=self.encoder_size, 
            bidirectional=True
            )
            
    def forward(self, input, hidden):
        encoder_init = hidden
        steps_size, batch_size, state_size = input.shape
        leftpad = torch.zeros(self.pad, batch_size, state_size).to(input.device)
        input = torch.cat((leftpad, input))
        
        # Run bidirectional RNN over data
        out_gru, hidden_gru = self.gru(
            self.dropout(input), encoder_init.contiguous()
            )
        out_gru = out_gru.clamp(min=-self.clip_val, max=self.clip_val)
        
        return out_gru
        
class Calcium_Generator(nn.Module):
    def __init__(self, input_size, output_size, parameters, dropout, 
                 device='cpu'):
        super(Calcium_Generator, self).__init__()
        
        self.input_size  = input_size
        self.output_size = output_size
        self.device      = device
        
        self.spike_generator = Spike_Generator(
            input_size=input_size, 
            output_size=output_size, 
            dropout=dropout, 
            device=device
            )

        self.calcium_generator = AR1_Calcium(
            parameters=parameters, device=device
            )
        
    def forward(self, input, hidden):
        calcium_state = hidden
        spike_state = self.spike_generator(input)
        calcium_state = self.calcium_generator(spike_state, calcium_state)
        return calcium_state, spike_state

class Spike_Generator(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0, device='cpu'):
        super(Spike_Generator, self).__init__()
    
        self.fc_logspike = nn.Linear(
            in_features=input_size, out_features=output_size
            )
        self.dropout     = nn.Dropout(dropout)
        self.device      = device
    
    def forward(self, input):
        return torch.clamp(
            self.fc_logspike(self.dropout(input)).exp() - 1, min=0.0
            )
    
class AR1_Calcium(nn.Module):
    
    def __init__(self, parameters=PARAMS, device='cpu'):
        
        super(AR1_Calcium, self).__init__()
        
        self.device= device
        
        self.gain = value_to_tensor(
            parameters['gain']['value'], 
            parameters['gain']['learnable'], 
            device=device
            )
        self.bias = value_to_tensor(
            parameters['bias']['value'], 
            parameters['bias']['learnable'], 
            device=device
            )
        self.logvar = value_to_tensor(
            np.log(parameters['var']['value']), 
            parameters['var']['learnable'], 
            device=device
            )
        tau = value_to_tensor(
            parameters['tau']['value'], 
            parameters['tau']['learnable'], 
            device=device
            )
        self.damp = torch.clamp((tau - 1) / tau, min=0.0)

        param_names = ['gain', 'bias', 'tau', 'var']
        params = []
        for param_name in param_names:
            if parameters[param_name] ['learnable']:
                param = nn.Parameter(torch.tensor(
                    parameters[param_name]['value'], 
                    device=device, 
                    dtype=torch.float32)
                    ) 
            else:
                param = torch.tensor(
                    parameters['gain']['value'], 
                    device=device, 
                    dtype=torch.float32
                    )
            params.append(param)

    def forward(self, input, hidden):
        return hidden * self.damp + self.gain * input + self.bias


def value_to_tensor(value, learnable, device, dtype=torch.float32):
    value = torch.tensor(value, device=device, dtype=dtype)
    if learnable:
        return nn.Parameter(value)
    else:
        return value

        