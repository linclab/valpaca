import logging
import sys

import torch
import torch.nn as nn

sys.path.extend(['.', '..'])
from models import lfads, calcium


logger = logging.getLogger(__name__)


PRIOR = {
    'obs' : {'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
                     'var'  : {'value': 0.1, 'learnable' : True}}},
    'deep': {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                     'var'  : {'value': 0.1, 'learnable' : False}},
             'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                     'var'  : {'value': 0.1, 'learnable' : True},
                     'tau'  : {'value': 10,  'learnable' : True}}},
    }

PARAMS = {
    'gain' : {'value' : 1.0, 'learnable' : False},
    'bias' : {'value' : 0.0, 'learnable' : False},
    'tau'  : {'value' : 10., 'learnable' : False},
    'var'  : {'value' : 0.1, 'learnable' : True}
    }


class VaLPACa_Net(nn.Module):
    
    def __init__(self, input_size,
                 deep_g_encoder_size=64, deep_c_encoder_size=64,
                 obs_encoder_size=128, obs_latent_size=64, 
                 deep_g_latent_size=32, deep_u_latent_size=1,
                 obs_controller_size=64, deep_controller_size=32,
                 generator_size=64, factor_size=4, prior=PRIOR,
                 obs_params=PARAMS, clip_val=5.0, dropout=0.0, max_norm=200, 
                 generator_burn=0, deep_unfreeze_step=2000, obs_pad=3, 
                 do_normalize_factors=True, factor_bias=False, device='cpu'):
        
        super(VaLPACa_Net, self).__init__()
        
        self.input_size           = input_size
        self.obs_encoder_size     = obs_encoder_size
        self.obs_latent_size      = obs_latent_size
        self.obs_controller_size  = obs_controller_size
        
        self.deep_g_encoder_size  = deep_g_encoder_size
        self.deep_c_encoder_size  = deep_c_encoder_size
        self.deep_g_latent_size   = deep_g_latent_size
        self.deep_u_latent_size   = deep_u_latent_size
        self.deep_controller_size = deep_controller_size
        
        self.factor_size          = factor_size
        self.generator_size       = generator_size
        
        self.obs_pad              = obs_pad
        self.generator_burn       = generator_burn
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        
        self.deep_unfreeze_step   = deep_unfreeze_step
        
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias          = factor_bias
        
        self.device               = device
        
        self.dropout              = torch.nn.Dropout(dropout)
                
        self.obs_model = calcium.Calcium_Net(
            input_size      = self.input_size,
            encoder_size    = self.obs_encoder_size,
            latent_size     = self.obs_latent_size,
            controller_size = self.obs_controller_size,
            factor_size     = self.factor_size,
            parameters      = obs_params,
            prior           = prior['obs'],
            dropout         = dropout,
            pad             = self.obs_pad,
            clip_val        = self.clip_val,
            device          = self.device
            )
        
        self.deep_model = lfads.LFADS_Net(
            input_size      = self.input_size,
            g_encoder_size  = self.deep_g_encoder_size,
            c_encoder_size  = self.deep_c_encoder_size,
            g_latent_size   = self.deep_g_latent_size,
            u_latent_size   = self.deep_u_latent_size,
            controller_size = self.deep_controller_size,
            generator_size  = self.generator_size,
            factor_size     = self.factor_size,
            prior           = prior['deep'],
            clip_val        = self.clip_val,
            dropout         = dropout,
            max_norm        = self.max_norm,
            do_normalize_factors = self.do_normalize_factors,
            factor_bias     = self.factor_bias,
            device          = self.device
            )
        
        self.deep_model.add_module(
            'fc_logrates', nn.Linear(self.factor_size, self.input_size)
            )
        
        self.initialize_weights()
        
        if self.deep_unfreeze_step > 0:
            for p in self.deep_model.parameters():
                p.requires_grad = False
            
    def forward(self, input):
        
        input = input.permute(1, 0, 2)
        self.steps_size, self.batch_size, input_size = input.shape
        if input_size != self.input_size:
            raise ValueError('input size does not match self.input_size.')
        
        obs_encoder_state, obs_controller_state = \
            self.obs_model.initialize_hidden_states(input)
        
        out_obs_enc = self.obs_model.encoder(input, obs_encoder_state)

        input_deep =  input
        [deep_g_encoder_state, 
         deep_c_encoder_state, 
         deep_controller_state] = self.deep_model.initialize_hidden_states(
              input_deep
              )
        
        [self.deep_model.g_posterior_mean, 
         self.deep_model.g_posterior_logvar, 
         out_deep_g_enc, 
         out_deep_c_enc] = self.deep_model.encoder(
             input_deep, (deep_g_encoder_state, deep_c_encoder_state)
             )
        
        deep_generator_state = self.deep_model.fc_genstate(
            self.deep_model.sample_gaussian(
                self.deep_model.g_posterior_mean, 
                self.deep_model.g_posterior_logvar
                )
            )
        
        factor_state = self.deep_model.generator.fc_factors(
            self.deep_model.dropout(deep_generator_state)
            )
        
        factors = torch.empty(
            0, self.batch_size, self.factor_size, device=self.device
            )
        
        obs_state = torch.zeros(
            self.batch_size, self.input_size, device=self.device
            )
        spike_state = torch.zeros(
            self.batch_size, self.input_size, device=self.device
            )
        
        spikes = torch.empty(
            0, self.batch_size, self.input_size, device=self.device
            )
        obs    = torch.empty(
            0, self.batch_size, self.input_size, device=self.device
            )
        
        self.obs_model.u_posterior_mean   = torch.empty(
            self.batch_size, 0, self.obs_latent_size, device=self.device
            )
        self.obs_model.u_posterior_logvar = torch.empty(
            self.batch_size, 0, self.obs_latent_size, device=self.device
            )
        
        if (self.deep_c_encoder_size > 0 and self.deep_controller_size > 0 and 
            self.deep_u_latent_size > 0):
            deep_gen_inputs = torch.empty(
                0, self.batch_size, self.deep_u_latent_size, device=self.device
                )
            
            # initialize u posterior store
            self.deep_model.u_posterior_mean   = torch.empty(
                self.batch_size, 0, self.deep_u_latent_size, device=self.device
                )
            self.deep_model.u_posterior_logvar = torch.empty(
                self.batch_size, 0, self.deep_u_latent_size, device=self.device
                )
        
        for s in range(self.obs_pad):
            [obs_u_mean, 
             obs_u_logvar, 
             obs_controller_state] = self.obs_model.controller(
                 torch.cat((out_obs_enc[s], obs_state), dim=1), 
                 obs_controller_state
                 )
            self.obs_model.u_posterior_mean   = torch.cat(
                (self.obs_model.u_posterior_mean, obs_u_mean.unsqueeze(1)), 
                dim=1
                )
            self.obs_model.u_posterior_logvar = torch.cat(
                (self.obs_model.u_posterior_logvar, obs_u_logvar.unsqueeze(1)), 
                dim=1
                )
    
            obs_generator_state = self.obs_model.sample_gaussian(
                obs_u_mean, obs_u_logvar
                )
            obs_state, spike_state = self.obs_model.generator(
                torch.cat(
                    (obs_generator_state, torch.zeros_like(factor_state)), dim=1
                    ), 
                obs_state)
        
        for t in range(self.generator_burn):
            deep_generator_state, factor_state = self.deep_model.generator(
                None, deep_generator_state
                )
        
        for t in range(self.steps_size):
            
            if (self.deep_c_encoder_size > 0 and 
                self.deep_controller_size > 0 and self.deep_u_latent_size > 0):

                [deep_u_mean, 
                 deep_u_logvar, 
                 deep_controller_state] = self.deep_model.controller(
                     torch.cat((out_deep_c_enc[t], factor_state), dim=1), 
                     deep_controller_state
                     )

                self.deep_model.u_posterior_mean = torch.cat(
                    (self.deep_model.u_posterior_mean, deep_u_mean.unsqueeze(1)), 
                    dim=1
                    )
                self.deep_model.u_posterior_logvar = torch.cat(
                    (self.deep_model.u_posterior_logvar, deep_u_logvar.unsqueeze(1)), 
                    dim=1
                    )

                deep_generator_input = self.deep_model.sample_gaussian(
                    deep_u_mean, deep_u_logvar
                    )
                deep_gen_inputs = torch.cat(
                    (deep_gen_inputs, deep_generator_input.unsqueeze(0)), dim=0
                    )
            else:
                deep_generator_input = torch.empty(
                    self.batch_size, self.deep_u_latent_size, device=self.device
                    )
                deep_gen_inputs = None
            
            [obs_u_mean, 
             obs_u_logvar, 
              obs_controller_state] = self.obs_model.controller(
                  torch.cat((out_obs_enc[self.obs_pad + t], obs_state), dim=1), 
                  obs_controller_state
                  )
            self.obs_model.u_posterior_mean   = torch.cat(
                (self.obs_model.u_posterior_mean, obs_u_mean.unsqueeze(1)), 
                dim=1
                )
            self.obs_model.u_posterior_logvar = torch.cat(
                (self.obs_model.u_posterior_logvar, obs_u_logvar.unsqueeze(1)), 
                dim=1
                )
            
            obs_generator_state = self.obs_model.sample_gaussian(
                obs_u_mean, obs_u_logvar
                )
            
            deep_generator_state, factor_state = self.deep_model.generator(
                deep_generator_input, deep_generator_state
                )
            
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
            obs_state, spike_state = self.obs_model.generator(
                torch.cat((obs_generator_state, factor_state), dim=1), 
                obs_state
                )
            
            obs = torch.cat((obs, obs_state.unsqueeze(0)), dim=0)
            spikes = torch.cat((spikes, spike_state.unsqueeze(0)), dim=0)
            
        if (self.deep_c_encoder_size > 0 and self.deep_controller_size > 0 and 
            self.deep_u_latent_size > 0):
            # Instantiate AR1 process as mean and variance per time step
            [self.deep_model.u_prior_mean, 
             self.deep_model.u_prior_logvar] = self.deep_model._gp_to_normal(
                 self.deep_model.u_prior_gp_mean, 
                 self.deep_model.u_prior_gp_logvar, 
                 self.deep_model.u_prior_gp_logtau, 
                 deep_gen_inputs
                 )
            
        recon = {
            'rates' : self.deep_model.fc_logrates(factors).exp(),
            'data'  : obs.permute(1, 0, 2),
            'spikes': spikes,
            }
        
        return recon, (factors, deep_gen_inputs)
    


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
                
        with torch.no_grad():
            self.deep_model.initialize_weights()
            self.obs_model.initialize_weights()
    

    def change_parameter_grad_status(self, step, optimizer, scheduler, 
                                     loading_checkpoint=False):
        
        def step_condition(run_step, status_step, loading_checkpoint):
            if status_step is not None:
                if loading_checkpoint:
                    return run_step >= status_step
                else:
                    return run_step == status_step
            else:
                return False
        
        if step_condition(step, self.deep_unfreeze_step, loading_checkpoint):
            logger.info('Unfreezing deep model parameters.')
            optimizer.add_param_group({
                'params' : [p for p in self.deep_model.parameters() if not p.requires_grad],
                'lr' : optimizer.param_groups[0]['lr']
                })
            scheduler.min_lrs.append(scheduler.min_lrs[0])
            for p in self.deep_model.parameters():
                p.requires_grad_(True)
        
        return optimizer, scheduler
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def normalize_factors(self):
        self.deep_model.normalize_factors()
        
        