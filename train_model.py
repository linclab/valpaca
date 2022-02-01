#!/usr/bin/env python

import argparse
import os

import torch
import torch.optim as opt
import pickle
import numpy as np

from ar1 import estimate_parameters

from run_manager import RunManager
from scheduler import LFADS_Scheduler

from utils import read_data, load_parameters, save_parameters
from plotter import Plotter

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-p', '--config', type=str)
parser.add_argument('-o', '--output_dir', default='/tmp', type=str)
parser.add_argument('--max_epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--data_suffix', default='data', type=str)
parser.add_argument('--detect_local_minima', action='store_true')

parser.add_argument('-t', '--use_tensorboard', action='store_true')
parser.add_argument('--orion', action='store_true')
parser.add_argument('-r', '--restart', action='store_true')
parser.add_argument('-c', '--do_health_check', action='store_true')

parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--kl_deep_max', type=float, default=None)
parser.add_argument('--kl_obs_max', type=float, default=None)
parser.add_argument('--kl_obs_dur', type=int, default=None)
parser.add_argument('--kl_obs_dur_scale', type=int, default=1.0)
parser.add_argument('--deep_start_p', type=int, default=None)
parser.add_argument('--deep_start_p_scale', type=float, default=1.0)
parser.add_argument('--l2_gen_scale', type=float, default=None)
parser.add_argument('--l2_con_scale', type=float, default=None)
parser.add_argument('--seed', type=int, default=None)


def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device = %s'%device)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    hyperparams = load_parameters(args.config)
    
    
    hp_string, hyperparams = adjust_hyperparams(args, hyperparams)

    if args.model in ["valpaca", "svlae"]:
        print(f'FACTOR SIZE = {hyperparams["model"]["factor_size"]} -- OBS ENCODER SIZE = {hyperparams["model"]["obs_encoder_size"]} -- DEEP g ENCODER SIZE = {hyperparams["model"]["deep_g_encoder_size"]}')

    save_loc, hyperparams = generate_save_loc(args, hyperparams, hp_string)
    
    save_parameters(save_loc, hyperparams)
    
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        
    data_dict   = read_data(args.data_path)
    
    train_dl, valid_dl, plotter, model, objective = prep_model(model_name  = args.model,
                                                               data_dict   = data_dict,
                                                               data_suffix = args.data_suffix,
                                                               batch_size  = args.batch_size,
                                                               device = device,
                                                               hyperparams = hyperparams)
    
    print_model_description(model)
    
    optimizer, scheduler = prep_optimizer(model, hyperparams)
        
    if args.use_tensorboard:
        writer, rm_plotter = prep_tensorboard(save_loc, plotter, args.restart)
    else:
        writer = None
        rm_plotter = None
        
    run_manager = RunManager(model      = model,
                             objective  = objective,
                             optimizer  = optimizer,
                             scheduler  = scheduler,
                             train_dl   = train_dl,
                             valid_dl   = valid_dl,
                             writer     = writer,
                             plotter    = rm_plotter,
                             max_epochs = args.max_epochs,
                             save_loc   = save_loc,
                             do_health_check = args.do_health_check,
                             detect_local_minima = args.detect_local_minima,
                             load_checkpoint=(not args.restart))

    run_manager.run()
        
    save_figs(save_loc, run_manager.model, run_manager.valid_dl, plotter)
    pickle.dump(run_manager.loss_dict, open(os.path.join(save_loc, 'loss.pkl'), 'wb'))
    
    if args.orion:
        # import orion
        # from orion.client import report_objective
        from orion.client import report_results
        
        report_results([{'name':'val_loss', 'type':'objective', 'value':run_manager.loss_dict['valid']['total'][-1]}])
#         report_objective(run_manager.loss_dict['valid']['total'][-1], name='objective')

#-------------------------------------------------------------------
#-------------------------------------------------------------------
def prep_model(model_name, data_dict, data_suffix, batch_size, device, hyperparams):
    print(model_name)
    if model_name == 'lfads':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device)
        model, objective = prep_lfads(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=train_dl.dataset.tensors[0].dtype,
                                      dt= data_dict['dt']
                                      )
        
    elif model_name == 'lfads-gaussian':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device)
        model, objective = prep_lfads_gaussian(input_dims = input_dims,
                                               hyperparams = hyperparams,
                                               device= device,
                                               dtype=train_dl.dataset.tensors[0].dtype
                                              )
        
    elif model_name == 'svlae':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device)
        
        if 'obs_gain_init' in data_dict.keys():
            print('gain= %.4f'%data_dict['obs_gain_init'].mean())
            hyperparams['model']['obs']['gain']['value'] = data_dict['obs_gain_init']
        if 'obs_bias_init' in data_dict.keys():
            print('bias= %.4f'%data_dict['obs_bias_init'].mean())
            hyperparams['model']['obs']['bias']['value'] = data_dict['obs_bias_init']
        if 'obs_var_init' in data_dict.keys():
            print('var= %.4f'%data_dict['obs_var_init'].mean())
            hyperparams['model']['obs']['var']['value'] = data_dict['obs_var_init']
        if 'obs_tau_init' in data_dict.keys():
            print('tau= %.4f'%data_dict['obs_tau_init'].mean())
            hyperparams['model']['obs']['tau']['value'] = data_dict['obs_tau_init']
        
        model, objective = prep_svlae(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=train_dl.dataset.tensors[0].dtype,
                                      dt=data_dict['dt']
                                      )
    elif model_name == 'valpaca':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device)
        
#         g, var = estimate_ar1_parameters(data_dict['train_fluor'])
#         gain = np.sqrt(var)*hyperparams['model']['obs']['snr']
        
#         hyperparams['model']['obs']['gain']['value'] = gain
#         hyperparams['model']['obs']['tau']['value'] = -data_dict['dt']/np.log(g)
#         hyperparams['model']['obs']['var']['value'] = var
        
#         print(np.median(hyperparams['model']['obs']['gain']['value']),
#               np.median(hyperparams['model']['obs']['var']['value']),
#               np.median(hyperparams['model']['obs']['tau']['value']))
        
        if 'obs_gain_init' in data_dict.keys():
            print('gain= %.4f'%data_dict['obs_gain_init'].mean())
            hyperparams['model']['obs']['gain']['value'] = data_dict['obs_gain_init']
        if 'obs_bias_init' in data_dict.keys():
            print('bias= %.4f'%data_dict['obs_bias_init'].mean())
            hyperparams['model']['obs']['bias']['value'] = data_dict['obs_bias_init']
        if 'obs_var_init' in data_dict.keys():
            print('var= %.4f'%data_dict['obs_var_init'].mean())
            hyperparams['model']['obs']['var']['value'] = data_dict['obs_var_init']
        if 'obs_tau_init' in data_dict.keys():
            print('tau= %.4f'%data_dict['obs_tau_init'].mean())
            hyperparams['model']['obs']['tau']['value'] = data_dict['obs_tau_init']
                
        model, objective = prep_valpaca(input_dims = input_dims,
                                        hyperparams=hyperparams,
                                        device= device,
                                        dtype=train_dl.dataset.tensors[0].dtype,
                                        dt=data_dict['dt']
                                        )
        
    else:
        raise NotImplementedError('Model must be one of \'lfads\', \'conv3d_lfads\', or \'svlae\'')
        
    return train_dl, valid_dl, plotter, model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
        
def prep_lfads(input_dims, hyperparams, device, dtype, dt):
    from objective import LFADS_Loss, LogLikelihoodPoisson
    from lfads import LFADS_SingleSession_Net

    model = LFADS_SingleSession_Net(input_size           = input_dims,
                                    factor_size          = hyperparams['model']['factor_size'],
                                    g_encoder_size       = hyperparams['model']['g_encoder_size'],
                                    c_encoder_size       = hyperparams['model']['c_encoder_size'],
                                    g_latent_size        = hyperparams['model']['g_latent_size'],
                                    u_latent_size        = hyperparams['model']['u_latent_size'],
                                    controller_size      = hyperparams['model']['controller_size'],
                                    generator_size       = hyperparams['model']['generator_size'],
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    obs                  = 'poisson',
                                    device               = device).to(device)
    
    loglikelihood = LogLikelihoodPoisson(dt=float(dt))

    objective = LFADS_Loss(loglikelihood            = loglikelihood,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)

    return model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_lfads_gaussian(input_dims, hyperparams, device, dtype):
    from objective import LFADS_Loss, LogLikelihoodGaussian
    from lfads import LFADS_SingleSession_Net
    
    model = LFADS_SingleSession_Net(input_size           = input_dims,
                                    factor_size          = hyperparams['model']['factor_size'],
                                    g_encoder_size       = hyperparams['model']['g_encoder_size'],
                                    c_encoder_size       = hyperparams['model']['c_encoder_size'],
                                    g_latent_size        = hyperparams['model']['g_latent_size'],
                                    u_latent_size        = hyperparams['model']['u_latent_size'],
                                    controller_size      = hyperparams['model']['controller_size'],
                                    generator_size       = hyperparams['model']['generator_size'],
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    obs                  = 'gaussian',
                                    device               = device).to(device)
        
    loglikelihood = LogLikelihoodGaussian()
    
    objective = LFADS_Loss(loglikelihood            = loglikelihood,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_svlae(input_dims, hyperparams, device, dtype, dt):
    from svlae import SVLAE_Net
    from objective import LogLikelihoodGaussian, LogLikelihoodPoissonSimplePlusL1, SVLAE_Loss

    loglikelihood_obs  = LogLikelihoodGaussian()
    loglikelihood_deep = LogLikelihoodPoissonSimplePlusL1(dt=float(dt))
    
    objective = SVLAE_Loss(loglikelihood_obs        = loglikelihood_obs,
                           loglikelihood_deep       = loglikelihood_deep,
                           loss_weight_dict         = {'kl_deep'    : hyperparams['objective']['kl_deep'],
                                                       'kl_obs'     : hyperparams['objective']['kl_obs'],
                                                       'l2'         : hyperparams['objective']['l2'],
                                                       'recon_deep' : hyperparams['objective']['recon_deep']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    hyperparams['model']['obs']['tau']['value']/=float(dt)
    
    model = SVLAE_Net(input_size            = input_dims,
                      factor_size           = hyperparams['model']['factor_size'],
                      obs_encoder_size      = hyperparams['model']['obs_encoder_size'],
                      obs_latent_size       = hyperparams['model']['obs_latent_size'],
                      obs_controller_size   = hyperparams['model']['obs_controller_size'],
                      deep_g_encoder_size   = hyperparams['model']['deep_g_encoder_size'],
                      deep_c_encoder_size   = hyperparams['model']['deep_c_encoder_size'],
                      deep_g_latent_size    = hyperparams['model']['deep_g_latent_size'],
                      deep_u_latent_size    = hyperparams['model']['deep_u_latent_size'],
                      deep_controller_size  = hyperparams['model']['deep_controller_size'],
                      generator_size        = hyperparams['model']['generator_size'],
                      prior                 = hyperparams['model']['prior'],
                      clip_val              = hyperparams['model']['clip_val'],
                      generator_burn        = hyperparams['model']['generator_burn'],
                      dropout               = hyperparams['model']['dropout'],
                      do_normalize_factors  = hyperparams['model']['normalize_factors'],
                      factor_bias           = hyperparams['model']['factor_bias'],
                      max_norm              = hyperparams['model']['max_norm'],
                      deep_unfreeze_step    = hyperparams['model']['deep_unfreeze_step'],
                      obs_params            = hyperparams['model']['obs'],
                      device                = device).to(device)
    
    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_valpaca(input_dims, hyperparams, device, dtype, dt):
    from valpaca import VaLPACa_Net
    from objective import LogLikelihoodGaussian, LogLikelihoodPoissonSimplePlusL1, SVLAE_Loss

    loglikelihood_obs  = LogLikelihoodGaussian()
    loglikelihood_deep = LogLikelihoodPoissonSimplePlusL1(dt=float(dt))
    
    objective = SVLAE_Loss(loglikelihood_obs        = loglikelihood_obs,
                           loglikelihood_deep       = loglikelihood_deep,
                           loss_weight_dict         = {'kl_deep'    : hyperparams['objective']['kl_deep'],
                                                       'kl_obs'     : hyperparams['objective']['kl_obs'],
                                                       'l2'         : hyperparams['objective']['l2'],
                                                       'recon_deep' : hyperparams['objective']['recon_deep']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    hyperparams['model']['obs']['tau']['value']/=float(dt)
    
    model = VaLPACa_Net(input_size            = input_dims,
                        factor_size           = hyperparams['model']['factor_size'],
                        obs_encoder_size      = hyperparams['model']['obs_encoder_size'],
                        obs_latent_size       = hyperparams['model']['obs_latent_size'],
                        obs_controller_size   = hyperparams['model']['obs_controller_size'],
                        deep_g_encoder_size   = hyperparams['model']['deep_g_encoder_size'],
                        deep_c_encoder_size   = hyperparams['model']['deep_c_encoder_size'],
                        deep_g_latent_size    = hyperparams['model']['deep_g_latent_size'],
                        deep_u_latent_size    = hyperparams['model']['deep_u_latent_size'],
                        deep_controller_size  = hyperparams['model']['deep_controller_size'],
                        generator_size        = hyperparams['model']['generator_size'],
                        prior                 = hyperparams['model']['prior'],
                        clip_val              = hyperparams['model']['clip_val'],
                        obs_pad               = hyperparams['model']['obs_pad'],
                        generator_burn        = hyperparams['model']['generator_burn'],
                        dropout               = hyperparams['model']['dropout'],
                        do_normalize_factors  = hyperparams['model']['normalize_factors'],
                        factor_bias           = hyperparams['model']['factor_bias'],
                        max_norm              = hyperparams['model']['max_norm'],
                        deep_unfreeze_step    = hyperparams['model']['deep_unfreeze_step'],
                        obs_params            = hyperparams['model']['obs'],
                        device                = device).to(device)
    
    return model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_data(data_dict, data_suffix, batch_size, device):
    train_data  = torch.Tensor(data_dict['train_%s'%data_suffix]).to(device)
    valid_data  = torch.Tensor(data_dict['valid_%s'%data_suffix]).to(device)
    
    num_trials, num_steps, input_size = train_data.shape
    
    train_ds    = torch.utils.data.TensorDataset(train_data)
    valid_ds    = torch.utils.data.TensorDataset(valid_data)
    
    train_dl    = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size = valid_data.shape[0])
    
    TIME = np.arange(0, num_steps*data_dict['dt'], data_dict['dt'])
    
    train_truth = {}
    if 'train_rates' in data_dict.keys():
        train_truth['rates'] = data_dict['train_rates']
    if 'train_latent' in data_dict.keys():
        train_truth['latent'] = data_dict['train_latent']
    if 'valid_spikes' in data_dict.keys():
        train_truth['spikes'] = data_dict['train_spikes']
        
    valid_truth = {}
    if 'valid_rates' in data_dict.keys():
        valid_truth['rates'] = data_dict['valid_rates']
    if 'valid_latent' in data_dict.keys():
        valid_truth['latent'] = data_dict['valid_latent']
    if 'valid_spikes' in data_dict.keys():
        valid_truth['spikes'] = data_dict['valid_spikes']

    plotter = {'train' : Plotter(time=TIME, truth=train_truth),
               'valid' : Plotter(time=TIME, truth=valid_truth)}
    
    return train_dl, valid_dl, input_size, plotter

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_optimizer(model, hyperparams):
    
    optimizer = opt.Adam([p for p in model.parameters() if p.requires_grad],
                         lr=hyperparams['optimizer']['lr_init'],
                         betas=(hyperparams['optimizer']['beta1'],hyperparams['optimizer']['beta2']),
                         eps=hyperparams['optimizer']['eps'])
    
    scheduler = LFADS_Scheduler(optimizer      = optimizer,
                                mode           = 'min',
                                factor         = hyperparams['scheduler']['scheduler_factor'],
                                patience       = hyperparams['scheduler']['scheduler_patience'],
                                verbose        = True,
                                threshold      = 1e-4,
                                threshold_mode = 'abs',
                                cooldown       = hyperparams['scheduler']['scheduler_cooldown'],
                                min_lr         = hyperparams['scheduler']['lr_min'])
    
    return optimizer, scheduler

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def print_model_description(model):
    total_params = 0
    for ix, (name, param) in enumerate(model.named_parameters()):
        print(ix, name, list(param.shape), param.numel(), param.requires_grad)
        total_params += param.numel()
    
    print('Total parameters: %i'%total_params)

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_tensorboard(save_loc, plotter, restart):
    import importlib
    if importlib.util.find_spec('torch.utils.tensorboard'):
        tb_folder = os.path.join(save_loc, 'tensorboard')
        if not os.path.exists(tb_folder):
            os.mkdir(tb_folder)
        elif os.path.exists(tb_folder) and restart:
            os.system('rm -rf %s'%tb_folder)
            os.mkdir(tb_folder)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_folder)
        rm_plotter = plotter
    else:
        writer = None
        rm_plotter = None
            
    return writer, rm_plotter

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def adjust_hyperparams(args, hyperparams):

    hp_string = ''
    if args.lr:
        lr = args.lr
        hyperparams['optimizer']['lr_init'] = lr
        hyperparams['scheduler']['lr_min']  = lr * 1e-3
        hp_string += 'lr= %.4f\n'%lr
        
    if args.kl_obs_dur:
        hyperparams['objective']['kl_obs']['schedule_dur'] = args.kl_obs_dur * args.kl_obs_dur_scale
        hp_string += 'kl_obs_dur= %i\n'%(args.kl_obs_dur*args.kl_obs_dur_scale)

    if args.kl_obs_max:
        hyperparams['objective']['kl_obs']['max'] = args.kl_obs_max
        hp_string += 'kl_obs_max= %.3f\n'%(args.kl_obs_max)
        
    if args.kl_deep_max:
        hyperparams['objective']['kl_deep']['max'] = args.kl_deep_max
        hp_string += 'kl_deep_max= %.3f\n'%(args.kl_deep_max)
    
    if args.deep_start_p:
        deep_start = int(args.deep_start_p * args.deep_start_p_scale * hyperparams['objective']['kl_obs']['schedule_dur'])
        hyperparams['objective']['kl_deep']['schedule_start'] = deep_start
        hyperparams['objective']['l2']['schedule_start'] = deep_start
        hyperparams['model']['deep_unfreeze_step'] = deep_start
        hp_string += 'deep_start= %i\n'%deep_start
        
    if args.l2_gen_scale:
        l2_gen_scale = args.l2_gen_scale
        hyperparams['objective']['l2_gen_scale'] = l2_gen_scale
        hp_string += 'l2_gen_scale= %.3f\n'%l2_gen_scale
    
    if args.l2_con_scale:
        l2_con_scale = args.l2_con_scale
        hyperparams['objective']['l2_con_scale'] = l2_con_scale
        hp_string += 'l2_con_scale= %.3f\n'%l2_con_scale
    
    if args.seed:
        hp_string += 'seed= %s'%args.seed

    if hyperparams['model']['deep_width'] is not None:
        if "obs" in hyperparams['model'].keys(): # valpaca or svlae
            hyperparams['model']['deep_g_encoder_size'] = hyperparams['model']['deep_width']
            hyperparams['model']['deep_c_encoder_size'] = hyperparams['model']['deep_width']

        else: # lfads
            hyperparams['model']['g_encoder_size'] = hyperparams['model']['deep_width']
            hyperparams['model']['c_encoder_size'] = hyperparams['model']['deep_width']
    
    if 'obs' in hyperparams['model'].keys() and hyperparams['model']['obs_width'] is not None: # valpaca or svlae
        hyperparams['model']['obs_controller_size'] = hyperparams['model']['obs_width']
        hyperparams['model']['obs_encoder_size'] = hyperparams['model']['obs_width']
        
    hp_string = hp_string.replace('\n', '-').replace(' ', '').replace('=', '')
    hp_string = '_hp-'+ hp_string
        
    return hp_string, hyperparams

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def generate_save_loc(args, hyperparams, orion_hp_string):
    data_name = args.data_path.split('/')[-1]
    model_name = hyperparams['model_name']
    if 'ospikes' in args.data_suffix:
        model_name += '_oasis'
    mhp_list = [key.replace('size', '').replace('deep', 'd').replace('obs', 'o').replace('_', '')[:4] + str(val) for key, val in hyperparams['model'].items() if 'size' in key]
    mhp_list.sort()
    hyperparams['run_name'] = '_'.join(mhp_list)
    hyperparams['run_name'] += orion_hp_string
    save_loc = '%s/%s/%s/%s/'%(args.output_dir, data_name, model_name, hyperparams['run_name'])
    return save_loc, hyperparams

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def save_figs(save_loc, model, dl, plotter):
    fig_folder = os.path.join(save_loc, 'figs')
    
    if os.path.exists(fig_folder):
        os.system('rm -rf %s'%fig_folder)
    os.mkdir(fig_folder)
    
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    fig_dict = plotter['valid'].plot_summary(model= model, dl= dl)
    for k, v in fig_dict.items():
        if type(v) == Figure:
            v.savefig(os.path.join(fig_folder, f'{k}.svg'))

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def estimate_ar1_parameters(F, top_k=1):
    n_trials, n_steps, n_cells = F.shape
    damp = np.zeros(n_cells)
    var = np.zeros(n_cells)
    for i in range(n_cells):
        s = np.argsort(F[:, :, i].max(axis=1) - F[:, :, i].min(axis=1))
        top_k_trials = F[s][-top_k:, :, i]
        g_tmp, sn_tmp = [], []
        for trial in top_k_trials:
            g, sn = estimate_parameters(trial, p=1, fudge_factor=.98)
            g_tmp.append(g)
            sn_tmp.append(sn)
        damp[i] = np.median(g_tmp)
        var[i] = np.median(sn_tmp)**2
    return damp, var
            
#-------------------------------------------------------------------
#-------------------------------------------------------------------

if __name__ == '__main__':
    main()