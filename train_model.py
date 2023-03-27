#!/usr/bin/env python

import argparse
import copy
import logging
from pathlib import Path
import shutil
import sys

import torch
import torch.optim as opt
import pickle as pkl
import numpy as np

sys.path.extend(['.', '..'])
from models import ar1
from utils import run_manager, scheduler, util, plotter


logger = logging.getLogger(__name__)


def main(args):

    # set logger to the specified level
    util.set_logger_level(logger, level=args.log_level)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')
    
    util.seed_all(args.seed)

    if args.output_dir is None:
        args.output_dir = Path(args.data_path).parent

    hyperparams = util.load_parameters(args.config)
    hyperparams, hp_str = adjust_hyperparams(args, hyperparams)
    save_loc, run_name = generate_save_loc(
        hyperparams['model'], args.data_path, args.output_dir, 
        model_name=hyperparams['model_name'], hp_str=hp_str
        )
    hyperparams['run_name'] = run_name

    data_dict = util.read_data(args.data_path)
    util.save_parameters(save_loc, hyperparams)
    
    train_dl, valid_dl, plotter_dict, model, objective = prep_model(
        data_dict   = data_dict,
        hyperparams = hyperparams,
        model_name  = args.model_name,
        data_suffix = args.data_suffix,
        batch_size  = args.batch_size,
        device      = device,
        log_model   = True,
        )    
    
    optimizer, sched = prep_optimizer(model, hyperparams)
        
    writer, rm_plotter = prep_tensorboard(
        save_loc, plotter_dict, args.restart, use_tb=args.use_tensorboard
        )

    run_mng = run_manager.RunManager(
        model      = model,
        objective  = objective,
        optimizer  = optimizer,
        scheduler  = sched,
        train_dl   = train_dl,
        valid_dl   = valid_dl,
        writer     = writer,
        plotter    = rm_plotter,
        max_epochs = args.max_epochs,
        save_loc   = save_loc,
        do_health_check = args.do_health_check,
        detect_local_minimum = args.detect_local_minimum,
        load_checkpoint = (not args.restart)
        )

    run_mng.run()
        
    save_figs(save_loc, run_mng.model, run_mng.valid_dl, plotter_dict)

    with open(Path(save_loc, 'loss.pkl'), 'wb') as f:
        pkl.dump(run_mng.loss_dict, f)

    if args.orion:
        from orion.client import report_results, report_objective
        
        # valid_loss = run_mng.loss_dict['valid']['total'][-1]
        valid_loss = run_mng.best

        results_dict = {
            'name' : 'val_loss', 
            'type' : 'objective', 
            'value': valid_loss,
            }

        report_results([results_dict])
        # report_objective(valid_loss, name='objective')

#-------------------------------------------------------------------
#-------------------------------------------------------------------
def prep_model(data_dict, hyperparams, model_name='valpaca', 
               data_suffix='data', batch_size=None, device='cpu', 
               log_model=False):
    
    model_str = f'\n{model_name.capitalize()} model parameters:'

    train_dl, valid_dl, input_dims, plotter_dict = prep_data(
        data_dict=data_dict, 
        data_suffix=data_suffix, 
        batch_size=batch_size, 
        device=device
        )

    if model_name in ['lfads', 'lfads-gaussian']:
        model, obj = prep_lfads(
            input_dims=input_dims,
            hyperparams=hyperparams,
            dt=data_dict['dt'],
            gauss=(model_name == 'lfads-gaussian'),
            device=device,
            )
                
    elif model_name in ['svlae', 'valpaca']:
        for val_name in ['gain', 'bias', 'var', 'tau']:
            key = f'obs_{val_name}_init'
            if key in data_dict.keys():
                vals = data_dict[key]
                model_str = f'{model_str}\n{val_name}={vals.mean():.4f}'
                hyperparams['model']['obs'][val_name]['value'] = vals
        model_str = f"{model_str}\n"

        model, obj = prep_svlae_valpaca(
            input_dims=input_dims,
            hyperparams=hyperparams,
            dt=data_dict['dt'],
            model_name=model_name,
            device=device,
            )
            
    else:
        raise NotImplementedError(
            'Model must be one of \'lfads\', \'lfads-gaussian\', \'svlae\', '
            'or \'valpaca\'.'
            )

    if log_model:
        logger.info(f'{model_str}')
        log_model_description(model)
 

    return train_dl, valid_dl, plotter_dict, model, obj
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
        
def prep_lfads(input_dims, hyperparams, dt, gauss=False, device='cpu'):
    from models import objective, lfads

    obs = 'gaussian' if gauss else 'poisson'

    model = lfads.LFADS_SingleSession_Net(
        input_size           = input_dims,
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
        obs                  = obs,
        device               = device
        ).to(device)
    
    loglikelihood = objective.LogLikelihoodPoisson(dt=float(dt))

    obj = objective.LFADS_Loss(
        loglikelihood    = loglikelihood,
        loss_weight_dict = {'kl': hyperparams['objective']['kl'], 
                            'l2': hyperparams['objective']['l2']},
        l2_con_scale     = hyperparams['objective']['l2_con_scale'],
        l2_gen_scale     = hyperparams['objective']['l2_gen_scale']
        ).to(device)

    return model, obj

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_svlae_valpaca(input_dims, hyperparams, dt, model_name='valpaca', 
                       device='cpu'):
    from models import objective, svlae, valpaca

    loglikelihood_obs  = objective.LogLikelihoodGaussian()
    loglikelihood_deep = objective.LogLikelihoodPoissonSimplePlusL1(dt=float(dt))
    
    obj = objective.SVLAE_Loss(
        loglikelihood_obs  = loglikelihood_obs,
        loglikelihood_deep = loglikelihood_deep,
        loss_weight_dict   = {'kl_deep'    : hyperparams['objective']['kl_deep'],
                              'kl_obs'     : hyperparams['objective']['kl_obs'],
                              'l2'         : hyperparams['objective']['l2'],
                              'recon_deep' : hyperparams['objective']['recon_deep']},
        l2_con_scale       = hyperparams['objective']['l2_con_scale'],
        l2_gen_scale       = hyperparams['objective']['l2_gen_scale']
        ).to(device)
    
    hyperparams['model']['obs']['tau']['value'] /= float(dt)
    
    if model_name == 'svlae':
        model_fct = svlae.SVLAE_Net
        extra_kwargs = dict()
    elif model_name == 'valpaca':
        model_fct = valpaca.VaLPACa_Net   
        extra_kwargs = {'obs_pad': hyperparams['model']['obs_pad']}
    else:
        raise ValueError('model_name must be \'svlae\' or \'valpaca\'.')

    model = model_fct(
        input_size            = input_dims,
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
        device                = device,
        **extra_kwargs
        ).to(device)
    
    return model, obj

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_data(data_dict, data_suffix='data', batch_size=None, device='cpu'):

    if f'train_{data_suffix}' not in data_dict.keys():
        raise ValueError(
            f'\'{data_suffix}\' data_suffix not found in the data keys.'
            )

    train_data  = torch.Tensor(data_dict[f'train_{data_suffix}']).to(device)
    valid_data  = torch.Tensor(data_dict[f'valid_{data_suffix}']).to(device)
    
    _, num_steps, input_size = train_data.shape
    
    train_ds = torch.utils.data.TensorDataset(train_data)
    valid_ds = torch.utils.data.TensorDataset(valid_data)
    
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
        )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=valid_data.shape[0]
        )
    
    time = np.arange(0, num_steps * data_dict['dt'], data_dict['dt'])
    
    train_grdtruth = dict()
    for data_type in ['rates', 'latent', 'spikes']:
        if f'train_{data_type}' in data_dict.keys():
            train_grdtruth[data_type] = data_dict[f'train_{data_type}']
        
    valid_grdtruth = dict()
    for data_type in ['rates', 'latent', 'spikes']:
        if f'valid_{data_type}' in data_dict.keys():
            train_grdtruth[data_type] = data_dict[f'valid_{data_type}']

    plotter_dict = {
        'train': plotter.Plotter(time=time, grdtruth=train_grdtruth),
        'valid': plotter.Plotter(time=time, grdtruth=valid_grdtruth)
        }
    
    return train_dl, valid_dl, input_size, plotter_dict

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_optimizer(model, hyperparams):
    
    betas = (
        hyperparams['optimizer']['beta1'], 
        hyperparams['optimizer']['beta2']
        )

    optimizer = opt.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hyperparams['optimizer']['lr_init'],
        betas=betas,
        eps=hyperparams['optimizer']['eps'])
    
    sched = scheduler.LFADS_Scheduler(
        optimizer      = optimizer,
        mode           = 'min',
        factor         = hyperparams['scheduler']['scheduler_factor'],
        patience       = hyperparams['scheduler']['scheduler_patience'],
        verbose        = True,
        threshold      = 1e-4,
        threshold_mode = 'abs',
        cooldown       = hyperparams['scheduler']['scheduler_cooldown'],
        min_lr         = hyperparams['scheduler']['lr_min']
        )

    return optimizer, sched

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def log_model_description(model):
    total_params = 0
    param_items = []
    for ix, (name, param) in enumerate(model.named_parameters()):
        param_items.append(
            f'{ix} {name} {list(param.shape)} '
            f'{param.numel()} {param.requires_grad}'
            )
        total_params += param.numel()
    
    param_str = '\n'.join(param_items)
    
    logger.info(f'{param_str}\nTotal number of parameters: {total_params}\n')

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_tensorboard(save_loc, plotter_dict, restart=True, use_tb=True):

    import importlib
    if use_tb and importlib.util.find_spec('torch.utils.tensorboard'):
        tb_folder = Path(save_loc, 'tensorboard')
        if tb_folder.is_dir() and restart:
            shutil.rmtree(str(tb_folder))
        tb_folder.mkdir(exist_ok=True, parents=True)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_folder)
        rm_plotter = plotter_dict
    else:
        writer = None
        rm_plotter = None
            
    return writer, rm_plotter

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def adjust_hyperparams(args, hyperparams, log_changed=True):

    hyperparams = copy.deepcopy(hyperparams)

    # Shared hyperparameters
    shared_hps = []
    if hyperparams['model_name'] in ['svlae', 'valpaca']:
        if hyperparams['model']['deep_width'] is not None:
            deep_width = hyperparams['model']['deep_width']
            hyperparams['model']['deep_g_encoder_size'] = deep_width
            hyperparams['model']['deep_c_encoder_size'] = deep_width
            shared_hps.append(f'deep_g_encoder_size={deep_width}')
            shared_hps.append(f'deep_c_encoder_size={deep_width}')
    
        if hyperparams['model']['obs_width'] is not None: # valpaca or svlae        
            obs_width = hyperparams['model']['obs_width']
            hyperparams['model']['obs_controller_size'] = obs_width
            hyperparams['model']['obs_encoder_size'] = obs_width
            shared_hps.append(f'obs_controller_size={obs_width}')
            shared_hps.append(f'obs_encoder_size={obs_width}')

    # Other hyperparameters
    hps = []
    if args.lr:
        lr = args.lr
        hyperparams['optimizer']['lr_init'] = lr
        hyperparams['scheduler']['lr_min']  = lr * 1e-3
        hps.append(f'lr={lr:.4f}')
        
    if args.kl_obs_dur:
        sched_dur = int(args.kl_obs_dur * args.kl_obs_dur_scale)
        hyperparams['objective']['kl_obs']['schedule_dur'] = sched_dur
        hps.append(f'kl_obs_dur={sched_dur}')

    if args.kl_obs_max:
        hyperparams['objective']['kl_obs']['max'] = args.kl_obs_max
        hps.append(f'kl_obs_max={args.kl_obs_max:.3f}')
        
    if args.kl_deep_max:
        hyperparams['objective']['kl_deep']['max'] = args.kl_deep_max
        hps.append(f'kl_deep_max={args.kl_deep_max:.3f}')
    
    if args.deep_start_p:
        deep_start = int(args.deep_start_p * args.deep_start_p_scale * \
            hyperparams['objective']['kl_obs']['schedule_dur'])
        hyperparams['objective']['kl_deep']['schedule_start'] = deep_start
        hyperparams['objective']['l2']['schedule_start'] = deep_start
        hyperparams['model']['deep_unfreeze_step'] = deep_start
        hps.append(f'deep_start={deep_start}')
        
    if args.l2_gen_scale:
        l2_gen_scale = args.l2_gen_scale
        hyperparams['objective']['l2_gen_scale'] = l2_gen_scale
        hps.append(f'l2_gen_scale={l2_gen_scale:.3f}')
    
    if args.l2_con_scale:
        l2_con_scale = args.l2_con_scale
        hyperparams['objective']['l2_con_scale'] = l2_con_scale
        hps.append(f'l2_con_scale={l2_con_scale:.3f}')
    
    if args.seed:
        hps.append(f'seed={args.seed}')

    if log_changed:
        # include all shared parameters in the log
        hp_str_pr = '\n'.join(shared_hps + hps)
        log_str = f'\nAdjusted hyperparameters:\n{hp_str_pr}'
        logging.info(log_str)

    # don't include the shared parameters in the string
    hp_str = '-'.join(hps).replace(' ', '').replace('=', '')
    hp_str = f'hp-{hp_str}'

    return hyperparams, hp_str


#-------------------------------------------------------------------
#-------------------------------------------------------------------

def generate_save_loc(model_params, data_path, output_dir, model_name='valpaca', 
                      hp_str='hp-', log_save_loc=True):

    data_name = Path(data_path).name
    if 'ospikes' in args.data_suffix:
        model_name = f'{model_name}_oasis'

    srcs = ['size', 'deep', 'obs', '_']
    targs = ['', 'd', 'o', '']
    hp_list = []
    for key, val in model_params.items():
        if 'size' not in key:
            continue
        for src, targ in zip(srcs, targs):
            key = key.replace(src, targ)
        hp_list.append(f'{key[:4]}{val}')

    run_name = '{}_{}'.format('_'.join(sorted(hp_list)), hp_str)
    save_loc = Path(output_dir, data_name, model_name, run_name)

    if log_save_loc:
        logger.info(
            f'\nModel training results will be saved under:\n{save_loc}.'
            )
    
    return save_loc, run_name

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def save_figs(save_loc, model, dl, plotter_dict):
    fig_folder = Path(save_loc, 'figs')
    
    if fig_folder.is_dir():
        shutil.rmtree(fig_folder)

    fig_folder.mkdir(parents=True)
    
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use('agg')
    fig_dict = plotter_dict['valid'].plot_summary(model= model, dl= dl)
    for k, v in fig_dict.items():
        if isinstance(v, plt.Figure):
            v.savefig(Path(fig_folder, f'{k}.svg'), bbox_inches="tight")

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def estimate_ar1_parameters(F, top_k=1):
    _, _, n_cells = F.shape
    damp = np.zeros(n_cells)
    var = np.zeros(n_cells)
    for i in range(n_cells):
        s = np.argsort(F[:, :, i].max(axis=1) - F[:, :, i].min(axis=1))
        top_k_trials = F[s][-top_k:, :, i]
        g_tmp, sn_tmp = [], []
        for trial in top_k_trials:
            g, sn = ar1.estimate_parameters(trial, p=1, fudge_factor=.98)
            g_tmp.append(g)
            sn_tmp.append(sn)
        damp[i] = np.median(g_tmp)
        var[i] = np.median(sn_tmp) ** 2
    return damp, var
            
#-------------------------------------------------------------------
#-------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=Path)
    parser.add_argument('-c', '--config', type=Path, 
                        help='path to hyperparameters')

    # optional parameters
    parser.add_argument('-o', '--output_dir', default=None, type=Path, 
                        help='data_path directory is used if output_dir is None'
                        )
    parser.add_argument('-m', '--model_name', default='valpaca')

    parser.add_argument('-t', '--use_tensorboard', action='store_true')
    parser.add_argument('-r', '--restart', action='store_true')
    
    parser.add_argument('--do_health_check', action='store_true')
    parser.add_argument('--max_epochs', default=2000, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--data_suffix', default='data', type=str)
    parser.add_argument('--detect_local_minimum', action='store_true')

    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--kl_deep_max', type=float, default=None)
    parser.add_argument('--kl_obs_max', type=float, default=None)
    parser.add_argument('--kl_obs_dur', type=int, default=None)
    parser.add_argument('--kl_obs_dur_scale', type=int, default=1)
    parser.add_argument('--deep_start_p', type=int, default=None)
    parser.add_argument('--deep_start_p_scale', type=float, default=1.0)
    parser.add_argument('--l2_gen_scale', type=float, default=None)
    parser.add_argument('--l2_con_scale', type=float, default=None)

    parser.add_argument('--orion', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')

    args = parser.parse_args()

    logger = util.get_logger_with_basic_format(level=args.log_level)

    main(args)

