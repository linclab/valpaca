#!/usr/bin/env python

import argparse
import copy
from pathlib import Path
import logging

import numpy as np
import pickle as pkl
import torch
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')

from utils import util
from train_model import prep_model


logger = logging.getLogger(__name__)


def main(args):

    # set logger to the specified level
    util.set_logger_level(logger, level=args.log_level)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_dict = util.read_data(args.data_path)

    infer_latents_from_model(
        data_dict, model_dir=args.model_dir, output_dir=args.output_dir, 
        data_suffix=args.data_suffix, num_average=args.num_average, 
        checkpoint=args.checkpoint, device=device
        )


def infer_latents_from_model(data_dict, model_dir, output_dir=None, 
                             data_suffix='data', num_average=200, 
                             checkpoint='best', device='cpu'):

    model_name = get_model_name(model_dir)
    hyperparams = util.load_parameters(
        Path(model_dir, 'hyperparameters.yaml')
        )
    
    if output_dir is None:
        output_dir = model_dir
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Re-instantiate model
    train_dl, valid_dl, _, model, _ = prep_model(
        model_name  = model_name,
        data_dict   = data_dict,
        data_suffix = data_suffix,
        batch_size  = num_average,
        device      = device,
        hyperparams = hyperparams
        )
    
    # Load parameters 
    state_dict = torch.load(
        Path(model_dir, 'checkpoints', f'{checkpoint}.pth'), 
        map_location=torch.device(device)
        )
    logger.info(
        f'Checkpoint ({checkpoint}): {state_dict["run_manager"]["epoch"]}'
        )
    
    model.load_state_dict(state_dict['net'])

    model.eval()

    # generate latent dict
    latent_dict = get_latent_dict(
        data_dict, model, train_dl, valid_dl, model_name=model_name, 
        data_suffix=data_suffix, num_average=num_average
        )
    
    # generate result dict, add data to latent dict and save dictionaries
    get_save_results_latent_dicts(
        model, data_dict, latent_dict, output_dir=output_dir, 
        model_name=model_name, data_suffix=data_suffix
        )


def get_save_results_latent_dicts(model, data_dict, latent_dict, output_dir='.', 
                                  model_name='valpaca', data_suffix='data'):

    latent_dict = copy.deepcopy(latent_dict)

    # generate ground truth dict, and add data to latent dict
    grdtruth_dict, latent_dict = get_grdtruth_dict(
        data_dict, latent_dict, data_suffix=data_suffix
        )
    
    results_dict = dict()
    figs_dict = dict()
    for key in ['train', 'valid']:
        results_dict[key], figs_dict[key] = compare_recon_grdtruth(
            latent_dict[key], grdtruth_dict[key]
            )
        for var, sub_dict in figs_dict[key].items():
            save_path = Path(output_dir, 'figs', f'{key}_{var}_rsq.svg')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            sub_dict['fig'].savefig(save_path, bbox_inches="tight")
            logger.info(f'Figure saved under: {save_path}.')
                
    factor_size = model.factor_size
    if hasattr(model, 'u_latent_size'):
        u_size = model.u_latent_size
    elif hasattr(model, 'lfads'):
        u_size = model.lfads.u_latent_size
    elif hasattr(model, 'deep_model'):
        u_size = model.deep_model.u_latent_size

    train_size, steps_size, state_size = data_dict[f'train_{data_suffix}'].shape
    valid_size, steps_size, state_size = data_dict[f'valid_{data_suffix}'].shape

    data_size = train_size + valid_size

    factors = np.zeros((data_size, steps_size, factor_size))
    rates   = np.zeros((data_size, steps_size, state_size))
        
    if 'train_idx' in data_dict.keys() and 'valid_idx' in data_dict.keys():
        if u_size > 0:
            inputs  = np.zeros((data_size, steps_size, state_size))

        if (model_name == 'svlae' or model_name == 'valpaca' or 
            'ospikes' in data_suffix):
            spikes  = np.zeros((data_size, steps_size, state_size))
            fluor   = np.zeros((data_size, steps_size, state_size))
        
        train_idx = data_dict['train_idx']
        valid_idx = data_dict['valid_idx']
        
        latent_dict['ordered'] = dict()

        factors[train_idx] = latent_dict['train']['latent']
        factors[valid_idx] = latent_dict['valid']['latent']
        latent_dict['ordered']['factors'] = factors

        rates[train_idx] = latent_dict['train']['rates']
        rates[valid_idx] = latent_dict['valid']['rates']
        latent_dict['ordered']['rates'] = rates

        if u_size > 0:
            inputs[train_idx] = latent_dict['train']['inputs']
            inputs[valid_idx] = latent_dict['valid']['inputs']
            latent_dict['ordered']['inputs'] = inputs

        if model_name == 'svlae' or model_name =='valpaca' or 'ospikes' in data_suffix:
            spikes[train_idx] = latent_dict['train']['spikes']
            spikes[valid_idx] = latent_dict['valid']['spikes']
            latent_dict['ordered']['spikes'] = spikes
            fluor[train_idx] = latent_dict['train']['fluor']
            fluor[valid_idx] = latent_dict['valid']['fluor']
            latent_dict['ordered']['fluor'] = fluor
        
    if factor_size == 3:
        for key in ['train', 'valid']:
            fig = plot_3d(
                X=latent_dict[key]['latent_aligned'].T, 
                title=f'rsq={results_dict[key]["latent_aligned"]["rsq"]:.3f}'
            )
            save_path = Path(output_dir, 'figs', f'{key}_factors3d_rsq.svg')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_path, bbox_inches="tight")

    # save data dictionaries
    with open(Path(output_dir, 'latent.pkl'), 'wb') as f:
        pkl.dump(latent_dict, f)

    with open(Path(output_dir, 'results.yaml'), 'w') as f:
        yaml.dump(results_dict, f, default_flow_style=False)
    
    return results_dict, latent_dict


def get_grdtruth_dict(data_dict, latent_dict, data_suffix='data'):

    latent_dict = copy.deepcopy(latent_dict)

    grdtruth_dict = dict()
    for key in ['train', 'valid']:
        grdtruth_dict[key] = dict()
        for var in ['rates', 'spikes', 'latent', 'fluor']:
            if var == 'fluor':
                data_suffix_use = data_suffix
                if 'ospikes' in data_suffix:
                    data_suffix_use = data_suffix.split('_')[:2]
                data_dict_key = f'{key}_{data_suffix_use}'
            else:
                data_dict_key = f'{key}_{var}'
                 
            if data_dict_key in data_dict.keys():
                grdtruth_dict[key][var] = data_dict[data_dict_key]
                if var == 'latent':
                    if data_dict_key == 'train_latent':
                        L_train = fit_linear_model(
                            np.concatenate(latent_dict[key][var]),
                            np.concatenate(grdtruth_dict[key][var])
                            )
                    latent_dict[key]['latent_aligned'] = L_train.predict(
                        np.concatenate(latent_dict[key][var])
                        )
                    grdtruth_dict[key]['latent_aligned'] = \
                        np.concatenate(grdtruth_dict[key]['latent'])

    return grdtruth_dict, latent_dict


def get_latent_dict(data_dict, model, train_dl, valid_dl, model_name='valpaca', 
                    data_suffix='data', num_average=200):

    # Dictionary for storing inferred states
    latent_dict = {'train' : dict(), 'valid' : dict()}
    latent_dict['train']['latent'] = []
    latent_dict['valid']['latent'] = []
    latent_dict['train']['recon'] = []
    latent_dict['valid']['recon'] = []
    if model_name == 'svlae' or model_name == 'valpaca':
        latent_dict['train']['spikes'] = []
        latent_dict['valid']['spikes'] = []

    with torch.no_grad():
        for dl, key in ((train_dl, 'train'), (valid_dl, 'valid')):
            latent_dict[key]['latent'] = []
            latent_dict[key]['rates'] = []
            if model_name == 'svlae' or model_name =='valpaca':
                latent_dict[key]['spikes'] = []
                latent_dict[key]['fluor'] = []
            for x in dl.dataset:
                x = x[0]
                result = infer_and_recon(
                    x, batch_size=num_average, model=model
                    )
                latent_dict[key]['latent'].append(result['latent'])
                latent_dict[key]['rates'].append(result['rates'])

                if model_name == 'svlae' or model_name == 'valpaca':
                    latent_dict[key]['spikes'].append(result['spikes'])
                    latent_dict[key]['fluor'].append(result['fluor'])
                    
                if 'inputs' in result.keys():
                    if 'inputs' not in latent_dict[key].keys():
                        latent_dict[key]['inputs'] = []
                    latent_dict[key]['inputs'].append(result['inputs'])
                    
            if 'ospikes' in data_suffix:
                latent_dict[key]['spikes'] = data_dict[f'{key}_{data_suffix}']
                calc_suffix = data_suffix.replace('spikes', 'calcium')
                latent_dict[key]['fluor'] = data_dict[f'{key}_{calc_suffix}']
    
    for dataset, latent_dict_k in latent_dict.items():
        for variable in latent_dict_k.keys():
            latent_dict[dataset][variable] = np.array(
                latent_dict[dataset][variable]
                )

    return latent_dict


def infer_and_recon(sample, batch_size, model):
    batch = util.batchify_sample(sample, batch_size)
    recon, (factors, inputs) = model(batch)
    result = dict()
    result['latent'] = factors.mean(dim=1).cpu().numpy()
    result['rates'] = recon['rates'].mean(dim=1).cpu().numpy()
    if inputs is not None:
        result['inputs'] = inputs.mean(dim=1).cpu().numpy()
    if 'spikes' in recon.keys():
        result['spikes'] = recon['spikes'].mean(dim=1).cpu().numpy()
        result['fluor'] = recon['data'].mean(dim=0).cpu().numpy()
    return result


def align_linear(x, y=None, L=None):
    if L is not None:
        return L.predict(x)
    elif y is not None:
        return fit_linear_model(x, y).predict(x)
    L = fit_linear_model(x, y)


def fit_linear_model(x, y):
    from sklearn.linear_model import LinearRegression
    L = LinearRegression().fit(x, y)
    return L
    

def compute_rsquared(x, y, model=None):
    
    if model is not None:
        return model(x, y).score()
    else:
        return np.corrcoef(x, y)[0, 1] ** 2
        

def plot_3d(X, Y=None, figsize=(12, 12), view=(None, None), title=None):

    if not X.shape[0] == 3:
        raise ValueError('The first dimension of X must have length 3.')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0], X[1], X[2], lw=0.1)

    if Y:
        if not Y.shape[0] == 3:
            raise ValueError('The first dimension of Y must have length 3.')
        ax.plot(Y[0], Y[1], Y[2], lw=0.1, alpha=0.7)

    ax.view_init(view[0], view[1])
    if title:
        ax.set_title(title, fontsize=16)
    
    return fig
    

def plot_rsquared(x, y, figsize=(4, 4), ms=1, title=''):

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, '.', ms=ms, color='dimgrey', rasterized=True)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Reconstruction')
    ax.set_ylabel('Ground Truth')

    return fig


def compare_keys(key, x_dict, y_dict):
    results_dict = dict()
    figs_dict = dict()
    results_dict['rsq'] = compute_rsquared(
        x=x_dict[key].flatten(),
        y=y_dict[key].flatten()
        )
    figs_dict['fig'] = plot_rsquared(
        x_dict[key].flatten(),
        y_dict[key].flatten(),
        title=f'rsq={results_dict["rsq"]:.3f}'
        )
    
    return results_dict, figs_dict


def compare_recon_grdtruth(latent_dict, grdtruth_dict):
    
    results_dict = dict()
    figs_dict = dict()
    for var in ['rates', 'spikes', 'fluor', 'latent_aligned']:
        if var in latent_dict.keys() and var in grdtruth_dict.keys():
            results_dict[var], figs_dict[var] = compare_keys(
                var, latent_dict, grdtruth_dict
                )
            
    return results_dict, figs_dict


def get_model_name(model_dir):

    model_dir_parts = Path(model_dir).parts
    if len(model_dir_parts) < 2:
        raise(
            'Expected the next to last subdirectory in model_dir to start '
            'with the name of the model, e.g. ../MODELNAME_suffix/sub_direc.'
            )
    
    model_name = model_dir_parts[-2].split('_')[0]

    return model_name

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=Path)

    # optional parameters
    parser.add_argument('-m', '--model_dir', default='models', type=Path)
    parser.add_argument('-o', '--output_dir', default=None, type=Path, 
                        help='model_dir is used if output_dir is None')
    parser.add_argument('-n', '--num_average', default=200, type=int)

    parser.add_argument('--data_suffix', default='data', type=str)
    parser.add_argument('--checkpoint', default='best', type=str)
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')

    args = parser.parse_args()

    logger = util.get_logger_with_basic_format(level=args.log_level)

    main(args)

