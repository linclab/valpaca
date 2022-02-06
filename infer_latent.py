#!/usr/bin/env python

import argparse
import os
import pickle
import yaml

import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')

from utils import utils
from train_model import prep_model

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_dir', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-n', '--num_average', default=200, type=int)
parser.add_argument('--data_suffix', default='data', type=str)
parser.add_argument('--checkpoint', default='best', type=str)

def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model_name = os.path.split(os.path.dirname(args.model_dir.rstrip('/')))[1].split('_')[0]
    except:
        raise ValueError("Expected the next to last subdirectory in args.model_dir to be or "
            "to start with the name of the model, e.g. ../MODELNAME/sub_direc.")
    
    hp_path = os.path.join(args.model_dir, 'hyperparameters.yaml')
    hyperparams = utils.load_parameters(hp_path)
    
    data_dict = utils.read_data(args.data_path)
    
    # Re-instantiate model
    train_dl, valid_dl, plotter, model, objective = prep_model(model_name  = model_name,
                                                               data_dict   = data_dict,
                                                               data_suffix = args.data_suffix,
                                                               batch_size  = args.num_average,
                                                               device = device,
                                                               hyperparams = hyperparams)
    
    # Load parameters
    state_dict = torch.load(os.path.join(args.model_dir, 'checkpoints', '%s.pth'%args.checkpoint))
    print('checkpoint= %i'%state_dict['run_manager']['epoch'])
    
    model.load_state_dict(state_dict['net'])
    model.eval()
    
    # Dictionary for storing inferred states
    latent_dict = {'train' : {}, 'valid' : {}}
    latent_dict['train']['latent'] = []
    latent_dict['valid']['latent'] = []
    latent_dict['train']['recon'] = []
    latent_dict['valid']['recon'] = []
    if model_name == 'svlae' or model_name =='valpaca':
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
                result = infer_and_recon(x, batch_size=args.num_average, model=model)
                latent_dict[key]['latent'].append(result['latent'])
                latent_dict[key]['rates'].append(result['rates'])
#                 pdb.set_trace()
                if model_name == 'svlae' or model_name == 'valpaca':
                    latent_dict[key]['spikes'].append(result['spikes'])
                    latent_dict[key]['fluor'].append(result['fluor'])
                    
                if 'inputs' in result.keys():
                    if 'inputs' not in latent_dict[key].keys():
                        latent_dict[key]['inputs'] = []
                    latent_dict[key]['inputs'].append(result['inputs'])
                    
            if 'ospikes' in args.data_suffix:
                latent_dict[key]['spikes'] = data_dict[key+'_%s'%(args.data_suffix)]
                latent_dict[key]['fluor'] = data_dict[key+'_%s'%(args.data_suffix.replace('spikes', 'calcium'))]
    
    for dataset, latent_dict_k in latent_dict.items():
        for variable, val in latent_dict_k.items():
            latent_dict[dataset][variable] = np.array(latent_dict[dataset][variable])
    
    truth_dict = {}
    for key in ['train', 'valid']:
        truth_dict[key] = {}
        for var in ['rates', 'spikes', 'latent', 'fluor']:
            if var == 'fluor':
                if 'ospikes' in args.data_suffix:
                    data_dict_key = key + '_' + '_'.join(args.data_suffix.split('_')[:2])
                else:
                    data_dict_key = key + '_' + args.data_suffix
            else:
                data_dict_key = key + '_' + var
            
            
            if data_dict_key in data_dict.keys():
                truth_dict[key][var] = data_dict[data_dict_key]
                if var == 'latent':
                    if data_dict_key == 'train_latent':
                        L_train = fit_linear_model(np.concatenate(latent_dict[key][var]),
                                                   np.concatenate(truth_dict[key][var]))
                    latent_dict[key]['latent_aligned'] = L_train.predict(np.concatenate(latent_dict[key][var]))
                    truth_dict[key]['latent_aligned'] = np.concatenate(truth_dict[key]['latent'])
    
    results_dict = {}
    figs_dict = {}
    for key in ['train', 'valid']:
        results_dict[key], figs_dict[key] = compare_truth(latent_dict[key], truth_dict[key])
        for var, sub_dict in figs_dict[key].items():
            save_path = os.path.join(args.model_dir, 'figs', '%s_%s_rsq.svg'%(key, var))
            sub_dict['fig'].savefig(save_path)
            print('saved figure at ' + save_path)
                
    factor_size = model.factor_size
    if hasattr(model, 'u_latent_size'):
        u_size = model.u_latent_size
    elif hasattr(model, 'lfads'):
        u_size = model.lfads.u_latent_size
    elif hasattr(model, 'deep_model'):
        u_size = model.deep_model.u_latent_size

    train_size, steps_size, state_size = data_dict['train_%s'%args.data_suffix].shape
    valid_size, steps_size, state_size = data_dict['valid_%s'%args.data_suffix].shape

    data_size = train_size + valid_size

    factors = np.zeros((data_size, steps_size, factor_size))
    rates   = np.zeros((data_size, steps_size, state_size))
        
    if 'train_idx' in data_dict.keys() and 'valid_idx' in data_dict.keys():
        
        if u_size > 0:
            inputs  = np.zeros((data_size, steps_size, state_size))

        if model_name == 'svlae' or model_name=='valpaca' or 'ospikes' in args.data_suffix:
            spikes  = np.zeros((data_size, steps_size, state_size))
            fluor   = np.zeros((data_size, steps_size, state_size))
        
        train_idx = data_dict['train_idx']
        valid_idx = data_dict['valid_idx']
        
        latent_dict['ordered'] = {}

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

        if model_name == 'svlae' or model_name =='valpaca' or 'ospikes' in args.data_suffix:
            spikes[train_idx] = latent_dict['train']['spikes']
            spikes[valid_idx] = latent_dict['valid']['spikes']
            latent_dict['ordered']['spikes'] = spikes
            fluor[train_idx] = latent_dict['train']['fluor']
            fluor[valid_idx] = latent_dict['valid']['fluor']
            latent_dict['ordered']['fluor'] = fluor
        
    if factor_size == 3:
        for key in ['train', 'valid']:
            fig = plot_3d(X=latent_dict[key]['latent_aligned'].T, title='rsq= %.3f'%results_dict[key]['latent_aligned']['rsq'])
            fig.savefig(os.path.join(args.model_dir, 'figs', '%s_factors3d_rsq.svg'%(key)))
        
    pickle.dump(latent_dict, file=open(os.path.join(args.model_dir, 'latent.pkl'), 'wb'))
    yaml.dump(results_dict, open(os.path.join(args.model_dir, 'results.yaml'), 'w'), default_flow_style=False)
    
def infer_and_recon(sample, batch_size, model):
    batch = utils.batchify_sample(sample, batch_size)
    recon, (factors, inputs) = model(batch)
    result = {}
    result['latent'] = factors.mean(dim=1).cpu().numpy()
    result['rates'] = recon['rates'].mean(dim=1).cpu().numpy()
    if inputs is not None:
        result['inputs'] = inputs.mean(dim=1).cpu().numpy()
    if 'spikes' in recon.keys():
        result['spikes'] = recon['spikes'].mean(dim=1).cpu().numpy()
        result['fluor'] = recon['data'].mean(dim=0).cpu().numpy()
    return result
    
from sklearn.linear_model import LinearRegression

def align_linear(x, y=None, L=None):
    if L is not None:
        return L.predict(x)
    elif y is not None:
        return fit_linear_model(x, y).predict(x)
    L = fit_linear_model(x, y)

def fit_linear_model(x, y):
    L = LinearRegression().fit(x, y)
    return L
    
def compute_rsquared(x, y, model=None):
    
    if model is not None:
        return model(x, y).score()
    else:
        return np.corrcoef(x, y)[0, 1]**2
        

def plot_3d(X, Y=None, figsize = (12, 12), view = (None, None), title=None):

    '''TBC'''

    assert X.shape[0] == 3, 'X data must be 3 dimensional'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0], X[1], X[2], lw=0.1)
    if Y:
        assert Y.shape[0] == 3, 'Y data must be 3 dimensional'
        ax.plot(Y[0], Y[1], Y[2], lw=0.1, alpha=0.7)
    ax.view_init(view[0], view[1])
    if title:
        ax.set_title(title, fontsize=16)
    
    return fig
    
def plot_rsquared(x, y, figsize=(4,4), ms=1, title=''):

    '''
    TBC
    '''

    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, '.', ms=ms, color='dimgrey', rasterized=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Reconstruction')
    plt.ylabel('Truth')

    return fig

def compare_truth(latent_dict, truth_dict):
    results_dict = {}
    figs_dict = {}
    def compare(key, x_dict, y_dict, save=True):
        results_dict = {}
        figs_dict = {}
        results_dict['rsq'] = compute_rsquared(x= x_dict[key].flatten(),
                                               y= y_dict[key].flatten())
        figs_dict['fig'] = plot_rsquared(x_dict[key].flatten(),
                                            y_dict[key].flatten(),
                                            title='rsq= %.3f'%results_dict['rsq'])
        
        return results_dict, figs_dict
    
    for var in ['rates', 'spikes', 'fluor', 'latent_aligned']:
        if var in latent_dict.keys() and var in truth_dict.keys():
            results_dict[var], figs_dict[var] = compare(var, latent_dict, truth_dict)
            
    return results_dict, figs_dict

    
if __name__ == '__main__':
    main()