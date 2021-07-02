import h5py
import os
import yaml
import pickle

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('../')
from utils import read_data

import argparse

linclab_red = '#e84924ff'  # ground-truth
linclab_blue = '#37a1d0ff' # reconstructed
linclab_grey = '#969696ff' # other

parser = argparse.ArgumentParser()
parser.add_argument('--dt_sys', default=0.1, type=float)
parser.add_argument('--dt_cal', default=0.1, type=float)
parser.add_argument('--rate', default=2.0, type=float)
parser.add_argument('--sigma', default=5.0, type=float)
parser.add_argument('--model_dir',default='models', type=str )

def main():
    args = parser.parse_args()
    sys_name = 'lorenz'
    
    project_dir = '/'.join((os.environ['HOME'], 'valpaca'))
    model_dir = '/'.join((project_dir, 'models'))
    
    s = 1.0/args.sigma

    ou = 'ou_t0.3_s%.1f'%s

    lfads_model_desc = 'cenc0_cont0_fact3_genc64_gene64_glat64_ulat0_hp-'
    valpa_model_desc = 'dcen0_dcon0_dgen64_dgla64_dula0_fact3_gene64_ocon32_oenc32_olat64_hp-'
    base_name = 'results.yaml'

    lfads_name = 'lfads'
    gauss_name = 'lfads-gaussian'
    oasis_name = '_'.join((lfads_name, 'oasis'))
    valpa_name = 'valpaca'
    
    ar1  = 'fluor_ar1'
    hill = 'fluor_hillar1'
    
    conditions = 'sys%.4f_cal%.4f_sig%.1f_base%.1f'%(args.dt_sys, args.dt_cal, args.sigma, args.rate)

    rsq = {}
    
    seeds = range(1000, 17000, 1000)

    for seed in seeds:
        data_name = '_'.join((sys_name, 'seed'+str(seed), conditions))
        ar1_name  = '_'.join((data_name, ar1, ou, 'n'))
        hill_name = '_'.join((data_name, hill, ou, 'n'))
        
        lfads_filename = '/'.join((model_dir, ar1_name, lfads_name, lfads_model_desc, base_name))
        
        gauss_ar1_filename = '/'.join((model_dir, ar1_name, gauss_name, lfads_model_desc, base_name))
        oasis_ar1_filename = '/'.join((model_dir, ar1_name, oasis_name, lfads_model_desc, base_name))
        valpa_ar1_filename = '/'.join((model_dir, ar1_name, valpa_name, valpa_model_desc, base_name))
        
        gauss_hill_filename = '/'.join((model_dir, hill_name, gauss_name, lfads_model_desc, base_name))
        oasis_hill_filename = '/'.join((model_dir, hill_name, oasis_name, lfads_model_desc, base_name))
        valpa_hill_filename = '/'.join((model_dir, hill_name, valpa_name, valpa_model_desc, base_name))

        print(seed)
        
        for filename, column in zip([lfads_filename, gauss_ar1_filename, oasis_ar1_filename, valpa_ar1_filename, gauss_hill_filename, oasis_hill_filename, valpa_hill_filename], 
                                     ['lfads', 'gauss_ar1', 'oasis_ar1', 'valpaca_ar1', 'gauss_hill','oasis_hill', 'valpaca_hill']):
            if os.path.exists(filename):
                myfile = open(filename, 'rb')
                results = yaml.load(myfile, Loader=yaml.Loader)
                myfile.close()
                if column not in rsq.keys():
                    rsq[column] = {}
                for key in results['valid'].keys():
                    if key not in rsq[column].keys():
                        rsq[column][key] = {}
                    rsq[column][key][seed] = results['valid'][key]['rsq']
#                     print(rsq[column][key][seed])
            else:
                print('%s does not exist'%filename)
            
    yaml.dump(rsq, open('./%s_%s_rsq.yaml'%(sys_name, conditions), 'w'))
    
    print('AR1 t-test')
    for var in ['latent_aligned', 'spikes', 'rates', 'fluor']:
        a = [rsq['oasis_ar1'][var][s] for s in seeds]
        b = [rsq['valpaca_ar1'][var][s] for s in seeds]
        t, p = ttest_rel(a=a, b=b)
        print('%s: t_%i=%.3f, p=%.3f'%(var, len(seeds)-1, t, p)) 
        
    print('Hill t-test')
    for var in ['latent_aligned', 'spikes', 'rates', 'fluor']:
        a = [rsq['oasis_hill'][var][s] for s in seeds]
        b = [rsq['valpaca_hill'][var][s] for s in seeds]
        t, p = ttest_rel(a=a, b=b)
        print('%s: t_%i=%.3f, p=%.3f'%(var, len(seeds)-1, t, p)) 
    
    rows = list(rsq.keys())
    rows.sort()
    
    df_mean = pd.DataFrame([pd.DataFrame(rsq[row]).mean() for row in rows], index=rows)
    df_std  = pd.DataFrame([pd.DataFrame(rsq[row]).std() for row in rows], index=rows)
    
    df = df_mean.round(3).astype(str) + ' $\pm$ ' + (df_std/np.sqrt(20)).round(3).astype(str)
    
    df = df.reindex(axis=1, labels=['latent_aligned', 'spikes', 'rates', 'fluor'])
    df = df.reindex(axis=0, labels=['lfads', 'gauss_ar1', 'oasis_ar1', 'valpaca_ar1', 'gauss_hill','oasis_hill', 'valpaca_hill'])

    df = df.rename(index={'lfads' : 'LFADS',
                          'gauss_ar1'   : 'Linear Gaussian-LFADS',
                          'oasis_ar1'   : 'Linear OASIS+LFADS',
                          'valpaca_ar1' : 'Linear VaLPACa',
                          'gauss_hill'   : 'Nonlinear Gaussian-LFADS',
                          'oasis_hill'   : 'Nonlinear OASIS+LFADS',
                          'valpaca_hill' : 'Nonlinear VaLPACa'},
                   columns={'latent_aligned' : 'Lorenz state', 'rates' : 'Firing Rates', 'fluor' : 'Fluorescence', 'spikes': 'Spike Counts'})
    
    df.to_latex(open('%s_%s_rsq.tex'%(sys_name, conditions), 'w+'))
    
    seed = 2000
    data_name = '_'.join((sys_name, 'seed'+str(seed), conditions))
    
    data_ar1_name = data_name + '_fluor_ar1_ou_t0.3_s%s_n'%('2.0' if args.sigma=='5.0' else '2.0')
    data_hill_name = data_name + '_fluor_hillar1_ou_t0.3_s%s_n'%('2.0' if args.sigma=='5.0' else '2.0')
    ar1_name  = '_'.join((data_name, ar1, ou, 'n'))
    hill_name = '_'.join((data_name, hill, ou, 'n'))
    
    base_name = 'latent.pkl'

    oasis_ar1_filename = '/'.join((model_dir, ar1_name, oasis_name, lfads_model_desc, base_name))
    valpa_ar1_filename = '/'.join((model_dir, ar1_name, valpa_name, valpa_model_desc, base_name))
    oasis_hill_filename = '/'.join((model_dir, hill_name, oasis_name, lfads_model_desc, base_name))
    valpa_hill_filename = '/'.join((model_dir, hill_name, valpa_name, valpa_model_desc, base_name))
    
    data_ar1_dict = read_data('../synth_data/%s'%data_ar1_name)
    data_hill_dict = read_data('../synth_data/%s'%data_hill_name)
#     import pdb; pdb.set_trace()
    
    oasis_ar1_latent_dict = pickle.load(open(oasis_ar1_filename, 'rb'))
    valpa_ar1_latent_dict = pickle.load(open(valpa_ar1_filename, 'rb'))
    oasis_hill_latent_dict = pickle.load(open(oasis_hill_filename, 'rb'))
    valpa_hill_latent_dict = pickle.load(open(valpa_hill_filename, 'rb'))
    
    fig_valpa_ar1 = plot_lorenz_examples(latent_dict=valpa_ar1_latent_dict, data_dict=data_ar1_dict, hill=False)
    fig_valpa_ar1.savefig('./lorenz_examples_valpa_ar1_%s_%s.svg'%(sys_name, conditions))
    fig_oasis_ar1 = plot_lorenz_examples(latent_dict=oasis_ar1_latent_dict, data_dict=data_ar1_dict, hill=False)
    fig_oasis_ar1.savefig('./lorenz_examples_oasis_ar1_%s_%s.svg'%(sys_name, conditions))
    
    fig_valpa_hill = plot_lorenz_examples(latent_dict=valpa_hill_latent_dict, data_dict=data_hill_dict, hill=True)
    fig_valpa_hill.savefig('./lorenz_examples_valpa_hill_%s_%s.svg'%(sys_name, conditions))
    fig_oasis_hill = plot_lorenz_examples(latent_dict=oasis_hill_latent_dict, data_dict=data_hill_dict, hill=True)
    fig_oasis_hill.savefig('./lorenz_examples_oasis_hill_%s_%s.svg'%(sys_name, conditions))
    
#     import pdb; pdb.set_trace()
                          
def plot_lorenz_examples(latent_dict, data_dict, hill, num_traces_to_show=8, figsize=(5.5,2)):
    fig, axs = plt.subplots(nrows=2, ncols =4, figsize=figsize)

    time = np.linspace(0, 3, 90)

    num_traces_to_show=8
    ax = axs[0,0]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)
    plt.plot(time, s + data_dict['valid_fluor_' + '%s'%('hillar1' if hill else 'ar1')][0][:, :num_traces_to_show], color=linclab_red, lw=1)
    plt.plot(time, s + latent_dict['valid']['fluor'][0][:, :num_traces_to_show], color=linclab_blue, lw=0.75)
    plt.yticks(np.concatenate([s, s[1:]-1, [s[-1] + s[1]-1]]), [s[0], s[1]-1, ''*6])
    ax.set_xticklabels([])
    plt.ylabel('normalized dF/F', fontsize=6)

    ax = axs[0,1]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)*10
    plt.plot(time, s + data_dict['valid_rates'][0][:, :num_traces_to_show], color=linclab_red, lw=1)
    plt.plot(time, s + latent_dict['valid']['rates'][0][:, :num_traces_to_show], color=linclab_blue, lw=0.75)
    plt.yticks(np.concatenate([s, s[1:]-2, [s[-1] + s[1]-2]]), [s[0], s[1]-2, ''*6])
    ax.set_xticklabels([])
    plt.ylabel('Spike Rate (Hz)', fontsize=6)

    ax = axs[1,0]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)*4
    plt.plot(time, s + data_dict['valid_spikes'][0][:, :num_traces_to_show], color=linclab_red, lw=1)
    plt.plot(time, s + latent_dict['valid']['spikes'][0][:, :num_traces_to_show], color=linclab_blue, lw=0.75)
    plt.yticks(np.concatenate([s, s[1:]-1, [s[-1] + s[1]-1]]), [s[0], s[1]-1, ''*6])
    plt.ylabel('Spike Counts', fontsize=6)
    plt.xlabel('Time (s)', fontsize=6)

    ax = axs[1,1]
    plt.sca(ax)
    s = np.arange(3)
    plt.plot(time, s + data_dict['valid_latent'][0], color=linclab_red, lw=1)
    plt.plot(time, s + latent_dict['valid']['latent_aligned'][:90], color=linclab_blue, lw=0.75)
    plt.ylabel('Factors', fontsize=6)
    plt.xlabel('Time (s)', fontsize=6)

    ax.yaxis.set_ticks_position('none')
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    
    for ax in axs.ravel():
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plt.sca(ax)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

    ax = axs[0, 2]
    plt.sca(ax)
    plt.plot(latent_dict['valid']['fluor'].ravel(), data_dict['valid_fluor_' + '%s'%('hillar1' if hill else 'ar1')].ravel(), '.', ms=0.5, color=linclab_grey, rasterized=True)
    plt.title('Fluorescence', fontsize=8)
    plt.ylabel('Truth', fontsize=6)
    plt.xlabel('Reconstruction', fontsize=6)
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    ax = axs[0, 3]
    plt.sca(ax)
    plt.plot(latent_dict['valid']['rates'].ravel(), data_dict['valid_rates'].ravel(), '.', ms=0.5, color=linclab_grey, rasterized=True)
    plt.title('Rates', fontsize=8)
    plt.ylabel('Truth', fontsize=6)
    plt.xlabel('Reconstruction', fontsize=6)
    plt.xticks([0, 40])
    plt.yticks([0, 40])

    ax = axs[1, 2]
    plt.sca(ax)
    plt.plot(latent_dict['valid']['spikes'].ravel(), data_dict['valid_spikes'].ravel(), '.', ms=0.5, color=linclab_grey, rasterized=True)
    plt.title('Spike Counts', fontsize=8)
    plt.ylabel('Truth', fontsize=6)
    plt.xlabel('Reconstruction', fontsize=6)
    plt.xticks([0, 9])
    plt.yticks([0, 9])

    ax = axs[1, 3]
    plt.sca(ax)
    plt.plot(latent_dict['valid']['latent_aligned'].ravel(), data_dict['valid_latent'].ravel(), '.', ms=0.5, color=linclab_grey, rasterized=True)
    plt.title('Lorenz State', fontsize=8)
    plt.ylabel('Truth', fontsize=6)
    plt.xlabel('Reconstruction', fontsize=6)
    plt.xticks([-1, 1])
    plt.yticks([-1, 1])

    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    return fig
        
if __name__ == '__main__':
    main()