#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
import yaml
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import matplotlib.pyplot as plt
import matplotlib
from zmq import fd_sockopts
matplotlib.use('agg')

sys.path.append('..')
from utils import util


COLORS = {
    'linc_red'  : '#E84924', # ground truth
    'linc_blue' : '#37A1D0', # reconstructed
    'linc_grey' : '#969696', # other
}

SAVE_KWARGS = {
    'bbox_inches': 'tight',
    'facecolor'  : 'w',
    'transparent': False,
    'format'     : 'svg',
}

PLUS_MIN = u'\u00B1'

MODEL_DICT = {
    'lfads'       : 'LFADS',
    'gauss_ar1'   : 'Linear Gaussian-LFADS',
    'oasis_ar1'   : 'Linear OASIS+LFADS',
    'valpaca_ar1' : 'Linear VaLPACa',
    'gauss_hill'  : 'Nonlinear Gaussian-LFADS',
    'oasis_hill'  : 'Nonlinear OASIS+LFADS',
    'valpaca_hill': 'Nonlinear VaLPACa',
}

DATA_DICT = {
    'latent_aligned': 'Lorenz state', 
    'rates'         : 'Firing Rates', 
    'fluor'         : 'Fluorescence', 
    'spikes'        : 'Spike Counts',
    }

CALC_DYN_DICT = {
    'ar1' : 'AR1',
    'hill': 'Hill',
}

MODEL_NAME_DICT = {
    'lfads'  : 'lfads',
    'gauss'  : 'lfads-gaussian',
    'oasis'  : 'lfads_oasis',
    'valpaca': 'valpaca'
}

PARAM_NAME_DICT = {
    'lfads'  : 'cenc0_cont0_fact3_genc64_gene64_glat64_ulat0_hp-',
    'valpaca': 'dcen0_dcon0_dgen64_dgla64_dula0_fact3_gene64_ocon32_oenc32_olat64_hp-',
}

RESULTS_FILENAME = 'results.yaml'
LATENTS_FILENAME = 'latent.pkl'
DATA_NAME = 'lorenz'
SEEDS = np.arange(1, 17) * 1000
TIME = np.linspace(0, 3, 90)

# specific analysis/plotting parameters
ANALYSIS_MODEL_NAMES = ['oasis', 'valpaca']
PLOT_SEED = 2000


logger = logging.getLogger(__name__)


##--------MAIN--------##
#############################################
def main(args):

    # set logger to the specified level
    util.set_logger_level(logger, level=args.log_level)
    
    seeds = SEEDS

    # get some parameter strings
    ou_str = get_ou_str(args.sigma)
    cond_str = get_conditions_str(
        dt_sys=args.dt_sys, dt_cal=args.dt_cal, sigma=args.sigma, 
        rate=args.rate
        )

    if args.output_dir is None:
        args.output_dir = args.model_dir

    # collect and save RSQ dictionary
    savepath = Path(args.output_dir, f'{DATA_NAME}_{cond_str}_rsq.yaml')
    rsq = collect_rsq(
        cond_str, ou_str, model_dir=args.model_dir, seeds=seeds, 
        savepath=savepath
        )

    # log t-test results
    util.set_logger_level(logger, level="info")
    run_ttests(rsq, seeds=seeds, model_names=ANALYSIS_MODEL_NAMES)
    util.set_logger_level(logger, level=args.log_level)

    # compile and save statistics
    savepath = f'{DATA_NAME}_{cond_str}_rsq.tex'
    compile_df(rsq, savepath=savepath)

    # plot results
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    for model in ANALYSIS_MODEL_NAMES:
        for calc_dyn in CALC_DYN_DICT.keys():
            latent_dict = util.load_latent(args.model_dir)

            latent_path = get_model_filepath(
                cond_str, ou_str, model_dir=args.model_dir, model=model, 
                calc_dyn=calc_dyn, seed=PLOT_SEED, filetype='latents')
            
            latent_dict = util.load_latent(latent_path)

            data_dict = get_data_dict(args.data_dir, calc_dyn=calc_dyn)

            fig = plot_lorenz_examples(
                latent_dict=latent_dict, data_dict=data_dict, calc_dyn=calc_dyn
                )

            savepath = Path(
                args.output_dir, 
                f'lorenz_examples_{model}_{calc_dyn}_{DATA_NAME}_{cond_str}.svg'
                )
            fig.savefig(savepath, **SAVE_KWARGS)


##--------ANALYSIS FUNCTIONS--------##
#############################################
def collect_rsq(conditions_str, ou_str, model_dir='.', seeds=SEEDS, 
                savepath=None):

    logger.info("Collecting RSQ")

    rsq = dict()
    for seed in seeds:
        logger.debug(f'R-squared results')        
        logger.info(f'Seed: {seed}')

        for model in MODEL_NAME_DICT.keys():
            for calc_dyn in CALC_DYN_DICT.keys():                
                filepath, model_name = get_model_filepath(
                    conditions_str, ou_str, model_dir=model_dir, model=model, 
                    calc_dyn=calc_dyn, seed=seed, filetype='results'
                    )

            if Path(filepath).is_file():
                with open(filepath, 'rb') as f:
                    results = yaml.load(f, Loader=yaml.Loader)

                if model_name not in rsq.keys():
                    rsq[model_name] = dict()
                    
                for key in results['valid'].keys():
                    if key not in rsq[model_name].keys():
                        rsq[model_name][key] = dict()
                    rsq[model_name][key][seed] = results['valid'][key]['rsq']
                    logger.debug(
                        f'{model_name}, {key} {seed} '
                        f'{rsq[model_name][key][seed]}'
                        )
            else:
                logger.info(f'{filepath} skipped, as it does not exist.')
    
    if len(rsq) == 0:
        raise RuntimeError("No data was found.")
    
    if savepath is not None:
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        with open(savepath, 'w+') as f:
            yaml.dump(rsq, f)

    return rsq


#############################################
def run_ttests(rsq, seeds=SEEDS, model_names=['oasis', 'valpaca']):

    if len(model_names) != 2:
        raise ValueError('Must provide exactly 2 model names for comparison.')
    model_1, model_2 = model_names

    for calc_dyn in CALC_DYN_DICT.keys():
        logger.info(f'{CALC_DYN_DICT[calc_dyn]} t-test')
        for data_type in DATA_DICT.keys():
            a = [rsq[f'{model_1}_{calc_dyn}'][data_type][s] for s in seeds]
            b = [rsq[f'{model_2}_{calc_dyn}'][data_type][s] for s in seeds]
            t, p = ttest_rel(a=a, b=b)
            logger.info(f'{data_type}: t_{len(seeds) - 1}={t:.3f}, p={p:.3f}') 


#############################################
def compile_df(rsq, savepath=None):

    model_names = sorted(list(rsq.keys()))
    
    df_mean = pd.DataFrame(
        [pd.DataFrame(rsq[model_name]).mean() for model_name in model_names], 
        index=model_names
        )
    df_std  = pd.DataFrame(
        [pd.DataFrame(rsq[model_name]).std() for model_name in model_names], 
        index=model_names
        )

    # combine statistic dataframes     
    df = df_mean.round(3) + PLUS_MIN + (df_std / np.sqrt(20)).round(3)
    
    df = df.reindex(axis=1, labels=DATA_DICT.keys())
    df = df.reindex(axis=0, labels=model_names)

    df = df.rename(index=MODEL_DICT.keys(), columns=DATA_DICT.keys())
    
    if savepath is not None:
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        with open(savepath, 'w+') as f:
            df.to_latex(f)

    return df
    

##--------DATA AND PATH FUNCTIONS--------##
#############################################
def get_model_filepath(conditions_str, ou_str, model_dir='.', model='valpaca', 
                       calc_dyn='ar1', seed=1000, filetype='results'):

    if calc_dyn not in ['ar1', 'hill']:
        raise ValueError('calc_dyn must be either \'ar1\' or \'hill\'.')

    calc_dyn_str = 'fluor_ar1' if calc_dyn == 'ar1' else 'fluor_hill'
    data_name = f'{DATA_NAME}_seed{seed}_{conditions_str}'
    calc_dyn_name = f'{data_name}_{calc_dyn_str}_{ou_str}_n'

    if model not in MODEL_NAME_DICT.keys():
        model_keys = list(MODEL_NAME_DICT.keys())
        raise ValueError('model must be in {}.'.format(', '.join(model_keys)))
        
    param_model = 'valpaca' if model == 'valpaca' else 'lfads'

    if filetype == 'results':
        filename = RESULTS_FILENAME
    elif filetype == 'latents':
        filename = LATENTS_FILENAME
    else:
        raise ValueError('filename should be \'results\' or \'latents\'.')

    filepath = Path(
        model_dir, 
        calc_dyn_name, 
        MODEL_NAME_DICT[model], 
        PARAM_NAME_DICT[param_model], 
        filename
        )
    
    model_name = f'{model}_{calc_dyn}'
    
    return filepath, model_name


#############################################
def get_data_dict(data_dir, calc_dyn='ar1'):

    if calc_dyn not in ['ar1', 'hill']:
        raise ValueError('calc_dyn must be either \'ar1\' or \'hill\'.')
    calc_dyn_str = 'fluor_ar1' if calc_dyn == 'ar1' else 'fluor_hill'
    
    ou_str = get_ou_str(sigma=0.5) # s should be 2.0

    calc_dyn_name = f'{DATA_NAME}_{calc_dyn_str}_{ou_str}_n'

    data_dict = util.read_data(data_dir, calc_dyn_name)

    return data_dict


#############################################
def get_conditions_str(dt_sys=0.1, dt_cal=0.1, sigma=5.0, rate=2.0):

    conditions_str = (
        f'sys{dt_sys:.4f}_'
        f'cal{dt_cal:.4f}_'
        f'sig{sigma:.1f}_'
        f'base{rate:.1f}'
    )

    return conditions_str


#############################################
def get_ou_str(sigma=5.0):

    s = 1 / sigma
    ou = f'ou_t0.3_s{s:.1f}'
    return ou


##--------PLOT FUNCTIONS--------##
#############################################
def adjust_axes(ax, min_val=0, axis='both', lw=2.5):

    if axis not in ['x', 'y', 'both']:
        raise ValueError('axis must be \'x\', \'y\', or \'both\'.')

    if axis in ['x', 'both']:
        xticks = [min_val, int(np.around(ax.get_xlim[1]))]
        ax.set_xticks(xticks)
        ax.spines['bottom'].set_bounds(xticks)
        ax.xaxis.set_tick_params(length=lw * 2, width=lw)

    if axis in ['y', 'both']:
        yticks = [min_val, int(np.around(ax.get_ylim[1]))]
        ax.set_yticks(yticks)
        ax.spines['left'].set_bounds(yticks)
        ax.yaxis.set_tick_params(length=lw * 2, width=lw)


#############################################
def plot_lorenz_examples(latent_dict, data_dict, calc_dyn='ar1', 
                         num_traces_to_show=8, figsize=(9.6, 4)):

    fig, axs = plt.subplots(
        nrows=2, ncols=4, figsize=figsize,
        gridspec_kw={'wspace': 0.6, 'hspace': 0.4}
        )
    
    # some plotting parameters
    grdtr_lw = 2
    lw = 1.5
    fs = 12
    grdtr_color = COLORS['linc_red']
    recon_color = COLORS['linc_blue']
    grey = COLORS['linc_grey']

    # plot ground truth vs reconstructed data
    data_type_dict = {
        'fluor' : ['Normalized dF/F', 1], # y label, y shift
        'rates' : ['Spike Rate (Hz)', 15],
        'spikes': ['Spike Counts', 4],
        'latent': ['Factors', 1],
    }
    for d, (data_type, (ylabel, shift)) in enumerate(data_type_dict.items()):
        r = d // 2
        c = d % 2
        ax = ax[r, c]

        data_suffix = f'_{calc_dyn}' if data_type == 'fluor' else ''
        latent_suffix = '_aligned' if data_type == 'latent' else ''
        n = 3 if data_type == 'latent' else num_traces_to_show
        
        s = np.arange(n) * shift
        ax.plot(
            TIME, 
            s + data_dict[f'valid_{data_type}{data_suffix}'][0][:, : n], 
            color=grdtr_color, lw=grdtr_lw
            )
        ax.plot(
            TIME, 
            s + latent_dict['valid'][f'{data_type}{latent_suffix}'][0][:, : n], 
        color=recon_color, lw=lw
        )

        ax.set_ylabel(ylabel, fontsize=fs)
        if data_type == 'latent':
            adjust_axes(ax, 0, axis='x')
            ax.get_yaxis().set_visible(False)
            ax.spines['left'].set_visible(False)
        else:
            adjust_axes(ax, 0)

        if r == 0:
            ax.set_xticklabels('')
        else:
            ax.set_xlabel('Time (s)', fontsize=fs)
        
    
    # plot ground truth vs reconstructed data in scatterplot
    data_type_dict = {
        'fluor' : 'Fluorescence',
        'rates' : 'Spike Rates',
        'spikes': 'Spike Counts',
        'latent': 'Lorenz State',
    }
    for d, (data_type, title) in enumerate(data_type_dict.items()):
        r = d // 2
        c = d % 2
        ax = ax[r, c + 2]

        data_suffix = f'_{calc_dyn}' if data_type == 'fluor' else ''
        latent_suffix = '_aligned' if data_type == 'latent' else ''
        
        ax.plot(
            latent_dict['valid'][f'{data_type}{latent_suffix}'].ravel(), 
            data_dict[f'valid_{data_type}{data_suffix}'].ravel(), 
            marker='.', ms=0.5, color=grey, rasterized=True
            )
        
        ax.set_title(title, fontsize=fs)
        ax.set_ylabel('Ground Truth', fontsize=fs)
        ax.set_xlabel('Reconstruction', fontsize=fs)

        min_val = -1 if data_type == 'latent' else 0
        adjust_axes(ax, min_val=min_val)


    # adjust plot formatting
    for ax in axs.ravel():
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(labelsize=fs)

    return fig
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # optional parameters
    parser.add_argument('-d', '--data_dir', default=Path('..', 'synth_data'), 
                        type=Path)
    parser.add_argument('-m', '--model_dir', default='models', type=Path)
    parser.add_argument('-o', '--output_dir', default=None, type=Path, 
                        help='model_dir is used if output_dir is None')

    parser.add_argument('--dt_sys', default=0.1, type=float)
    parser.add_argument('--dt_cal', default=0.1, type=float)
    parser.add_argument('--sigma', default=5.0, type=float)
    parser.add_argument('--rate', default=2.0, type=float)
    parser.add_argument('--log_level', default='info', 
                        help='log level, e.g. debug, info, error')

    args = parser.parse_args()

    logger = util.get_logger_with_basic_format(level=args.log_level)
    
    main(args)
