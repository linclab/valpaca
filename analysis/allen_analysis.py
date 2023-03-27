#!/usr/bin/env python

import argparse
import copy
import logging
import multiprocessing
from pathlib import Path
import sys

from joblib import delayed, Parallel
import matplotlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
from scipy.signal import savgol_filter
matplotlib.use('agg')

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, \
    RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import networkx as nx

sys.path.extend(['.', '..'])
from utils import util
from models import supervised


logger = logging.getLogger(__name__)

# Data from https://dandiarchive.org/dandiset/000039/
# See metadata: https://github.com/AllenInstitute/Contrast_Analysis/blob/master/targeted_manifest.csv


##--------HYPERPARAMETERS--------##
#############################################

# for plotting
COLORS = {'linc_red'  : '#E84924',
          'linc_blue' : '#37A1D0'}

SAVE_KWARGS = {
    'bbox_inches': 'tight',
    'facecolor'  : 'w',
    'transparent': False,
    'format'     : 'svg',
}

# unicodes
DEG = u'\u00B0'
PLUS_MIN = u'\u00B1'

# non linear decoder arguments
TRAIN_P = 0.8

MODEL_KWARGS = {
    'classify': False, # only affects non-linear predictor
    'hidden_size': 16,
    'num_layers': 1,
    'bias': True,
    'dropout': 0.25,
    'device': 'cpu',
    }

FIT_KWARGS = {
    'batch_size': 68,
    'learning_rate': 5e-5,
    'pos_weight': None,
    'log_freq': 200,
    'max_epochs': 5000,
    'max_iter': 1,
    'save_loc': '.'
}

# Trial related parameters
TIME_EDGES = [0, 2]
FEATURES = ['direction', 'contrast']


##--------MAIN--------##
#############################################
def main(args):

    # set logger to the specified level
    util.set_logger_level(logger, level=args.log_level)

    data_dict = util.read_data(args.data_path)
    latent_dict = None
    if not args.raw:
        latent_dict = util.load_latent(args.model_dir)

    if args.output_dir is None:
        args.output_dir = args.model_dir

    # plot examples
    if latent_dict is not None:
        savepath = Path(args.output_dir, 'allen_examples')
        plot_examples(data_dict, latent_dict, trial_ix=15, savepath=savepath)
    
    # plot factor PCA
    for plot_2d in [True, False]:
        plot_save_factors_all(
            data_dict, latent_dict, 
            output_dir=Path(args.output_dir, 'factor_plots'), 
            plot_2d=plot_2d,
            seed=args.seed
            )

    # run decoders
    if args.num_runs == 0:
        return

    for scale in [True, False]:
        scale_str = 'scaling' if scale else 'no scaling'

        for feature in ['both']:
            feat_str = ' and '.join(FEATURES) if feature == 'both' else feature
            logger.info(
                f'{feat_str.capitalize()} decoders: {scale_str}', 
                extra={'spacing': '\n'}
                )

            run_decoders(
                data_dict, latent_dict, args.output_dir, 
                run_linreg=args.run_linreg, run_nl_decoder=args.run_nl_decoder, 
                num_runs=args.num_runs, seed=args.seed, scale=scale, 
                feature=feature, log_scores=True, parallel=args.parallel
                )


##--------GENERAL FUNCTIONS--------##
#############################################
def load_trial_data(data_dict, latent_dict=None):

    if latent_dict is None:
        idxs = np.concatenate([data_dict['train_idx'], data_dict['valid_idx']])
        data_unsorted = np.concatenate(
            [data_dict['train_fluor'], data_dict['valid_fluor']
            ], axis=0)
        data = data_unsorted[np.argsort(idxs)]
    else:
        data = latent_dict['ordered']['factors']

    num_trials = len(data)
    direction = np.zeros(num_trials)
    direction[data_dict['train_idx']] = data_dict['train_direction']
    direction[data_dict['valid_idx']] = data_dict['valid_direction']

    contrast = np.zeros(num_trials)
    contrast[data_dict['train_idx']] = data_dict['train_contrast']
    contrast[data_dict['valid_idx']] = data_dict['valid_contrast']
    
    return data, direction, contrast



##--------DECODER FUNCTIONS--------##

#############################################
def run_decoders(data_dict, latent_dict, output_dir, run_linreg=True, 
                 run_nl_decoder=False, num_runs=10, scale=True, 
                 feature='direction', seed=None, log_scores=True, 
                 parallel=False):

    scale_str = "_scaled" if scale else ""

    if latent_dict is None:
        X_train = data_dict['train_fluor']
        X_test = data_dict['valid_fluor']
    else:
        X_train = latent_dict['train']['latent']
        X_test = latent_dict['valid']['latent']
    
    X = np.concatenate([X_train, X_test], axis=0)

    y_train, y_test = [], []
    features = FEATURES if feature == 'both' else [feature]
    for feat in features:
        if feat not in FEATURES:
            raise ValueError(f'{feat} not recognized. Should be in {FEATURES}')
        y_train.append(data_dict[f'train_{feat}'])
        y_test.append(data_dict[f'valid_{feat}'])
    y = np.concatenate([np.vstack(y_train), np.vstack(y_test)], axis=1).T

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if run_linreg:
        linreg_dir = Path(output_dir, 'linear_predictor')
        if log_scores:
            logger.info(
                '  Linear regression R2 scores:', extra={'spacing': '\n'}
                )

        linreg_scores = perform_decoder_runs(
            linreg_eval, X=X, y=y, feature=feature, num_runs=num_runs, 
            seed=seed, log_scores=log_scores, scale=scale, 
            output_dir=linreg_dir, parallel=parallel
            )
        np.save(Path(linreg_dir, f'{feature}{scale_str}_scores'), linreg_scores)


    if run_nl_decoder:
        decoder_kwargs = dict()

        model_kwargs = copy.deepcopy(MODEL_KWARGS)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs['device'] = device

        fit_kwargs = copy.deepcopy(FIT_KWARGS)

        nl_dir = Path(output_dir, 'non_linear_predictor')
        if log_scores:
            if model_kwargs["classify"]:
                score_str = 'decoding bal. acc.'
            else:
                score_str = 'regression R2'
            logger.info(
                f'  Non-linear {score_str} scores:', extra={'spacing': '\n'}
                )

        decoder_kwargs = {
            'scale'       : scale,
            'model_kwargs': model_kwargs,
            'fit_kwargs'  : fit_kwargs,
        }

        nl_decoder_scores = perform_decoder_runs(
            recurrent_net_eval, X=X, y=y, feature=feature, num_runs=num_runs, 
            seed=seed, log_scores=log_scores, output_dir=nl_dir, 
            parallel=parallel, max_jobs=4, **decoder_kwargs,
            )

        np.save(Path(nl_dir, f'{feature}{scale_str}_scores'), nl_decoder_scores)


#############################################
def perform_decoder_run(decoder_fct, X, y, train_p=0.8, seed=None, 
                        feature='direction', classify=True, **decoder_kwargs):

    # get a new split
    train_data, test_data = train_test_split(
        list(zip(X, y)), train_size=train_p, stratify=y, 
        random_state=seed,
        )

    decoder_kwargs['X_train'] = np.asarray(list(zip(*train_data))[0])
    decoder_kwargs['X_test'] = np.asarray(list(zip(*test_data))[0])
    decoder_kwargs['y_train'] = format_y(
        np.asarray(list(zip(*train_data))[1]), feature=feature, 
        as_class=classify
    )
    decoder_kwargs['y_test'] = format_y(
        np.asarray(list(zip(*test_data))[1]), feature=feature, 
        as_class=classify
    )

    run_score, y_test, y_pred = decoder_fct(seed=seed, **decoder_kwargs)
    return run_score, y_test, y_pred


#############################################
def perform_decoder_runs(decoder_fct, X, y, feature='direction', num_runs=10, 
                         seed=None, log_scores=True, train_p=TRAIN_P, 
                         output_dir=None, parallel=False, max_jobs=-1,
                         **decoder_kwargs):

    util.seed_all(seed)

    sub_seeds = [np.random.choice(int(2e5)) for _ in range(num_runs)]

    classify = False
    if 'model_kwargs' in decoder_kwargs.keys():
        classify = decoder_kwargs['model_kwargs']['classify']
    
    scale_str = "_scaled" if decoder_kwargs['scale'] else ""
    feat_str = '' if feature == 'both' else f'{feature} '
    title_str = f'{feat_str}decoding' if classify else f'{feat_str}regression'

    plot_data = {'y_test': -np.inf, 'y_pred': -np.inf, 'idx': -1}

    if parallel:
        n_jobs = min(multiprocessing.cpu_count(), num_runs)
        if n_jobs <= 1:
            parallel = False
        if max_jobs != -1:
            n_jobs = min(n_jobs, max_jobs)

    if parallel:
        scores, y_tests, y_preds = zip(*Parallel(n_jobs=n_jobs)(
            delayed(perform_decoder_run)(
                decoder_fct, X, y, train_p=train_p, seed=sub_seed, 
                feature=feature, classify=classify, **decoder_kwargs
                ) for sub_seed in sub_seeds
            ))
    else:
        scores = []

    for i, sub_seed in enumerate(sub_seeds):
        if parallel:
            run_score, y_test, y_pred = scores[i], y_tests[i], y_preds[i]
        else:
            run_score, y_test, y_pred = perform_decoder_run(
                decoder_fct, X, y, train_p=train_p, seed=sub_seed, 
                feature=feature, classify=classify, **decoder_kwargs
                )
            scores.append(run_score)

        if run_score >= max(scores):
            plot_data['y_test'] = y_test
            plot_data['y_pred'] = y_pred
            plot_data['idx'] = i

            if classify:
                score_str = f'bal. acc.={run_score * 100:.2f}%' 
            else: 
                score_str = f'R$^2$={run_score:.3f}'
            suptitle = f'{title_str.capitalize()} ({score_str})'

            if not classify:
                fig = plot_regression_results(
                    y_test, y_pred, feature=feature, suptitle=suptitle, 
                    )
                if output_dir is not None:
                    savepath = Path(
                        output_dir, f'{feature}{scale_str}_regression'
                        )
                    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        f'{savepath}.{SAVE_KWARGS["format"]}', **SAVE_KWARGS
                        )


        if log_scores:
            score_mean = np.mean(scores[:i+1])
            score_sem = np.std(scores[:i+1]) / np.sqrt(i + 1)

            running_score_str = u'running score: {:.3f} {} {:.3f}'.format(
                score_mean, PLUS_MIN, score_sem
                )
            score_str = u'n={}, score: {:.3f}, {}'.format(
                i+1, run_score, running_score_str
                )

            logger.info(score_str)


#############################################
def plot_regression_results(y_test, y_pred, feature='direction', suptitle=None, 
                            jitter=True):
    
    num_plots = 1
    if feature == 'both':
        num_plots = 2

    fig, ax = plt.subplots(1, num_plots, figsize=[num_plots * 5, 4.5])
    if feature != 'both':
        ax = [ax]

    if feature in ['direction', 'both']:
        valid = ((y_pred[:, : 2] < 1) * (y_pred[:, : 2] > -1)).min(axis=1)
        d_test, d_pred = [
            (np.arctan2(y[:, 0], y[:, 1]) / np.pi * 180 + 360) % 360
            for y in [y_test, y_pred]
        ]
        plot_test = jitter_data(d_test) if jitter else d_test
        for mask, col in [(valid, 'blue'), (~valid, 'red')]:
            ax[0].plot(
                plot_test[mask], d_pred[mask], lw=0, marker='.', color=col, 
                alpha=0.6
                )

        id = [d_test.min(), d_test.max()]
        ax[0].plot(id, id, color='black', alpha=0.8, zorder=-5)

        if feature == 'both':
            ax[0].set_title('Direction (in deg.)')
    
    if feature in ['contrast', 'both']:
        if feature == 'both':
            c_test, c_pred = y_test[:, -1], y_pred[:, -1]
        else:
            c_test, c_pred = y_test, y_pred
        plot_test = jitter_data(c_test) if jitter else c_test
        ax[-1].plot(
            plot_test, c_pred, lw=0, marker='.', color='blue', alpha=0.6
            )
        
        id = [c_test.min(), c_test.max()]
        ax[-1].plot(id, id, color='black', alpha=0.8, zorder=-5)

        if feature == 'both':
            ax[-1].set_title('Contrast')
    
    jit_str = ' (jittered)' if jitter else ''
    ax[0].set_ylabel('Prediction')
    for sub_ax in ax:
        sub_ax.spines['right'].set_visible(False)
        sub_ax.spines['top'].set_visible(False)
        sub_ax.set_xlabel(f'Target{jit_str}')


    if suptitle is not None:
        y = 1.01 if feature == 'both' else 0.98
        fig.suptitle(suptitle, y=y)
    
    return fig


#############################################
def jitter_data(data):

    n = len(data)
    width = np.diff(np.sort(np.unique(data))).min()

    jitter = np.random.normal(0, 0.2, n)
    jitter = np.minimum(np.maximum(jitter, -0.5), 0.5)
    jitter *= width * 0.75

    jittered = data + jitter

    return jittered


#############################################
def linreg_eval(X_train, y_train, X_test, y_test, scale=True, seed=None, 
                grid_search=False, log_params=True):

    # reshape data
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # initialize pipeline
    model = Ridge(
        alpha=1e2, random_state=seed, solver='auto', max_iter=1000, 
        fit_intercept=True
    )

    if len(y_train.shape) != 1:
        model = MultiOutputRegressor(model)

    scale_steps = [StandardScaler()] if scale else []
    pipeline = make_pipeline(*scale_steps, model)

    # select parameters with cross-validation grid search
    if grid_search:
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        param_dist = {'ridge__alpha': loguniform(1e-2, 1e2)}
        n_grid_iter = 20 * len(param_dist) # scaling partially by # of grid dims
        pipeline = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, 
            cv=cv, n_iter=n_grid_iter, random_state=seed, n_jobs=-1
            )
    
    pipeline.fit(X_train, y_train)

    if grid_search and log_params:
        param_str = '\n    '.join(
            [f'{k}: {v}' for k, v in pipeline.best_params_.items()]
            )
        logger.info(f'Best parameters (linreg):\n    {param_str}')

    # predict on the held out test set
    y_pred = pipeline.predict(X_test)
    score = metrics.r2_score(y_test, y_pred)

    return score, y_test, y_pred


#############################################
def recurrent_net_eval(X_train, y_train, X_test, y_test, model_kwargs, 
                       fit_kwargs, scale=True, seed=None, train_p=TRAIN_P):

    # get a validation set from the training set
    train_data, valid_data = train_test_split(
        list(zip(X_train, y_train)), train_size=train_p, stratify=y_train, 
        random_state=seed,
        )

    X_train, y_train = [np.asarray(data) for data in zip(*train_data)]
    X_valid, y_valid = [np.asarray(data) for data in zip(*valid_data)]

    input_size = X_train.shape[-1]
    if model_kwargs['classify']:
        dense_size = max(y_train) + 1
    else:
        dense_size = 1 if len(y_train.shape) == 1 else y_train.shape[1]

    util.seed_all(seed)
    model = supervised.Supervised_BiRecurrent_Net(
        input_size=input_size, dense_size=dense_size, **model_kwargs
        )
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(len(X_train), -1)
            ).reshape(X_train.shape)

        # apply
        X_valid = scaler.transform(
            X_valid.reshape(len(X_valid), -1)
            ).reshape(X_valid.shape)

        X_test = scaler.transform(
            X_test.reshape(len(X_test), -1)
            ).reshape(X_test.shape)

    # fit with a validation set
    model.fit(X_train, y_train, X_valid, y_valid, **fit_kwargs)
    
    # predict on the held out test set
    y_pred = model.predict(X_test)
    if model_kwargs['classify']:
        score = metrics.balanced_accuracy_score(
            y_test.squeeze(), y_pred.squeeze()
            )
    else:
        score = metrics.r2_score(y_test, y_pred)

    return score, y_test, y_pred
    

#############################################
def format_y(y, feature='direction', as_class=False):

    if feature != 'both' and len(y.shape) == 2:
        y = np.squeeze(y, axis=1)

    if as_class:
        y_class = np.empty(len(y)).astype(int)
        if feature == 'both': # both features
            y = np.asarray([
                float(f'{y1}.{y2}') 
                for y1, y2 in zip(
                    format_y(y[:, 0], as_class=True), 
                    format_y(y[:, 1], as_class=True)
                    )
                ])
        for i, val in enumerate(np.unique(y)): 
            y_class[y == val] = i
        y = y_class

    else:
        if feature == 'contrast':
            pass
        elif feature == 'direction':
            circ_y = np.empty((len(y), 2))
            circ_y[:, 0] = np.sin(np.pi * y / 180)
            circ_y[:, 1] = np.cos(np.pi * y / 180)
            y = circ_y
        elif feature == 'both':
            ys = []
            for f, feature in enumerate(FEATURES):
                ys.append(format_y(y[:, f], feature=feature).reshape(len(y), -1))
            y = np.concatenate(ys, axis=1)
        else:
            raise ValueError('feature should be \'direction\' or \'contrast\'.')

    return y


##--------PLOT EXAMPLES--------##
#############################################
def adjust_axes(ax, min_val=0, max_val=None, axis='both', lw=1, length=4, 
                n_sig=2):

    if axis not in ['x', 'y', 'both']:
        raise ValueError('axis must be \'x\', \'y\', or \'both\'.')

    if axis == 'both':
        axes = ['x', 'y']
    else:
        axes = [axis]

    for axis in axes:
        ticks = [min_val, max_val]
        round_o = None
        new_vals = [None, None]
        for v, val in enumerate(ticks):
            if val is not None:
                continue
            if axis == 'x':
                new_vals[v] = ax.get_xlim()[v]
            else:
                new_vals[v] = ax.get_ylim()[v]
            if n_sig is not None:
                order = int(np.floor(np.log10(np.absolute(new_vals[v]))))
                o = int(-order + n_sig - 1)
                if round_o is None:
                    round_o = o
                elif o < 0:
                    round_o = max(round_o, o)
                else:
                    round_o = min(round_o, o)

        if round_o is not None:
            for v, val in enumerate(new_vals):
                if val is not None:
                    ticks[v] = np.around(val, round_o)

        if axis == 'x':    
            ax.set_xticks(ticks)
            ax.spines['bottom'].set_bounds(ticks)
            ax.xaxis.set_tick_params(length=length, width=lw)
        else:
            ax.set_yticks(ticks)
            ax.spines['left'].set_bounds(ticks)
            ax.yaxis.set_tick_params(length=length, width=lw)


#############################################
def plot_examples(data_dict, latent_dict, trial_ix=0, num_traces=8, 
                  figsize=(4.8, 4), savepath=None):

    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=figsize, sharex=True, 
        gridspec_kw={'wspace': 0.6, 'hspace': 0.4}
        )

    # gather data
    grdtr_fluor = data_dict['valid_fluor'][trial_ix] # ground truth
    recon_fluor = latent_dict['valid']['fluor'][trial_ix]
    recon_rates = latent_dict['valid']['rates'][trial_ix]
    recon_spikes = latent_dict['valid']['spikes'][trial_ix]
    latents = latent_dict['valid']['latent'][trial_ix]

    x = np.linspace(*TIME_EDGES, len(grdtr_fluor)) # time in seconds
    
    # some plotting parameters
    grdtr_lw = 2
    lw = 1.5
    fs = 12
    grdtr_color = COLORS['linc_red']
    recon_color = COLORS['linc_blue']

    # select indices of sequences with higher standard deviations
    ix_incl = np.argsort(grdtr_fluor.std(axis=0))[-num_traces : ]
    num_traces = len(ix_incl)

    # plot fluorescence pre/post reconstruction
    fl_ax = axs[0, 0]
    s = np.arange(num_traces) * 5
    gain = data_dict['obs_gain_init'].mean()
    fl_ax.plot(x, s + grdtr_fluor[:, ix_incl] / gain, color=grdtr_color, 
               lw=grdtr_lw)
    fl_ax.plot(x, s + recon_fluor[:, ix_incl] / gain, color=recon_color, lw=lw)
    fl_ax.set_ylabel('dF/F', fontsize=fs)
    adjust_axes(fl_ax, min_val=0, axis='y')

    # plot reconstructed spike rates
    rt_ax = axs[0, 1]
    s = np.arange(num_traces) * 20
    rt_ax.plot(x, s + recon_rates[:, ix_incl], color=recon_color, lw=lw)
    rt_ax.set_ylabel('Spike Rate (Hz)', fontsize=fs)
    adjust_axes(rt_ax, min_val=0, axis='y')

    # plot reconstructed spike counts
    ct_ax = axs[1, 0]
    s = np.arange(num_traces)
    ct_ax.plot(x, s + recon_spikes[:, ix_incl], color=recon_color, lw=lw)
    ct_ax.set_ylabel('Spike Counts', fontsize=fs)
    ct_ax.set_xlabel('Time (s)', fontsize=fs)
    adjust_axes(ct_ax, min_val=0, axis='y')

    # plot factors
    fact_ax = axs[1, 1]
    s = np.arange(num_traces) * 2
    fact_ax.plot(x, s + latents[:, : num_traces], color=recon_color, lw=lw)
    fact_ax.set_xlabel('Time (s)', fontsize=fs)
    fact_ax.set_ylabel('Factors', fontsize=fs)
    fact_ax.set_yticks([])
    fact_ax.set_yticklabels([])
    fact_ax.spines['left'].set_visible(False)

    # adjust axis formatting
    for ax in axs.ravel():
        adjust_axes(ax, min_val=TIME_EDGES[0], max_val=TIME_EDGES[-1], axis='x')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(labelsize=fs)

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{savepath}.{SAVE_KWARGS["format"]}', **SAVE_KWARGS)

    return fig


##--------PLOT PCA FACTORS--------##
#############################################
def plot_save_factors_all(data_dict, latent_dict, output_dir='factors_3d_plots', 
                          plot_2d=False, seed=None, close=True):

    latents, direction, contrast = load_trial_data(data_dict, latent_dict)

    n_dim = 2 if plot_2d else 3

    savename = f'factors_{n_dim}d'

    fact_fig = plot_factors_all(
        latents, direction, contrast, plot_2d=plot_2d, seed=seed
        )
    
    savepath = Path(output_dir, savename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fact_fig.savefig(
        f'{savepath}.{SAVE_KWARGS["format"]}', **SAVE_KWARGS
        )
    
    if close:
        plt.close('all')


#############################################
def plot_factors_all(latents, direction, contrast, plot_2d=False, seed=None):
    
    fig, ax_mean, ax_single, ax_dist = init_factors_fig(plot_2d=plot_2d)
    
    # create model with full latents
    model = None
    n_comps = 2 if plot_2d else 3
    model = fit_PCA_Regression(
        latents, 
        num_components=n_comps,
        seed=seed
        )

    feature_data = {
        'direction': direction,
        'contrast': contrast,
    }
    for feature, ax_m, ax_s, ax_d in zip(FEATURES, ax_mean, ax_single, ax_dist):

        title = feature.capitalize()
        ax_m.set_title(title, fontsize='x-large')

        proj_factors = plot_factors(
            latents,
            feature_data[feature],
            ax_m, 
            ax_s, 
            model=model,
            plot_2d=plot_2d,
            # circ_col=(feature == 'direction'),
            seed=seed
            )

        dist_df = get_distance_df(proj_factors, direction, contrast)
        plot_distance_graph(dist_df, feature=feature, ax=ax_d, seed=seed)

    if plot_2d:
        adjust_trajectory_axes(ax_mean, shared_axes=True)
        adjust_trajectory_axes(ax_single, shared_axes=True)

    return fig


#############################################
def init_factors_fig(plot_2d=False):

    fig, ax = plt.subplots(3, 2, figsize=(10, 14))

    ax_mean, ax_single, ax_dist = ax

    if not plot_2d:
        for i in range(2):
            ax_mean[i].remove()
            ax_mean[i] = fig.add_subplot(3, 2, i + 1, projection='3d')

            ax_single[i].remove()
            ax_single[i] = fig.add_subplot(3, 2, i + 3, projection='3d')

        for sub_ax in np.concatenate([ax_mean, ax_single]):
            sub_ax.view_init(elev=40., azim=50.)
        
    
    return fig, ax_mean, ax_single, ax_dist


#############################################
def plot_factors(latents, feature, ax_mean, ax_single, model=None, 
                 plot_2d=False, circ_col=False, seed=None):
    

    n_comps = 2 if plot_2d else 3
    if model is None:
        model = fit_PCA_Regression(
            latents, 
            num_components=n_comps,
            seed=seed
            )

    proj_factors = predict_PCA_Regression(latents, model)
    if proj_factors.shape[-1] != n_comps:
        raise ValueError(
            f'Expected \'model\' to produce {n_comps} components, '
            f'but found {proj_factors.shape[-1]}.'
            )

    plot_trajectories(
        ax_mean, ax_single, proj_factors, feature, circ_col=circ_col
        )

    return proj_factors


#############################################
def predict_PCA_Regression(X, model):

    pred_shape = (-1, X.shape[-1])
    targ_shape = X.shape[:2] + (-1, )
    proj_factors = model.predict(X.reshape(pred_shape)).reshape(targ_shape)

    return proj_factors


#############################################
def fit_PCA_Regression(X, num_components=3, regr_scale=False, seed=None):

    num_trials, num_steps, num_items = X.shape
    X = X.transpose(0, 2, 1)
    
    X_flat = X.reshape(num_trials * num_items, num_steps)
    
    alphas = np.logspace(-2, 1, 30)

    # perform PCA along the trial x item dimension (together)
    pca_pipeline = make_pipeline(
        StandardScaler(), 
        PCA(num_components, random_state=seed)
        )
    pca_results = pca_pipeline.fit(X_flat)

    # regress the components from the items, treating their mean activities 
    # (across trials) at each time step as a separate trial
    V = pca_results['pca'].components_

    # NOTE: scaling tends to obscure trends in raw data
    scale_steps = [StandardScaler()] if regr_scale else []
    regr_pipeline = make_pipeline(
        *scale_steps, 
        RidgeCV(alphas=alphas, fit_intercept=True)
        )

    regr_pipeline.fit(
        X.mean(axis=0).transpose(1, 0), 
        V.transpose(1, 0)
        )

    return regr_pipeline


#############################################
def plot_trajectories(ax_mean, ax_single, proj_factors, feature, 
                      single_marks=False, circ_col=False):

    feature_vals = np.sort(np.unique(feature))
    num_features = len(feature_vals)

    all_idxs, all_means = [], []
    for val in feature_vals:
        all_idxs.append(feature == val)
        all_means.append(proj_factors[all_idxs[-1]].mean(axis=0))

    if circ_col:
        colors = get_colors(None, None, n=num_features, circ_col=True)
    else:
        colors = get_colors(
            COLORS['linc_blue'], COLORS['linc_red'], n=num_features
            )

    n_comps = proj_factors.shape[-1]

    lw = 1.5
    ms = 4
    for mean, idxs, color in zip(all_means, all_idxs, colors):
        ax_mean.plot(*mean.T, color=color, lw=lw)
        ax_mean.plot(*mean[0], color=color, marker='o', lw=0, ms=ms)
        ax_mean.plot(*mean[-1], color=color, marker='^', lw=0, ms=ms)

        step = max(1, int(sum(idxs) / 20))
        sample_slice = slice(0, sum(idxs), step)
        for trial in proj_factors[idxs][sample_slice]:
            
            ftrial = savgol_filter(trial, 9, 2, axis=0)
            ax_single.plot(*ftrial.T, color=color, lw=0.15, alpha=0.8)
            if single_marks:
                ax_single.plot(
                    *ftrial[0], color=color, marker='o', ms=1, lw=0, alpha=0.8
                    )
                ax_single.plot(
                    *ftrial[-1], color=color, marker='^', ms=1, lw=0, alpha=0.8
                    )
    

    # format axes
    for ax in [ax_mean, ax_single]:
        if n_comps == 2:
            lw = 2.5
            ax.xaxis.set_tick_params(length=lw * 2, width=lw)
            ax.yaxis.set_tick_params(length=lw * 2, width=lw)

            for spine in ['right', 'top']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)
            
        elif n_comps == 3:
            for axis in ['x', 'y', 'z']:
                ax.locator_params(nbins=5, axis=axis)
            for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
                axis.line.set_linewidth(1)
                axis.set_ticklabels([])
                axis._axinfo['grid'].update({'linewidth': 0.5})
                axis._axinfo['tick'].update({'inward_factor': 0})
                axis._axinfo['tick'].update({'outward_factor': 0})

        else:
            raise NotImplementedError('Expected 2 or 3 components.')


##--------PLOT DISTANCES--------##
#############################################
def get_distance_df(proj_factors, direction, contrast):

    columns=[
        'feature', 'indices', 'values', 'dist_mean', 'dist_std', 
        'end_dist_mean', 'end_dist_std'
        ]

    all_dfs = []
    for feature, feat_str in [(direction, 'direction'), (contrast, 'contrast')]:   
        dist_df = pd.DataFrame(columns=columns)
        unique_vals = np.sort(np.unique(feature))
        val_idx = [np.where(feature == f)[0] for f in unique_vals]

        pw_idx  = [
            (i, j) for i in range(len(unique_vals)) 
            for j in range(len(unique_vals))[i + 1:]
            ]

        dists = np.asarray([
            calc_distance(proj_factors, val_idx[i], val_idx[j], norm=False)[0]
            for i, j in pw_idx
        ])
        num_end = dists.shape[1] // 4

        dist_df['dist_mean'] = dists.mean(axis=1).tolist()
        dist_df['dist_std'] = dists.std(axis=1).tolist()
        dist_df['end_dist_mean'] = dists[:, -num_end:].mean(axis=1).tolist()
        dist_df['end_dist_std'] = dists[:, -num_end:].std(axis=1).tolist()

        dist_df['indices'] = pw_idx
        dist_df['values'] = [
            (unique_vals[i], unique_vals[j]) for i, j in pw_idx
        ]
        dist_df['feature'] = feat_str
        all_dfs.append(dist_df)
    
    dist_df = pd.concat(all_dfs)

    return dist_df


#############################################
def plot_distance_graph(df, feature='direction', ax=None, neigh=True, 
                        end=False, seed=None):

    g = nx.Graph()
    sub_df = df.loc[df['feature'] == feature]
    
    dist_key = 'end_dist_mean' if end else 'dist_mean'

    node_idxs, node_values = [], []
    seps, weights = [], []
    for idx in sub_df.index:
        i, j = sub_df.loc[idx, 'indices']
        val1, val2 = sub_df.loc[idx, 'values']
        weight = 1 / sub_df.loc[idx, dist_key]
        g.add_edge(i, j, color='k', weight=weight, label=f'{val1}, {val2}')
        seps.append(np.absolute(j - i))
        weights.append(weight)

        if i not in node_idxs:
            node_idxs.append(i)
            node_values.append(val1)
        if j not in node_idxs:
            node_idxs.append(j)
            node_values.append(val2)

    if feature == 'direction':
        seps = [min([sep, len(node_idxs) - sep]) for sep in seps] # circular

    if neigh:
        weights = np.asarray(weights) 
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min()) + 0.5
        widths = np.asarray(
            [weights_norm[s] if sep == 1 else 0 for s, sep in enumerate(seps)]
        ) * 3
        alpha = 1
    else: # 0.25 to 0.75
        seps = (len(node_idxs) - np.asarray(seps))
        seps = (seps - seps.min()) / (seps.max() - seps.min()) * 0.5 + 0.25 
        widths = seps * 3
        alpha = seps

    node_colors = get_colors(
        COLORS['linc_blue'], COLORS['linc_red'], len(node_idxs)
    )
    node_labels = {i: label for i, label in zip(node_idxs, node_values)}
    
    pos = nx.spring_layout(g, dim=2, weight='weight', seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=[6, 6])
    else:
        fig = ax.figure

    fs, ns = 9, 1200
    nx.draw(
        g, pos, ax=ax, edge_color='k', width=widths, node_size=ns, 
        alpha=alpha
        )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, linewidths=2, edgecolors='k', 
        node_size=ns
        )
    nx.draw_networkx_labels(
        g, pos, ax=ax, labels=node_labels, font_size=fs, font_weight='bold'
        )
    
    return fig, ax
    

#############################################
def calc_distance(proj_factors, idx_1, idx_2, seed=None, norm=False):

    proj_1_mean = proj_factors[idx_1].mean(axis=0)
    proj_2_mean = proj_factors[idx_2].mean(axis=0)

    dist = np.sqrt(np.sum((proj_2_mean - proj_1_mean)**2, axis=-1))
    if norm:
        dist = dist / np.sum(dist)

    boot_dist_std = bootstrapped_diff_std(
        proj_factors[idx_1], proj_factors[idx_2], seed=seed
    )

    return dist, boot_dist_std
    

#############################################
def bootstrapped_diff_std(vals1, vals2, n_samples=1000, seed=None):
    """
    bootstrapped_std(data)
    
    Returns bootstrapped standard deviation of the mean.

    Required args:
        - vals1 (3D array): first set of data
        - vals2 (3D array): second set of data
    
    Optional args:
        - n_samples (int): number of samplings to take for bootstrapping
                           default: 1000

    Returns:
        - boot_dist_std (float): bootstrapped distance (data2 mean - data1 mean)
    """
    
    rng = np.random.RandomState(seed)

    # random values
    n_vals1 = len(vals1)
    n_vals2 = len(vals2)
    choices_exp = rng.choice(
        np.arange(n_vals1), (n_vals1, n_samples), replace=True
        ).reshape(n_vals1, -1)
    choices_unexp = rng.choice(
        np.arange(n_vals2), (n_vals2, n_samples), replace=True
        ).reshape(n_vals2, -1)
    
    vals1_resampled = np.mean(vals1[choices_exp], axis=0)
    vals2_resampled = np.mean(vals2[choices_unexp], axis=0)

    boot_dist_std = np.std(
        np.sqrt(np.sum((vals1_resampled - vals2_resampled)**2, axis=-1)), axis=0)
    
    return boot_dist_std
    

#############################################
def adjust_trajectory_axes(axs, shared_axes=True):

    for ax in axs:
        ax.autoscale()

    min_xlim = np.min([ax.get_xlim()[0] for ax in axs])
    max_xlim = np.max([ax.get_xlim()[1] for ax in axs])

    min_ylim = np.min([ax.get_ylim()[0] for ax in axs])
    max_ylim = np.max([ax.get_ylim()[1] for ax in axs])

    for a, ax in enumerate(axs[::-1]):
        if not shared_axes:
            min_xlim, max_xlim = ax.get_xlim()
            min_ylim, max_ylim = ax.get_ylim()

        # expand limits a bit
        exp_min_xlim = min_xlim - (max_xlim - min_xlim) * 0.03
        exp_max_xlim = max_xlim + (max_xlim - min_xlim) * 0.03

        exp_min_ylim = min_ylim - (max_ylim - min_ylim) * 0.03
        exp_max_ylim = max_ylim + (max_ylim - min_ylim) * 0.03

        # make adjustments
        ax.set_xlim(exp_min_xlim, exp_max_xlim)
        ax.set_ylim(exp_min_ylim, exp_max_ylim)
        adjust_axes(
            ax, min_val=None, max_val=None, axis='both', lw=2.5, length=5, 
            n_sig=1
            )
        ax.set_xticklabels(ax.get_xticks(), fontsize='large')
        if shared_axes and a != (len(axs) - 1):
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(ax.get_yticks(), fontsize='large')


#############################################
def get_colors(start, end, n=3, circ_col=False):

    if n < 2:
        raise ValueError('Must request at least 2 colors.')

    if circ_col:
        cm = plt.get_cmap('twilight')
        samples = np.linspace(0, 1, n + 1)[:n]
    else:
        cm = LinearSegmentedColormap.from_list('Custom', [start, end], N=n)
        samples = np.linspace(0, 1, n)

    colors = [cm(v) for v in samples]
    
    return colors


#############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', type=Path)

    # optional parameters
    parser.add_argument('-r', '--raw', action='store_true', 
                        help='analyses raw data, instead of latents')
    parser.add_argument('-m', '--model_dir', default='models', type=Path, 
                        help='latent.pkl directory')
    parser.add_argument('-o', '--output_dir', default=None, type=Path, 
                        help='model_dir is used if output_dir is None')
    parser.add_argument('-n', '--num_runs', default=10, type=int, 
                        help='number of decoders to run per decoder type')

    parser.add_argument('--run_linreg', action='store_true')
    parser.add_argument('--run_nl_decoder', action='store_true')

    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--log_level', default='info', 
                        help='log level, e.g. debug, info, error')

    args = parser.parse_args()

    logger = util.get_logger_with_basic_format(level=args.log_level)

    main(args)

