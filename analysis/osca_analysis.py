#!/usr/bin/env python

import argparse
import copy
import logging
import multiprocessing
from pathlib import Path
import sys

from joblib import Parallel, delayed
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
from scipy.signal import savgol_filter
matplotlib.use('agg')

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, \
    RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

sys.path.extend(['.', '..'])
from utils import util
from models import supervised


logger = logging.getLogger(__name__)


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
    'hidden_size': 16,
    'num_layers': 1,
    'bias': True,
    'dropout': 0.25,
    'device': 'cpu'
    }

FIT_KWARGS = {
    'batch_size': 68,
    'learning_rate': 5e-5,
    'pos_weight': 'auto',
    'log_freq': 200,
    'max_epochs': 5000,
    'max_iter': 1,
    'save_loc': '.'
}

# Gabor sequence related parameters
FRAMES_PER = 9
N_GAB_FR = 5 # A, B, C, D/U, gr
TIME_EDGES = [0, 1.5]
THETAS = [0, 45, 90, 135]
THETAS_MATCH_DU = [0, 45]


##--------MAIN--------##
#############################################
def main(args):

    # set logger to the specified level
    util.set_logger_level(logger, level=args.log_level)

    data_dict = update_keys(util.read_data(args.data_path))
    latent_dict = None
    if not args.raw:
        latent_dict = util.load_latent(args.model_dir)

    if args.output_dir is None:
        args.output_dir = args.model_dir

    # plot examples
    if latent_dict is not None:
        savepath = Path(args.output_dir, 'osca_examples')
        plot_examples(data_dict, latent_dict, trial_ix=15, savepath=savepath)
    
    # plot factor PCA
    n_dim = 2 if args.plot_2d else 3
    plot_save_factors_all(
        data_dict, latent_dict, 
        output_dir=Path(args.output_dir, f'factors_{n_dim}d_plots'), 
        shared_model=args.shared_model,
        plot_2d=args.plot_2d,
        projections=args.projections, 
        folded=True, 
        seed=args.seed
        )

    # run decoders
    if args.num_runs == 0:
        return

    for scale in [True, False]:
        scale_str = "scaling" if scale else "no scaling"

        logger.info(f'Decoders: {scale_str}', extra={'spacing': '\n'})
        run_decoders(
            data_dict, latent_dict, args.output_dir, run_logreg=args.run_logreg, 
            run_svm=args.run_svm, run_nl_decoder=args.run_nl_decoder, 
            num_runs=args.num_runs, seed=args.seed, scale=scale, 
            log_scores=True, parallel=args.parallel
            )


##--------GENERAL FUNCTIONS--------##
#############################################
def update_keys(data_dict):
    # update keys, if needed (surp -> unexp)
    for prefix in ['train', 'valid']:
        if f'{prefix}_surp' in data_dict.keys():
            data_dict[f'{prefix}_unexp'] = data_dict.pop(f'{prefix}_surp')
    return data_dict


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
    unexp = np.zeros(num_trials)
    unexp[data_dict['train_idx']] = data_dict['train_unexp']
    unexp[data_dict['valid_idx']] = data_dict['valid_unexp']
    unexp = unexp.astype('bool')

    ori = np.zeros(num_trials)
    ori[data_dict['train_idx']] = data_dict['train_ori']
    ori[data_dict['valid_idx']] = data_dict['valid_ori']
    ori = ori.astype('int')

    return data, unexp, ori


#############################################
def get_thetas(compare_same_abc=True):

    if compare_same_abc:
        thetas = THETAS
    else:
        thetas = THETAS_MATCH_DU

    return thetas


#############################################
def get_exp_theta(theta, compare_same_abc=True):

    if compare_same_abc:
        exp_theta = theta
    else:
        exp_theta = theta + 90
    
    return exp_theta

#############################################
def get_data_idxs_from_theta(unexp, ori, theta=0, compare_same_abc=True):

    exp_theta = get_exp_theta(theta, compare_same_abc)

    exp_idxs = np.logical_and(~unexp, ori==exp_theta)
    unexp_idxs = np.logical_and(unexp, ori==theta)
    
    return exp_idxs, unexp_idxs, exp_theta


##--------DECODER FUNCTIONS--------##
#############################################
def run_decoders(data_dict, latent_dict, output_dir, run_logreg=True, 
                 run_svm=True, run_nl_decoder=False, num_runs=10, scale=True, 
                 seed=None, log_scores=True, parallel=False):

    if latent_dict is None:
        X_train = data_dict['train_fluor']
        X_test = data_dict['valid_fluor']
    else:
        X_train = latent_dict['train']['latent']
        X_test = latent_dict['valid']['latent']
    
    X = np.concatenate([X_train, X_test], axis=0)

    y_train = data_dict['train_unexp'].astype('int')
    y_test = data_dict['valid_unexp'].astype('int')
    y = np.concatenate([y_train, y_test], axis=0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if run_logreg:
        if log_scores:
            logger.info(
                '  Logistic regression bal. acc. scores:', extra={'spacing': '\n'}
                )
        logreg_scores = perform_decoder_runs(
            logreg_eval, X=X, y=y, num_runs=num_runs, seed=seed, 
            log_scores=log_scores, scale=scale, parallel=parallel
            )
        np.save(Path(output_dir, 'logreg_decoder_scores'), logreg_scores)

    if run_svm:
        if log_scores:
            logger.info('RBF SVM scores:', extra={'spacing': '\n'})
        logreg_scores = perform_decoder_runs(
            rbf_svm_eval, X=X, y=y, num_runs=num_runs, seed=seed, 
            log_scores=log_scores, scale=scale, parallel=parallel
            )
        np.save(Path(output_dir, 'rbf_svm_scores'), logreg_scores)

    if run_nl_decoder:
        if log_scores:
            logger.info(
                '  Non-linear decoder bal. acc.scores:', extra={'spacing': '\n'}
                )

        decoder_kwargs = dict()

        model_kwargs = copy.deepcopy(MODEL_KWARGS)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs['device'] = device

        fit_kwargs = copy.deepcopy(FIT_KWARGS)
        fit_kwargs['save_loc'] = output_dir

        decoder_kwargs = {
            'scale'       : scale,
            'model_kwargs': model_kwargs,
            'fit_kwargs'  : fit_kwargs,
        }

        nl_decoder_scores = perform_decoder_runs(
            recurrent_net_eval, X=X, y=y, num_runs=num_runs, seed=seed, 
            log_scores=log_scores, parallel=parallel, max_jobs=4, 
            **decoder_kwargs
            )

        np.save(Path(output_dir, 'non_linear_decoder_scores'), nl_decoder_scores)


#############################################
def perform_decoder_run(decoder_fct, X, y, train_p=0.8, seed=None, 
                        **decoder_kwargs):

    # get a new split
    train_data, test_data = train_test_split(
        list(zip(X, y)), train_size=train_p, stratify=y, 
        random_state=seed,
        )

    decoder_kwargs['X_train'] = np.asarray(list(zip(*train_data))[0])
    decoder_kwargs['y_train'] = np.asarray(list(zip(*train_data))[1])
    decoder_kwargs['X_test'] = np.asarray(list(zip(*test_data))[0])
    decoder_kwargs['y_test'] = np.asarray(list(zip(*test_data))[1])

    run_score = decoder_fct(seed=seed, **decoder_kwargs)

    return run_score


#############################################
def perform_decoder_runs(decoder_fct, X, y, num_runs=10, seed=None, 
                         log_scores=True, train_p=TRAIN_P, parallel=False, 
                         max_jobs=-1, **decoder_kwargs):

    util.seed_all(seed)

    sub_seeds = [np.random.choice(int(2e5)) for _ in range(num_runs)]

    if parallel:
        n_jobs = min(multiprocessing.cpu_count(), num_runs)
        if n_jobs <= 1:
            parallel = False
        if max_jobs != -1:
            n_jobs = min(n_jobs, max_jobs)

    if parallel:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(perform_decoder_run)(
                decoder_fct, X, y, train_p=train_p, seed=sub_seed, 
                **decoder_kwargs
                ) for sub_seed in sub_seeds
            )
    else:
        scores = []

    for i, sub_seed in enumerate(sub_seeds):
        if parallel:
            run_score = scores[i]
        else:
            run_score = perform_decoder_run(
                decoder_fct, X, y, train_p=train_p, seed=sub_seed, 
                **decoder_kwargs
                )
            scores.append(run_score)

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
def logreg_eval(X_train, y_train, X_test, y_test, scale=True, seed=None, 
                grid_search=False, log_params=True):

    # reshape data
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # initialize pipeline
    model = LogisticRegression(
        C=1, class_weight='balanced', random_state=seed, solver='lbfgs', 
        penalty='l2', max_iter=1000, fit_intercept=True
        )
    scale_steps = [StandardScaler()] if scale else []
    pipeline = make_pipeline(*scale_steps, model)

    # select parameters with cross-validation grid search
    if grid_search:
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        param_dist = {'logisticregression__C': loguniform(1e-2, 1e2)}
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
        logger.info(f'Best parameters (logreg):\n    {param_str}')

    # predict on the held out test set
    score = metrics.balanced_accuracy_score(y_test, pipeline.predict(X_test))

    return score


#############################################
def rbf_svm_eval(X_train, y_train, X_test, y_test, scale=True, seed=None, 
                 grid_search=False, log_params=True):

    # reshape data
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # initialize pipeline
    model = SVC(
        C=1, gamma='scale', class_weight='balanced', kernel='rbf', 
        random_state=seed
        )
    scale_steps = [StandardScaler()] if scale else []
    pipeline = make_pipeline(*scale_steps, model)

    # select parameters with cross-validation grid search
    if grid_search:
        cv = StratifiedShuffleSplit(
            n_splits=5, test_size=0.2, random_state=seed
            )
        param_dist = {'svc__C': loguniform(1e-2, 1e10),
                      'svc__gamma': loguniform(1e-9, 1e2)}
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
        logger.info(f'Best parameters (rbf SVM):\n    {param_str}')

    # predict on the held out test set
    score = metrics.balanced_accuracy_score(y_test, pipeline.predict(X_test))

    return score
    

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
    dense_size = max(y_train) + 1

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
    y_hat_test = model.predict(X_test)
    score = metrics.balanced_accuracy_score(
        y_test.squeeze(), y_hat_test.squeeze()
        )

    return score
    

#############################################
def balanced_accuracy_score(targets, predictions):
    C = metrics.confusion_matrix(targets, predictions)
    return np.mean(C.diagonal() / C.sum(axis=1))



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
                          shared_model=False, plot_2d=False, projections=False, 
                          folded=True, seed=None, close=True):

    latents, unexp, ori = load_trial_data(data_dict, latent_dict)

    n_dim = 2 if plot_2d else 3

    projections_str = '_withproj' if projections else ''
    share_str = '_shared' if shared_model else ''
    basename = f'factors_{n_dim}d_orisplit{share_str}'
    for plot_single_trials in [False, True]:
        suffix = '_withtrials' if plot_single_trials else projections_str
        for compare_same_abc in [True, False]:
            match = 'abc' if compare_same_abc else 'du'
            savename = f'{basename}_match_{match}{suffix}'

            fact_fig = plot_factors_all_ori(
                latents, unexp, ori, compare_same_abc=compare_same_abc, 
                plot_single_trials=plot_single_trials, 
                shared_model=shared_model, plot_2d=plot_2d,
                projections=projections, folded=folded, seed=seed
                )
            
            savepath = Path(output_dir, savename)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fact_fig.savefig(
                f'{savepath}.{SAVE_KWARGS["format"]}', **SAVE_KWARGS
                )
    
    if close:
        plt.close("all")


#############################################
def plot_factors_all_ori(latents, unexp, ori, compare_same_abc=True, 
                         plot_single_trials=False, shared_model=False, 
                         plot_2d=False, projections=False, folded=True, 
                         seed=None):
    
    fig, ax_ups, ax_dns = init_factors_fig(
        compare_same_abc=compare_same_abc, plot_2d=plot_2d, 
        shared_axes=shared_model
        )
    
    # create model with full latents
    model = None
    if shared_model:
        n_comps = 2 if plot_2d else 3
        model = fit_PCA_Regression(
            latents, 
            num_components=n_comps,
            seed=seed
            )

    thetas = get_thetas(compare_same_abc)
    for theta, ax_up, ax_dn in zip(thetas, ax_ups, ax_dns):
    
        exp_idxs, unexp_idxs, exp_theta = get_data_idxs_from_theta(
            unexp, ori, theta, compare_same_abc=compare_same_abc
            )

        plot_factors(
            latents,
            exp_idxs,
            unexp_idxs,
            ax_up, 
            ax_dn, 
            model=model,
            plot_single_trials=plot_single_trials, 
            plot_2d=plot_2d,
            projections=projections, 
            folded=folded, 
            seed=seed
            )
        
        if compare_same_abc:
            title = u'ABC: {}{}'.format(theta, DEG)
        else:
            title = u'D/U: {}{}'.format(exp_theta, DEG)

        ax_up.set_title(title, fontsize='x-large')

    if plot_2d:
        adjust_trajectory_axes(ax_ups, shared_axes=shared_model)

    adjust_distance_axes(ax_dns)

    return fig


#############################################
def init_factors_fig(compare_same_abc=True, plot_2d=False, shared_axes=False):
    
    if plot_2d:
        subplot_kw = dict()
    else:
        subplot_kw = {'projection': '3d'}


    figsize = [8.75, 4.2]
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(241, **subplot_kw)
    ax2 = fig.add_subplot(242, **subplot_kw)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)

    if compare_same_abc:
        ax3 = fig.add_subplot(243, **subplot_kw)
        ax4 = fig.add_subplot(244, **subplot_kw)
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)
        ax_ups = [ax1, ax2, ax3, ax4]
        ax_dns = [ax5, ax6, ax7, ax8]
    else:
        ax_ups = [ax1, ax2]
        ax_dns = [ax5, ax6]

    if not plot_2d:
        for ax_up in ax_ups:
            ax_up.view_init(elev=40., azim=155.)
    
    if plot_2d:
        fig.subplots_adjust(hspace=0.4)
        if not shared_axes:
            fig.subplots_adjust(wspace=0.5)

    return fig, ax_ups, ax_dns


#############################################
def plot_factors(latents, exp_idxs, unexp_idxs, ax_up, ax_dn, model=None,
                 plot_single_trials=False, plot_2d=False, projections=False, 
                 folded=True, seed=None):
    
    if plot_single_trials:
        projections = False

    n_comps = 2 if plot_2d else 3
    if model is None:
        model = fit_PCA_Regression(
            latents[np.logical_or(exp_idxs, unexp_idxs)], 
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
        ax_up, proj_factors, exp_idxs, unexp_idxs, 
        plot_single_trials=plot_single_trials, projections=projections, 
        folded=folded
        )

    dist, boot_dist_std = calc_distance(
        proj_factors, exp_idxs, unexp_idxs, seed=seed
        )
    plot_distance(ax_dn, dist, dist_err=boot_dist_std)


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
def plot_trajectories(ax, proj_factors, exp_idxs, unexp_idxs, 
                      plot_single_trials=False, projections=True, folded=True, 
                      single_marks=True):

    proj_exp_mean = proj_factors[exp_idxs].mean(axis=0)
    proj_unexp_mean = proj_factors[unexp_idxs].mean(axis=0)

    n_comps = proj_factors.shape[-1]

    lw = 1.5
    ms = 4
    for mean, color_name in zip(
        [proj_exp_mean, proj_unexp_mean], ['linc_blue', 'linc_red']
        ):
        color = COLORS[color_name]

        ax.plot(*mean.T, color=color, lw=lw)
        ax.plot(*mean[0], color=color, marker='o', lw=0, ms=ms)
        ax.plot(*mean[-1], color=color, marker='^', lw=0, ms=ms)
        ax.plot(
            *mean[FRAMES_PER::FRAMES_PER].T, color=color, marker='s', lw=0, 
            ms=ms
            )
    
    if plot_single_trials:
        for grp, color_name in zip(
            [exp_idxs, unexp_idxs], ['linc_blue', 'linc_red']
            ):

            keep_slice = slice(None)
            if sum(grp) > sum(unexp_idxs): # retain only every 10
                keep_slice = slice(None, None, 10)

            color = COLORS[color_name]
            for trial in proj_factors[grp][keep_slice]:
                ftrial = savgol_filter(trial, 9, 2, axis=0)
                ax.plot(*ftrial.T, color=color, lw=0.15)
                if single_marks:
                    ax.plot(*ftrial[0], color=color, marker='o', ms=1, lw=0)
                    ax.plot(*ftrial[-1], color=color, marker='^', ms=1, lw=0)
                    ax.plot(
                        *ftrial[FRAMES_PER::FRAMES_PER].T, 's', color=color, 
                        ms=1, lw=0
                        )
    
    else: # plot errors
        if n_comps == 2:
            add_stats_2d(
                ax, proj_factors[exp_idxs], proj_factors[unexp_idxs], 
                folded=folded
                )

        elif n_comps == 3 and projections:
            lims = add_stats_projections(
                ax, proj_factors[exp_idxs], proj_factors[unexp_idxs], 
                folded=folded
                )
            set_all_lims(ax, lims)

    # format axes
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
        if hasattr(ax, "zaxis"):
            axes = [ax.xaxis, ax.yaxis, ax.zaxis]
        else:
            axes = [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis] # deprecated
        for axis in axes:
            axis.line.set_linewidth(1)
            axis.set_ticklabels([])
            axis._axinfo['grid'].update({'linewidth': 0.5})
            axis._axinfo['tick'].update({'inward_factor': 0})
            axis._axinfo['tick'].update({'outward_factor': 0})

    else:
        raise NotImplementedError('Expected 2 or 3 components.')
            

##--------PLOT DISTANCES--------##
#############################################
def adjust_distance_axes(axs):
    
    # y axis is shared for distances
    min_ylim = np.min([ax.get_ylim()[0] for ax in axs])
    max_ylim = np.max([ax.get_ylim()[1] for ax in axs])
    for a, ax in enumerate(axs):
        ax.set_ylim([min_ylim, max_ylim])

    max_ytick = np.floor(max_ylim * 20) / 20
    yticks = [0, max_ytick]

    # reverse order to avoid erasing ytick labels
    for a, ax in enumerate(axs[::-1]):
        ax.set_yticks(yticks)
        ax.spines['left'].set_bounds(yticks)
        if a == len(axs) - 1:
            ax.set_ylabel('Distance', fontsize='large')
            ax.set_yticklabels(
                [f'{ytick:.2f}' for ytick in yticks], fontsize='large'
                )
        else:
            ax.set_yticklabels(['', '']) 


#############################################
def plot_distance(ax, dist, dist_err=None, color='darkslategrey', 
                  add_marks=True, alpha=1.0, label_x_axis=True):

    x = np.arange(len(dist))
    length = FRAMES_PER * N_GAB_FR
    if len(x) != length:
        raise ValueError(
            f'Plotting designed for Gabor sequences of length {length}, but '
            f'found {len(x)}.'
            )

    # plot distance
    lw = 2.5
    ms = 5

    # calculate and plot error first
    if dist_err is not None:
        ax.fill_between(
            x, dist - dist_err, dist + dist_err, color=color, alpha=0.3, lw=0
            )

    ax.plot(dist, color=color, lw=lw, alpha=alpha)
    if add_marks:
        ax.plot(x[0], dist[0], 'o', color=color, ms=ms, alpha=alpha)
        ax.plot(x[-1], dist[-1], '^', color=color, ms=ms, alpha=alpha)
        ax.plot(
            x[FRAMES_PER::FRAMES_PER], dist[FRAMES_PER::FRAMES_PER], 's', 
            color=color, ms=ms, alpha=alpha
            )
        
        # amend the plot format a bit
        for i in x[FRAMES_PER::FRAMES_PER]:
            ax.axvline(
                i, ls='dashed', lw=1, zorder=-13, alpha=0.6, color='k', 
                ymin=0.1
                )

    ax.set_xlim([x[0] - 5, x[-1] + 2]) # some padding
    ax.set_xticks([x[0], x[-1]])
    ax.xaxis.set_tick_params(length=lw * 2, width=lw)
    ax.yaxis.set_tick_params(length=lw * 2, width=lw)

    if label_x_axis:
        ax.set_xticklabels([str(t) for t in TIME_EDGES], fontsize='large')
        ax.set_xlabel('Time (s)', fontsize='large')

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
    ax.spines['bottom'].set_bounds([x[0], x[-1]])
    

#############################################
def calc_distance(proj_factors, exp_idxs, unexp_idxs, seed=None, norm=False):

    proj_exp_mean = proj_factors[exp_idxs].mean(axis=0)
    proj_unexp_mean = proj_factors[unexp_idxs].mean(axis=0)

    dist = np.sqrt(np.sum((proj_unexp_mean - proj_exp_mean)**2, axis=-1))
    if norm:
        dist = dist / np.sum(dist)

    boot_dist_std = bootstrapped_diff_std(
        proj_factors[exp_idxs], proj_factors[unexp_idxs], seed=seed
    )

    return dist, boot_dist_std
    

#############################################
def bootstrapped_diff_std(exp, unexp, n_samples=1000, seed=None):
    """
    bootstrapped_std(data)
    
    Returns bootstrapped standard deviation of the mean.

    Required args:
        - exp (3D array): regular data
        - unexp (3D array): unexprise data
    
    Optional args:
        - n_samples (int): number of samplings to take for bootstrapping
                           default: 1000

    Returns:
        - boot_dist_std (float): bootstrapped distance (data2 mean - data1 mean)
    """
    
    rng = np.random.RandomState(seed)

    # random values
    n_exp = len(exp)
    n_unexp = len(unexp)
    choices_exp = rng.choice(
        np.arange(n_exp), (n_exp, n_samples), replace=True).reshape(n_exp, -1)
    choices_unexp = rng.choice(
        np.arange(n_unexp), (n_unexp, n_samples), replace=True
        ).reshape(n_unexp, -1)
    
    exp_resampled = np.mean(exp[choices_exp], axis=0)
    unexp_resampled = np.mean(unexp[choices_unexp], axis=0)

    boot_dist_std = np.std(
        np.sqrt(np.sum((unexp_resampled - exp_resampled)**2, axis=-1)), axis=0)
    
    return boot_dist_std


##--------PLOT PROJECTIONS--------##
#############################################
def add_stats_2d(ax, exp, unexp, error='std', folded=True):
    
    exp_mean, exp_err = obtain_statistics(exp, error=error)
    unexp_mean, unexp_err = obtain_statistics(unexp, error=error)
    
    length = FRAMES_PER * N_GAB_FR
    for (mean, error, color_name) in zip(
        [exp_mean, unexp_mean], [exp_err, unexp_err], ['linc_blue', 'linc_red']
        ):

        if len(mean) != length:
            raise ValueError(
                'Plotting designed for Gabor sequences of '
                f'length {length}, but found {len(mean)}.'
                )

        create_polygon_fill_between(
            ax, mean, error, color_name=color_name, alpha=0.2, folded=folded, 
            zorder=-13
            )


#############################################
def add_stats_projections(ax, exp, unexp, error='std', folded=True):
    
    exp_mean, exp_err = obtain_statistics(exp, error=error)
    unexp_mean, unexp_err = obtain_statistics(unexp, error=error)
    
    projections, lims = get_lims(
        ax, exp_mean, exp_err, unexp_mean, unexp_err, pad=0.05
        )
    
    alpha = 0.4
    ms = 3.5
    length = FRAMES_PER * N_GAB_FR
    for a, (axis, offset) in enumerate(zip(['x', 'y', 'z'], projections)):
        for (mean, error, color_name) in zip(
            [exp_mean, unexp_mean], 
            [exp_err, unexp_err], 
            ['linc_blue', 'linc_red']
            ):

            if len(mean) != length:
                raise ValueError(
                    'Plotting designed for Gabor sequences of '
                    f'length {length}, but found {len(mean)}.'
                    )

            create_polygon_fill_between(
                ax, mean, error, offset=offset, color_name=color_name, 
                alpha=0.2, axis=axis, folded=folded
                )

            kwargs = {
                'zdir'  : axis,
                'zs'    : offset,
                'color' : COLORS[color_name],
                'alpha' : alpha,
            }

            data = [mean[:, i] for i in range(3) if i != a]
            ax.plot(*data, lw=1.5, **kwargs)
            
            data = [mean[0, i] for i in range(3) if i != a]
            ax.plot(*data, marker='o', lw=0, ms=ms, **kwargs)
            
            data = [mean[-1, i] for i in range(3) if i != a]
            ax.plot(*data, marker='^', lw=0, ms=ms, **kwargs)
            
            data = [mean[FRAMES_PER::FRAMES_PER, i] for i in range(3) if i != a]
            ax.plot(*data, marker='s', lw=0, ms=ms, **kwargs)
            
    return lims


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
def obtain_statistics(data, error='std'):
    # trials x frames x axis
    data_mean = data.mean(axis=0)
    
    if error == 'std':
        data_err = data.std(axis=0)
    elif error == 'var':
        data_err = data.var(axis=0)
    else:
        raise ValueError(f'{error} error not recognized. Must be std or var.')
    
    return data_mean, data_err


#############################################
def set_all_lims(ax, axis_lims):
    for a, axis in enumerate(['X', 'Y', 'Z']):
        if axis == 'X':
            ax.set_xlim(axis_lims[a])
        elif axis == 'Y':
            ax.set_ylim(axis_lims[a])
        elif axis == 'Z':
            ax.set_zlim(axis_lims[a])


#############################################          
def get_lims(ax, exp_mean, exp_err, unexp_mean, unexp_err, pad=0):
    
    low_exp = (exp_mean - exp_err).min(axis=0) 
    high_exp = (exp_mean + exp_err).max(axis=0)
    pad_exp = (high_exp - low_exp) * pad
    low_exp_pad, high_exp_pad = (low_exp - pad_exp), (high_exp + pad_exp)

    low_unexp = (unexp_mean - unexp_err).min(axis=0)
    high_unexp = (unexp_mean + unexp_err).max(axis=0)
    pad_unexp = (high_unexp - low_unexp) * pad
    low_unexp_pad = low_unexp - pad_unexp
    high_unexp_pad = high_unexp + pad_unexp
    
    projections = []
    all_lims = []
    for a, axis in enumerate(['X', 'Y', 'Z']):
        
        min_val = np.min([low_exp_pad[a], low_unexp_pad[a]]) 
        max_val = np.max([high_exp_pad[a], high_unexp_pad[a]])
        
        if axis == 'X':
            lims = list(ax.get_xlim())
        elif axis == 'Y':
            lims = list(ax.get_ylim())
        elif axis == 'Z':
            lims = list(ax.get_zlim())
        
        lims[0] = np.min([min_val, lims[0]])
        lims[1] = np.max([max_val, lims[1]])
            
        if axis in ['Y', 'Z']:
            projections.append(lims[0]) # low
        elif axis == 'X':
            projections.append(lims[1]) # high
        
        all_lims.append(lims)
    
    return projections, all_lims
        

#############################################
def create_polygon_fill_between(ax, mean, error, offset=0, 
                                color_name='linclab_blue', alpha=0.2, 
                                axis='x', folded=True, zorder=None):
    
    from shapely import geometry
    from shapely.ops import unary_union
    
    n_dims = mean.shape[-1]
    if n_dims not in [2, 3]:
        raise NotImplementedError('Expected 2 or 3 dimensions.')

    if n_dims == 3:
        axes_keep = [v for v, val in enumerate(['x', 'y', 'z']) if val != axis]
    else:
        axes_keep = [0, 1]

    top = mean + error
    bottom = mean - error
    
    # https://stackoverflow.com/questions/59167152/project-a-3d-surfacegenerated-by-plot-trisurf-to-xy-plane-and-plot-the-outline
    polygons = []
    for i in range(len(mean) - 1):
        try:
            polygons.append(geometry.Polygon(
                [top[i, axes_keep], 
                 top[i + 1, axes_keep],
                 bottom[i + 1, axes_keep],
                 bottom[i, axes_keep]]))
            
        except (ValueError, Exception) as err:
            logger.warning(f'Polygon creation error: {err}')
            pass
    
    if folded:
        alpha /= 2
    else:
        # Check for self intersection while building up the cascaded union
        union = geometry.Polygon([])
        for polygon in polygons:
            try:
                union = unary_union([polygon, union])
            except ValueError as err:
                logger.warning(f'Union creation error: {err}')
                pass

        polygons = union
        if isinstance(polygons, geometry.Polygon):
            polygons = [polygons]
    
    for single_polygon in polygons:
        x, y = single_polygon.exterior.xy

        plg = PolyCollection(
            [list(zip(x, y))], alpha=alpha, facecolor=COLORS[color_name],
            zorder=zorder
            )
        if n_dims == 3:
            ax.add_collection3d(plg, zs=offset, zdir=axis)    
        else:
            ax.add_collection(plg)


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
    parser.add_argument('-p', '--projections', action='store_true', 
                        help='plot error as projections, if plotting 3d')

    parser.add_argument('--shared_model', action='store_true', 
                help="calculate PCA for full dataset.")
    parser.add_argument('--plot_2d', action='store_true', 
                help="calculate and plot PCA components in 2D instead of 3D.")

    parser.add_argument('--run_logreg', action='store_true')
    parser.add_argument('--run_svm', action='store_true')
    parser.add_argument('--run_nl_decoder', action='store_true')

    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--log_level', default='info', 
                        help='log level, e.g. debug, info, error')

    args = parser.parse_args()

    logger = util.get_logger_with_basic_format(level=args.log_level)

    main(args)

