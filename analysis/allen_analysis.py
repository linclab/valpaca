import argparse
import copy
from pathlib import Path
import sys
import warnings


import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, \
    RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from scipy.signal import savgol_filter

sys.path.extend(['.', '..'])
import utils
from supervised import Supervised_BiRecurrent_Net


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
    'learning_rate': 0.00005,
    'pos_weight': None,
    'print_freq': 250,
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

    latent_dict = utils.load_latent(args.model_dir)
    data_dict = update_keys(utils.read_data(args.data_path))

    # plot examples
    savepath = Path(args.model_dir, 'allen_examples.svg')
    plot_examples(data_dict, latent_dict, trial_ix=15, savepath=savepath)
    
    # plot factor PCA
    plot_save_factors_3d_all(
        data_dict, latent_dict, Path(args.model_dir, 'factors_3d_plots'), 
        projections=args.projections, 
        folded=True, 
        seed=args.seed
        )

    # run decoders
    run_decoders(
        data_dict, latent_dict, args.model_dir, run_logreg=args.run_logreg, 
        run_svm=args.run_svm, run_nl_decoder=args.nl_decoder, 
        num_runs=args.num_runs, seed=args.seed, log_scores=True
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
def seed_all(seed):

    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


##--------DECODER FUNCTIONS--------##
#############################################
def run_decoders(data_dict, latent_dict, savedir, run_logreg=True, 
                 run_svm=True, run_nl_decoder=False, num_runs=10, scale=True, 
                 seed=None, log_scores=True):

    decoder_kwargs = {
        'X_train'   : latent_dict['train']['latent'],
        'y_train'   : data_dict['train_unexp'].astype('int'),
        'X_test'    : latent_dict['valid']['latent'],
        'y_test'    : data_dict['valid_unexp'].astype('int'),
        'scale'     : scale,
    }

    Path(savedir).mkdir(parents=True, exist_ok=True)

    if run_logreg:
        if log_scores:
            print("\nLogistic regression scores:")
        logreg_scores = perform_decoder_runs(
            logreg_eval, num_runs=num_runs, seed=seed, 
            log_scores=log_scores, **decoder_kwargs
            )
        np.save(Path(savedir, 'logreg_decoder_scores'), logreg_scores)

    if run_svm:
        if log_scores:
            print("\nRBF SVM scores:")
        logreg_scores = perform_decoder_runs(
            rbf_svm_eval, num_runs=num_runs, seed=seed, 
            log_scores=log_scores, **decoder_kwargs
            )
        np.save(Path(savedir, 'rbf_svm_scores'), logreg_scores)

    if run_nl_decoder:
        if log_scores:
            print("\nNon-linear decoder scores:")

        nl_decoder_kwargs = copy.deepcopy(decoder_kwargs)

        model_kwargs = copy.deepcopy(MODEL_KWARGS)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs['device'] = device
        nl_decoder_kwargs['model_kwargs'] = model_kwargs

        fit_kwargs = copy.deepcopy(FIT_KWARGS)
        fit_kwargs['save_loc'] = savedir
        nl_decoder_kwargs['fit_kwargs'] = fit_kwargs

        nl_dec_scores = perform_decoder_runs(
            recurrent_net_eval, num_runs=num_runs, seed=seed, 
            log_scores=log_scores, **nl_decoder_kwargs
            )

        np.save(Path(savedir, 'non_linear_decoder_scores'), nl_dec_scores)


#############################################
def perform_decoder_runs(decoder_fct, num_runs=10, seed=None, log_scores=True, 
                         **decoder_kwargs):

    seed_all(seed)

    scores = []
    for i in range(num_runs):
        sub_seed = np.random.choice(int(2e5))
        run_score = decoder_fct(seed=sub_seed, **decoder_kwargs)
        scores.append(run_score)

        if log_scores:
            score_mean = np.mean(scores)
            score_sem = np.std(scores) / np.sqrt(i+1)

            running_score_str = u'running score {:.3f} {} {:.3f}'.format(
                score_mean, PLUS_MIN, score_sem
                )
            score_str = u'n={}, score: {:.3f}, {}'.format(
                i+1, run_score, running_score_str
                )

            print(score_str)


#############################################
def logreg_eval(X_train, y_train, X_test, y_test, scale=True, seed=None, 
                log_params=True):

    # reshape data
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # initialize pipeline
    model = LogisticRegression(
        class_weight='balanced', random_state=seed, solver='lbfgs', 
        max_iter=1000, fit_intercept=True
        )
    scale_steps = [StandardScaler()] if scale else []
    pipeline = make_pipeline(*scale_steps, model)

    # fit with cross-validation and grid search
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    param_dist = {'logisticregression__C': loguniform(1e-2, 1e2)}
    n_grid_iter = 20 * len(param_dist) # partial scaling with num of grid dims
    grid = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        cv=cv, n_iter=n_grid_iter, n_jobs=-1
        )
    grid.fit(X_train, y_train)

    if log_params:
        param_str = '\n    '.join(
            [f'{k}: {v}' for k, v in grid.best_params_.items()]
            )
        print(f'best parameters (logreg):\n    {param_str}')

    # predict on the held out test set
    score = balanced_accuracy_score(y_test, grid.predict(X_test))

    return score


#############################################
def rbf_svm_eval(X_train, y_train, X_test, y_test, scale=True, seed=None, 
                log_params=True):

    # reshape data
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # initialize pipeline
    model = SVC(class_weight='balanced', random_state=seed)
    scale_steps = [StandardScaler()] if scale else []
    pipeline = make_pipeline(*scale_steps, model)

    # fit with cross-validation and grid search
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    param_dist = {'svc__C': loguniform(1e-2, 1e10),
                  'svc__gamma': loguniform(1e-9, 1e2)}
    n_grid_iter = 20 * len(param_dist) # partial scaling with num of grid dims
    grid = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        cv=cv, n_iter=n_grid_iter, n_jobs=-1
        )
    grid.fit(X_train, y_train)

    if log_params:
        param_str = '\n    '.join(
            [f'{k}: {v}' for k, v in grid.best_params_.items()]
            )
        print(f'best parameters (rbf SVM):\n    {param_str}')

    # predict on the held out test set
    score = balanced_accuracy_score(y_test, grid.predict(X_test))

    return score
    

#############################################
def recurrent_net_eval(X_train, y_train, X_test, y_test, model_kwargs, 
                       fit_kwargs, scale=True, seed=None, train_p=TRAIN_P):

    # get a validation set from the training set
    train_data, valid_data = train_test_split(
        list(zip(X_train, y_train)), train_size=train_p, random_state=seed,
        )

    X_train, y_train = [np.asarray(data) for data in zip(*train_data)]
    X_valid, y_valid = [np.asarray(data) for data in zip(*valid_data)]

    input_size = X_train.shape[-1]
    dense_size = max(y_train) + 1

    if fit_kwargs['pos_weight'] is None: # derive weights from train set
        fit_kwargs['pos_weight'] = [
            sum(y_train != v) / sum(y_train == v) for v in [0, 1]
            ]

    model = Supervised_BiRecurrent_Net(
        input_size=input_size, dense_size=dense_size, **model_kwargs
        )
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(len(X_train), -1)
            ).reshape(X_train.shape)

        X_test = scaler.transform(
            X_test.reshape(len(X_test), -1)
            ).reshape(X_test.shape)

    # fit with a validation set
    model.fit(X_train, y_train, X_valid, y_valid, **fit_kwargs)
    
    # predict on the held out test set
    y_hat_test = model.predict(X_test)
    score = balanced_accuracy_score(y_test.squeeze(), y_hat_test.squeeze())

    return score
    

#############################################
def balanced_accuracy_score(targets, predictions):
    C = metrics.confusion_matrix(targets, predictions)
    return np.mean(C.diagonal() / C.sum(axis=1))



##--------PLOT EXAMPLES--------##
#############################################
def plot_examples(data_dict, latent_dict, trial_ix=0, num_traces=8, 
                  figsize=(2.75, 2), savepath=None):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    # gather data
    gt_fluor = data_dict['valid_fluor'][trial_ix] # ground truth
    recon_fluor = latent_dict['valid']['fluor'][trial_ix]
    recon_rates = latent_dict['valid']['rates'][trial_ix]
    recon_spikes = latent_dict['valid']['spikes'][trial_ix]
    latents = latent_dict['valid']['latent'][trial_ix]

    x = np.linspace(*TIME_EDGES, len(gt_fluor)) # time in seconds
    
    # some plotting parameters
    gt_lw = 2
    lw = 1.5
    fs = 6
    gt_color = COLORS['linc_red']
    recon_color = COLORS['linc_blue']

    # select indices of sequences with higher standard deviations
    ix_incl = np.argsort(gt_fluor.std(axis=0))[-num_traces : ]
    num_traces = len(ix_incl)

    # plot fluorescence pre/post reconstruction
    ax = axs[0, 0]
    s = np.arange(num_traces) * 5
    gain = data_dict['obs_gain_init'].mean()
    ax.plot(x, s + gt_fluor[:, ix_incl] / gain, color=gt_color, lw=gt_lw)
    ax.plot(x, s + recon_fluor[:, ix_incl] / gain, color=recon_color, lw=lw)

    ax.set_xticklabels([])
    ax.set_ylabel('dF/F', fontsize=fs)

    # plot reconstructed spike rates
    ax = axs[0, 1]
    s = np.arange(num_traces) * 20
    ax.plot(x, s + recon_rates[:, ix_incl], color=recon_color, lw=lw)
    ax.set_xticklabels([])
    ax.set_ylabel('Spike Rate (Hz)', fontsize=fs)

    # plot reconstructed spike counts
    ax = axs[1, 0]
    s = np.arange(num_traces)
    ax.plot(x, s + recon_spikes[:, ix_incl], color=recon_color, lw=lw)
    ax.set_ylabel('Spike Counts', fontsize=fs)
    ax.set_xlabel('Time (s)', fontsize=fs)

    ax = axs[1, 1]
    s = np.arange(num_traces)
    ax.plot(x, s + latents[:, : num_traces], color=recon_color, lw=lw)
    ax.set_ylabel('Factors', fontsize=6)
    ax.set_xlabel('Time (s)', fontsize=6)
    ax.yaxis.set_ticks_position('none')
    ax.spines['left'].set_visible(False)
    
    for ax in axs.ravel():
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(labelsize=fs)

    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, **SAVE_KWARGS)

    return fig


##--------PLOT PCA FACTORS--------##
#############################################
def plot_save_factors_3d_all(data_dict, latent_dict, savedir='factors_3d_plots', 
                             projections=False, folded=True, seed=None):

    latents, unexp, ori = load_trial_data(data_dict, latent_dict)

    basename = 'factors_3d_orisplit'
    for compare_same_abc in [True, False]:
        match = 'abc' if compare_same_abc else 'du'
        basename_match = f'{basename}_match_{match}'
        for plot_single_trials in [True, False]:
            projections_str = '_withproj' if projections else ''
            suffix = '_withtrials' if plot_single_trials else projections_str
            savename = f'{basename_match}{suffix}'

            fact_fig = plot_factors_3d_all_ori(
                latents, unexp, ori, compare_same_abc=compare_same_abc, 
                plot_single_trials=plot_single_trials, 
                projections=projections, folded=folded, seed=seed
                )
            
            savepath = Path(savedir, savename)
            Path(savedir).mkdir(parents=True, exist_ok=True)
            fact_fig.savefig(savepath, **SAVE_KWARGS)    
            
    plt.close()


#############################################
def plot_factors_3d_all_ori(latents, unexp, ori, compare_same_abc=True, 
                            plot_single_trials=False, projections=False, 
                            folded=True, seed=None):
    
    fig, ax_ups, ax_dns = init_factors_3d_fig(compare_same_abc=compare_same_abc)
    
    if compare_same_abc:
        thetas = THETAS
        exp_thetas_f = lambda x: x
    else:
        thetas = THETAS_MATCH_DU
        exp_thetas_f = lambda x: (x + 90)
        
    for theta, ax_up, ax_dn in zip(thetas, ax_ups, ax_dns):
    
        plot_factors_3d(
            latents,
            np.logical_and(unexp, ori==theta),
            np.logical_and(~unexp, ori==exp_thetas_f(theta)),
            ax_up, ax_dn, 
            plot_single_trials=plot_single_trials, 
            projections=projections, 
            folded=folded, 
            seed=seed
            )
        
        if theta == exp_thetas_f(theta):
            title = u'ABC: {}{}'.format(theta, DEG)
        else:
            title = u'D/U: {}{}'.format(exp_thetas_f(theta), DEG)
        ax_up.set_title(title, fontsize='x-large')

    adjust_distance_axes(ax_dns)

    return fig


#############################################
def init_factors_3d_fig(compare_same_abc=True):
    
    figsize = [8.75, 4.2]
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(241, projection='3d')
    ax2 = fig.add_subplot(242, projection='3d')
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)

    if compare_same_abc:
        ax3 = fig.add_subplot(243, projection='3d')
        ax4 = fig.add_subplot(244, projection='3d')
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)
        ax_ups = [ax1, ax2, ax3, ax4]
        ax_dns = [ax5, ax6, ax7, ax8]
    else:
        ax_ups = [ax1, ax2]
        ax_dns = [ax5, ax6]

    for ax_up in ax_ups:
        ax_up.view_init(elev=40., azim=155.)

    return fig, ax_ups, ax_dns


#############################################
def plot_factors_3d(latents, unexp_grp, exp_grp, ax_up, ax_dn, 
                    plot_single_trials=False, projections=False, folded=True, 
                    seed=None):
    
    if plot_single_trials:
        projections = False

    model = fit_PCA_Regression(
        latents[np.logical_or(unexp_grp, exp_grp)], 
        num_components=3,
        seed=seed
        )

    proj_factors = predict_PCA_Regression(latents, model)

    plot_trajectories(
        ax_up, proj_factors, unexp_grp, exp_grp, 
        plot_single_trials=plot_single_trials
        )

    calc_plot_distance(ax_dn, proj_factors, unexp_grp, exp_grp, seed=seed)

    if projections:
        lims = add_stats_projections(
            ax_up, proj_factors[exp_grp], proj_factors[unexp_grp], 
            folded=folded
            )
        set_all_lims(ax_up, lims)


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
def plot_trajectories(ax, proj_factors, unexp_grp, exp_grp, 
                      plot_single_trials=False):

    proj_unexp_mean = proj_factors[unexp_grp].mean(axis=0)
    proj_exp_mean = proj_factors[exp_grp].mean(axis=0)

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
            [exp_grp, unexp_grp], ['linc_blue', 'linc_red']
            ):

            keep_slice = slice(None)
            if sum(grp) > sum(unexp_grp): # retain only every 10
                keep_slice = slice(None, None, 10)

            color = COLORS[color_name]
            for trial in proj_factors[grp][keep_slice]:
                ftrial = savgol_filter(trial, 9, 2, axis=0)
                ax.plot(*ftrial.T, color=color, lw=0.15)
                ax.plot(*ftrial[0], color=color, marker='o', ms=1, lw=0)
                ax.plot(*ftrial[-1], color=color, marker='^', ms=1, lw=0)
                ax.plot(
                    *ftrial[FRAMES_PER::FRAMES_PER].T, 's', color=color, ms=1, 
                    lw=0
                    )

    # reset the axis ticks
    for axis in ['x', 'y', 'z']:
        ax.locator_params(nbins=5, axis=axis)
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(1)
        axis.set_ticklabels([])
        axis._axinfo['grid'].update({'linewidth': 0.5})
        axis._axinfo['tick'].update({'inward_factor': 0})
        axis._axinfo['tick'].update({'outward_factor': 0})


##--------PLOT DISTANCES--------##
#############################################
def adjust_distance_axes(axs):
    # share y axis for distances
    min_ylim = np.min([ax.get_ylim()[0] for ax in axs])
    max_ylim = np.max([ax.get_ylim()[1] for ax in axs])
    for a, ax in enumerate(axs):
        ax.set_ylim([min_ylim, max_ylim])

    max_ytick = np.floor(max_ylim * 20) / 20
    yticks = [0, max_ytick]
    for a, ax in enumerate(axs):
        ax.set_yticks(yticks)
        ax.spines['left'].set_bounds(yticks)
        if a == 0:
            ax.set_ylabel('Distance', fontsize='large')
            ax.set_yticklabels(
                [f'{ytick:.2f}' for ytick in yticks], fontsize='large'
                )
        else:
            ax.set_yticklabels(['', '']) 


#############################################
def calc_plot_distance(ax, proj_factors, unexp_grp, exp_grp, seed=None):

    proj_unexp_mean = proj_factors[unexp_grp].mean(axis=0)
    proj_exp_mean = proj_factors[exp_grp].mean(axis=0)

    # calculate distance between mean traces in 3D with bootstrapped std
    dist = np.sqrt(np.sum((proj_unexp_mean - proj_exp_mean)**2, axis=-1))
    boot_dist_std = bootstrapped_diff_std(
        proj_factors[exp_grp], proj_factors[unexp_grp], seed=seed
        )
    x = np.arange(len(dist))
    length = FRAMES_PER * N_GAB_FR
    if len(x) != length:
        raise ValueError(
            f'Plotting designed for Gabor sequences of length {length}, but '
            f'found {len(x)}.'
            )

    # plot distance
    color = 'darkslategrey'
    lw = 2.5
    ms = 5
    ax.fill_between(
        x, dist - boot_dist_std, dist + boot_dist_std, color=color, alpha=0.3, 
        lw=0
        )

    ax.plot(dist, color=color, lw=lw)
    ax.plot(x[0], dist[0], 'o', color=color, ms=ms)
    ax.plot(x[-1], dist[-1], '^', color=color, ms=ms)
    ax.plot(
        x[FRAMES_PER::FRAMES_PER], dist[FRAMES_PER::FRAMES_PER], 's', 
        color=color, ms=ms
        )
    
    # amend the plot format a bit
    for i in x[FRAMES_PER::FRAMES_PER]:
        ax.axvline(
            i, ls='dashed', lw=1, zorder=-13, alpha=0.6, color='k', ymin=0.1
            )
    ax.set_xlim([x[0] - 5, x[-1] + 2]) # some padding
    ax.set_xticks([x[0], x[-1]])
    ax.set_xticklabels([str(t) for t in TIME_EDGES], fontsize='large')
    ax.xaxis.set_tick_params(length=lw * 2, width=lw)
    ax.yaxis.set_tick_params(length=lw * 2, width=lw)
    ax.set_xlabel('Time (s)', fontsize='large')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
    ax.spines['bottom'].set_bounds([x[0], x[-1]])
    

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
                ax, mean, error, offset, color_name=color_name, alpha=0.2, 
                axis=axis, folded=folded
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
def create_polygon_fill_between(ax, mean, error, offset, color_name, alpha=0.2, 
                                axis='x', folded=True):
    
    from shapely import geometry
    from shapely.ops import unary_union
    
    axes_keep = [v for v, val in enumerate(['x', 'y', 'z']) if val != axis]
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
            print(f'Polygon creation error: {err}')
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
                print(f'Union creation error: {err}')
                pass

        polygons = union
        if isinstance(polygons, geometry.Polygon):
            polygons = [polygons]
    
    for single_polygon in polygons:
        x, y = single_polygon.exterior.xy

        plg = PolyCollection(
            [list(zip(x, y))], alpha=alpha, facecolor=COLORS[color_name]
            )
        ax.add_collection3d(plg, zs=offset, zdir=axis)


#############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_dir', type=str)
    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-n', '--num_runs', default=10, type=int, 
        help='number of decoders to run per type')
    parser.add_argument('-l', '--run_logreg', action='store_true')
    parser.add_argument('-s', '--run_svm', action='store_true')
    parser.add_argument('-s', '--run_nl_dec', action='store_true')
    parser.add_argument('-p', '--projections', action='store_true')
    parser.add_argument('-r', '--seed', default=100, type=int)

    args = parser.parse_args()

    main(args)

