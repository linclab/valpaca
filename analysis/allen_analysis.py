import pickle
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics
import sklearn.linear_model as models

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scipy.signal import savgol_filter

from supervised import Supervised_BiRecurrent_Net

import torch
import torch.nn as nn

import argparse

import sys
sys.path.extend(['.', '../'])
import utils

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_dir', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-n', '--num_runs', default=0, type=int)
parser.add_argument('-s', '--run_svm', action='store_true')
parser.add_argument('-p', '--projections', action='store_true')

colors = {'linc_red'  : '#E84924',
          'linc_blue' : '#37A1D0'}

def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    latent_filename = os.path.join(args.model_dir, 'latent.pkl')
    
    if not os.path.exists(latent_filename):
        raise ValueError(f'latent.pkl file not found in {args.model_dir}. Run infer_latent.py')

    latent_dict = pickle.load(open(latent_filename, 'rb'))

    data_dict = utils.read_data(args.data_path)
                
    num_trials, num_steps, num_cells = latent_dict['ordered']['rates'].shape
    num_trials, num_steps, num_factors = latent_dict['ordered']['factors'].shape
    
    surp = np.zeros(num_trials)
    surp[data_dict['train_idx']] = data_dict['train_surp']
    surp[data_dict['valid_idx']] = data_dict['valid_surp']
    surp = surp.astype('bool')

    ori = np.zeros(num_trials)
    ori[data_dict['train_idx']] = data_dict['train_ori']
    ori[data_dict['valid_idx']] = data_dict['valid_ori']
    ori = ori.astype('int')
    
    factors = latent_dict['ordered']['factors']
    
    seed = 100
    
    fig1 = plot_factors_3d_all_ori(factors, surp, ori, compare_same_abc=True, with_single_trials=False, 
                                   projections=args.projections, seed=seed)
    fig2 = plot_factors_3d_all_ori(factors, surp, ori, compare_same_abc=True, with_single_trials=True, 
                                   seed=seed)
    fig3 = plot_factors_3d_all_ori(factors, surp, ori, compare_same_abc=False, with_single_trials=False, 
                                   projections=args.projections, seed=seed)
    fig4 = plot_factors_3d_all_ori(factors, surp, ori, compare_same_abc=False, with_single_trials=True, 
                                   seed=seed)
    

#     plt.legend([l1, l2], ['Surprise', 'Regular'])

    projections_str = "_withproj" if args.projections else ""
    
    fig1.savefig(os.path.join(args.model_dir, 'factors_3d_orisplit{}.svg'.format(projections_str)))
    fig2.savefig(os.path.join(args.model_dir, 'factors_3d_orisplit_withtrials.svg'))
    fig3.savefig(os.path.join(args.model_dir, 'factors_3d_orisplit_matchori{}.svg'.format(projections_str)))
    fig4.savefig(os.path.join(args.model_dir, 'factors_3d_orisplit_matchori_withtrials.svg'))
    
    plt.close()
    
    fig5 = plot_examples(latent_dict, data_dict, trial_ix=15)
    
    fig5.savefig(os.path.join(args.model_dir, 'allen_examples.svg'))
        
    trainp = 0.8
    
    seed = 100
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scores = []
    
    hyperparams = {'batch_size':68,
                   'learning_rate':0.00005,
                   'pos_weight':[1.0, 9.0],
                   'print_freq':250,
                   'max_epochs':5000,
                   'max_iter':1,
                   'save_loc':args.model_dir,
                   'hidden_size':16,
                   'num_layers':1,
                   'bias':True,
                   'dropout':0.25,
                   'device':device}
    
    if args.run_svm:
        print('rbf eval: %.3f'%rbf_svm_eval(latent_dict['train']['latent'].reshape(int(0.8*num_trials), num_factors*num_steps),
                                            data_dict['train_surp'].astype('int'),
                                            latent_dict['valid']['latent'].reshape(int(0.2*num_trials), num_factors*num_steps),
                                            data_dict['valid_surp'].astype('int')))
    
    for i in range(args.num_runs):
        
        train_data, test_data = train_test_split(list(zip(factors, surp)), train_size=trainp)
        
        x_train, y_train = zip(*train_data)
        x_test, y_test = zip(*test_data)
        
        x_train = np.array(x_train).reshape(int(np.round(num_trials*trainp)), num_steps, num_factors)
        x_test =np.array(x_test).reshape(int(np.round(num_trials*(1-trainp))), num_steps, num_factors)
        
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        scores.append(recurrent_net_eval(x_train, y_train,
                                         x_test, y_test, 
                                         args = hyperparams))
        
        print('n= %i, score: %.3f \u00B1 %.3f'%(i+1, np.mean(scores), np.std(scores)/np.sqrt(i+1)))
        
    if args.num_runs > 0: 
        np.save(os.path.join(args.model_dir, 'surp_scores'), scores)
    
def plot_factors_3d_all_ori(factors, surp, ori, compare_same_abc=True, with_single_trials=False, projections=False, seed=None):
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(241, projection='3d')
    ax2 = fig.add_subplot(242, projection='3d')
    ax3 = fig.add_subplot(243, projection='3d')
    ax4 = fig.add_subplot(244, projection='3d')
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.view_init(elev=40., azim=155.)

    if compare_same_abc:
        f = lambda x: x
    else:
        f = lambda x: (x+90)%180
    
    for theta, ax_up, ax_dn in zip([0, 45, 90, 135],
                                   [ax1, ax2, ax3, ax4],
                                   [ax5, ax6, ax7, ax8]):
    
        plot_factors_3d(factors,
                        np.logical_and(surp,ori==theta),
                        np.logical_and(~surp,ori==f(theta)),
                        ax_up, ax_dn, with_single_trials, projections, seed)
        ax_up.set_title('Surp: %ideg, Reg: %ideg'%(theta, f(theta)))

    # share y axis for distances
    sharey_axes = [ax5, ax6, ax7, ax8]
    min_ylim = np.min([ax.get_ylim()[0] for ax in sharey_axes])
    max_ylim = np.max([ax.get_ylim()[1] for ax in sharey_axes])
    for ax in sharey_axes:
        ax.set_ylim([min_ylim, max_ylim])

    return fig
    
def plot_factors_3d(factors, surp_grp, regl_grp, ax1, ax2, with_single_trials=False, projections=False, seed=None):
    
    if with_single_trials:
        projections = False

    W, b = fit_PCA_Regression(factors[np.logical_or(surp_grp, regl_grp)], num_components=3)
    
    proj = factors @ W.T + b
    
    proj_surp_mean = proj[surp_grp].mean(axis=0)
    proj_regl_mean = proj[regl_grp].mean(axis=0)

    for mean, color in zip([proj_regl_mean, proj_surp_mean], ['linc_blue', 'linc_red']):
        ax1.plot(*mean.T, color=colors[color], lw=2)
        ax1.plot(*mean[0], color=colors[color], marker='o', lw=0, ms=6)
        ax1.plot(*mean[-1], color=colors[color], marker='^', lw=0, ms=6)
        ax1.plot(*mean[9::9].T, color=colors[color], marker='s', lw=0, ms=6)
    
    if with_single_trials:
        lims = None
        for trial in proj[regl_grp][::10]:
            ftrial = savgol_filter(trial, 9, 2, axis=0)
            ax1.plot(*ftrial.T, color=colors['linc_blue'], lw=0.15)
            ax1.plot(*ftrial[0], color=colors['linc_blue'], marker='o', ms=1, lw=0)
            ax1.plot(*ftrial[-1], color=colors['linc_blue'], marker='^', ms=1, lw=0)
            ax1.plot(*ftrial[9::9].T, 's', color=colors['linc_blue'], ms=1, lw=0)
    
        for trial in proj[surp_grp]:
            ftrial = savgol_filter(trial, 9, 2, axis=0)
            ax1.plot(*ftrial.T, color=colors['linc_red'], lw=0.15)
            ax1.plot(*ftrial[0], color=colors['linc_red'], marker='o', ms=1, lw=0)
            ax1.plot(*ftrial[-1], color=colors['linc_red'], marker='^', ms=1, lw=0)
            ax1.plot(*ftrial[9::9].T, 's', color=colors['linc_red'], ms=1, lw=0)

    elif projections:
        lims = add_stats_projections(ax1, proj[regl_grp], proj[surp_grp])
        set_all_lims(ax1, lims)

    # to reset the axis ticks
    ax1.locator_params(nbins=5, axis='x')
    ax1.locator_params(nbins=5, axis='y')
    ax1.locator_params(nbins=5, axis='z')

    dist = np.sqrt(np.sum((proj_surp_mean - proj_regl_mean)**2, axis=-1))

    # bootstrapped st deviation
    boot_dist_std = bootstrapped_diff_std(proj[regl_grp], proj[surp_grp], seed=seed)
    ax2.fill_between(range(len(dist)), dist - boot_dist_std, dist + boot_dist_std, 
                     color='darkslategrey', alpha=0.3)

    ax2.plot(dist, color='darkslategrey', lw=2)
    ax2.plot(0, dist[0], 'o', color='darkslategrey', ms=6)
    ax2.plot(44,dist[-1], '^', color='darkslategrey', ms=6)
    ax2.plot([9, 18, 27, 36], dist[9::9], 's', color='darkslategrey', ms=6)

    
def plot_examples(latent_dict, data_dict, trial_ix=0, num_traces_to_show=8, figsize=(2.75,2)):
    fig, axs = plt.subplots(nrows=2, ncols =2, figsize=figsize, )

    time = np.linspace(0, 1.5, 45)
    
    ix_to_show = np.argsort(data_dict['valid_fluor'][trial_ix].std(axis=0))[-num_traces_to_show:]
#     pdb.set_trace()

    num_traces_to_show=8
    ax = axs[0,0]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)*5
    plt.plot(time, s + data_dict['valid_fluor'][trial_ix][:, ix_to_show]/data_dict['obs_gain_init'].mean(), color=colors['linc_red'], lw=2)
    plt.plot(time, s + latent_dict['valid']['fluor'][trial_ix][:, ix_to_show]/data_dict['obs_gain_init'].mean(), color=colors['linc_blue'], lw=1.5)
    
#     pdb.set_trace()
    
#     plt.yticks(np.concatenate([s, s[1:]-1, [s[-1] + s[1]-1]]), [s[0], s[1]-1, ''*6])
    ax.set_xticklabels([])
    plt.ylabel('dF/F', fontsize=6)

    ax = axs[0,1]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)*20
    plt.plot(time, s + latent_dict['valid']['rates'][trial_ix][:, ix_to_show], color=colors['linc_blue'], lw=1.5)
#     plt.yticks(np.concatenate([s, s[1:]-2, [s[-1] + s[1]-2]]), [s[0], s[1]-2, ''*6])
    ax.set_xticklabels([])
    plt.ylabel('Spike Rate (Hz)', fontsize=6)

    ax = axs[1,0]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)
    plt.plot(time, s + latent_dict['valid']['spikes'][trial_ix][:, ix_to_show], color=colors['linc_blue'], lw=1.5)
#     plt.yticks(np.concatenate([s, s[1:]-1, [s[-1] + s[1]-1]]), [s[0], s[1]-1, ''*6])
    plt.ylabel('Spike Counts', fontsize=6)
    plt.xlabel('Time (s)', fontsize=6)

    ax = axs[1,1]
    plt.sca(ax)
    s = np.arange(num_traces_to_show)
    plt.plot(time, s + latent_dict['valid']['latent'][trial_ix][:, :num_traces_to_show], color=colors['linc_blue'], lw=1.5)
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


    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    return fig

def fit_PCA_Regression(X, num_components, conditions=None):
    num_trials, num_steps, num_cells = X.shape
    X = X.transpose(0, 2, 1)
    Y = X.reshape(num_trials*num_cells, num_steps)
    
    pca_results = PCA().fit(StandardScaler().fit_transform(Y))
    V = pca_results.components_[:num_components]
    
    alphas = np.logspace(-2, 1, 30)
    model = RidgeCV(alphas = alphas, fit_intercept=True).fit(X.mean(axis=0).transpose(1, 0), V.transpose(1, 0))
    
    return model.coef_, model.intercept_
        
def rbf_svm_eval(X_train, y_train, X_test, y_test):
    param_dist = {'C': loguniform(1e-2, 1e10),
                  'gamma': loguniform(1e-9, 1e2)}
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=100)
    grid = RandomizedSearchCV(SVC(class_weight={0: 1.0, 1: 9.0}), param_distributions=param_dist, cv=cv, n_iter=200)
    grid.fit(X_train, y_train)
    return balanced_accuracy_score(y_test, grid.predict(X_test))
    
def recurrent_net_eval(x_train, y_train, x_test, y_test, args= {'batch_size' : 25, 'learning_rate' : 0.0001, 'pos_weight' : [1.0, 9.0], 'print_freq' : 250, 'max_epochs' : 1000, 'max_iter' : 1, 'save_loc' : None, 'hidden_size' : 16, 'num_layers' : 1, 'bias' : True, 'dropout' : 0.05, 'device' : 'cpu'}):
    
    input_size = x_train.shape[-1]
    dense_size = max(y_train) + 1
    model = Supervised_BiRecurrent_Net(input_size=input_size,
                                       hidden_size=args['hidden_size'],
                                       dense_size=dense_size,
                                       num_layers=args['num_layers'],
                                       bias=args['bias'],
                                       dropout=args['dropout'],
                                       device=args['device'])
    model.fit(x_train, y_train, x_test, y_test,
              batch_size=args['batch_size'],
              learning_rate=args['learning_rate'],
              pos_weight=args['pos_weight'],
              print_freq=args['print_freq'],
              max_epochs=args['max_epochs'],
              max_iter=args['max_iter'],
              save_loc=args['save_loc'])
    y_hat_test = model.predict(x_test)
    return balanced_accuracy_score(y_test.squeeze(), y_hat_test.squeeze())
    
def balanced_accuracy_score(targets, predictions):
    C = metrics.confusion_matrix(targets, predictions)
    return np.mean(C.diagonal() / C.sum(axis=1))

def bootstrapped_diff_std(regl, surp, n_samples=1000, seed=None):
    """
    bootstrapped_std(data)
    
    Returns bootstrapped standard deviation of the mean.

    Required args:
        - regl (3D array): regular data
        - surp (3D array): surprise data
    
    Optional args:
        - n_samples (int): number of samplings to take for bootstrapping
                           default: 1000

    Returns:
        - boot_dist_std (float): bootstrapped distance (data2 mean - data1 mean)
    """
    
    rng = np.random.RandomState(seed)

    # random values
    n_regl = len(regl)
    n_surp = len(surp)
    choices_regl = rng.choice(
        np.arange(n_regl), (n_regl, n_samples), replace=True).reshape(n_regl, -1)
    choices_surp = rng.choice(
        np.arange(n_surp), (n_surp, n_samples), replace=True).reshape(n_surp, -1)
    
    regl_resampled = np.mean(regl[choices_regl], axis=0)
    surp_resampled = np.mean(surp[choices_surp], axis=0)

    boot_dist_std = np.std(
        np.sqrt(np.sum((surp_resampled - regl_resampled)**2, axis=-1)), axis=0)
    
    return boot_dist_std


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


def set_all_lims(ax, axis_lims):
    for a, axis in enumerate(['X', 'Y', 'Z']):
        if axis == 'X':
            ax.set_xlim(axis_lims[a])
        elif axis == 'Y':
            ax.set_ylim(axis_lims[a])
        elif axis == 'Z':
            ax.set_zlim(axis_lims[a])

            
def get_lims(ax, regl_mean, regl_err, surp_mean, surp_err, pad=0):
    low_regl, high_regl = (regl_mean - regl_err).min(axis=0), (regl_mean + regl_err).max(axis=0)
    pad_regl = (high_regl - low_regl) * pad
    low_regl_pad, high_regl_pad = (low_regl - pad_regl), (high_regl + pad_regl)

    low_surp, high_surp = (surp_mean - surp_err).min(axis=0), (surp_mean + surp_err).max(axis=0)
    pad_surp = (high_surp - low_surp) * pad
    low_surp_pad, high_surp_pad = (low_surp - pad_surp), (high_surp + pad_surp)
    
    projections = []
    all_lims = []
    for a, axis in enumerate(['X', 'Y', 'Z']):
        
        min_val = np.min([low_regl_pad[a], low_surp_pad[a]]) 
        max_val = np.max([high_regl_pad[a], high_surp_pad[a]])
        
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
        
    
def create_polygon_fill_between(ax, mean, error, offset, color, alpha=0.2, axis='x'):
    
    from shapely import geometry
    
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
    
    alpha /= 2
    
    for single_polygon in polygons:
        x, y = single_polygon.exterior.xy

        plg = PolyCollection([list(zip(x, y))], alpha=alpha, facecolor=colors[color])
        ax.add_collection3d(plg, zs=offset, zdir=axis)

    
def add_stats_projections(ax, regl, surp, error='std'):
    regl_mean, regl_err = obtain_statistics(regl, error=error)
    surp_mean, surp_err = obtain_statistics(surp, error=error)
    
    projections, lims = get_lims(ax, regl_mean, regl_err, surp_mean, surp_err, pad=0.05)
    
    alpha = 0.4
    for a, (axis, offset) in enumerate(zip(['x', 'y', 'z'], projections)):
        for (mean, error, color) in zip(
            [regl_mean, surp_mean], [regl_err, surp_err], ['linc_blue', 'linc_red']):

            create_polygon_fill_between(ax, mean, error, offset, color, alpha=0.2, axis=axis)

            data = [mean.T[i] for i in range(3) if i != a]
            ax.plot(*data, zdir=axis, zs=offset, color=colors[color], lw=1.5, alpha=alpha)
            
            data = [mean[0, i] for i in range(3) if i != a]
            ax.plot(*data, zdir=axis, zs=offset, color=colors[color], alpha=alpha, marker='o', lw=0, ms=3.5)
            
            data = [mean[-1, i] for i in range(3) if i != a]
            ax.plot(*data, zdir=axis, zs=offset, color=colors[color], alpha=alpha, marker='^', lw=0, ms=3.5)
            
            data = [mean[9::9, i] for i in range(3) if i != a]
            ax.plot(*data, zdir=axis, zs=offset, color=colors[color], alpha=alpha, marker='s', lw=0, ms=3.5)
            
    return lims


if __name__ == '__main__':
    main()
    