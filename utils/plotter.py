import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.extend(['.', '..'])
from utils import util


logger = logging.getLogger(__name__)


SAVE_KWARGS = {
    'bbox_inches': 'tight',
    'facecolor'  : 'w',
    'transparent': False,
    'format'     : 'svg',
}

def get_suptitle_y(fig, fact=0.015):
    nrows = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0]
    y = 1 - fact * (nrows - 1)
    return y

class Plotter(object):
    def __init__(self, time, grdtruth=None, base_fontsize=14):
        self.dt = np.diff(time)[0]
        self.time = time
        self.fontsize = {
            'ticklabel': base_fontsize - 2,
            'label'    : base_fontsize,
            'title'    : base_fontsize + 2,
            'suptitle' : base_fontsize + 4,
            }
        
        self.colors = {
            'linc_red' : '#E84924',
            'linc_blue': '#37A1D0',
            }
        
        self.grdtruth = grdtruth
    
    
    #-------------------------------------------------------------------------
    def plot_summary(self, model, dl, num_average=200, ix=None, mode='traces', 
                     output_dir=None):
        
        '''
        self.plot_summary(model)
        
        Plot summary figures for dataset and ground truth if available. Create a batch
        from one sample by repeating a certain number of times, and average across them.
        
        Arguments:
            - num_average (int) : number of samples from posterior to average over
            - ix (int) : index of data samples to make summary plot from
            
        Returns:
            - fig_dict : dict of summary figures
        '''

        plt.close('all')
        
        figs_dict = dict()
        
        data = dl.dataset.tensors[0]
        
        batch_example, ix = util.batchify_random_sample(
            data=data, batch_size=num_average, ix=ix
            )
        batch_example = batch_example.to(model.device)
        figs_dict['ix'] = ix
        
        model.eval()
        with torch.no_grad():
            recon, (factors, inputs) = model(batch_example)
        
        orig = batch_example[0].cpu().numpy()
        logger.debug(
            f'Example shape: {batch_example.shape}, '
            f'Data shape: {data.shape}, ' 
            f'Reconstructed data shape: {recon["data"].shape}'
            )
                
        if mode == 'traces':
            figs_dict['traces'] = self.plot_traces(
                recon['data'].mean(dim=0).detach().cpu().numpy(), 
                orig, 
                mode='activity', 
                norm=True
                )
            
            y = get_suptitle_y(figs_dict['traces'], fact=0.01)
            figs_dict['traces'].suptitle(
                'Actual fluorescence trace vs.\n'
                'estimated mean for a sampled trial',
                y=y
                )
        
        elif mode == 'video':
            save_video_dir = (output_dir, 'videos')
            save_video_dir.mkdir(exist_ok=True, parents=True)
            self.plot_video(
                recon['data'].mean(dim=0).detach().cpu().numpy(), 
                orig, 
                save_folder=save_video_dir
                )
        
        else:
            raise ValueError('mode must be \'traces\' or \'video\'.')
        
        if self.grdtruth:
            if 'rates' in self.grdtruth.keys():
                recon_rates = recon['rates'].mean(dim=1).cpu().numpy()
                true_rates  = self.grdtruth['rates'][ix]
                figs_dict['grdtruth_rates'] = self.plot_traces(
                    recon_rates, true_rates, mode='rand'
                    )

                y = get_suptitle_y(figs_dict['grdtruth_rates'])
                figs_dict['grdtruth_rates'].suptitle(
                    'Reconstructed vs ground-truth rate function', y=y
                    )
            
            if 'latent' in self.grdtruth.keys():
                pred_factors = factors.mean(dim=1).cpu().numpy()
                true_factors = self.grdtruth['latent'][ix]
                figs_dict['grdtruth_factors'] = self.plot_traces(
                    pred_factors, true_factors, 
                    num_traces=true_factors.shape[-1], ncols=1
                    )
                
                y = get_suptitle_y(figs_dict['grdtruth_factors'])
                figs_dict['grdtruth_factors'].suptitle(
                    'Reconstructed vs ground-truth factors', y=y
                    )
            else:
                figs_dict['factors'] = self.plot_factors(
                    factors.mean(dim=1).cpu().numpy()
                    )
                
            if 'spikes' in self.grdtruth.keys():
                if 'spikes' in recon.keys():
                    recon_spikes = recon['spikes'].mean(dim=1).cpu().numpy()
                    true_spikes  = self.grdtruth['spikes'][ix]
                    figs_dict['grdtruth_spikes'] = self.plot_traces(
                        recon_spikes, true_spikes, mode='rand'
                        )
                    
                    y = get_suptitle_y(figs_dict['grdtruth_spikes'])
                    figs_dict['grdtruth_spikes'].suptitle(
                        'Reconstructed vs ground-truth spiking', y=y
                        )
        
        else:
            figs_dict['factors'] = self.plot_factors(
                factors.mean(dim=1).cpu().numpy()
                )
        
        if inputs is not None:
            figs_dict['inputs'] = self.plot_inputs(
                inputs.mean(dim=1).cpu().numpy()
                )

        return figs_dict
    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    
    def plot_traces(self, pred, true, figsize=(8, 8), num_traces=12, ncols=2, 
                    mode=None, norm=False, pred_logvar=None):
        '''
        Plot trace and compare to ground truth
        
        Arguments:
            - pred (np.array)   : array of predicted values to plot (dims: num_steps x num_cells)
            - true (np.array)   : array of true values to plot (dims: num_steps x num_cells)
            - figsize (2-tuple) : figure size (width, height) in inches (default = (8, 8))
            - num_traces (int)  : number of traces to plot (default = 24)
            - ncols (int)       : number of columns in figure (default = 2)
            - mode (string)     : mode to select subset of traces. Options: 'activity', 'rand', None.
                                  'Activity' plots the the num_traces/2 most active traces and num_traces/2
                                  least active traces defined sorted by mean value in trace
            - norm (bool)       : normalize predicted and actual values (default=True)
            - pred_logvar (np.array) : array of predicted values log-variance (dims: num_steps x num_cells) (default= None)
        
        '''
        
        num_cells = pred.shape[-1]
        
        nrows = int(num_traces / ncols)
        fig, axs = plt.subplots(
            figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True,
            gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
            )
        
        y = get_suptitle_y(fig)
        fig.suptitle('Trace reconstruction for a sampled trial.', y=y)
        
        if mode == 'rand':  
            idxs  = np.random.choice(
                list(range(num_cells)), size=num_traces, replace=False
                )
            idxs.sort()
                
        elif mode == 'activity':
            idxs = true.max(axis=0).argsort()[-num_traces:]
        
        else:
            idxs = list(range(num_cells))
        
        for j, ax in enumerate(axs.ravel()):
            # format
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if j % ncols == 0:
                ax.set_ylabel('Activity')
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            if (j - j % ncols) / ncols == (nrows - 1):
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if j < len(idxs):
                idx = idxs[j]
            else:
                continue

            if norm is True:
                true_plot = (
                    (true[:, idx] - np.mean(true[:, idx])) /
                    np.std(true[:, idx])
                    )
                pred_plot = (
                    (pred[:, idx] - np.mean(pred[:, idx])) /
                    np.std(pred[:, idx])
                    )
            else:
                true_plot = true[:, idx]
                pred_plot = pred[:, idx]

            ax.plot(
                self.time, true_plot, lw=2, 
                color=self.colors['linc_red'], label='Actual'
                )
            ax.plot(
                self.time, pred_plot, lw=2, 
                color=self.colors['linc_blue'], label='Reconstructed'
                )
                
        ax.legend()
        
        return fig
    
    #-------------------------------------------------------------------------
    def plot_video(self, pred, true, save_folder): #
        num_frames = true.shape[1]
        num_frames_pred = pred.shape[1]
        
        for t in range(num_frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            neg1 = ax1.imshow(pred[0, t]) 
            neg2 = ax2.imshow(true[0, t])
            neg1.set_clim(vmin=0, vmax=2)
            neg2.set_clim(vmin=0, vmax=2)
            fig.savefig(Path(save_folder, f'{t}.png'), bbox_inches="tight")
            plt.close(fig)

    
    #-------------------------------------------------------------------------
    
    def plot_factors(self, factors, max_in_col=5, figsize=(8, 8)):
        
        '''
        plot_factors(max_in_col=5, figsize=(8,8))
        
        Plot inferred factors in a grid
        
        Arguments:
            - max_in_col (int) : maximum number of subplots in a column
            - figsize (tuple of 2 ints) : figure size in inches
        Returns
            - figure
        '''
        
        _, factors_size = factors.shape
        
        nrows = min(max_in_col, factors_size)
        ncols = int(np.ceil(factors_size / max_in_col))
        
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, sharey=True, sharex=True,
            gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
            )
        
        y = get_suptitle_y(fig)
        fig.suptitle(f'Factors 1-{factors.shape[1]} for a sampled trial', y=y)

        fmin = factors.min()
        fmax = factors.max()
        
        for j, ax in enumerate(axs.ravel()):
            # format
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if j % ncols == 0:
                ax.set_ylabel('Activity')
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            if (j - j % ncols) / ncols == (nrows - 1):
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if j >= factors_size:
                continue
            
            ax.plot(self.time, factors[:, j])
            ax.set_ylim(fmin - 0.1, fmax + 0.1)
                
        return fig
    
    #-------------------------------------------------------------------------
    def plot_inputs(self, inputs, fig_width=8, fig_height=2):
        
        '''
        plot_inputs(fig_width=8, fig_height=2)
        
        Plot inferred inputs
        
        Arguments:
            - fig_width (int) : figure width in inches
            - fig_height (int) : figure height in inches
        '''

        _, inputs_size = inputs.shape
    
        figsize = (fig_width, fig_height * inputs_size)
        fig, axs = plt.subplots(
            nrows=inputs_size, figsize=figsize, squeeze=False, sharex=True
            )

        y = get_suptitle_y(fig)
        fig.suptitle('Input to the generator for a sampled trial', y=y)

        for j, ax in enumerate(axs.ravel()):
            # format
            if j == inputs_size - 1:
                ax.set_xlabel('Time (s)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.plot(self.time, inputs[:, j])

        return fig
    
