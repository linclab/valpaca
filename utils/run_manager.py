import logging
from pathlib import Path
import time

import torch


logger = logging.getLogger(__name__)


class RunManager():
    def __init__(self, model, objective, optimizer, scheduler, train_dl, 
                 valid_dl, plotter=None, writer=None, do_health_check=False, 
                 detect_local_minimum=False, max_epochs=1000, save_loc='/tmp/', 
                 load_checkpoint=False):
    
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model      = model
        self.objective  = objective
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.train_dl   = train_dl
        self.valid_dl   = valid_dl
        self.writer     = writer
        self.plotter    = plotter
            
        self.max_epochs      = max_epochs
        self.do_health_check = do_health_check
        self.detect_local_minimum = detect_local_minimum
        self.save_loc        = save_loc
        
        self.epoch = 0
        self.step  = 0
        self.best  = float('inf')
        
        self.loss_dict = {'train' : dict(),
                          'valid' : dict(),
                          'l2'    : []
                          }
        
        if load_checkpoint:
            self.load_checkpoint('recent')
            
    def run(self):
        for epoch in range(self.epoch, self.max_epochs):
            if self.optimizer.param_groups[0]['lr'] < self.scheduler.min_lrs[0]:
                break
            self.epoch = epoch + 1
            tic = time.perf_counter()
            loss_dict_list = []
            
            self.model.train()
            logger.debug(f'Length of training dataloader: {len(self.train_dl)}')
            for _, x in enumerate(self.train_dl):
                x = x[0]
                
                self.optimizer.zero_grad()

                fwd_tic = time.perf_counter()
                recon, latent = self.model(x)
                fwd_time = time.perf_counter() - fwd_tic
                logger.debug(f'Fwd time: {fwd_time}')

                loss_tic = time.perf_counter()
                loss, loss_dict = self.objective(
                    x_orig=x, x_recon=recon, model=self.model
                    )
                loss_time = time.perf_counter() - loss_tic
                logger.debug(f'Loss time: {loss_time}')
                loss_dict_list.append(loss_dict)
                
                bwd_tic = time.perf_counter()
                loss.backward()
                bwd_time = time.perf_counter() - bwd_tic
                logger.debug(f'Bwd time: {bwd_time}')

                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.model.max_norm
                    )

                # update the weights
                self.optimizer.step()

                self.objective.weight_schedule_fn(self.step)
                
                if self.model.do_normalize_factors:
                    # Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
                    self.model.normalize_factors()
                    
                self.optimizer, self.scheduler = \
                    self.model.change_parameter_grad_status(
                        self.step, self.optimizer, self.scheduler
                        )
                
                self.step += 1
            
            loss_dict = dict() 
            
            for d in loss_dict_list: 
                for k in d.keys(): 
                    loss_dict[k] = loss_dict.get(k, 0) + d[k] / len(loss_dict_list)
            for key, val in loss_dict.items():
                if key in self.loss_dict['train'].keys():
                    self.loss_dict['train'][key].append(loss_dict[key])
                elif key == 'l2':
                    self.loss_dict[key].append(loss_dict[key])
                else:
                    self.loss_dict['train'][key] = [loss_dict[key]]

            self.scheduler.step(self.loss_dict['train']['total'][-1])
            loss_dict_list = []
            self.model.eval()
            logger.debug(f'Length of validation dataloader: {len(self.valid_dl)}')
            for _, x in enumerate(self.valid_dl):
                x = x[0]
                with torch.no_grad():
                    fwd_val_tic = time.perf_counter()
                    recon, latent = self.model(x)
                    fwd_val_time = time.perf_counter() - fwd_val_tic
                    logger.debug(f'Valid fwd time: {fwd_val_time}')

                    loss, loss_dict = self.objective(
                        x_orig=x, x_recon=recon, model=self.model
                        )
                    loss_dict_list.append(loss_dict)
                    
            loss_dict = dict() 
            for d in loss_dict_list: 
                for k in d.keys(): 
                    loss_dict[k] = loss_dict.get(k, 0) + d[k] / len(loss_dict_list)

            for key, val in loss_dict.items():
                if key in self.loss_dict['valid'].keys():
                    self.loss_dict['valid'][key].append(loss_dict[key])
                elif key == 'l2':
                    pass
                else:
                    self.loss_dict['valid'][key] = [loss_dict[key]]
                    
            if not self.objective.any_zero_weights():
                # recalculate total validation loss
                adj_valid_loss = 0
                for key, val in self.loss_dict['valid'].items():
                    if 'recon' in key:
                        adj_valid_loss += val[-1]
                    if 'kl' in key:
                        full_kl_val = val[-1] / self.objective.loss_weights[key]['weight']
                        adj_valid_loss += full_kl_val

                if adj_valid_loss < self.best:
                    self.save_checkpoint('best')
                    self.best = adj_valid_loss
                
            self.save_checkpoint()
            if self.writer is not None:
                self.write_to_tensorboard()
                if self.plotter is not None:
                    if self.epoch % 25 == 0:
                        self.plot_to_tensorboard()
                        
                if self.do_health_check:
                    self.health_check(self.model)
                    
            toc = time.perf_counter()
            
            results_str = (
                f'Epoch {self.epoch:5}, '
                f'Epoch time={toc - tic:.3f}s, '
                'Loss (train, valid):'
            )

            for key in self.loss_dict['train'].keys():
                train_loss = self.loss_dict['train'][key][self.epoch-1]
                valid_loss = self.loss_dict['valid'][key][self.epoch-1]
                results_str = \
                    f'{results_str} {key} ({train_loss:.3f}, {valid_loss:.3f}),'
            
            l2_loss = self.loss_dict['l2'][self.epoch - 1]
            results_str = f'{results_str} l2 ({l2_loss:.3f})'
            if not self.objective.any_zero_weights():
                results_str = (f'{results_str}, '
                    f'adj. total valid loss ({adj_valid_loss:.4f})')
            
            logger.info(results_str)
            
            # Check if local minimum is reached with 0 KL or L2 loss
            if self.detect_local_minimum:
                in_local_minimum = False
                if not self.objective.any_zero_weights():
                    for key, val in self.loss_dict['valid'].items():
                        if ('kl' in key or 'l2' in key):
                            if torch._np.abs(val[-1] / self.objective.loss_weights[key]['weight']) < 0.1:
                                in_local_minimum = True
                        else:
                            if torch._np.abs(val[-1]) < 0.1:
                                in_local_minimum = True

                if in_local_minimum:
                    logger.warning('Stuck in a local minimum.')
                    break


    def write_to_tensorboard(self):
        
        # Write loss to tensorboard
        for ix, key in enumerate(self.loss_dict['train'].keys()):
            train_loss = self.loss_dict['train'][key][self.epoch-1]
            valid_loss = self.loss_dict['valid'][key][self.epoch-1]
            
            loss_dict = {
                'Training' : float(train_loss),
                'Validation' : float(valid_loss)
                }

            self.writer.add_scalars(
                f'1_Loss/{ix+1}_{key}', loss_dict, self.epoch
                 )

        l2_loss = self.loss_dict['l2'][self.epoch - 1]
        self.writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss), self.epoch)

        for jx, grp in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'2_Optimizer/1.{jx + 1}_Learning_Rate_PG', grp['lr'], 
                self.epoch
                )
        
        for kx, key in enumerate(self.objective.loss_weights.keys()):
            weight = self.objective.loss_weights[key]['weight']
            self.writer.add_scalar(
                f'2_Optimizer/2.{ix+1}_{key}_weight', weight, self.epoch
                )
        
    def plot_to_tensorboard(self):
        figs_dict_train = self.plotter['train'].plot_summary(
            model=self.model, dl=self.train_dl
            )
        
        figs_dict_valid = self.plotter['valid'].plot_summary(
            model=self.model, dl=self.valid_dl
            )
        
        fig_names = ['traces', 'inputs', 'factors', 'rates', 'spikes']
        for fn in fig_names:
            if fn in figs_dict_train.keys():
                self.writer.add_figure(
                    f'{fn}/train', figs_dict_train[fn], self.epoch, close=True
                    )
            elif f'grdtruth_{fn}' in figs_dict_train.keys():
                self.writer.add_figure(
                    f'{fn}/train', figs_dict_train[f'grdtruth_{fn}'], 
                    self.epoch, close=True
                    )

            if fn in figs_dict_valid.keys():
                self.writer.add_figure(
                    f'{fn}/valid', figs_dict_valid[fn], self.epoch, close=True
                    )
            elif f'grdtruth_{fn}' in figs_dict_valid.keys():
                self.writer.add_figure(
                    f'{fn}/valid', figs_dict_valid[f'grdtruth_{fn}'], 
                    self.epoch, close=True
                    )
            
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------

    def health_check(self, model):
        '''
        Gets gradient norms for each parameter and writes to tensorboard
        '''
        
        for ix, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                self.writer.add_scalar(
                    f'3_Gradient_norms/{ix}_{name}', param.grad.data.norm(), 
                    self.epoch
                    )
            else:
                self.writer.add_scalar(
                    f'3_Gradient_norms/{ix}_{name}', 0.0, self.epoch
                    )
                
            if 'weight' in name:
                self.writer.add_scalar(
                    f'4_Weight_norms/{ix}_{name}', param.data.norm(), 
                    self.epoch
                    )
        
    def save_checkpoint(self, output_filename='recent'):
        # Create dictionary of training variables
        train_dict = {
            'best'        : self.best,
            'loss_dict'   : self.loss_dict,
            'loss_weights': self.objective.loss_weights,
            'epoch'       : self.epoch, 
            'step'        : self.step,
            }
        
        # Save network parameters, optimizer state, and training variables
        checkpoint_direc = Path(self.save_loc, 'checkpoints')
        checkpoint_direc.mkdir(exist_ok=True, parents=True)
        
        model_dict = {
            'net'        : self.model.state_dict(), 
            'opt'        : self.optimizer.state_dict(),
            'sched'      : self.scheduler.state_dict(), 
            'run_manager': train_dict,
        }

        torch.save(model_dict, Path(checkpoint_direc, f'{output_filename}.pth'))
        
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
        
    def load_checkpoint(self, input_filename='recent'):
        input_path = Path(self.save_loc, 'checkpoints', f'{input_filename}.pth')
        if input_path.is_file():
            state_dict = torch.load(
                input_path, map_location=torch.device(self.device)
                )
            self.model.load_state_dict(state_dict['net'])

            epoch = state_dict['run_manager']['epoch']
            step = state_dict['run_manager']['step']
            logger.info(f"Loaded checkpoint: epoch {epoch}, step {step}.")
            if len(state_dict['opt']['param_groups']) > 1:
                self.optimizer, self.scheduler = \
                    self.model.change_parameter_grad_status(
                        state_dict['run_manager']['step'], 
                        self.optimizer, 
                        self.scheduler, 
                        loading_checkpoint=True
                        )
            self.optimizer.load_state_dict(state_dict['opt'])
            self.scheduler.load_state_dict(state_dict['sched'])

            self.best = state_dict['run_manager']['best']
            self.loss_dict = state_dict['run_manager']['loss_dict']
            self.objective.loss_weights = \
                state_dict['run_manager']['loss_weights']
            self.epoch = state_dict['run_manager']['epoch']
            self.step  = state_dict['run_manager']['step']
            
