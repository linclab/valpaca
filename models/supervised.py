import logging
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)


##--------GENERIC SUPERVISED NET--------##

CONV_PARAMS = {
    'padding'    : (0, ),
    'kernel_size': (3, ),
    'dilation'   : (1, ),
    'stride'     : (1, ),
}

POOL_PARAMS = {
    'padding'       : (0, ),
    'kernel_size'   : (2, ),
    'dilation'      : (1, ),
    'stride'        : (2, ),
    'return_indices': False,
}

def calc_pos_weight(labels):
    if set(labels) != {0, 1}:
        raise NotImplementedError('Expected labels to only include {0, 1}.')

    labels = np.asarray(labels)
    pos_weight = [sum(labels != v) / sum(labels == v) for v in [0, 1]]

    return pos_weight


class Supervised_Net(nn.Module):
    def __init__(self):
        super(Supervised_Net, self).__init__()
    
    def _get_dataloader(self, x, y, batch_size=25):
        x = torch.Tensor(x).to(self.device)
        y = y.reshape(-1, 1)
        y = torch.Tensor(
            OneHotEncoder(sparse=False).fit(y).transform(y)
            ).to(self.device)
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        return dl


    def _run_epoch(self, train_dl, valid_dl, optimizer, criterion, it=0, 
                   epoch=0, best_valid_loss=np.inf, num_ep_since_best=0, 
                   save_loc=".", log_freq=0):
        
        self.train()
        running_loss = 0
        tic = time.perf_counter()
        for i, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            y_hat = self(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        train_epoch_loss = running_loss / (i + 1)

        self.eval()
        running_loss = 0
        for i, (x, y) in enumerate(valid_dl):
            with torch.no_grad():
                y_hat = self(x)
                loss = criterion(y_hat, y)
                running_loss += loss.data
        valid_epoch_loss = running_loss / (i + 1)
        toc = time.perf_counter()

        if valid_epoch_loss < best_valid_loss:
            num_ep_since_best = 0
            best_valid_loss = valid_epoch_loss
            torch.save({
                'net': self.state_dict(), 
                'opt': optimizer.state_dict()
                }, 
                Path(save_loc, f'best_{it}.pth'))
        else:
            num_ep_since_best += 1

        if log_freq > 0:
            if (epoch + 1) % log_freq == 0:
                logger.info(
                    f'Epoch {epoch + 1:4}: '
                    f'Epoch time={toc-tic:.3f}s, '
                    f'train loss={train_epoch_loss:.3f}, '
                    f'valid_loss={valid_epoch_loss:.3f}, '
                    f'{num_ep_since_best} epochs since best, '
                    f'best={best_valid_loss:.3f}'
                    )
        
        return best_valid_loss, num_ep_since_best


    def fit(self, x_train, y_train, x_valid, y_valid, batch_size=25, 
            learning_rate=0.0001, pos_weight='auto', log_freq=0, 
            max_epochs=1000, max_iter=1, save_loc=None, 
            model_name='supervised'):

        if isinstance(pos_weight, str) and pos_weight == 'auto':
            # infer from training data
            pos_weight = calc_pos_weight(y_train)
        
        save_loc = Path(save_loc, 'analysis')
        Path(save_loc).mkdir(exist_ok=True, parents=True)

        best_it_loss = np.inf
        for it in range(max_iter):
            train_dl = self._get_dataloader(
                x_train, y_train, batch_size=batch_size
                )
            valid_dl = self._get_dataloader(
                x_valid, y_valid, batch_size=batch_size
                )

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor(pos_weight)
                ).to(self.device)
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            
            best_valid_loss = np.inf
            num_ep_since_best = 0
            for epoch in range(max_epochs):
                best_valid_loss, num_ep_since_best = self._run_epoch(
                    train_dl, valid_dl, optimizer, criterion, it=it, 
                    epoch=epoch, best_valid_loss=best_valid_loss, 
                    num_ep_since_best=num_ep_since_best, save_loc=save_loc, 
                    log_freq=log_freq
                    )

                if num_ep_since_best >= 500:
                    break

            if best_valid_loss < best_it_loss:
                best_it_loss = best_valid_loss
                best_it = it
                
        state_dict = torch.load(
            Path(save_loc, f'best_{best_it}.pth'), 
            map_location=torch.device(self.device)
            )
        self.load_state_dict(state_dict['net'])
        
        if save_loc is not None:
            torch.save(state_dict, Path(save_loc, f'{model_name}.pth'))
        

    def predict(self, x):
        self.eval()
        if not torch.is_tensor(x):
            x = torch.Tensor(x).to(self.device)
        with torch.no_grad():
            y_hat = self(x)
        softmax = torch.nn.functional.softmax(
            y_hat, dim=1
            ).argmax(dim=1).cpu().numpy()

        return softmax
    

##--------SUPERVISED CONV NET--------##

def out_dim(in_dim, padding, dilation, kernel_size, stride):
    dim =  int(
        (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
    return dim

def layer_out_dim(in_dim, layer):
    padding = layer.padding
    kernel_size = layer.kernel_size
    dilation = layer.dilation
    stride = layer.stride

    dim = [
        out_dim(i, p, d, k, s) for (i, p, d, k, s) in 
        zip(in_dim, padding, dilation, kernel_size, stride)
    ]
    return dim


class _ConvNd_Block(nn.ModuleList):
    def __init__(self, input_dims):
        super(_ConvNd_Block, self).__init__()
        
        self.input_dims = input_dims
        
    def forward(self, x):
        ind = None
        for layer in self:
            if (layer.return_indices and 
                nn.modules.pooling._MaxPoolNd in type(layer).__bases__):
                x, ind = layer(x)
            else:
                x = layer(x)

        return x, ind
    
    def get_output_dims(self):
        dims = self.input_dims
        for m in self:
            parents = type(m).__bases__
            if (nn.modules.conv._ConvNd in parents or 
                nn.modules.pooling._MaxPoolNd in parents):
                dims = layer_out_dim(dims, m)

        return dims
    

class Conv_Block(_ConvNd_Block):
    def __init__(self, input_dims, in_channels, out_channels, 
                 conv_params=CONV_PARAMS, pool_params=POOL_PARAMS, 
                 negative_slope=0.1, dropout=0.5):

        super(Conv_Block, self).__init__(input_dims)
        self.add_module('conv', nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, **conv_params
            ))
        self.add_module('pool', nn.MaxPool1d(**pool_params))
        self.add_module('nonlin', nn.LeakyReLU(negative_slope=negative_slope))
        self.add_module('dropout', nn.Dropout(dropout))


class Supervised_Conv1d_Net(Supervised_Net):
    def __init__(self, input_dims=(96, 45), channel_dims=(32, 8, 1),
                 conv_params=CONV_PARAMS, pool_params=POOL_PARAMS,
                 dropout=0.5, dense_features=1, device='cpu'):
    
        super(Supervised_Conv1d_Net, self).__init__()
        self.conv_blocks = nn.ModuleList()
        in_channels = input_dims[0]
        layer_dims = (input_dims[1],)
        for out_channels in channel_dims:
            self.conv_blocks.append(
                Conv_Block(
                    input_dims=layer_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_params=conv_params,
                    pool_params=pool_params,
                    dropout=dropout
                    )
                )
            in_channels = out_channels
            layer_dims = self.conv_blocks[-1].get_output_dims()
            
        self.dense = nn.Linear(
            in_features=layer_dims[0], out_features=dense_features
            )
        self.device = device
        self.to(self.device)
        
    def forward(self, x):
        x = (x,)
        for conv_block in self.conv_blocks:
            x = conv_block(x[0])
        return self.dense(x[0]).squeeze()
    

##--------SUPERVISED RECURRENT NET--------##

class Supervised_BiRecurrent_Net(Supervised_Net):
    def __init__(self, input_size=100, hidden_size=16, dense_size=2, 
                 num_layers=1, bias=True, dropout=0.05, device='cpu'):
        super(Supervised_BiRecurrent_Net, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, 
            num_layers=num_layers, bidirectional=True, bias=bias
            )
        self.dense = nn.Linear(
            in_features=hidden_size*2, out_features=dense_size
            )
        self.hidden_state_init = nn.Parameter(torch.zeros(2, hidden_size))
        self.device=device
        self.to(self.device)
        
    def forward(self, x):
        x = x.permute(1, 0, 2)
        hidden_state = self.initialize_hidden_states(x)
        rnn_out, rnn_hidden = self.rnn(self.dropout(x), hidden_state)
        rnn_hidden = rnn_hidden.permute(1, 0, 2)
        rnn_hidden = rnn_hidden.reshape(
            rnn_hidden.shape[0], 
            rnn_hidden.shape[1] * rnn_hidden.shape[2]
            )
        return self.dense(self.dropout(rnn_hidden)).squeeze()
        
    def initialize_hidden_states(self, x):
        self.steps_size, self.batch_size, input_size = x.shape
        self.num_directions, self.hidden_size = self.hidden_state_init.shape
        hidden_state = (torch.ones(
            self.batch_size, self.num_directions,  self.hidden_size, 
            device=self.device) * self.hidden_state_init
            ).permute(1, 0, 2)
        return hidden_state.contiguous()

