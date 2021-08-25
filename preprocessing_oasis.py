import numpy as np
import oasis
import argparse
import os

from utils import read_data, write_data

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-o', '--output', default=None, type=str)
parser.add_argument('-t', '--tau', default=0.1, type=float)
parser.add_argument('-s', '--scale', default=1.0, type=float)
parser.add_argument('-k', '--known', action='store_true', default=False)
parser.add_argument('-f', '--flatten', action='store_true', default=False)
parser.add_argument('-n', '--normalize', action='store_true', default=False)
parser.add_argument('-z', '--undo_train_test_split', action='store_true', default=False)
parser.add_argument('--data_suffix', default='fluor', type=str)

def main():
    args = parser.parse_args()
    data_name = os.path.basename(args.data_path)
    if args.output is None:
        dir_name = os.path.dirname(args.data_path)
    else:
        dir_name = args.output
    
    data_dict = read_data(args.data_path)
    dt = data_dict['dt']
    g = np.exp(-dt/args.tau)
    
    train_size, steps_size, state_size = data_dict['train_%s'%(args.data_suffix)].shape
    valid_size, steps_size, state_size = data_dict['valid_%s'%(args.data_suffix)].shape
    data_size = train_size + valid_size
    data = np.zeros((data_size, steps_size, state_size))
    
    if args.undo_train_test_split:
        train_idx = data_dict['train_idx']
        valid_idx = data_dict['valid_idx']
        data[train_idx] = data_dict['train_%s'%(args.data_suffix)]
        data[valid_idx] = data_dict['valid_%s'%(args.data_suffix)]
        
    else:
        data[:train_size] = data_dict['train_%s'%(args.data_suffix)]
        data[train_size:] = data_dict['valid_%s'%(args.data_suffix)]
    
    if args.flatten:
        data = data.reshape(data_size * steps_size, state_size).transpose()
    else:
        data = data.transpose(0, 2, 1)
        data = data.reshape(data_size * state_size, steps_size)
        data = np.hstack((np.zeros((data_size * state_size, 1)), data))

    if args.known:
        S, C = deconvolve_calcium_known(data, 
                                        g=g, 
                                        s_min=args.scale/2)
    else:
        if args.normalize:
            data = max_normalize(data.T, axis=0).T
        S, C, bias, G, gain, rval = deconvolve_calcium_unknown(data,
                                                               g=g,
                                                               snr_thresh=args.scale)
        
#         tau = -dt/(np.log(G))
        
    if args.flatten:
        data = data.reshape(state_size, data_size, steps_size).transpose(1, 2, 0)
        S = S.reshape(state_size, data_size, steps_size).transpose(1, 2, 0)
        C = C.reshape(state_size, data_size, steps_size).transpose(1, 2, 0)
        
    else:
        data = data.reshape(data_size, state_size, steps_size+1).transpose(0, 2, 1)[:, 1:]
        S = S.reshape(data_size, state_size, steps_size+1).transpose(0, 2, 1)[:, 1:]
        C = C.reshape(data_size, state_size, steps_size+1).transpose(0, 2, 1)[:, 1:]
        
        if not args.known:
            bias = np.median(bias.reshape(data_size, state_size), axis=0)
            tau = -dt/(np.log(np.median(G.reshape(data_size, state_size), axis=0)))
            gain = np.median(gain.reshape(data_size, state_size), axis=0)
            print(bias.mean(), tau.mean(), gain.mean())
            
    if args.undo_train_test_split:
        train_fluor = data[train_idx]
        valid_fluor = data[valid_idx]
        train_ospikes  = S[train_idx]
        valid_ospikes  = S[valid_idx]
        train_ocalcium = C[train_idx]
        valid_ocalcium = C[valid_idx]
        
    else:
        train_fluor = data[:train_size]
        valid_fluor = data[train_size:]
        train_ospikes  = S[:train_size]
        valid_ospikes  = S[train_size:]
        train_ocalcium = C[:train_size]
        valid_ocalcium = C[train_size:]
        
    data_dict['train_%s'%(args.data_suffix)] = train_fluor
    data_dict['valid_%s'%(args.data_suffix)] = valid_fluor
    
    data_dict['train_%s_ospikes'%(args.data_suffix)] = train_ospikes
    data_dict['valid_%s_ospikes'%(args.data_suffix)] = valid_ospikes
    
    data_dict['train_%s_ocalcium'%(args.data_suffix)] = train_ocalcium
    data_dict['valid_%s_ocalcium'%(args.data_suffix)] = valid_ocalcium
    
    if not args.known:
        data_dict['obs_gain_init'] = gain
        data_dict['obs_bias_init'] = bias
        data_dict['obs_tau_init'] = tau
        data_dict['obs_var_init'] = (gain/args.scale)**2
    
    arg_string =  '_o%s'%('k' if args.known else 'u')
    arg_string += '_t%s'%(str(args.tau))
    arg_string += '_s%s'%(str(args.scale))
    arg_string += '_f' if args.flatten else ''
    arg_string += '_z' if args.undo_train_test_split else ''
    arg_string += '_n' if args.normalize else ''
    
    write_data(os.path.join(dir_name, data_name + '_%s'%args.data_suffix + arg_string), data_dict)

def deconvolve_calcium_known(X, g=0.9, s_min=0.5):
    S = np.zeros_like(X)
    C = np.zeros_like(X)
    for ix, x in enumerate(X):
            c,s = oasis.functions.oasisAR1(x, g=g, s_min=0.5)
            S[ix] = s.round()
            C[ix] = c
    return S, C

def deconvolve_calcium_unknown(X, g=0.9, snr_thresh=3):
    '''
    Deconvolve calcium traces to spikes
    '''
    
    S = np.zeros_like(X)
    C = np.zeros_like(X)
    
    B = []
    G = []
    L = []
    M = []
    R = []
    
    b_init = compute_mode(X)
    
    for ix, x in enumerate(X):
        c, s, b, g, lam = oasis.functions.deconvolve(x, b=0, penalty=1, max_iter=5, optimize_g=1)
        sn = (x-c).std(ddof=1)
        c, s, b, g, lam = oasis.functions.deconvolve(x, b=b, penalty=1, sn=sn, max_iter=5, g=[g])
        sn = (x-c).std(ddof=1)
        c, s = oasis.oasis_methods.oasisAR1(x-b, g=g, lam=lam, s_min=sn*snr_thresh)
        r = np.corrcoef(c, x)[0, 1]
        
        S[ix] = np.round(s/(sn*snr_thresh))
        C[ix] = c
        
        B.append(b)
        G.append(g)
        L.append(lam)
        M.append(sn * snr_thresh)
        R.append(r)
            
    B = np.array(B)
    G = np.array(G) 
    L = np.array(L)
    M = np.array(M)
    R = np.array(R)
    
    return S, C, B, G, M, R

def max_normalize(X, axis=0):
    X = X - compute_mode(X)
    return X/X.max()

def compute_mode(X):
    h, b  = np.histogram(X.ravel(), bins='auto')
    xvals = (b[1:] + b[:-1])/2
    return xvals[h.argmax()]

if __name__ == '__main__':
    main()
    