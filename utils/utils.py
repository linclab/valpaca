import os

import h5py
import json
import numpy as np
import pickle
import torch
import yaml

def write_data(data_fname, data_dict, use_json=False, compression=None):
    """Write data in HD5F format.

    Args:
    data_fname: The filename of teh file in which to write the data.
    data_dict:  The dictionary of data to write. The keys are strings
      and the values are numpy arrays.
    use_json (optional): human readable format for simple items
    compression (optional): The compression to use for h5py (disabled by
      default because the library borks on scalars, otherwise try 'gzip').
    """

    dir_name = os.path.dirname(data_fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if use_json:
        the_file = open(data_fname,'w')
        json.dump(data_dict, the_file)
        the_file.close()
    else:
        try:
            with h5py.File(data_fname, 'w') as hf:
                for k, v in data_dict.items():
                    if type(k) is str:
                        clean_k = k.replace('/', '_')
                        if clean_k is not k:
                            print('Warning: saving variable with name: ', k, ' as ', clean_k)
                        else:
                            print('Saving variable with name: ', clean_k)
                    else:
                        clean_k = k

                    hf.create_dataset(clean_k, data=v, compression=compression)
        except IOError:
            print("Cannot open %s for writing.", data_fname)
                    

def read_data(data_fname):
    
    """ Read saved data in HDF5 format.

    Args:
        data_fname: The filename of the file from which to read the data.
    Returns:
        A dictionary whose keys will vary depending on dataset (but should
        always contain the keys 'train_data' and 'valid_data') and whose
        values are numpy arrays.
    """
    try:
        with h5py.File(data_fname, 'r') as hf:
            data_dict = {k: np.array(v) for k, v in hf.items()}
            return data_dict
    except IOError:
        print("Cannot open %s for reading." % data_fname)
        raise
        
def batchify_sample(sample, batch_size):
    if not torch.is_tensor(sample):
        sample = torch.Tensor(sample)
    batch = sample.unsqueeze(0).repeat(batch_size, *(1,)*(sample.ndimension()))
    return batch
        
def batchify_random_sample(data, batch_size, ix=None):
    
    """
    Randomly select sample from data, and turn into a batch of size batch_size to generate multiple samples
    from model to average over
    
    Args:
        data (torch.Tensor) : dataset to randomly select sample from
        batch_size (int) : number of sample repeats
    Optional:
        ix (int) : index to select sample. Randomly generated if None
    
    Returns:
        batch (torch.Tensor) : sample repeated batch
        ix (int) : index of repeated sample
    """
    
    num_trials = data.shape[0]
    if type(ix) is not int:
        ix = np.random.randint(num_trials)
    sample = data[ix]
    batch = batchify_sample(sample, batch_size)
    return batch, ix

def update_param_dict(prev_params, new_params):
    params = prev_params.copy()
    for k in prev_params.keys():
        if k in new_params.keys():
            params[k] = new_params[k]
    return params

def load_parameters(path):
    return yaml.load(open(path), Loader=yaml.FullLoader)

def save_parameters(save_loc, params):
    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)
    yaml.dump(params, open(os.path.join(save_loc, 'hyperparameters.yaml'), 'w'), default_flow_style=False)

def load_latent(model_dir):
    latent_filename = os.path.join(model_dir, 'latent.pkl')
    if not os.path.exists(latent_filename):
        raise ValueError(
            f'latent.pkl file not found in {model_dir}. Run infer_latent.py'
            )
    with open(latent_filename, 'rb') as f:
        latent_dict = pickle.load(f)
    return latent_dict


def seed_all(seed):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # for cuda > 10.2
