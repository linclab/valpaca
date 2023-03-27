import logging
from pathlib import Path
import logging
import os
import sys

import h5py
import json
import numpy as np
import pickle as pkl
import torch
import yaml


logger = logging.getLogger(__name__)


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

    Path(data_fname).parent.mkdir(exist_ok=True, parents=True)

    if use_json:
        with open(data_fname, 'w') as f:
            json.dump(data_dict, f)
    else:
        try:
            with h5py.File(data_fname, 'w') as hf:
                for k, v in data_dict.items():
                    if isinstance(k, str):
                        clean_k = k.replace('/', '_')
                        if clean_k != k:
                            logger.warning(
                                f'Saving variable {k} with name {clean_k}.'
                                )
                    else:
                        clean_k = k

                    hf.create_dataset(clean_k, data=v, compression=compression)
        except OSError as err:
            raise OSError(f'Cannot open {data_fname} for writing due to: {err}.')
                    

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
    except OSError as err:
        raise OSError(f'Cannot open {data_fname} for reading due to: {err}.')


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

def load_parameters(hyperpath):
    with open(hyperpath, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return params

def save_parameters(save_loc, params):
    Path(save_loc).mkdir(exist_ok=True, parents=True)
    with open(Path(save_loc, 'hyperparameters.yaml'), 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

def load_latent(model_dir):
    if len(Path(model_dir).suffix):
        latent_filename = Path(model_dir)
    else:
        latent_filename = Path(model_dir, 'latent.pkl')

    if not latent_filename.is_file:
        raise ValueError(
            f'latent.pkl file not found in {model_dir}. Run infer_latent.py'
            )

    with open(latent_filename, 'rb') as f:
        latent_dict = pkl.load(f)

    return latent_dict

def seed_all(seed):

    if seed is not None:
        logger.debug(f'Random state seed: {seed}')

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # for cuda > 10.2


class BasicLogFormatter(logging.Formatter):
    """
    BasicLogFormatter()

    Basic formatting class that formats different level logs differently. 
    Allows a spacing extra argument to add space at the beginning of the log.
    """

    dbg_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(spacing)s%(msg)s"
    wrn_fmt  = "%(spacing)s%(levelname)s: %(msg)s"
    err_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    crt_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(spacing)s%(levelname)s: %(msg)s"):
        """
        Optional args:
            - fmt (str): default format style.
        """
        super().__init__(fmt=fmt, datefmt=None, style="%") 

    def format(self, record):

        if not hasattr(record, "spacing"):
            record.spacing = ""

        # Original format as default
        format_orig = self._style._fmt

        # Replace default as needed
        if record.levelno == logging.DEBUG:
            self._style._fmt = BasicLogFormatter.dbg_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = BasicLogFormatter.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = BasicLogFormatter.wrn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = BasicLogFormatter.err_fmt
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = BasicLogFormatter.crt_fmt

        # Call the original formatter class to do the grunt work
        formatted_log = logging.Formatter.format(self, record)

        # Restore default format
        self._style._fmt = format_orig

        return formatted_log


def set_logger_level(logger, level="info"):

    if str(level).isdigit():
        logger.setLevel(int(level))
    elif isinstance(level, str) and hasattr(logging, level.upper()):
        logger.setLevel(getattr(logging, level.upper()))
    else:
        raise ValueError(f'{level} logging level not recognized.')


def get_logger(name=None, level="info", fmt=None, skip_exists=True):
    """
    get_logger()

    Returns logger. 

    Optional args:
        - name (str)        : logger name. If None, the root logger is returned.
                              default: None
        - level (str)       : level of the logger ("info", "error", "warning", 
                               "debug", "critical")
                              default: "info"
        - fmt (Formatter)   : logging Formatter to use for the handlers
                              default: None
        - skip_exists (bool): if a logger with the name already has handlers, 
                              does nothing and returns existing logger
                              default: True

    Returns:
        - logger (Logger): logger object
    """

    # create one instance
    logger = logging.getLogger(name)

    if not skip_exists:
        logger.handlers = []

    # skip if logger already has handlers
    add_handler = True
    for hd in logger.handlers:
        if isinstance(hd, logging.StreamHandler):
            add_handler = False
            if fmt is not None:
                hd.setFormatter(fmt)

    if add_handler:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    set_logger_level(logger, level)

    return logger


def get_logger_with_basic_format(**logger_kw):
    """
    get_logger_with_basic_format()

    Returns logger with basic formatting, defined by BasicLogFormatter class.

    Keyword args:
        - logger_kw (dict): keyword arguments for get_logger()
        
    Returns:
        - logger (Logger): logger object
    """


    basic_formatter = BasicLogFormatter()

    logger = get_logger(fmt=basic_formatter, **logger_kw)

    return logger
