#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
import sys

import yaml

sys.path.extend(['.', '..'])
from data import synthetic_data
from utils import utils


logger = logging.getLogger(__name__)


def main(args):

    # set logger to the specified level
    utils.set_logger_level(logger, level=args.log_level)

    filename = (
        f'lorenz_seed{args.seed}_'
        f'sys{args.dt_sys}_'
        f'cal{args.dt_spike}_'
        f'sig{args.dt_spike}_'
        f'base{args.rate_scale}'
    )
    filepath = Path(args.output_dir, filename)

    if filepath.is_file():
        logger.warning(f'{filepath} already exists.')
        return

    if args.config:
        with open(args.config, 'r') as f:
            params_dict = yaml.load(f, Loader=yaml.FullLoader)
        for key, val in params_dict.items():
            if getattr(args, key) is None:
                args.__setattr__(key, val)
            logger.info(f'{key}: {args.__getattribute__(key)}')

    from synthetic_data import LorenzSystem, EmbeddedLowDNetwork

    lorenz = LorenzSystem(num_inits=args.inits, dt=args.dt_sys)

    net = EmbeddedLowDNetwork(
        low_d_system=lorenz,
        net_size=args.cells,
        base_rate=args.rate_scale,
        dt=args.dt_sys
        )

    # generate data
    generator = synthetic_data.SyntheticCalciumDataGenerator(
        system     = net,
        seed       = args.seed,
        trainp     = args.trainp,
        burn_steps = args.burn_steps,
        num_steps  = args.steps,
        num_trials = args.trials,
        tau_cal    = 0.3,
        dt_cal     = args.dt_spike,
        n          = 2.0,
        A          = 1.0,
        gamma      = 0.01,
        sigma      = args.sigma
        )

    data_dict = generator.generate_dataset()

    Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    logger.info(f'Saving synthetic data to {filepath}.')
    utils.write_data(filepath, data_dict)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_dir', type=Path)

    # optional parameters
    parser.add_argument('-c', '--config', default=None, type=Path, 
                        help='path to hyperparameters')
    parser.add_argument('-s', '--seed', default=100, type=int)

    parser.add_argument('--trials', type=int)
    parser.add_argument('--inits', type=int)
    parser.add_argument('--cells', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--rate_scale', type=float)
    parser.add_argument('--trainp', type=float)
    parser.add_argument('--dt_spike', type=float)
    parser.add_argument('--dt_sys', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--burn_steps', type=int)
    parser.add_argument('--log_level', default='info', 
                        help='log level, e.g. debug, info, error')

    args = parser.parse_args()

    logger = utils.get_logger_with_basic_format(level=args.log_level)

    main(args)

