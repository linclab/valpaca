#!/usr/bin/env python

import argparse
import os
import sys

import yaml

sys.path.extend(['.', '..'])
from data import synthetic_data
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='./', type=str)
parser.add_argument('-s', '--seed', default=100, type=int)
parser.add_argument('-p', '--parameters', type=str)
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

def main():
    args = parser.parse_args()
    
    if os.path.exists('%s/lorenz_seed%s_sys%s_cal%s_sig%s_base%s'%(args.output,
                                                                   str(args.seed),
                                                                   str(args.dt_sys),
                                                                   str(args.dt_spike),
                                                                   str(args.sigma),
                                                                   str(args.rate_scale))):
        pass
    
    else:
        if args.parameters:
            params_dict = yaml.load(open(args.parameters), Loader=yaml.FullLoader)
            for key, val in params_dict.items():
                if getattr(args, key) is None:
                    args.__setattr__(key, val)
                print('%s : %s'%(key, str(args.__getattribute__(key))), flush=True)

        from synthetic_data import LorenzSystem, EmbeddedLowDNetwork

        lorenz = LorenzSystem(num_inits= args.inits,
                              dt= args.dt_sys)

        net = EmbeddedLowDNetwork(low_d_system = lorenz,
                                  net_size = args.cells,
                                  base_rate = args.rate_scale,
                                  dt = args.dt_sys)

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
            sigma      = args.sigma)

        data_dict = generator.generate_dataset()
        # save

        print('Saving to %s/lorenz_seed%s_sys%s_cal%s_sig%s_base%s'%(args.output,
                                                                     str(args.seed),
                                                                     str(args.dt_sys),
                                                                     str(args.dt_spike),
                                                                     str(args.sigma),
                                                                     str(args.rate_scale)), flush=True)
        utils.write_data('%s/lorenz_seed%s_sys%s_cal%s_sig%s_base%s'%(args.output,
                                                                      str(args.seed),
                                                                      str(args.dt_sys),
                                                                      str(args.dt_spike),
                                                                      str(args.sigma),
                                                                      str(args.rate_scale)), data_dict)
    
if __name__ == '__main__':
    main()