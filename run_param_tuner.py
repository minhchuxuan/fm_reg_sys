
# import sys
# sys.path.append("/mnt/public/lhh/code/")
import sys

sys.path.append("/home/lhh/code")
from datetime import datetime
import gc
import argparse
from fuxictr import autotuner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ECN_tuner_config_KKBox.yaml',
                        help='The config file for para tuning.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Use the tag to determine which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu', nargs='+', default=[0, 1, 2, 3, 2, 1, 0],
                        help='The list of gpu indexes, -1 for cpu.')
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    expid_tag = args['tag']

    # generate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'])
    autotuner.grid_search(config_dir, gpu_list, expid_tag)
