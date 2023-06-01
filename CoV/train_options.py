import argparse
from .base_options import boolstr


class TrainOptions:

    def __init__(self):
        pass

    def get_arguments(self, parser):
        # parser = BaseOptions.get_arguments(self, parser)
        parser.add_argument('--method', type=str, help='which method [static|cov-weighting|uncertainty|gradnorm|multi-objective] to run', default='cov-weighting')
        # parser.add_argument('--device',             type=str,   help='choose cpu or cuda:0 device"', default='cuda:0')
        parser.add_argument('--num_workers',        type=int,   help='number of threads to use for data loading', default=16)

        parser.add_argument('--epochs',             type=int,   help='number of total epochs to run', default=30)
        parser.add_argument('--learning_rate',      type=float, help='initial learning rate (default: 1e-4)', default=1e-4)
        parser.add_argument('--adjust_lr',          type=boolstr,  help='apply learning rate decay or not', default=True)
        parser.add_argument('--optimizer',          type=str,   help='Optimizer to use [adam|sgd|rmsprop]', default='adam')

        # Specific to CoV-Weighting
        parser.add_argument('--mean_sort',          type=str,   help='full or decay', default='full')
        parser.add_argument('--mean_decay_param',   type=float, help='What decay to use with mean decay', default=1.0)

        return parser

    @staticmethod
    def print_options(args):
        print('=== ARGUMENTS ===')
        for key, val in vars(args).items():
            print('{0: <20}: {1}'.format(key, val))
        print('=================')

    def parse(self):
        parser = argparse.ArgumentParser(description='CoV-Weighting PyTorch Implementation')
        parser = self.get_arguments(parser)

        args = parser.parse_args()

        # Print the options.
        self.print_options(args)

        return args

