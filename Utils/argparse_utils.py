import argparse


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def args_to_file(path, args, ):
    with open(path, 'w') as target_file:
        for arg in vars(args):
            target_file.write('--{} {}\n'.format(arg, getattr(args, arg)))


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue
