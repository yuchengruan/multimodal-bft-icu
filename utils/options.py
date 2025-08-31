import argparse
import yaml

def get_hparams(conf_path):
    parser = argparse.ArgumentParser()

    # all arguments
    parser = get_hparams_from_yaml(parser, conf_path)

    args = parser.parse_args()

    return args


def get_hparams_from_yaml(parser, conf_path):

    with open(conf_path, "r") as file:
        configs = yaml.safe_load(file)

    for key, val in configs.items():
        if type(val) is bool:
            parser.add_argument(f'--{key}',
                            action="store_true",
                            default=val)
        else:
            parser.add_argument(f'--{key}',
                            type=type(val),
                            default=val)
    
    return parser

def add_hparams2parser(hparams_dict):
    parser = argparse.ArgumentParser()

    # all arguments
    for key, val in hparams_dict.items():
        if type(val) is bool:
            parser.add_argument(f'--{key}',
                            action="store_true",
                            default=val)
        else:
            parser.add_argument(f'--{key}',
                            type=type(val),
                            default=val)
            
    args = parser.parse_args()

    return args

def add_hparams2parser_jn(hparams_dict):
    parser = argparse.ArgumentParser()

    # all arguments
    for key, val in hparams_dict.items():
        if type(val) is bool:
            parser.add_argument(f'--{key}',
                            action="store_true",
                            default=val)
        else:
            parser.add_argument(f'--{key}',
                            type=type(val),
                            default=val)
            
    args = parser.parse_args(args = [])

    return args
