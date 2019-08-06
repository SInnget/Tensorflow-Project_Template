import argparse
import json
import re

import commentjson
from tensorflow.python.client import device_lib


class InvalidClassNameError(Exception):
    pass


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c',
        '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def load_json(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


def load_commentjson(json_path):
    with open(json_path, 'r') as fp:
        return commentjson.loads(fp.read())


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    num_underline = 0
    if column_name.startswith('_'):
        raw_l = len(column_name)
        column_name = column_name.lstrip('_')
        num_underline = raw_l - len(column_name)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return f'{"_"* num_underline}{s1}'
